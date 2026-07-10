"""State-kNN datastore eval — the census recipe as parametrized architecture.

Keys each corpus position by its frozen-substrate CONTINUOUS-STREAM state
(``LinearStateStreamer``), stores the byte that followed, and at inference
retrieves nearest past states and votes their successors (``StateKNNMemory``),
mixed with the body's own windowed logits (kNN-LM). This is the round-3 census
recipe that hit 2.97 bpb kNN-alone and a verified negative kNN-LM lift, promoted
out of ``experiments/harnesses/knn_stream*.py`` — the three scripts collapse to
one config here, and every mechanism (FFT streaming, truncate-then-whiten,
tiled search, marginal backoff) now lives in the decepticons kernel.

The deliberate asymmetry the census fixed on: query KEYS are continuous-stream
states (unbounded history — what the substrate actually remembers), while the
kNN-LM BASE logits are the body's ordinary windowed forward pass (the model as
it deploys today). Store size is the campaign knob; store_device="cpu" lifts the
store into host RAM (~180M keys) past the ~16M VRAM ceiling.

Run:  python -m chronohorn.eval.knn_datastore --store-bytes 8000000
      python -m chronohorn.eval.knn_datastore --store-bytes 32000000 --store-device cpu
"""
from __future__ import annotations

import argparse
import functools
import time
from dataclasses import dataclass

import numpy as np
import torch

from decepticons.loader import load_checkpoint
from decepticons.models.state_knn_torch import (
    StateKNNConfig,
    StateKNNMemory,
    interpolate,
)
from decepticons.models.state_stream_torch import LinearStateStreamer

from .harness_util import bpb_from_logp, enwik8_bytes, paired_window_delta

DEFAULT_CHECKPOINT = (
    "/home/asuramaya/code/REPOS/chronohorn/out/results/"
    "fo-noadapt-enwik90-50k.checkpoint.pt"
)


@dataclass
class KNNDatastoreConfig:
    checkpoint: str = DEFAULT_CHECKPOINT
    store_bytes: int = 8_000_000       # enwik8[:store_bytes] keyed into the store
    test_start: int = 95_000_000       # held-out query region (enwik8 offset)
    test_len: int = 1_500_000
    window: int = 1024                 # L: base-logit forward window
    burn: int = 256                    # positions discarded per window (warm-up)
    n_cal: int = 12                    # calibration windows (tune k/tau/eps/lam)
    n_test: int = 32                   # test windows (paired-delta unit)
    key_dim: int = 128
    chunk: int = 32768                 # FFT stream chunk
    store_device: str | None = None    # None -> compute device; "cpu" -> RAM tier
    device: str = "cuda"
    seed: int = 0
    k_grid: tuple = (8, 16, 32, 64)
    tau_grid: tuple = (5.0, 10.0, 20.0, 40.0)
    eps_grid: tuple = (0.05, 0.1, 0.25)
    lam_grid: tuple = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5)


def run_knn_datastore(cfg: KNNDatastoreConfig, log=functools.partial(print, flush=True)) -> dict:
    t0 = time.time()
    def _log(m):
        log(f"[{time.time() - t0:7.1f}s] {m}")

    dev = cfg.device
    enwik = enwik8_bytes()
    train = enwik[: cfg.store_bytes]
    test = enwik[cfg.test_start : cfg.test_start + cfg.test_len]
    marginal = torch.tensor(np.bincount(train, minlength=256) / len(train),
                            device=dev, dtype=torch.float32)

    inf = load_checkpoint(cfg.checkpoint, device=dev)
    W = inf.weights()
    emb = torch.tensor(W["linear_embedding.weight"], device=dev)
    in_proj = torch.tensor(W["linear_in_proj"], device=dev)
    decays = torch.tensor(W["linear_decays"], device=dev, dtype=torch.float64)
    n_modes = decays.shape[0]
    _log(f"bank lifted: M={n_modes}, store_bytes={cfg.store_bytes}")

    streamer = LinearStateStreamer.from_bank(
        emb, in_proj, decays, chunk=cfg.chunk, device=dev)
    kcfg = StateKNNConfig(
        key_dim=cfg.key_dim, k=max(cfg.k_grid), metric="cosine",
        key_transform="pca_whiten", store_dtype="float16",
        store_device=cfg.store_device)
    mem = StateKNNMemory(n_modes, kcfg, device=dev)

    # pass 1: whitening basis over the continuous store stream
    for st, _ in streamer.stream(train):
        mem.observe(st)
    mem.finalize()
    share = float(mem.eigenvalues[: cfg.key_dim].sum() / mem.eigenvalues.sum())
    _log(f"whitening basis ready; top-{cfg.key_dim} variance share {share:.3f}")

    # pass 2: the store — key at t predicts byte t+1 (drop the final unpaired state)
    for st, s in streamer.stream(train):
        hi = min(s + len(st), cfg.store_bytes - 1)
        if hi <= s:
            break
        vals = torch.as_tensor(train[s + 1 : hi + 1], device=dev)
        mem.add(st[: hi - s], vals)
    _log(f"store built: {len(mem.keys) if mem.keys is not None else '(lazy)'} keys "
         f"on {mem.store_device}")

    # continuous-stream query keys at the window positions
    g = np.random.default_rng(cfg.seed)
    L, BURN = cfg.window, cfg.burn
    q_starts = g.integers(0, cfg.test_len - L - 1, size=cfg.n_cal + cfg.n_test)

    def _positions(starts):
        return np.concatenate([np.arange(s + BURN, s + L - 1) for s in starts])

    needed = np.unique(_positions(q_starts))
    qbuf = torch.empty(len(needed), n_modes, device=dev)
    for st, s in streamer.stream(test):
        lo = int(np.searchsorted(needed, s))
        hi = int(np.searchsorted(needed, s + len(st)))
        if hi > lo:
            qbuf[lo:hi] = st[needed[lo:hi] - s]

    def _slots(starts):
        return torch.as_tensor(np.searchsorted(needed, _positions(starts)),
                               device=dev)

    def _base(starts):
        """Windowed forward logits + targets, per the model-as-deployed."""
        Y, B = [], []
        for s in starts:
            seq = test[s : s + L]
            cap = inf.forward_captured(seq[None, :])
            B.append(torch.as_tensor(cap["logits"][0], device=dev)[BURN:-1])
            Y.append(torch.as_tensor(seq[BURN + 1 :], device=dev))
        return torch.cat(Y), torch.cat(B)

    S_cal, S_tst = qbuf[_slots(q_starts[: cfg.n_cal])], qbuf[_slots(q_starts[cfg.n_cal :])]
    y_cal, base_cal = _base(q_starts[: cfg.n_cal])
    y_tst, base_tst = _base(q_starts[cfg.n_cal :])
    base_tst_logp = torch.log_softmax(base_tst.float(), -1)
    base_bpb = bpb_from_logp(base_tst_logp, y_tst)
    _log(f"base (windowed logits): test {base_bpb:.4f}  ({len(y_tst)} positions)")

    # search once; sweep (k, tau, eps) on cal by re-voting cached neighbours
    d_cal, i_cal = mem.search(S_cal)
    d_tst, i_tst = mem.search(S_tst)
    _log("searched")

    def _vote(d, i, k, tau, eps):
        return mem.vote(d, i, k=k, temperature=1.0 / tau, eps=eps, marginal=marginal)

    best = None
    for k in cfg.k_grid:
        for tau in cfg.tau_grid:
            for eps in cfg.eps_grid:
                b = bpb_from_logp(torch.log(_vote(d_cal, i_cal, k, tau, eps).clamp_min(1e-12)), y_cal)
                if best is None or b < best[0]:
                    best = (b, k, tau, eps)
    b_cal, k, tau, eps = best
    p_tst = _vote(d_tst, i_tst, k, tau, eps)
    knn_alone = bpb_from_logp(torch.log(p_tst.clamp_min(1e-12)), y_tst)
    _log(f"kNN-alone (CONTINUOUS keys): cal {b_cal:.4f} (k={k} tau={tau} eps={eps}) "
         f"-> TEST {knn_alone:.4f}   [round-2 windowed: 4.62; census: 2.97]")

    # calibrate the kNN-LM mixing weight on cal, then the paired test delta
    p_cal = _vote(d_cal, i_cal, k, tau, eps)
    lam = min((bpb_from_logp(interpolate(p_cal, base_cal, lm), y_cal), lm)
              for lm in cfg.lam_grid)[1]
    mix_logp = interpolate(p_tst, base_tst, lam)
    stats = paired_window_delta(mix_logp, base_tst_logp, y_tst, cfg.n_test)
    _log(f"kNN-LM mix: lam={lam} -> TEST {stats['mix']:.4f} (base {stats['base']:.4f}, "
         f"paired delta {stats['delta']:+.4f} +/-{stats['ci']:.4f}, "
         f"windows improved {stats['improved']}/{stats['windows']})")

    return {
        "store_bytes": cfg.store_bytes, "store_device": str(mem.store_device),
        "base_bpb": base_bpb, "knn_alone": knn_alone,
        "k": k, "tau": tau, "eps": eps, "lam": lam, **stats,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--store-bytes", type=int, default=8_000_000)
    p.add_argument("--store-device", default=None,
                   help="'cpu' holds the key store in host RAM (the >16M-key tier)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--key-dim", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    cfg = KNNDatastoreConfig(
        checkpoint=a.checkpoint, store_bytes=a.store_bytes,
        store_device=a.store_device, device=a.device,
        key_dim=a.key_dim, seed=a.seed)
    run_knn_datastore(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
