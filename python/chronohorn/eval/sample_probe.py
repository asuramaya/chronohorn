"""Behavioral sampling probe — what does this body's bpb SOUND like?

Autoregressively samples bytes from a causal-bank checkpoint, optionally
mixing the state-kNN organ per token (kNN-LM at generation time), so every
rung on the bpb ladder gets an audible reading. The bpb->behavior map this
calibrates: ~2.0 = words, ~1.5 = sentences, ~1.2 = paragraphs, ~1.0 = pages.

Prompts come from enwik8[96.5M:] — past the eval query region [95M, 96.5M),
so the probe never rehearses text any ladder number was scored on.

Run:  python -m chronohorn.eval.sample_probe                     # base only
      python -m chronohorn.eval.sample_probe --store-bytes 8000000 --lam 0.1
"""
from __future__ import annotations

import argparse
import functools
from dataclasses import dataclass

import numpy as np
import torch

from decepticons.loader import load_checkpoint
from decepticons.models.state_knn_torch import StateKNNConfig, StateKNNMemory
from decepticons.models.state_stream_torch import LinearStateStreamer

from .harness_util import enwik8_bytes
from .knn_datastore import DEFAULT_CHECKPOINT

PROBE_OFFSET = 96_500_000   # first byte past the eval query region


@dataclass
class SampleProbeConfig:
    checkpoint: str = DEFAULT_CHECKPOINT
    prompt_len: int = 512
    steps: int = 384
    n_prompts: int = 3
    temps: tuple = (0.7, 1.0)
    window: int = 1024              # base forward context cap (model-as-deployed)
    seed: int = 0
    device: str = "cuda"
    # organ (0 = base only). Vote params pinned to the ladder's 8M selections.
    store_bytes: int = 0
    lam: float = 0.05
    k: int = 64
    vote_temp: float = 0.05
    eps: float = 0.1


def run_sample_probe(cfg: SampleProbeConfig, log=functools.partial(print, flush=True)) -> list[dict]:
    dev = cfg.device
    enwik = enwik8_bytes()
    inf = load_checkpoint(cfg.checkpoint, device=dev)
    W = inf.weights()

    mem = streamer = None
    if cfg.store_bytes:
        emb = torch.tensor(W["linear_embedding.weight"], device=dev)
        in_proj = torch.tensor(W["linear_in_proj"], device=dev)
        decays = torch.tensor(W["linear_decays"], device=dev, dtype=torch.float64)
        streamer = LinearStateStreamer.from_bank(emb, in_proj, decays, device=dev)
        mem = StateKNNMemory(decays.shape[0], StateKNNConfig(
            key_dim=128, k=cfg.k, metric="cosine", key_transform="pca_whiten",
            store_dtype="float16"), device=dev)
        train = enwik[: cfg.store_bytes]
        log(f"building {cfg.store_bytes} organ store (whitening pass)...")
        for st, _ in streamer.stream(train):
            mem.observe(st)
        mem.finalize()
        mem.reserve(cfg.store_bytes - 1)
        for st, s in streamer.stream(train):
            hi = min(s + len(st), cfg.store_bytes - 1)
            if hi <= s:
                break
            mem.add(st[: hi - s], torch.as_tensor(train[s + 1 : hi + 1], device=dev))
        log("store ready")

    rng = np.random.default_rng(cfg.seed)
    out = []
    for i in range(cfg.n_prompts):
        start = PROBE_OFFSET + i * 1_000_000
        prompt = enwik[start : start + cfg.prompt_len].astype(np.int64)
        for T in cfg.temps:
            window = prompt.tolist()
            h = streamer.states(prompt)[-1] if mem is not None else None
            for _ in range(cfg.steps):
                ctx = np.asarray(window[-cfg.window:], dtype=np.int64)
                logits = torch.as_tensor(inf.forward(ctx[None])[0, -1])
                p = torch.softmax(logits, -1)
                if mem is not None:
                    d, idx = mem.search(h[None])
                    p_knn = mem.vote(d, idx, k=cfg.k, temperature=cfg.vote_temp,
                                     eps=cfg.eps)[0].cpu()
                    p = (1.0 - cfg.lam) * p + cfg.lam * p_knn
                p = (p.double() ** (1.0 / T))
                p = (p / p.sum()).numpy()
                b = int(rng.choice(256, p=p))
                window.append(b)
                if mem is not None:
                    h = streamer.step(h, b)
            text = bytes(window[cfg.prompt_len:]).decode("utf-8", "replace")
            tag = f"prompt@{start} T={T}" + (f" organ(lam={cfg.lam})" if mem is not None else " base")
            log(f"\n===== {tag} =====\n{text}\n")
            out.append({"start": start, "temp": T, "organ": mem is not None,
                        "text": text})
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--steps", type=int, default=384)
    p.add_argument("--n-prompts", type=int, default=3)
    p.add_argument("--temps", nargs="+", type=float, default=[0.7, 1.0])
    p.add_argument("--store-bytes", type=int, default=0)
    p.add_argument("--lam", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)
    run_sample_probe(SampleProbeConfig(
        checkpoint=a.checkpoint, steps=a.steps, n_prompts=a.n_prompts,
        temps=tuple(a.temps), store_bytes=a.store_bytes, lam=a.lam,
        seed=a.seed, device=a.device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
