"""THE DECISIVE CONTROL: base + n-gram  vs  base + state-kNN organ.

The organ loses to a byte n-gram STANDALONE (ruling b2815ef5: order-7 count table on
8M bytes = 2.3128, organ on 150M bytes = 2.3892). But the ladder never claimed the organ
was a strong predictor — it measured what the organ buys MIXED INTO A TRAINED BASE.
Complementarity is not strength, and the two can come apart completely: the base is a
neural LM trained on this very corpus, so it may already contain every n-gram statistic
there is, in which case mixing them back in buys nothing at all.

So this asks the only question that decides the campaign:

    does a byte n-gram, mixed into the base, buy MORE than the organ does?

  n-gram mix >= organ mix   ->  the organ is DOMINATED. The ladder measured the cost of
                                retrieval, not its value. Say so, and give the budget to
                                the core step.
  n-gram mix <<  organ mix  ->  the organ's value was never prediction, it was
                                COMPLEMENTARITY to a trained body — a sharper and more
                                defensible claim than anything the ladder has made.

Both are run on the EVAL'S CANONICAL WINDOWS (n_cal=12 slice, seed 0) so the base bpb
must reproduce 2.0367 exactly; if it does not, the harness is lying and the run aborts.

Run:  python -m chronohorn.eval.ngram_mix --store-bytes 8000000
"""
from __future__ import annotations

import argparse
import functools

import numpy as np
import torch

from decepticons.loader import load_checkpoint

from .harness_util import bpb_from_logp, enwik8_bytes
from .knn_datastore import DEFAULT_CHECKPOINT
from .ngram_control import ByteNgram

EVAL_N_CAL, EVAL_N_TEST = 12, 32          # the eval's canonical draw — do not change
QUERY_LO, QUERY_HI = 95_000_000, 96_500_000


def _windows(test: np.ndarray, starts, window: int, burn: int, order: int):
    ctx, y, idx = [], [], []
    for s in starts:
        seq = test[s: s + window]
        for i in range(burn, window - 1):
            ctx.append(seq[i - order + 1: i + 1])
            y.append(seq[i + 1])
        idx.append(s)
    return np.stack(ctx).astype(np.uint8), np.array(y, dtype=np.int64)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store-bytes", type=int, default=8_000_000)
    ap.add_argument("--order", type=int, default=7)
    ap.add_argument("--alpha", type=float, default=64.0)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--burn", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--organ-delta", type=float, default=None,
                    help="the organ's known mix delta at this store size, for the verdict")
    a = ap.parse_args(argv)
    log = functools.partial(print, flush=True)

    enwik = enwik8_bytes()
    store = enwik[: a.store_bytes]
    test = enwik[QUERY_LO:QUERY_HI]

    # THE EVAL'S CANONICAL DRAW. cal = q[:12], test = q[12:44]. Anything else scores the
    # contenders on windows the ladder never saw (the bug that killed E3's first run).
    g = np.random.default_rng(a.seed)
    q = g.integers(0, len(test) - a.window - 1, size=EVAL_N_CAL + EVAL_N_TEST)
    cal_starts, tst_starts = q[:EVAL_N_CAL], q[EVAL_N_CAL:]

    log(f"base pass ({len(cal_starts)} cal + {len(tst_starts)} test windows)...")
    inf = load_checkpoint(DEFAULT_CHECKPOINT, device=a.device)

    def base_logits(starts):
        B, Y = [], []
        for s in starts:
            seq = test[s: s + a.window]
            cap = inf.forward_captured(seq[None, :])
            B.append(torch.as_tensor(cap["logits"][0], device=a.device)[a.burn:-1])
            Y.append(torch.as_tensor(seq[a.burn + 1:], device=a.device))
        return torch.cat(B).float(), torch.cat(Y)

    base_cal, y_cal = base_logits(cal_starts)
    base_tst, y_tst = base_logits(tst_starts)
    lp_cal = torch.log_softmax(base_cal, -1)
    lp_tst = torch.log_softmax(base_tst, -1)
    base_bpb = bpb_from_logp(lp_tst, y_tst)
    log(f"base: {base_bpb:.4f}  ({len(y_tst)} positions)")
    if abs(base_bpb - 2.0367) > 5e-4:
        raise SystemExit(f"ABORT: base {base_bpb:.4f} != the ladder's 2.0367 — wrong windows.")
    log("base reproduces the ladder's 2.0367 — the windows are the ladder's windows.\n")

    log(f"building byte n-gram (order {a.order}, alpha {a.alpha}, store {a.store_bytes:,})...")
    ng = ByteNgram(a.order, a.alpha).build(store, log=lambda *_: None)
    null = lambda *_a, **_k: None                                        # noqa: E731
    c_cal, yc = _windows(test, cal_starts, a.window, a.burn, a.order)
    c_tst, yt = _windows(test, tst_starts, a.window, a.burn, a.order)
    assert np.array_equal(yt, y_tst.cpu().numpy()), "target mismatch — windows disagree"

    p_cal = torch.as_tensor(ng.predict(c_cal, log=null), device=a.device, dtype=torch.float32)
    p_tst = torch.as_tensor(ng.predict(c_tst, log=null), device=a.device, dtype=torch.float32)
    ng_alone = bpb_from_logp(torch.log(p_tst.clamp_min(1e-12)), y_tst)
    log(f"n-gram ALONE: {ng_alone:.4f}\n")

    def mix_logp(lp, p, lam):
        return torch.log(((1 - lam) * lp.exp() + lam * p).clamp_min(1e-12))

    grid = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
    best = min(((bpb_from_logp(mix_logp(lp_cal, p_cal, l), y_cal), l) for l in grid),
               key=lambda t: t[0])
    lam = best[1]
    mix = bpb_from_logp(mix_logp(lp_tst, p_tst, lam), y_tst)
    delta = mix - base_bpb

    nb = -lp_tst.gather(-1, y_tst[:, None]).squeeze(-1)
    nn_ = -torch.log(p_tst.clamp_min(1e-12)).gather(-1, y_tst[:, None]).squeeze(-1)
    oracle = float(torch.minimum(nb, nn_).mean() / np.log(2))

    log("=" * 70)
    log(f"BASE + N-GRAM   lam={lam:.2f}  TEST {mix:.4f}   delta {delta:+.4f}")
    log(f"  n-gram oracle purse: {base_bpb - oracle:.4f}")
    log("=" * 70)
    if a.organ_delta is not None:
        od = a.organ_delta
        log(f"BASE + ORGAN (same store)          delta {od:+.4f}")
        log("")
        if delta <= od:
            log(f"VERDICT: THE N-GRAM DOMINATES. It buys {od - delta:+.4f} bpb MORE than the organ,")
            log("         and it is a count table. The ladder measured the COST of retrieval.")
        else:
            log(f"VERDICT: THE ORGAN SURVIVES. It buys {delta - od:+.4f} bpb MORE than the n-gram")
            log("         DESPITE being the weaker standalone predictor — so its value is")
            log("         COMPLEMENTARITY to a trained body, not raw prediction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
