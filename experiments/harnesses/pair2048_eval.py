"""Firm paired eval + depth curve for a seq2048 horizon body vs the seq1024
frontier. Durable home after /tmp ate the first copy (2026-07-09).

Usage: python pair2048_eval.py <body-name>   (default fo-adapt-enwik90-seq2048-100k)

A) Paired 48x1024 (same windows, burn 256): fo-learnable-50k vs <body>,
   both scored at L=1024 — model quality with eval length held fixed.
B) Depth curve at L=2048 for <body>: mean bits per position bin — does depth
   past 1024 pay inside this model?
"""
import sys

import numpy as np
from decepticons.loader import load_checkpoint

BODY = sys.argv[1] if len(sys.argv) > 1 else "fo-adapt-enwik90-seq2048-100k"
R = "/home/asuramaya/code/REPOS/chronohorn/out/results"
enwik = np.fromfile("/home/asuramaya/code/REPOS/chronohorn/data/roots/enwik/enwik8", dtype=np.uint8)
test = enwik[95_000_000:100_000_000].astype(np.int64)

def bits(lg, tgt, burn):
    lg = np.asarray(lg)[0][:-1].astype(np.float64)
    m = lg.max(-1, keepdims=True)
    logp = lg - m - np.log(np.exp(lg - m).sum(-1, keepdims=True))
    b = -np.take_along_axis(logp, tgt[:, None], -1)[:, 0] / np.log(2)
    return b[burn:] if burn else b

# ---- A: paired at L=1024 ----
N, L, BURN = 48, 1024, 256
starts = np.random.default_rng(0).integers(0, len(test) - L - 1, size=N)
per = {}
for arm, name in (("s1024", "fo-learnable-50k"), ("body", BODY)):
    inf = load_checkpoint(f"{R}/{name}.checkpoint.pt", device="cuda")
    per[arm] = np.array([bits(inf.forward_captured(test[s:s+L][None, :])["logits"],
                              test[s:s+L][1:], BURN).mean() for s in starts])
d = per["body"] - per["s1024"]
ci = 1.96 * d.std() / np.sqrt(len(d))
print(f"A) paired @1024: fo-learnable-50k {per['s1024'].mean():.4f}  {BODY} "
      f"{per['body'].mean():.4f}  diff {d.mean():+.4f} ±{ci:.4f}")

# ---- B: depth curve at L=2048 ----
N2, L2 = 24, 2048
starts2 = np.random.default_rng(1).integers(0, len(test) - L2 - 1, size=N2)
inf = load_checkpoint(f"{R}/{BODY}.checkpoint.pt", device="cuda")
rows = [bits(inf.forward_captured(test[s:s+L2][None, :])["logits"], test[s:s+L2][1:], 0)
        for s in starts2]
allb = np.stack(rows)
print(f"B) depth curve, {BODY} on 2048-windows:")
for lo, hi in [(4, 8), (8, 32), (32, 128), (128, 512), (512, 1024),
               (1024, 1536), (1536, 2047)]:
    seg = allb[:, lo:hi].mean(1)
    print(f"   pos {lo:>5}-{hi:<5} {seg.mean():.4f} ±{1.96*seg.std()/np.sqrt(len(seg)):.4f}")
