"""THE CONTROL WE SHOULD HAVE RUN FIRST: is the state-kNN organ just an expensive n-gram?

The operator asked the question that should have been asked at rung one, and the repo
makes it sting: decepticons already ships OnlineCausalMemory (a causal n-gram
accumulator) and polyhash ships NgramTable. We had n-grams the whole time and never
pointed them at the organ.

The reason it matters is recorded in our own letters: EVERYTHING ABOVE ~1.8 bpb IS
COMMODITY STATISTICS — PPM's well. The kNN ladder's best mix is 1.9530. That is INSIDE
the well. So the organ may be buying, with 150M keys and 38GB of RAM, exactly what a
byte n-gram buys from a count table.

This builds a proper interpolated byte n-gram (orders 0..K, Dirichlet/Witten-Bell style
backoff, 64-bit rolling context hashes) from the SAME store bytes a kNN rung uses, and
scores it on the SAME query windows. Two numbers decide it:

  n-gram ALONE  vs  kNN-alone   — which is the better standalone predictor?
  n-gram MIXED  vs  kNN mix     — which buys more when mixed into the same base?

If the n-gram matches or beats the organ, the ladder measured the cost of retrieval, not
its value, and the honest move is to say so.

Run:  python -m chronohorn.eval.ngram_control --store-bytes 8000000 --max-order 8
"""
from __future__ import annotations

import argparse
import functools

import numpy as np

MUL = np.uint64(0x100000001B3)      # FNV-ish 64-bit mixing constant
OFF = np.uint64(0xCBF29CE484222325)


def _context_hashes(data: np.ndarray, order: int) -> np.ndarray:
    """64-bit hash of the `order` bytes preceding each position i (i >= order).

    Returns h[j] = hash(data[j : j+order]) for j in [0, len(data)-order), i.e. the
    context ENDING just before position j+order. Rolling, vectorised, collision-safe
    at 64 bits for our corpus sizes.
    """
    n = len(data) - order
    if n <= 0:
        return np.empty(0, dtype=np.uint64)
    h = np.full(n, OFF, dtype=np.uint64)
    for t in range(order):
        h = (h ^ data[t: t + n].astype(np.uint64)) * MUL
    return h


class ByteNgram:
    """Interpolated byte n-gram over orders 0..K, built from a byte array.

    Backoff is Dirichlet/Witten-Bell style and recursive:
        p_c(x | ctx) = (count(ctx, x) + a * p_{c-1}(x | ctx')) / (total(ctx) + a)
    with the unigram (c=0) as the base case. No tuning beyond `alpha`, deliberately:
    this is a CONTROL, and a control that needs careful tuning to beat the thing it
    controls for is not a control any more.
    """

    def __init__(self, max_order: int = 8, alpha: float = 1.0, vocab: int = 256) -> None:
        self.max_order, self.alpha, self.vocab = max_order, alpha, vocab
        self.tables: dict[int, tuple[np.ndarray, np.ndarray]] = {}   # order -> (keys, counts)
        self.unigram = np.zeros(vocab, dtype=np.float64)

    def build(self, data: np.ndarray, log=print) -> "ByteNgram":
        data = np.ascontiguousarray(data, dtype=np.uint8)
        cnt = np.bincount(data, minlength=self.vocab).astype(np.float64)
        self.unigram = (cnt + 1.0) / (cnt.sum() + self.vocab)     # +1 smoothed

        for c in range(1, self.max_order + 1):
            h = _context_hashes(data, c)                # context of length c
            tgt = data[c:].astype(np.uint64)            # the byte that followed it
            n = min(len(h), len(tgt))
            # pack (context, target) into one key so np.unique gives counts directly
            key = (h[:n] * np.uint64(self.vocab)) + tgt[:n]
            key.sort()
            uniq, counts = np.unique(key, return_counts=True)
            self.tables[c] = (uniq, counts.astype(np.float64))
            log(f"  order {c}: {len(uniq):>12,} distinct (context,byte) pairs")
        return self

    def predict(self, contexts: np.ndarray, log=print) -> np.ndarray:
        """contexts: [N, max_order] bytes preceding each query position (left-padded).

        Returns [N, vocab] probabilities.
        """
        N = len(contexts)
        p = np.tile(self.unigram, (N, 1))               # c = 0 base case

        for c in range(1, self.max_order + 1):
            uniq, counts = self.tables[c]
            ctx = contexts[:, -c:]                      # last c bytes
            h = np.full(N, OFF, dtype=np.uint64)
            for t in range(c):
                h = (h ^ ctx[:, t].astype(np.uint64)) * MUL

            base = h * np.uint64(self.vocab)
            # gather counts for all 256 targets: one searchsorted per byte value
            row = np.zeros((N, self.vocab), dtype=np.float64)
            for v in range(self.vocab):
                idx = np.searchsorted(uniq, base + np.uint64(v))
                ok = (idx < len(uniq)) & (uniq[np.minimum(idx, len(uniq) - 1)] == base + np.uint64(v))
                row[ok, v] = counts[idx[ok]]

            total = row.sum(1, keepdims=True)
            p = (row + self.alpha * p) / (total + self.alpha)      # recursive interpolation
            log(f"  order {c}: {(total > 0).mean() * 100:5.1f}% of queries had this context")
        return p


def bpb(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log2(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())


def main(argv=None) -> int:
    from .harness_util import enwik8_bytes

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store-bytes", type=int, default=8_000_000)
    ap.add_argument("--max-order", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--n-test", type=int, default=32)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--burn", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    log = functools.partial(print, flush=True)

    enwik = enwik8_bytes()
    store = enwik[: a.store_bytes]
    test = enwik[95_000_000: 96_500_000]     # the eval's query region

    log(f"building byte n-gram: store={a.store_bytes:,} order<={a.max_order} alpha={a.alpha}")
    ng = ByteNgram(a.max_order, a.alpha).build(store, log=log)

    # SAME window draw as knn_datastore (seed 0, window 1024, burn 256, 32 test windows)
    g = np.random.default_rng(a.seed)
    starts = g.integers(0, len(test) - a.window - 1, size=12 + a.n_test)[12:]  # skip cal
    ctxs, ys = [], []
    for s in starts:
        seq = test[s: s + a.window]
        for i in range(a.burn, a.window - 1):
            ctxs.append(seq[i - a.max_order + 1: i + 1])
            ys.append(seq[i + 1])
    contexts = np.stack(ctxs).astype(np.uint8)
    y = np.array(ys, dtype=np.int64)
    log(f"scoring {len(y):,} positions (same windows as the kNN eval)")

    p = ng.predict(contexts, log=log)
    alone = bpb(p, y)

    log("")
    log("=" * 72)
    log(f"BYTE N-GRAM ALONE (order<={a.max_order}, store {a.store_bytes // 1_000_000}M): {alone:.4f} bpb")
    log("=" * 72)
    log("compare, on the SAME query region:")
    log(f"  base (the trained body)                          2.0367")
    log(f"  state-kNN organ ALONE,   8M store                2.9743")
    log(f"  state-kNN organ ALONE, 150M store                2.3892")
    log(f"  byte n-gram ALONE,    {a.store_bytes // 1_000_000:>4}M store  ->          {alone:.4f}")
    log("")
    log("If the n-gram alone beats the organ alone, the organ is an expensive n-gram")
    log("and the ladder measured the COST of retrieval, not its value.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
