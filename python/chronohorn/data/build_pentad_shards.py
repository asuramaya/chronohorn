"""Build interleaved multi-modal (pentad) byte shards for curriculum-free training.

Fold 1 (enwik) comes from enwik9[100M:500M] — the enwik8 prefix is SKIPPED so
the lineage eval anchor enwik8[95M:100M] stays held out. Folds 2-5 come from raw
byte files (the pentad_pull.py output: world / code / math / wikiml). Shards are
round-robin interleaved across folds by a global counter so the sequential
TokenStream never sees one fold as a contiguous block (ordering is curriculum
whether you want it or not); the last VAL bytes of each fold become its val
shard. Promoted from data/staging/pentad_shard.py.

Run: python -m chronohorn.data.build_pentad_shards --staging data/staging --out data/roots/pentad
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

FOLDS = ["enwik", "world", "code", "math", "wikiml"]
SHARD_BYTES = 20_000_000
VAL_BYTES = 2_000_000
FOLD_BYTES = 400_000_000
ENWIK9_MIN = 500_000_000


def write_interleaved_shards(folds: dict, out_dir, *, shard_bytes: int = SHARD_BYTES,
                             val_bytes: int = VAL_BYTES, log=print) -> dict[str, int]:
    """Write round-robin train shards + per-fold val shards; return {name: n_train_shards}.

    folds: ordered mapping name -> byte array. The last ``val_bytes`` of each
    fold become ``<name>_val_000000.bin``; the rest is cut into ``shard_bytes``
    chunks and interleaved across folds by a global counter, so shard g cycles
    through the folds rather than emitting one fold as a block. uint16 on disk.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spans: dict[str, int] = {}
    for name, b in folds.items():
        n = len(b)
        if n <= val_bytes:
            raise ValueError(f"fold {name}: {n} bytes <= val_bytes {val_bytes}")
        spans[name] = n - val_bytes
        np.asarray(b[n - val_bytes:n], dtype=np.uint16).tofile(
            out_dir / f"{name}_val_000000.bin")
        log(f"{name}: {n/1e6:.1f}MB ({(n - val_bytes)/1e6:.1f} train + {val_bytes/1e6:.0f} val)")

    n_shards = {name: span // shard_bytes for name, span in spans.items()}
    g = 0
    for i in range(max(n_shards.values(), default=0)):
        for name in folds:
            if i >= n_shards[name]:
                continue
            chunk = np.asarray(folds[name][i * shard_bytes:(i + 1) * shard_bytes],
                               dtype=np.uint16)
            chunk.tofile(out_dir / f"{g:06d}_{name}_train_{g:06d}.bin")
            g += 1
    log(f"wrote {g} train shards, {len(folds)} val shards -> {out_dir}")
    return n_shards


def byte_coverage(folds: dict, *, sample_bytes: int = 50_000_000) -> dict[str, dict]:
    """Per-fold >127-byte fraction and emoji-lead count — the script/emoji sanity."""
    out = {}
    for name, b in folds.items():
        sample = np.asarray(b[:min(len(b), sample_bytes)])
        out[name] = {
            "hi_byte_frac": float((sample > 127).mean()),
            "emoji_leads": sample.tobytes().count(b"\xf0\x9f"),  # utf-8 lead for U+1F000+
        }
    return out


def load_folds(staging_dir, *, fold_bytes: int = FOLD_BYTES) -> dict:
    """Memory-map the five pentad fold sources from a staging directory."""
    staging = Path(staging_dir)
    folds = {}
    for name in FOLDS:
        if name == "enwik":
            data = np.memmap(staging / "enwik9", dtype=np.uint8, mode="r")
            if len(data) < ENWIK9_MIN:
                raise ValueError("enwik9 truncated")
            folds[name] = data[100_000_000:100_000_000 + fold_bytes]
        else:
            folds[name] = np.memmap(staging / "pentad" / f"{name}.raw",
                                    dtype=np.uint8, mode="r")
    return folds


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--staging", default="data/staging")
    p.add_argument("--out", default="data/roots/pentad")
    p.add_argument("--shard-bytes", type=int, default=SHARD_BYTES)
    p.add_argument("--val-bytes", type=int, default=VAL_BYTES)
    a = p.parse_args(argv)
    folds = load_folds(a.staging)
    write_interleaved_shards(folds, a.out, shard_bytes=a.shard_bytes, val_bytes=a.val_bytes)
    print("\nbyte-coverage sanity (per fold):")
    for name, cov in byte_coverage(folds).items():
        print(f"  {name:7s} >127 bytes: {cov['hi_byte_frac']*100:5.1f}%   "
              f"emoji leads/50MB: {cov['emoji_leads']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
