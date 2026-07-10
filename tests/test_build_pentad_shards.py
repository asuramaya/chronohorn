"""Pentad shard builder — round-robin interleave + held-out val split.

Exercises the pure sharding logic on tiny in-memory folds (the real builder
memmaps 2GB of enwik9 + raws, out of scope for CI).
"""
from __future__ import annotations

import numpy as np

from chronohorn.data.build_pentad_shards import (
    byte_coverage,
    write_interleaved_shards,
)


def _fold(start, n):
    return np.arange(start, start + n, dtype=np.int64) % 256


def test_interleave_val_split_and_shard_names(tmp_path):
    folds = {"a": _fold(0, 35), "b": _fold(1000, 35)}   # span 30 each, 3 shards each
    n_shards = write_interleaved_shards(
        folds, tmp_path, shard_bytes=10, val_bytes=5, log=lambda *a: None)
    assert n_shards == {"a": 3, "b": 3}

    # round-robin: global counter alternates folds, so names cycle a,b,a,b,...
    names = sorted(p.name for p in tmp_path.glob("*_train_*.bin"))
    assert names == [
        "000000_a_train_000000.bin", "000001_b_train_000001.bin",
        "000002_a_train_000002.bin", "000003_b_train_000003.bin",
        "000004_a_train_000004.bin", "000005_b_train_000005.bin",
    ]

    # val = the LAST val_bytes of each fold, uint16
    a_val = np.fromfile(tmp_path / "a_val_000000.bin", dtype=np.uint16)
    assert np.array_equal(a_val, (_fold(0, 35)[30:35]).astype(np.uint16))

    # first train shard = the fold's first shard_bytes, uint16
    a0 = np.fromfile(tmp_path / "000000_a_train_000000.bin", dtype=np.uint16)
    assert np.array_equal(a0, (_fold(0, 35)[:10]).astype(np.uint16))


def test_rejects_fold_smaller_than_val(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        write_interleaved_shards({"a": _fold(0, 4)}, tmp_path,
                                 shard_bytes=10, val_bytes=5, log=lambda *a: None)


def test_byte_coverage_flags_high_bytes_and_emoji():
    ascii_fold = np.frombuffer(b"hello world " * 10, dtype=np.uint8)
    emoji_fold = np.frombuffer("grin 😀😀".encode("utf-8") * 5, dtype=np.uint8)
    cov = byte_coverage({"ascii": ascii_fold, "emoji": emoji_fold}, sample_bytes=10_000)
    assert cov["ascii"]["hi_byte_frac"] == 0.0 and cov["ascii"]["emoji_leads"] == 0
    assert cov["emoji"]["hi_byte_frac"] > 0.0 and cov["emoji"]["emoji_leads"] == 10
