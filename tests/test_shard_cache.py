"""Bounded LRU shard cache — the opt-in RAM tier for the token dataloader.

Verifies the cache is a no-op when disabled (default), memoizes decoded shards
when enabled, and never exceeds its byte budget (evicting least-recently-used).
The budget guard is the safety contract: it must protect tight cgroup/k8s jobs.
"""
from __future__ import annotations

import numpy as np
import pytest

from chronohorn.train import token_shard_dataset as tsd
from chronohorn.train.token_shard_dataset import (
    HEADER_BYTES,
    HEADER_INTS,
    TOKEN_SHARD_MAGIC,
    TOKEN_SHARD_VERSION,
    _load_token_shard,
    _ShardCache,
)


def _write_shard(path, tokens: np.ndarray) -> None:
    header = np.zeros(HEADER_INTS, dtype=np.int32)
    header[0], header[1], header[2] = TOKEN_SHARD_MAGIC, TOKEN_SHARD_VERSION, len(tokens)
    with open(path, "wb") as fh:
        fh.write(header.tobytes())
        fh.write(tokens.astype(np.uint16).tobytes())


def test_cache_disabled_is_a_noop():
    c = _ShardCache(0)
    c.put("a", np.zeros(10, np.int32))
    assert c.get("a") is None


def test_cache_memoizes_and_returns_same_object():
    c = _ShardCache(1 << 20)
    arr = np.arange(16, dtype=np.int32)
    c.put("a", arr)
    assert c.get("a") is arr  # identity — no re-decode, no re-copy


def test_cache_evicts_lru_within_budget():
    each = np.zeros(100, np.int32).nbytes           # 400 bytes
    c = _ShardCache(budget_bytes=2 * each + 1)       # holds exactly two shards
    a, b, d = (np.zeros(100, np.int32) for _ in range(3))
    c.put("a", a); c.put("b", b)
    assert c.get("a") is a and c.get("b") is b
    c.get("a")                                       # touch a -> b is now LRU
    c.put("d", d)                                    # evicts b, not a
    assert c.get("a") is a
    assert c.get("b") is None
    assert c.get("d") is d
    assert c._bytes <= c.budget


def test_cache_skips_oversized_shard():
    c = _ShardCache(10)
    c.put("big", np.zeros(100, np.int32))            # 400 > 10 budget
    assert c.get("big") is None


def test_load_token_shard_uses_cache_when_enabled(tmp_path, monkeypatch):
    shard = tmp_path / "s0.bin"
    _write_shard(shard, np.arange(64, dtype=np.uint16))
    # enabled: second load returns the identical cached object
    monkeypatch.setattr(tsd, "_SHARD_CACHE", _ShardCache(1 << 20))
    first = _load_token_shard(shard)
    second = _load_token_shard(shard)
    assert second is first
    assert first.dtype == np.int32 and first.tolist() == list(range(64))


def test_load_token_shard_no_cache_by_default(tmp_path, monkeypatch):
    shard = tmp_path / "s0.bin"
    _write_shard(shard, np.arange(64, dtype=np.uint16))
    monkeypatch.setattr(tsd, "_SHARD_CACHE", _ShardCache(0))
    a = _load_token_shard(shard)
    b = _load_token_shard(shard)
    assert a is not b                                # fresh decode each time
    assert np.array_equal(a, b)
