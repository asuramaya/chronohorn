"""Canonical enwik provisioning: splits, P6 checks, metered-download refusal."""
from __future__ import annotations

import numpy as np
import pytest

from chronohorn.data.enwik import (
    ENWIK8_SIZE,
    ENWIK8_SPLITS,
    build_shards_from_arrays,
    check_enwik,
    provision_enwik,
)


def _fake_text_bytes(n: int) -> bytes:
    # Wiki-shaped filler: printable ASCII with newlines, low zero fraction.
    rng = np.random.default_rng(0)
    body = rng.integers(97, 123, size=n, dtype=np.uint8)
    body[::80] = 10  # newlines
    return body.tobytes()


def test_split_registry_shapes() -> None:
    # The census offsets are frozen; a silent edit here would desynchronize
    # every ledger number.
    assert ENWIK8_SPLITS["train8m"] == (0, 8_000_000)
    assert ENWIK8_SPLITS["calib"] == (60_000_000, 262_144)
    assert ENWIK8_SPLITS["test"] == (70_000_000, 524_288)
    assert ENWIK8_SPLITS["val"] == (95_000_000, 524_288)
    assert ENWIK8_SPLITS["val_large"] == (95_000_000, 4_194_304)
    for name, (offset, length) in ENWIK8_SPLITS.items():
        assert offset + length <= ENWIK8_SIZE, name


def test_val_large_is_bigger_pool_same_region() -> None:
    # 8x val, 2048 slots of 2048 — the pool that makes deep-bucket SEMs honest.
    v_off, v_len = ENWIK8_SPLITS["val"]
    vl_off, vl_len = ENWIK8_SPLITS["val_large"]
    assert vl_off == v_off                       # same held-out region start
    assert vl_len % 2048 == 0 and vl_len // 2048 == 2048
    assert vl_len >= 8 * v_len


def test_holdouts_disjoint_from_train8m() -> None:
    train_end = sum(ENWIK8_SPLITS["train8m"])
    for name in ("calib", "test", "val"):
        offset, _ = ENWIK8_SPLITS[name]
        assert offset >= train_end, f"{name} overlaps train8m"


def test_val_disjoint_from_train90m() -> None:
    # train90m deliberately swallows calib/test; val (and val_large) are the
    # only holdouts, so both must sit past the fed-training boundary.
    train_end = sum(ENWIK8_SPLITS["train90m"])
    for name in ("val", "val_large"):
        offset, length = ENWIK8_SPLITS[name]
        assert offset >= train_end, f"{name} overlaps train90m"
        assert offset + length <= ENWIK8_SIZE, f"{name} runs past enwik8"


def test_check_enwik_flags_size_mismatch(tmp_path) -> None:
    path = tmp_path / "enwik8"
    path.write_bytes(_fake_text_bytes(1000))
    problems = check_enwik(path, expected_size=ENWIK8_SIZE, expected_md5=None)
    assert any("size" in p for p in problems)


def test_check_enwik_passes_healthy_text(tmp_path) -> None:
    path = tmp_path / "enwik8"
    payload = _fake_text_bytes(2_000_000)
    path.write_bytes(payload)
    problems = check_enwik(path, expected_size=len(payload), expected_md5=None)
    assert problems == []


def test_check_enwik_catches_null_interleaving(tmp_path) -> None:
    # The session-11 P6 bug class: uint16-widened bytes read as uint8.
    text = np.frombuffer(_fake_text_bytes(500_000), dtype=np.uint8)
    widened = np.zeros(text.size * 2, dtype=np.uint8)
    widened[0::2] = text
    path = tmp_path / "enwik8"
    path.write_bytes(widened.tobytes())
    problems = check_enwik(path, expected_size=widened.size, expected_md5=None)
    assert any("zero" in p for p in problems)


def test_provision_refuses_implicit_download(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="metered"):
        provision_enwik(root=tmp_path, which="enwik8", source=None, download=False)


def test_build_shards_roundtrip_and_glob_shape(tmp_path) -> None:
    from chronohorn.data.byte_reader import open_byte_shard

    train = np.frombuffer(_fake_text_bytes(2_500_000), dtype=np.uint8)
    val = np.frombuffer(_fake_text_bytes(100_000), dtype=np.uint8)
    paths = build_shards_from_arrays(tmp_path, train, val, shard_bytes=1_000_000)

    # 3 train shards + 1 val shard, named to match the trainer's
    # *_train_*.bin / *_val_*.bin globs.
    assert [p.name for p in paths] == [
        "enwik_train_000000.bin",
        "enwik_train_000001.bin",
        "enwik_train_000002.bin",
        "enwik_val_000000.bin",
    ]
    assert sorted(p.name for p in tmp_path.glob("*_train_*.bin")) == [p.name for p in paths[:3]]
    assert not list(tmp_path.glob("*.tmp"))

    # Payload survives the uint16 shard round trip byte-exactly.
    back = np.concatenate([np.asarray(open_byte_shard(str(p))) for p in paths[:3]])
    np.testing.assert_array_equal(back.astype(np.uint8), train)
    np.testing.assert_array_equal(
        np.asarray(open_byte_shard(str(paths[3]))).astype(np.uint8), val
    )


def test_provision_from_source_verifies(tmp_path) -> None:
    source = tmp_path / "somewhere" / "enwik8_copy"
    source.parent.mkdir()
    source.write_bytes(_fake_text_bytes(1000))
    # A source that fails the size check must not be installed.
    with pytest.raises(ValueError, match="size"):
        provision_enwik(root=tmp_path / "root", which="enwik8", source=source)
    assert not (tmp_path / "root" / "enwik8").exists()
    assert not list((tmp_path / "root").glob("*.tmp"))
