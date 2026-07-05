"""Canonical enwik8/enwik9 provisioning and split registry.

enwik files are RAW uint8 bytes (mattmahoney.net/dc) — no chronohorn
shard header, no uint16 widening. Do NOT read them with
`byte_reader.open_byte_shard`; use `load_enwik8`/`load_split` here.

The split offsets below are the frozen-substrate-census conventions
(decepticons/experiments/frozen_census_2026_07) and are the standard for
all ledger work: numbers are only comparable when measured on the same
bytes.

Provisioning never downloads by default — this machine's internet is
metered. Point --source at an existing copy, or pass --download to
explicitly fetch (~36MB zipped for enwik8, ~322MB for enwik9).

Usage:
    chronohorn data enwik --verify
    chronohorn data enwik --provision --source /path/to/enwik8
    chronohorn data enwik --provision --download            # metered!
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from chronohorn.data.byte_reader import check_sample

# enwik8 size/md5 verified 2026-07-05 against the census working copy
# and the published mattmahoney.net/dc value. enwik9's md5 is left
# unpinned until a copy is actually fetched and checked against the
# published listing — size alone still gates truncation.
ENWIK8_SIZE = 100_000_000
ENWIK8_MD5 = "a1fa5ffddb56f4953e226637dabbb36a"
ENWIK9_SIZE = 1_000_000_000
ENWIK9_MD5: str | None = None

ENWIK8_URL = "https://mattmahoney.net/dc/enwik8.zip"
ENWIK9_URL = "https://mattmahoney.net/dc/enwik9.zip"

# Census split registry: name -> (offset, length) into enwik8.
# train8m/calib/test are the census ladder's bytes. train90m is the
# fed-training split (m4b) and OVERLAPS calib and test by construction —
# models trained on it may only be evaluated on `val`.
ENWIK8_SPLITS: dict[str, tuple[int, int]] = {
    "train8m": (0, 8_000_000),
    "train90m": (0, 90_000_000),
    "calib": (60_000_000, 262_144),
    "test": (70_000_000, 524_288),
    "val": (95_000_000, 524_288),
}


def default_enwik_root() -> Path:
    env = os.environ.get("CHRONOHORN_ENWIK_ROOT")
    if env:
        return Path(env)
    # python/chronohorn/data/enwik.py -> repo root is parents[3]
    return Path(__file__).resolve().parents[3] / "data" / "roots" / "enwik"


def enwik8_path(root: Path | None = None) -> Path:
    return (root or default_enwik_root()) / "enwik8"


def enwik9_path(root: Path | None = None) -> Path:
    return (root or default_enwik_root()) / "enwik9"


def check_enwik(
    path: Path,
    *,
    expected_size: int | None = ENWIK8_SIZE,
    expected_md5: str | None = None,
) -> list[str]:
    """Verify an enwik file: size, optional md5, and P6 content checks.

    Returns a list of problems (empty = healthy). The P6 checks catch the
    session-11 bug class — wrong dtype, wrong encoding, contaminated or
    truncated data — before any run trains on it.
    """
    problems: list[str] = []
    if not path.exists():
        return [f"{path}: missing"]
    size = path.stat().st_size
    if expected_size is not None and size != expected_size:
        problems.append(f"{path.name}: size {size} != expected {expected_size}")
    if expected_md5 is not None:
        digest = hashlib.md5()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_md5:
            problems.append(f"{path.name}: md5 {digest.hexdigest()} != expected {expected_md5}")
    sample = np.fromfile(path, dtype=np.uint8, count=1_000_000)
    check = check_sample(sample, byte_level=True)
    if check.fatal:
        problems.extend(f"{path.name}: {w}" for w in check.warnings)
    return problems


def load_enwik8(root: Path | None = None) -> np.memmap:
    """Memory-map the full enwik8 as uint8. Verifies size, not content."""
    path = enwik8_path(root)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run `chronohorn data enwik --provision --source <copy>`"
        )
    size = path.stat().st_size
    if size != ENWIK8_SIZE:
        raise ValueError(f"{path}: size {size} != {ENWIK8_SIZE} — truncated or wrong file")
    return np.memmap(path, dtype=np.uint8, mode="r", shape=(ENWIK8_SIZE,))


def load_split(name: str, root: Path | None = None) -> np.ndarray:
    """Load a canonical enwik8 split by name (see ENWIK8_SPLITS)."""
    if name not in ENWIK8_SPLITS:
        raise KeyError(f"unknown enwik8 split {name!r}; known: {sorted(ENWIK8_SPLITS)}")
    offset, length = ENWIK8_SPLITS[name]
    data = load_enwik8(root)
    return data[offset : offset + length]


def provision_enwik(
    *,
    root: Path,
    which: str = "enwik8",
    source: Path | None = None,
    download: bool = False,
) -> Path:
    """Materialize enwik8/enwik9 at the canonical root and verify it.

    Copies from `source` if given; downloads only when `download=True`
    (metered internet — never implicit). Raises on verification failure.
    """
    expected_size, expected_md5, url = {
        "enwik8": (ENWIK8_SIZE, ENWIK8_MD5, ENWIK8_URL),
        "enwik9": (ENWIK9_SIZE, ENWIK9_MD5, ENWIK9_URL),
    }[which]
    dest = root / which
    if dest.exists():
        problems = check_enwik(dest, expected_size=expected_size, expected_md5=expected_md5)
        if not problems:
            return dest
        raise ValueError(f"{dest} exists but fails checks: {problems}")

    root.mkdir(parents=True, exist_ok=True)
    if source is not None:
        tmp = dest.with_name(dest.name + ".tmp")
        try:
            os.link(source, tmp)
        except OSError:
            shutil.copy2(source, tmp)
    elif download:
        import io
        import urllib.request
        import zipfile

        print(f"downloading {url} (metered!)", flush=True)
        with urllib.request.urlopen(url) as resp:
            payload = resp.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            with zf.open(which) as src, open(dest.with_name(dest.name + ".tmp"), "wb") as out:
                shutil.copyfileobj(src, out)
        tmp = dest.with_name(dest.name + ".tmp")
    else:
        raise FileNotFoundError(
            f"{dest} missing and no --source given. Internet is metered, so "
            f"downloads are explicit: rerun with --download to fetch {url}"
        )

    problems = check_enwik(tmp, expected_size=expected_size, expected_md5=expected_md5)
    if problems:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"provisioned copy fails checks: {problems}")
    os.replace(tmp, dest)
    return dest


def write_byte_shard(path: Path, byte_values: np.ndarray) -> None:
    """Write bytes as a chronohorn-format uint16 shard (header + payload)."""
    from chronohorn.data.byte_reader import HEADER_INTS, TOKEN_SHARD_MAGIC, TOKEN_SHARD_VERSION

    header = np.zeros(HEADER_INTS, dtype=np.int32)
    header[0] = TOKEN_SHARD_MAGIC
    header[1] = TOKEN_SHARD_VERSION
    header[2] = len(byte_values)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(header.tobytes() + byte_values.astype(np.uint16).tobytes())
    os.replace(tmp, path)


def build_shards_from_arrays(
    out_dir: Path,
    train_bytes: np.ndarray,
    val_bytes: np.ndarray,
    *,
    shard_bytes: int = 10_000_000,
) -> list[Path]:
    """Write train/val byte arrays as trainer-consumable shard files.

    Produces enwik_train_%06d.bin + enwik_val_000000.bin in the uint16
    header format `build_token_shard_dataset` globs, and P6-checks every
    shard after writing. Returns the shard paths.
    """
    from chronohorn.data.byte_reader import open_byte_shard_checked

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    n_shards = (len(train_bytes) + shard_bytes - 1) // shard_bytes
    for i in range(n_shards):
        path = out_dir / f"enwik_train_{i:06d}.bin"
        write_byte_shard(path, train_bytes[i * shard_bytes : (i + 1) * shard_bytes])
        paths.append(path)
    val_path = out_dir / "enwik_val_000000.bin"
    write_byte_shard(val_path, val_bytes)
    paths.append(val_path)
    for path in paths:
        open_byte_shard_checked(str(path))
    return paths


def build_enwik8_shards(
    root: Path | None = None,
    *,
    train_split: str = "train90m",
    shard_bytes: int = 10_000_000,
) -> Path:
    """Materialize canonical enwik8 training shards for the fleet.

    Shard root is ``<enwik_root>/shards_<train_split>``; the val shard is
    always the census `val` split — the only holdout disjoint from
    train90m. Idempotent: returns the existing dir if already complete.
    """
    enwik_root = root or default_enwik_root()
    out_dir = enwik_root / f"shards_{train_split}"
    marker = out_dir / "provenance.json"
    if marker.exists():
        return out_dir
    train = np.asarray(load_split(train_split, enwik_root))
    val = np.asarray(load_split("val", enwik_root))
    paths = build_shards_from_arrays(out_dir, train, val, shard_bytes=shard_bytes)
    import json as _json

    marker.write_text(_json.dumps({
        "source": "enwik8",
        "source_md5": ENWIK8_MD5,
        "train_split": train_split,
        "train_offsets": ENWIK8_SPLITS[train_split],
        "val_offsets": ENWIK8_SPLITS["val"],
        "shard_bytes": shard_bytes,
        "shards": [p.name for p in paths],
        "note": "val is the ONLY legal holdout for train90m models",
    }, indent=2))
    return out_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="chronohorn data enwik",
        description="Provision and verify canonical enwik8/enwik9 bytes.",
    )
    parser.add_argument("--root", default=None, help="Override the enwik root directory.")
    parser.add_argument("--enwik9", action="store_true", help="Operate on enwik9 instead of enwik8.")
    parser.add_argument("--provision", action="store_true", help="Materialize the file at the root.")
    parser.add_argument("--source", default=None, help="Existing local copy to provision from.")
    parser.add_argument("--download", action="store_true", help="Explicitly allow download (metered).")
    parser.add_argument("--verify", action="store_true", help="Verify the canonical copy (default action).")
    parser.add_argument("--build-shards", action="store_true", help="Build trainer shards from enwik8.")
    parser.add_argument("--train-split", default="train90m", help="Split to shard as training data.")
    args = parser.parse_args(argv)

    root = Path(args.root) if args.root else default_enwik_root()
    which = "enwik9" if args.enwik9 else "enwik8"

    if args.build_shards:
        out_dir = build_enwik8_shards(root, train_split=args.train_split)
        print(f"ok: shards at {out_dir} (P6-checked)")
        return 0

    if args.provision:
        source = Path(args.source) if args.source else None
        dest = provision_enwik(root=root, which=which, source=source, download=args.download)
        print(f"ok: {dest} provisioned and verified (size+md5+P6)")
        return 0

    expected_size, expected_md5 = (
        (ENWIK8_SIZE, ENWIK8_MD5) if which == "enwik8" else (ENWIK9_SIZE, ENWIK9_MD5)
    )
    problems = check_enwik(root / which, expected_size=expected_size, expected_md5=expected_md5)
    if problems:
        for problem in problems:
            print(f"FAIL: {problem}", file=sys.stderr)
        return 1
    print(f"ok: {root / which} verified (size+md5+P6)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
