"""Download and verify training data shards on the local machine.

Usage:
    chronohorn data provision                    # default: fineweb10B_sp1024, 80 train shards
    chronohorn data provision --train-shards 40  # fewer shards for dev
    chronohorn data provision --verify           # just check, don't download
    chronohorn data provision --data-root /data/chronohorn/fineweb10B_sp1024

Designed to be run directly on GPU nodes so they can self-provision.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

HF_REPO_ID = "willdepueoai/parameter-golf"
HF_REPO_TYPE = "dataset"
HF_TOKENIZER_SUBFOLDER = "datasets/tokenizers"

_VARIANT_CONFIG = {
    "sp1024": {
        "hf_subfolder": "datasets/datasets/fineweb10B_sp1024",
        "data_root": "/data/chronohorn/fineweb10B_sp1024",
        "tokenizer_dir": "/data/chronohorn/tokenizers",
        "train_shards": 80,
        "tokenizer_files": ("fineweb_1024_bpe.model", "fineweb_1024_bpe.vocab"),
    },
    "sp4096": {
        "hf_subfolder": "datasets/datasets/fineweb10B_sp4096",
        "data_root": "/data/chronohorn/fineweb10B_sp4096",
        "tokenizer_dir": "/data/chronohorn/tokenizers",
        "train_shards": 80,
        "tokenizer_files": ("fineweb_4096_bpe.model", "fineweb_4096_bpe.vocab"),
    },
    "bytes": {
        "source_variant": "sp1024",
        "data_root": "/data/chronohorn/fineweb10B_bytes",
        "tokenizer_dir": "/data/chronohorn/tokenizers",
        "train_shards": 80,
        "tokenizer_files": ("fineweb_1024_bpe.model", "fineweb_1024_bpe.vocab"),
        "is_byte_conversion": True,
    },
}
DEFAULT_VARIANT = "sp1024"
VAL_SHARDS = 1


def _train_shard_name(i: int) -> str:
    return f"fineweb_train_{i:06d}.bin"


def _val_shard_name(i: int) -> str:
    return f"fineweb_val_{i:06d}.bin"


def verify_data_root(data_root: Path, *, train_shards: int) -> list[str]:
    """Return list of missing files."""
    missing: list[str] = []
    for i in range(train_shards):
        path = data_root / _train_shard_name(i)
        if not path.exists() or path.stat().st_size == 0:
            missing.append(_train_shard_name(i))
    for i in range(VAL_SHARDS):
        path = data_root / _val_shard_name(i)
        if not path.exists() or path.stat().st_size == 0:
            missing.append(_val_shard_name(i))
    return missing


def verify_tokenizer(tokenizer_dir: Path, *, tokenizer_files: tuple[str, ...] = ("fineweb_1024_bpe.model", "fineweb_1024_bpe.vocab")) -> list[str]:
    """Return list of missing tokenizer files."""
    missing: list[str] = []
    for name in tokenizer_files:
        if not (tokenizer_dir / name).exists():
            missing.append(name)
    return missing


def _download_file(filename: str, subfolder: str, dest: Path) -> None:
    """Download a single file from HuggingFace and copy to dest."""
    from huggingface_hub import hf_hub_download

    cached = Path(hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        subfolder=subfolder,
        repo_type=HF_REPO_TYPE,
    )).resolve()
    try:
        os.link(cached, dest)
    except OSError:
        shutil.copy2(cached, dest)


def provision(
    *,
    data_root: Path,
    tokenizer_dir: Path,
    train_shards: int,
    hf_dataset_subfolder: str,
    tokenizer_files: tuple[str, ...] = ("fineweb_1024_bpe.model", "fineweb_1024_bpe.vocab"),
) -> int:
    """Download missing shards and tokenizer files. Returns count of files downloaded."""
    try:
        import huggingface_hub  # noqa: F401
    except ImportError as exc:
        print(
            "error: huggingface_hub is not installed.\n"
            "  Install it:  pip install huggingface_hub\n"
            "  Or in a venv: python3 -m venv /tmp/hf && /tmp/hf/bin/pip install huggingface_hub",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    data_root.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0

    # Train shards
    for i in range(train_shards):
        name = _train_shard_name(i)
        dest = data_root / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        print(f"  [{i + 1}/{train_shards}] {name}", flush=True)
        _download_file(name, hf_dataset_subfolder, dest)
        downloaded += 1

    # Val shards
    for i in range(VAL_SHARDS):
        name = _val_shard_name(i)
        dest = data_root / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        print(f"  val: {name}", flush=True)
        _download_file(name, hf_dataset_subfolder, dest)
        downloaded += 1

    # Tokenizer
    for name in tokenizer_files:
        dest = tokenizer_dir / name
        if dest.exists():
            continue
        print(f"  tokenizer: {name}", flush=True)
        _download_file(name, HF_TOKENIZER_SUBFOLDER, dest)
        downloaded += 1

    return downloaded


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="chronohorn data provision",
        description="Download and verify training data shards on this machine.",
    )
    parser.add_argument(
        "--variant",
        default=DEFAULT_VARIANT,
        choices=sorted(_VARIANT_CONFIG),
        help=f"Tokenizer variant (default: {DEFAULT_VARIANT})",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Where to store shards (default: variant-specific)",
    )
    parser.add_argument(
        "--tokenizer-dir",
        default=None,
        help="Where to store tokenizer files (default: variant-specific)",
    )
    parser.add_argument(
        "--train-shards",
        type=int,
        default=None,
        help="Number of training shards to download (default: variant-specific)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Check for missing files without downloading.",
    )
    args = parser.parse_args(argv)
    vcfg = _VARIANT_CONFIG[args.variant]
    data_root = Path(args.data_root or vcfg["data_root"])
    tokenizer_dir = Path(args.tokenizer_dir or vcfg["tokenizer_dir"])
    train_shards = args.train_shards or vcfg["train_shards"]
    tokenizer_files = vcfg["tokenizer_files"]

    missing_data = verify_data_root(data_root, train_shards=train_shards)
    missing_tok = verify_tokenizer(tokenizer_dir, tokenizer_files=tokenizer_files)

    if not missing_data and not missing_tok:
        total = train_shards + VAL_SHARDS
        print(f"ok: {total} shards + tokenizer present at {data_root}")
        return 0

    if args.verify:
        print(f"missing {len(missing_data)} shard(s), {len(missing_tok)} tokenizer file(s):")
        for name in missing_data + missing_tok:
            print(f"  {name}")
        return 1

    # Byte conversion: convert from source tokenizer shards using all cores
    if vcfg.get("is_byte_conversion"):
        source_variant = vcfg["source_variant"]
        source_cfg = _VARIANT_CONFIG[source_variant]
        source_root = Path(source_cfg["data_root"])
        tokenizer_path = tokenizer_dir / source_cfg["tokenizer_files"][0]

        # Ensure source shards exist first
        source_missing = verify_data_root(source_root, train_shards=train_shards)
        if source_missing:
            print(f"provisioning source {source_variant} shards first...")
            provision(
                data_root=source_root,
                tokenizer_dir=tokenizer_dir,
                train_shards=train_shards,
                hf_dataset_subfolder=source_cfg["hf_subfolder"],
                tokenizer_files=source_cfg["tokenizer_files"],
            )

        # Convert to bytes using all cores
        print(f"converting {source_variant} → bytes using all cores...")
        from chronohorn.data.build_byte_shards import main as convert_main
        convert_main([
            "--input-dir", str(source_root),
            "--output-dir", str(data_root),
            "--tokenizer", str(tokenizer_path),
            "--workers", "0",
        ])
        still_missing = verify_data_root(data_root, train_shards=train_shards)
        if still_missing:
            print(f"error: {len(still_missing)} byte shard(s) still missing", file=sys.stderr)
            return 1
        print(f"done: byte shards at {data_root}")
        return 0

    print(f"provisioning {len(missing_data)} shard(s) + {len(missing_tok)} tokenizer file(s) to {data_root}")
    downloaded = provision(
        data_root=data_root,
        tokenizer_dir=tokenizer_dir,
        train_shards=train_shards,
        hf_dataset_subfolder=vcfg["hf_subfolder"],
        tokenizer_files=tokenizer_files,
    )
    print(f"done: downloaded {downloaded} file(s)")

    # Final verify
    still_missing = verify_data_root(data_root, train_shards=args.train_shards)
    if still_missing:
        print(f"error: {len(still_missing)} file(s) still missing after download", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
