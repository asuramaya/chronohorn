"""Convert tokenized shards to raw byte shards using all CPU cores.

Supports any sentencepiece tokenizer (sp1024, sp4096, sp8192).
Reads tokenized shards, decodes to text via sentencepiece, encodes as raw
bytes (0-255). Output shards use the same header format with vocab_size=256.

Usage:
    python -m chronohorn.data.build_byte_shards \
        --input-dir /data/chronohorn/fineweb10B_sp1024 \
        --output-dir /data/chronohorn/fineweb10B_bytes \
        --tokenizer /data/chronohorn/tokenizers/fineweb_1024_bpe.model \
        --workers 0  # 0 = all cores
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

TOKEN_SHARD_MAGIC = 20240520
TOKEN_SHARD_VERSION = 1
HEADER_INTS = 256
HEADER_BYTES = HEADER_INTS * np.dtype(np.int32).itemsize


def _load_shard_tokens(path: Path) -> np.ndarray:
    blob = path.read_bytes()
    if len(blob) >= HEADER_BYTES:
        header = np.frombuffer(blob[:HEADER_BYTES], dtype=np.int32, count=HEADER_INTS)
        if header.size >= 3 and int(header[0]) == TOKEN_SHARD_MAGIC and int(header[1]) == TOKEN_SHARD_VERSION:
            token_count = int(header[2])
            return np.frombuffer(blob[HEADER_BYTES:], dtype=np.uint16, count=token_count).astype(np.int32)
    return np.frombuffer(blob, dtype=np.uint16).astype(np.int32)


def _write_byte_shard(path: Path, byte_tokens: np.ndarray) -> None:
    header = np.zeros(HEADER_INTS, dtype=np.int32)
    header[0] = TOKEN_SHARD_MAGIC
    header[1] = TOKEN_SHARD_VERSION
    header[2] = len(byte_tokens)
    path.write_bytes(header.tobytes() + byte_tokens.astype(np.uint16).tobytes())


def _convert_one(args_tuple: tuple) -> tuple[str, int]:
    """Worker function for multiprocessing. Each worker loads its own sp model."""
    input_path_str, output_path_str, tokenizer_path = args_tuple
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    input_path = Path(input_path_str)
    output_path = Path(output_path_str)
    tokens = _load_shard_tokens(input_path)
    text = sp.decode(tokens.tolist())
    raw_bytes = text.encode("utf-8", errors="replace")
    byte_tokens = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.uint16)
    _write_byte_shard(output_path, byte_tokens)
    return input_path.name, len(byte_tokens)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Convert tokenized shards to byte shards (parallel)")
    parser.add_argument("--input-dir", required=True, help="Directory with tokenized shards")
    parser.add_argument("--output-dir", required=True, help="Output directory for byte shards")
    parser.add_argument("--tokenizer", required=True, help="Path to sentencepiece model")
    parser.add_argument("--workers", type=int, default=0, help="Number of workers (0 = all cores)")
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shards = sorted(input_dir.glob("fineweb_*.bin"))
    if not shards:
        print(f"No shards found in {input_dir}", flush=True)
        return

    n_workers = args.workers or os.cpu_count() or 4
    n_workers = min(n_workers, len(shards))

    work = [
        (str(shard), str(output_dir / shard.name), args.tokenizer)
        for shard in shards
    ]

    print(f"Converting {len(shards)} shards with {n_workers} workers...", flush=True)
    t0 = time.time()

    total_bytes = 0
    with mp.Pool(n_workers) as pool:
        for name, n_bytes in pool.imap_unordered(_convert_one, work):
            total_bytes += n_bytes
            print(f"  {name} → {n_bytes:,} bytes", flush=True)

    elapsed = time.time() - t0
    print(f"\nTotal: {total_bytes:,} bytes in {output_dir} ({elapsed:.1f}s, {len(shards)/elapsed:.1f} shards/s)", flush=True)


if __name__ == "__main__":
    main()
