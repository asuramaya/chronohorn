"""Safe byte-shard reader with sanity checks.

Centralizes the lesson from heinrich session 11 P6: chronohorn writes byte
shards as uint16 (header + payload). Naive uint8 readers see header bytes
plus null-interleaved payload, producing apparently-random data and
catastrophically wrong predictions. Session 11 caught this in heinrich; the
fix lived in heinrich's CLI but didn't propagate. Session 13 hit the same
bug in a new pilot script.

This module is the canonical reader. All chronohorn code that reads byte
shards should import from here. The 4-point sanity check follows
heinrich's P6 spec.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

TOKEN_SHARD_MAGIC = 20240520
TOKEN_SHARD_VERSION = 1
HEADER_INTS = 256
HEADER_BYTES = HEADER_INTS * 4  # int32

# Sanity-check thresholds (from heinrich P6).
MAX_BYTE_VALUE = 255
MAX_ZERO_FRACTION = 0.30  # >30% zeros → null-interleaved or UTF-16 indicator
MIN_PRINTABLE_FRACTION = 0.25  # <25% printable ASCII → likely binary or wrong encoding
MAX_BYTE_ENTROPY = 7.5  # >7.5 bits/byte → near-uniform, not language


@dataclass(frozen=True)
class ByteShardCheck:
    n_tokens: int
    max_value: int
    zero_fraction: float
    printable_fraction: float
    entropy_bits: float
    warnings: tuple[str, ...]
    fatal: bool


def check_sample(tokens: np.ndarray, byte_level: bool = True) -> ByteShardCheck:
    """Check a token sample. Returns a ByteShardCheck dataclass.

    `tokens` is a 1-D array of token values (typically uint16 read from
    a chronohorn byte shard, which holds byte values 0-255).
    """
    n = int(tokens.size)
    sample = tokens
    if n > 5_000_000:
        sample = tokens[:5_000_000]

    counts = np.bincount(sample.astype(np.int64), minlength=256)
    n_sample = int(sample.size)
    zero_frac = float(counts[0]) / n_sample if n_sample else 0.0
    max_val = int(sample.max())

    # Printable ASCII: [32, 127) plus tab/lf/cr.
    printable = int(((sample >= 32) & (sample < 127)).sum()
                    + (sample == 9).sum()
                    + (sample == 10).sum()
                    + (sample == 13).sum())
    printable_frac = printable / n_sample if n_sample else 0.0

    # Byte-distribution Shannon entropy (effective vocab limited to 256 for byte data).
    p = counts[:256] / counts[:256].sum() if counts[:256].sum() else np.array([1.0])
    p = p[p > 0]
    entropy = float(-(p * np.log2(p)).sum()) if p.size else 0.0

    warnings = []
    fatal = False

    if byte_level and max_val > MAX_BYTE_VALUE:
        warnings.append(
            f"max value {max_val} > 255; file is probably sentencepiece-tokenized "
            f"or wrongly read (try uint16 reader with header skip)")
        fatal = True

    if zero_frac > MAX_ZERO_FRACTION:
        warnings.append(
            f"{zero_frac*100:.1f}% zero bytes (>30% threshold). Indicates UTF-16 "
            f"encoding or null-interleaved pattern from uint8-reader-on-uint16-file bug. "
            f"Real UTF-8 English is 1-5% zeros.")
        fatal = True

    if zero_frac < MAX_ZERO_FRACTION and printable_frac < MIN_PRINTABLE_FRACTION:
        warnings.append(
            f"{printable_frac*100:.1f}% printable ASCII (<25% threshold). Likely binary "
            f"or wrongly encoded for a text language model.")

    if entropy > MAX_BYTE_ENTROPY:
        warnings.append(
            f"byte entropy {entropy:.2f} bits (>7.5 threshold). Near-uniform; unsuitable "
            f"for language modeling.")

    return ByteShardCheck(
        n_tokens=n, max_value=max_val, zero_fraction=zero_frac,
        printable_fraction=printable_frac, entropy_bits=entropy,
        warnings=tuple(warnings), fatal=fatal,
    )


def open_byte_shard(path: str) -> np.memmap:
    """Memory-map a chronohorn byte shard, skipping the 1024-byte header.

    Returns a uint16 memmap of token IDs in 0-255 range. The chronohorn
    shard format stores byte values as uint16 (low byte = value, high byte =
    0) for format uniformity with sp-tokenized shards. Reading as uint8
    instead produces null-interleaved garbage — this is the bug that
    heinrich session 11 chapter 3 retracts 168 prior MRIs over.
    """
    file_size = os.path.getsize(path)
    if file_size < HEADER_BYTES:
        raise ValueError(f"{path}: file too small ({file_size} bytes < {HEADER_BYTES} header)")
    n_tokens = (file_size - HEADER_BYTES) // 2
    return np.memmap(path, dtype=np.uint16, mode="r", offset=HEADER_BYTES, shape=(n_tokens,))


def open_byte_shard_checked(path: str, sample_tokens: int = 1_000_000,
                            raise_on_fatal: bool = True) -> np.memmap:
    """Open a shard and run sanity checks. Raises ValueError on fatal warnings."""
    arr = open_byte_shard(path)
    sample = np.asarray(arr[:min(sample_tokens, arr.size)])
    check = check_sample(sample, byte_level=True)
    if check.warnings:
        for w in check.warnings:
            import sys
            print(f"[byte_reader] {path}: WARNING — {w}", file=sys.stderr, flush=True)
        if check.fatal and raise_on_fatal:
            raise ValueError(
                f"{path}: byte-shard sanity check failed. {len(check.warnings)} warnings "
                f"(see above). Pass raise_on_fatal=False to bypass.")
    return arr
