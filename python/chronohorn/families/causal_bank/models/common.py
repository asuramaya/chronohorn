from __future__ import annotations

import hashlib
import math

import numpy as np


def _stable_seed(base_seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFFFFFF


def _rng_for(base_seed: int, name: str) -> np.random.Generator:
    return np.random.default_rng(_stable_seed(base_seed, name))


def _xavier_uniform(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    if len(shape) != 2:
        raise ValueError(f"xavier_uniform expects rank-2 shape, got {shape}")
    fan_out, fan_in = shape
    limit = math.sqrt(6.0 / max(fan_in + fan_out, 1))
    return rng.uniform(-limit, limit, size=shape).astype(np.float32)


def _embedding_uniform(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    if len(shape) != 2:
        raise ValueError(f"embedding_uniform expects rank-2 shape, got {shape}")
    _, embedding_dim = shape
    limit = math.sqrt(1.0 / max(embedding_dim, 1))
    return rng.uniform(-limit, limit, size=shape).astype(np.float32)
