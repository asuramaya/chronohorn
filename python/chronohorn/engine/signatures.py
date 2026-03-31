from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np


def summarize_named_arrays(named_arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    ordered = sorted((name, np.asarray(value, dtype=np.float32)) for name, value in named_arrays.items())
    digest = hashlib.sha256()
    total_count = 0
    total_sum = 0.0
    total_sq_sum = 0.0
    first_values: list[float] = []
    for name, array in ordered:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(b"\0")
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.tobytes())
        flat = contiguous.reshape(-1)
        total_count += int(flat.size)
        total_sum += float(flat.astype(np.float64, copy=False).sum())
        total_sq_sum += float(np.square(flat, dtype=np.float64).sum())
        if len(first_values) < 8:
            take = min(8 - len(first_values), int(flat.size))
            first_values.extend(float(v) for v in flat[:take].tolist())
    rms = math.sqrt(total_sq_sum / total_count) if total_count > 0 else 0.0
    return {
        "tensor_count": len(ordered),
        "value_count": total_count,
        "sha256": digest.hexdigest(),
        "sum": total_sum,
        "rms": rms,
        "first_values": first_values,
    }
