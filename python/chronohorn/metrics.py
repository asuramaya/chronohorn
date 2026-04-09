from __future__ import annotations

from collections.abc import Mapping
from typing import Any

DEFAULT_TEXT_BYTES_PER_TOKEN = 2.436


def _positive_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric <= 0:
        return None
    return numeric


def text_bytes_per_token_from_payload(
    *,
    model: Mapping[str, Any] | None = None,
    training: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
    default: float = DEFAULT_TEXT_BYTES_PER_TOKEN,
) -> float:
    performance = training.get("performance", {}) if isinstance(training, Mapping) else {}
    train_cfg = config.get("train", {}) if isinstance(config, Mapping) else {}
    for candidate in (
        (model or {}).get("text_bytes_per_token"),
        performance.get("test_bytes_per_token"),
        performance.get("text_bytes_per_token"),
        train_cfg.get("text_bytes_per_token"),
        (config or {}).get("text_bytes_per_token"),
    ):
        numeric = _positive_float(candidate)
        if numeric is not None:
            return numeric
    return default


def bpb_from_bits_per_token(
    bits_per_token: Any,
    *,
    text_bytes_per_token: float = DEFAULT_TEXT_BYTES_PER_TOKEN,
) -> float | None:
    bpt = _positive_float(bits_per_token)
    if bpt is None:
        return None
    ratio = _positive_float(text_bytes_per_token)
    if ratio is None:
        return None
    return bpt / ratio


def probe_bpb_from_row(
    row: Mapping[str, Any],
    *,
    text_bytes_per_token: float = DEFAULT_TEXT_BYTES_PER_TOKEN,
) -> float | None:
    for key in ("bpb", "test_bpb"):
        numeric = _positive_float(row.get(key))
        if numeric is not None:
            return numeric
    for key in ("bits_per_token", "test_bits_per_token"):
        numeric = bpb_from_bits_per_token(row.get(key), text_bytes_per_token=text_bytes_per_token)
        if numeric is not None:
            return numeric
    return None
