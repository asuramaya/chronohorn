"""Shared eval-harness utilities — the blocks the kNN harnesses copy-pasted.

enwik loading (via the package's canonical reader, not a hardcoded path), the
bits-per-byte-from-log-probs reduction, and the paired-window delta CI. Kept
tiny and dependency-light so any eval harness can reuse them.
"""
from __future__ import annotations

import numpy as np
import torch

from ..data.enwik import load_enwik8


def enwik8_bytes() -> np.ndarray:
    """Full enwik8 as an int64 array (memmap materialized once)."""
    return np.asarray(load_enwik8()).astype(np.int64)


def bpb_from_logp(logp: torch.Tensor, y: torch.Tensor) -> float:
    """Mean bits-per-byte of targets y under log-probabilities logp (T, vocab)."""
    idx = torch.arange(len(y), device=logp.device)
    return float((-logp[idx, y] / np.log(2)).mean())


def paired_window_delta(mix_logp: torch.Tensor, base_logp: torch.Tensor,
                        y: torch.Tensor, n_windows: int) -> dict:
    """Per-window paired delta of two log-prob tensors, with a 95% CI.

    Reshapes the flat per-position bits into (n_windows, positions), means each
    window, and reports the paired mean delta, its 1.96-sigma CI, and the count
    of windows where the mix beat the base. Windows are the resampling unit —
    positions within a window are correlated, so the CI must be over windows.
    """
    idx = torch.arange(len(y), device=mix_logp.device)
    bits_mix = (-mix_logp[idx, y] / np.log(2)).reshape(n_windows, -1).mean(1)
    bits_base = (-base_logp[idx, y] / np.log(2)).reshape(n_windows, -1).mean(1)
    d = (bits_mix - bits_base).detach().cpu().numpy()
    ci = 1.96 * d.std() / np.sqrt(len(d))
    return {
        "mix": float(bits_mix.mean()),
        "base": float(bits_base.mean()),
        "delta": float(d.mean()),
        "ci": float(ci),
        "improved": int((d < 0).sum()),
        "windows": len(d),
    }
