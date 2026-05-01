"""Patch-at-readout loss correctness tests."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional


def _build_shifted_targets(y: torch.Tensor, n: int, ignore_index: int = -100) -> torch.Tensor:
    """Build [B, T, N] targets where targets[:, t, i] = y[:, t+i], ignore at tail."""
    b, t = y.shape
    out = torch.full((b, t, n), ignore_index, dtype=y.dtype, device=y.device)
    for i in range(n):
        if i == 0:
            out[:, :, 0] = y
        elif i < t:
            out[:, : t - i, i] = y[:, i:]
    return out


def _patch_loss(logits: torch.Tensor, y: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """Loss for [B, T, N, V] logits with N-shifted targets from y [B, T]."""
    b, t, n, v = logits.shape
    targets = _build_shifted_targets(y, n, ignore_index)
    return F.cross_entropy(
        logits.reshape(-1, v), targets.reshape(-1), ignore_index=ignore_index
    )


def test_n1_matches_baseline():
    """With N=1, patch loss must equal the baseline cross-entropy."""
    torch.manual_seed(0)
    b, t, v = 2, 8, 16
    logits_flat = torch.randn(b, t, v)
    y = torch.randint(0, v, (b, t))
    baseline = F.cross_entropy(logits_flat.reshape(-1, v), y.reshape(-1))
    patch = _patch_loss(logits_flat.unsqueeze(2), y)  # [B,T,1,V]
    assert torch.allclose(baseline, patch, atol=1e-6), f"{baseline.item()} vs {patch.item()}"


def test_shifted_targets_tail_is_ignored():
    """Last N-1 positions per row must be ignored for higher heads."""
    y = torch.arange(1, 9).unsqueeze(0)  # [1, 8]: values 1..8
    targets = _build_shifted_targets(y, n=4, ignore_index=-100)
    # Head 0 (i=0): full y
    assert (targets[0, :, 0] == y[0]).all()
    # Head 1 (i=1): y shifted left by 1, last position ignored
    assert (targets[0, :7, 1] == y[0, 1:]).all()
    assert targets[0, 7, 1].item() == -100
    # Head 3 (i=3): position 4 targets y[7] = 8; last 3 positions ignored
    assert targets[0, 4, 3].item() == 8
    assert (targets[0, -3:, 3] == -100).all()


def test_finite_loss_with_random_logits():
    """Sanity: patch loss is finite with random logits and N=4."""
    torch.manual_seed(0)
    b, t, n, v = 2, 16, 4, 32
    logits = torch.randn(b, t, n, v)
    y = torch.randint(0, v, (b, t))
    loss = _patch_loss(logits, y)
    assert torch.isfinite(loss).item()
    # Expected value around log(v) for uniform random logits.
    assert 3.0 < loss.item() < 4.5  # log(32) ≈ 3.46
