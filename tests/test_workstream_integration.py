"""Integration tests: verify causal-bank models from decepticons work."""
from __future__ import annotations

import pytest

try:
    from decepticons.causal_bank import CausalBankConfig, scale_config
    from decepticons.models.causal_bank_torch import CausalBankModel
    import torch
except ImportError:
    pytest.skip("decepticons or torch not installed", allow_module_level=True)


def _train_steps(model, steps=5, seq_len=64, batch_size=2, vocab=1024, lr=1e-3):
    """Run a few training steps and return initial/final loss."""
    torch.manual_seed(42)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        x = torch.randint(0, vocab, (batch_size, seq_len))
        logits = model(x)
        target = x[:, 1:]
        pred = logits[:, :-1, :]
        loss = torch.nn.functional.cross_entropy(
            pred.reshape(-1, vocab), target.reshape(-1)
        )
        if hasattr(model, 'substrate_regularization'):
            loss = loss + model.substrate_regularization()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def test_frozen_baseline_trains():
    """Frozen baseline should train (integration sanity check)."""
    cfg = scale_config(CausalBankConfig(
        substrate_mode="frozen", oscillatory_frac=0.875,
    ), 6.0)
    model = CausalBankModel(256, cfg)
    losses = _train_steps(model, steps=10, vocab=256)
    assert min(losses) < losses[0], f"loss should improve: {losses}"
