"""Integration tests: verify the 3 workstreams actually train."""
from __future__ import annotations

import pytest

try:
    from open_predictive_coder.causal_bank import CausalBankConfig, scale_config
except ImportError:
    pytest.skip("open_predictive_coder not installed", allow_module_level=True)

import torch
from chronohorn.models.causal_bank_torch import CausalBankModel


def _train_steps(model, steps=5, seq_len=64, batch_size=2, vocab=1024, lr=1e-3):
    """Run a few training steps and return initial/final loss."""
    torch.manual_seed(42)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        x = torch.randint(0, vocab, (batch_size, seq_len))
        logits = model(x)
        # Shift for next-token prediction
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


def _loss_improved(losses: list[float]) -> bool:
    """Check if training made progress: min loss is below first loss."""
    return min(losses) < losses[0]


def test_learnable_decays_train_and_change():
    """Learnable decays should have gradients and change during training."""
    cfg = scale_config(CausalBankConfig(
        substrate_mode="learnable_decays", oscillatory_frac=0.875,
    ), 6.0)
    model = CausalBankModel(256, cfg)

    initial_decays = model.linear_decays.data.clone()
    assert model.linear_decays.requires_grad, "decays should be trainable"

    losses = _train_steps(model, steps=20, vocab=256)
    assert _loss_improved(losses), f"loss should improve: {losses}"

    changed = (model.linear_decays.data - initial_decays).abs().sum().item()
    assert changed > 0, "decays should have changed after training"


def test_learnable_mixing_train_and_change():
    """Learnable mixing (input projection) should change during training."""
    cfg = scale_config(CausalBankConfig(
        substrate_mode="learnable_mixing", oscillatory_frac=0.875,
    ), 6.0)
    model = CausalBankModel(256, cfg)

    initial_proj = model.linear_in_proj.data.clone()
    assert model.linear_in_proj.requires_grad, "in_proj should be trainable"

    losses = _train_steps(model, steps=20, vocab=256)
    assert _loss_improved(losses), f"loss should improve: {losses}"

    changed = (model.linear_in_proj.data - initial_proj).abs().sum().item()
    assert changed > 0, "in_proj should have changed after training"


def test_online_memory_forward_and_train():
    """Model with online memory should forward and train without error."""
    cfg = scale_config(CausalBankConfig(
        memory_kind="ngram", oscillatory_frac=0.875,
    ), 6.0)
    model = CausalBankModel(256, cfg)

    assert model._use_online_memory, "memory should be active"

    losses = _train_steps(model, steps=3, vocab=256, seq_len=32)
    assert losses[-1] < losses[0] * 1.5, f"loss should not explode: {losses}"


def test_online_memory_gate_learns():
    """The memory gate should move away from zero with training."""
    cfg = scale_config(CausalBankConfig(
        memory_kind="ngram", oscillatory_frac=0.875,
    ), 6.0)
    model = CausalBankModel(256, cfg)

    initial_gate = model._memory_gate.data.item()
    _train_steps(model, steps=10, vocab=256, seq_len=32)
    final_gate = model._memory_gate.data.item()

    # Gate should have moved (doesn't matter which direction)
    assert abs(final_gate - initial_gate) > 1e-6, \
        f"memory gate should change: {initial_gate} -> {final_gate}"


def test_recurrent_readout_forward_and_train():
    """GRU readout should forward and train."""
    cfg = scale_config(CausalBankConfig(
        linear_readout_kind="recurrent", oscillatory_frac=0.875,
    ), 6.0)
    model = CausalBankModel(256, cfg)

    losses = _train_steps(model, steps=20, vocab=256)
    assert _loss_improved(losses), f"loss should improve: {losses}"


def test_frozen_baseline_comparison():
    """Frozen baseline should also train (sanity check)."""
    cfg = scale_config(CausalBankConfig(
        substrate_mode="frozen", oscillatory_frac=0.875,
    ), 6.0)
    model = CausalBankModel(256, cfg)

    assert not hasattr(model, '_recompute_kernel') or not model._recompute_kernel

    losses = _train_steps(model, steps=20, vocab=256)
    assert _loss_improved(losses), f"loss should improve: {losses}"
