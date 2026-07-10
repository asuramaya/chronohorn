"""Async checkpoint writer — the covenant-safety battery.

Proves offline (no training run, no GPU) that background writes are byte-faithful
to synchronous ones, that the snapshot is a consistent clone (immune to later
param mutation), that the write never touches the training RNG, that a resume
point round-trips, and that the prev-then-new ordering never drops the last
resumable state.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from chronohorn.train.async_checkpoint import (
    AsyncCheckpointWriter,
    snapshot_training_state,
)


def _model_and_opt():
    torch.manual_seed(0)
    model = torch.nn.Linear(8, 4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    # one step so the optimizer carries state (exp_avg, exp_avg_sq)
    loss = model(torch.randn(3, 8)).sum()
    loss.backward()
    opt.step()
    return model, opt


def _snap(model, opt, step=5):
    return snapshot_training_state(
        model, opt, step=step, gear={"phase": 1}, grad_scaler=None,
        rng_cpu=torch.random.get_rng_state(), rng_cuda=None)


def test_snapshot_is_consistent_clone():
    """Mutating the model AFTER snapshot must not change the snapshot."""
    model, opt = _model_and_opt()
    snap = _snap(model, opt)
    before = snap["model"]["weight"].clone()
    with torch.no_grad():
        model.weight.add_(100.0)                 # live mutation post-snapshot
    assert torch.equal(snap["model"]["weight"], before)   # snapshot frozen
    assert not torch.equal(snap["model"]["weight"], model.weight.detach().cpu())


def test_snapshot_does_not_consume_rng():
    """The covenant guard: snapshotting reads state, never advances the RNG."""
    model, opt = _model_and_opt()
    rng_before = torch.random.get_rng_state()
    _snap(model, opt)
    assert torch.equal(torch.random.get_rng_state(), rng_before)


@pytest.mark.parametrize("enabled", [False, True])
def test_write_is_faithful_and_reloads(tmp_path, enabled):
    model, opt = _model_and_opt()
    snap = _snap(model, opt)
    path = tmp_path / "run_step5.training_state.pt"
    w = AsyncCheckpointWriter(enabled=enabled)
    w.submit(snap, path)
    w.close()
    assert path.exists()
    loaded = torch.load(path, weights_only=False)
    assert loaded["step"] == 5 and loaded["gear"] == {"phase": 1}
    assert torch.equal(loaded["model"]["weight"], snap["model"]["weight"])
    # optimizer moments survive the round-trip
    assert torch.equal(
        loaded["optimizer"]["state"][0]["exp_avg"],
        snap["optimizer"]["state"][0]["exp_avg"])


def test_async_matches_sync_bytes(tmp_path):
    """Background write produces the identical reloaded state as the sync path."""
    model, opt = _model_and_opt()
    sync_p, async_p = tmp_path / "sync.pt", tmp_path / "async.pt"
    AsyncCheckpointWriter(enabled=False).submit(_snap(model, opt), sync_p)
    wa = AsyncCheckpointWriter(enabled=True)
    wa.submit(_snap(model, opt), async_p)
    wa.close()
    a, b = torch.load(sync_p, weights_only=False), torch.load(async_p, weights_only=False)
    assert torch.equal(a["model"]["weight"], b["model"]["weight"])
    assert torch.equal(a["rng_cpu"], b["rng_cpu"])


def test_prev_deleted_only_after_new_write(tmp_path):
    """write-then-delete-prev: the last resumable state is never dropped."""
    model, opt = _model_and_opt()
    p0, p1 = tmp_path / "s0.pt", tmp_path / "s1.pt"
    w = AsyncCheckpointWriter(enabled=True)
    w.submit(_snap(model, opt, step=0), p0, prev=None)
    w.submit(_snap(model, opt, step=1), p1, prev=p0)
    w.close()
    assert p1.exists() and not p0.exists()   # new kept, prev cleaned


def test_snapshots_frozen_across_a_training_loop(tmp_path):
    """The loop-integration covenant proof: async checkpoints written WHILE
    training continues each reload to the exact state at their step — later
    optimizer steps racing against the background writer never corrupt an
    in-flight snapshot."""
    model, opt = _model_and_opt()
    w = AsyncCheckpointWriter(enabled=True)
    expected, paths = {}, {}
    for step in range(1, 9):
        loss = model(torch.randn(3, 8)).sum()      # a real training step:
        opt.zero_grad(); loss.backward(); opt.step()   # mutates params + moments
        snap = snapshot_training_state(
            model, opt, step=step, gear={}, grad_scaler=None,
            rng_cpu=torch.random.get_rng_state(), rng_cuda=None)
        expected[step] = snap["model"]["weight"].clone()
        paths[step] = tmp_path / f"s{step}.pt"
        w.submit(snap, paths[step])                # no prev: keep all to compare
    w.close()
    for step, p in paths.items():
        loaded = torch.load(p, weights_only=False)
        assert torch.equal(loaded["model"]["weight"], expected[step]), f"step {step}"


def test_write_error_is_surfaced(tmp_path):
    model, opt = _model_and_opt()
    w = AsyncCheckpointWriter(enabled=True)
    w.submit(_snap(model, opt), tmp_path / "nonexistent_dir" / "x.pt")  # bad path
    with pytest.raises(Exception):
        w.close()
