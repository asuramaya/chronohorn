"""Resume round-trip: a crashed-and-resumed run must equal the straight run.

The resume contract (b1/c/#29): a periodic checkpoint carries the FULL
trajectory — weights, optimizer moments, RNG, AMP scale, gear — and the
sequential data stream fast-forwards to the same position, so training
steps 5..8 after a resume see bit-identical inputs and produce bit-identical
weights to a run that never stopped.

Simulates the crash shape periodic states exist for: run B stops at step 4
with NO final save (only the periodic _step4 training_state survives), then
resumes to step 8. Compared against run A which trains 0..8 uninterrupted.

CPU / fp32 / --lr-schedule none so equality is exact, not approximate.
(The gear-pinning branch itself is CUDA-only — on CPU the gearbox never
runs and the pinned gear is None — but the state key is asserted present.)
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

TRAINER = "chronohorn.families.causal_bank.training.train_causal_bank_torch"


def _write_toy_root(root: Path, rng: np.random.Generator) -> None:
    """Text-shaped byte shards (lowercase words) so data sanity checks pass."""
    root.mkdir(parents=True)
    letters = np.frombuffer(b"abcdefghijklmnopqrstuvwxyz", dtype=np.uint8)
    for split, n in (("train", 300_000), ("val", 40_000)):
        body = letters[rng.integers(0, 26, size=n)].copy()
        body[rng.random(n) < 0.18] = ord(" ")  # word boundaries
        body[rng.random(n) < 0.01] = ord("\n")
        body.astype(np.uint16).tofile(root / f"toy_{split}_000000.bin")


def _train(json_path: Path, root: Path, steps: int, *extra: str) -> None:
    cmd = [
        sys.executable, "-B", "-m", TRAINER,
        "--data-root", str(root), "--vocab-size", "256", "--seed", "42",
        "--steps", str(steps), "--seq-len", "32", "--batch-size", "2",
        "--scale", "1.0", "--learning-rate", "0.001",
        "--lr-schedule", "none", "--lr-warmup-steps", "0",
        "--device", "cpu", "--save-checkpoint-every", "4",
        "--json", str(json_path), *extra,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f"trainer failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"


def test_resume_roundtrip_bit_exact(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _write_toy_root(root, np.random.default_rng(0))

    # A: the straight run, 0..8, with final save.
    a_dir = tmp_path / "a"
    a_dir.mkdir()
    _train(a_dir / "run.json", root, 8, "--save-checkpoint")

    # B: "crash" at step 4 — periodic saves only, no final artifacts
    # (--no-save-checkpoint suppresses the default final save, leaving exactly
    # the file set a SIGKILL after step 4 would leave).
    b_dir = tmp_path / "b"
    b_dir.mkdir()
    _train(b_dir / "run.json", root, 4, "--no-save-checkpoint")
    periodic_ts = b_dir / "run_step4.training_state.pt"
    assert periodic_ts.exists(), "periodic training_state sidecar not written (#26)"
    state = torch.load(periodic_ts, map_location="cpu", weights_only=False)
    for key in ("model", "optimizer", "step", "rng_cpu", "grad_scaler", "gear"):
        assert key in state, f"periodic training_state missing {key!r}"
    assert state["step"] == 4

    # B resumed: continue 5..8 from the periodic state.
    _train(b_dir / "run2.json", root, 8, "--save-checkpoint",
           "--resume", str(periodic_ts))

    a_final = torch.load(a_dir / "run.checkpoint.pt", map_location="cpu", weights_only=False)
    b_final = torch.load(b_dir / "run2.checkpoint.pt", map_location="cpu", weights_only=False)
    assert sorted(a_final) == sorted(b_final)
    for name in a_final:
        assert torch.equal(a_final[name], b_final[name]), (
            f"{name} diverged after resume: straight-through and resumed runs "
            f"must be bit-identical (max abs diff "
            f"{(a_final[name] - b_final[name]).abs().max().item():.3e})"
        )


def test_periodic_states_roll_and_final_supersedes(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _write_toy_root(root, np.random.default_rng(1))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _train(run_dir / "run.json", root, 8, "--save-checkpoint")
    # Rolling keep-latest: step4 deleted when step8 was written; final
    # training_state supersedes step8. Model bodies all remain for forensics.
    assert not (run_dir / "run_step4.training_state.pt").exists()
    assert not (run_dir / "run_step8.training_state.pt").exists()
    assert (run_dir / "run.training_state.pt").exists()
    assert (run_dir / "run_step4.checkpoint.pt").exists()
    assert (run_dir / "run_step8.checkpoint.pt").exists()
    final_state = torch.load(run_dir / "run.training_state.pt",
                             map_location="cpu", weights_only=False)
    assert "gear" in final_state and "grad_scaler" in final_state
