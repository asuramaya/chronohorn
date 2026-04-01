from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def tmp_manifest(tmp_path: Path):
    """Write a temporary manifest JSONL and return its path."""
    def _write(rows: list[dict]) -> Path:
        p = tmp_path / "test_manifest.jsonl"
        with p.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        return p
    return _write


@pytest.fixture
def sample_job() -> dict:
    return {
        "name": "test-job-1",
        "family": "causal-bank",
        "backend": "cuda",
        "resource_class": "cuda_gpu",
        "launcher": "managed_command",
        "command": "echo hello",
        "scale": 14.0,
        "steps": 1000,
        "learning_rate": 0.0015,
        "oscillatory_schedule": "logspace",
        "input_proj_scheme": "random",
    }
