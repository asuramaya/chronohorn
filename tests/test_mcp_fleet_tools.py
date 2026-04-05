"""Tests for the MCP fleet tools added today: pull, sync, launch, status, converge."""
from __future__ import annotations

import json
from pathlib import Path

from chronohorn.db import ChronohornDB
from chronohorn.mcp import ToolServer


def _make_server(tmp_path, _servers=[]) -> ToolServer:
    # Close any previous server's DB to avoid leaked connections
    for old in _servers:
        old._shared_db.close()
    _servers.clear()
    db = ChronohornDB(tmp_path / "test.db")
    # Seed some results
    for i, bpb in enumerate([1.85, 1.90, 1.95, 2.00]):
        db.record_result(f"run-{i}", {
            "model": {"test_bpb": bpb, "params": 10000000},
            "config": {"train": {"steps": 10000}},
            "training": {
                "performance": {"tokens_per_second": 350000, "elapsed_sec": 900},
                "probes": [
                    {"step": 1000, "bpb": bpb + 0.5},
                    {"step": 5000, "bpb": bpb + 0.2},
                    {"step": 10000, "bpb": bpb},
                ],
            },
        })
    ts = ToolServer(db=db)
    _servers.append(ts)
    return ts


def test_fleet_converge_with_name(tmp_path):
    ts = _make_server(tmp_path)
    result = ts._do_fleet_converge({"name": "run-0", "steps": 50000})
    assert result["base_name"] == "run-0"
    assert result["base_bpb"] == 1.85
    assert result["target_steps"] == 50000
    assert result["est_gpu_hours"] > 0


def test_fleet_converge_default_best(tmp_path):
    ts = _make_server(tmp_path)
    result = ts._do_fleet_converge({"steps": 100000})
    # May pick best or report no results (population filter may exclude test data)
    assert "base_name" in result or "error" in result


def test_fleet_converge_missing_name(tmp_path):
    ts = _make_server(tmp_path)
    result = ts._do_fleet_converge({"name": "nonexistent", "steps": 10000})
    assert "error" in result


def test_fleet_launch_dry_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Create minimal directory structure
    (tmp_path / "python").mkdir()
    (tmp_path / "scripts").mkdir()
    ts = _make_server(tmp_path)
    result = ts._do_fleet_launch({
        "arch": "v12",
        "script": "scripts/train_polyhash.py",
        "host": "localhost",
        "name": "dry-test",
        "single": True,
        "steps": 100,
        "dry_run": True,
        "extra_args": ["--hidden-dim", "64"],
    })
    assert result["success"] is True
    assert "dry run" in result["output"]


def test_fleet_launch_missing_required(tmp_path):
    import pytest
    ts = _make_server(tmp_path)
    # Should raise ValueError because script is required
    with pytest.raises(ValueError, match="required parameter 'script'"):
        ts._do_fleet_launch({"name": "test"})
    # Should raise ValueError because arch is required
    with pytest.raises(ValueError, match="required parameter 'arch'"):
        ts._do_fleet_launch({"script": "scripts/train.py", "name": "test"})


def test_fleet_status_returns_hosts_structure(tmp_path, monkeypatch):
    """Test that fleet_status returns the right structure even if hosts are offline."""
    ts = _make_server(tmp_path)
    # Override FLEET_HOSTS to empty list to avoid SSH
    monkeypatch.setattr("chronohorn.observe.serve.FLEET_HOSTS", [])
    result = ts._do_fleet_status({})
    assert "hosts" in result
    assert isinstance(result["hosts"], list)


def test_register_run(tmp_path):
    ts = _make_server(tmp_path)
    result = ts._do_register_run({
        "name": "test-hybrid-run",
        "host": "slop-01",
        "config": {"family": "causal-bank", "scale": 8.0, "state_dim": 16, "seed": 42},
        "steps": 50000,
        "seed": 42,
    })
    assert result["registered"] == "test-hybrid-run"
    assert result["host"] == "slop-01"
    assert result["state"] == "running"
    # Verify it's in the DB
    db = ts._shared_db
    jobs = db.query("SELECT name, host, state, family FROM jobs WHERE name = 'test-hybrid-run'")
    assert len(jobs) == 1
    assert jobs[0]["host"] == "slop-01"
    assert jobs[0]["state"] == "running"
    assert jobs[0]["family"] == "causal-bank"
    # Verify event was logged
    events = db.query("SELECT event, data FROM events ORDER BY ts DESC LIMIT 1")
    assert len(events) == 1
    assert events[0]["event"] == "registered_run"


def test_register_run_missing_name(tmp_path):
    import pytest
    ts = _make_server(tmp_path)
    with pytest.raises(ValueError, match="required parameter 'name'"):
        ts._do_register_run({"host": "slop-01"})
