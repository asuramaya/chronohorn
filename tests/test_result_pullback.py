from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from chronohorn.db import ChronohornDB
from chronohorn.fleet.results import pull_remote_result


def test_pull_result_returns_payload_on_success(tmp_path: Path):
    fake_payload = {"model": {"test_bpb": 2.05}, "config": {}}
    fake_ssh_output = json.dumps(fake_payload)

    with patch("chronohorn.fleet.results._ssh_cat_file", return_value=fake_ssh_output):
        result = pull_remote_result(
            host="slop-01",
            remote_run="/tmp/chronohorn-runs/test-job",
            job_name="test-job",
            local_out_dir=tmp_path,
        )

    assert result.success is True
    assert result.local_path is not None
    assert result.local_path.exists()
    saved = json.loads(result.local_path.read_text())
    assert saved["model"]["test_bpb"] == 2.05


def test_pull_result_returns_failure_on_ssh_error(tmp_path: Path):
    with patch("chronohorn.fleet.results._ssh_cat_file", side_effect=RuntimeError("ssh failed")):
        result = pull_remote_result(
            host="slop-01",
            remote_run="/tmp/chronohorn-runs/test-job",
            job_name="test-job",
            local_out_dir=tmp_path,
        )

    assert result.success is False
    assert result.error is not None


def test_pull_result_skips_if_local_exists(tmp_path: Path):
    local_file = tmp_path / "test-job.json"
    local_file.write_text('{"already": "here"}')

    result = pull_remote_result(
        host="slop-01",
        remote_run="/tmp/chronohorn-runs/test-job",
        job_name="test-job",
        local_out_dir=tmp_path,
    )

    assert result.success is True
    assert result.skipped is True


def test_pull_result_reingests_existing_local_file_into_db(tmp_path: Path):
    db = ChronohornDB(tmp_path / "test.db")
    local_file = tmp_path / "test-job.json"
    local_file.write_text(
        json.dumps(
            {
                "model": {"test_bits_per_token": 4.872, "params": 1000},
                "config": {},
                "training": {"performance": {}, "probes": []},
            }
        )
    )

    result = pull_remote_result(
        host="slop-01",
        remote_run="/data/chronohorn/out",
        job_name="test-job",
        local_out_dir=tmp_path,
        db=db,
    )

    rows = db.query("SELECT bpb FROM results WHERE name = ?", ("test-job",))
    assert result.success is True
    assert result.skipped is True
    assert result.ingested is True
    assert len(rows) == 1
    assert abs(rows[0]["bpb"] - 2.0) < 0.01
    db.close()
