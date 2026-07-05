"""Local jobs complete when their --json result exists.

Without this, a finished local job looks pending forever and the
manifest queue relaunches it in a loop (M5 twin-1 burned a full rerun
this way on 2026-07-05).
"""
from __future__ import annotations

from chronohorn.fleet.dispatch import detect_completed_job


def _local_job(tmp_path, name: str = "twin") -> dict[str, object]:
    return {
        "name": name,
        "hosts": ["local"],
        "launcher": "local_command",
        "cwd": str(tmp_path),
        "command": f"python -m chronohorn train x --steps 10 --json out/results/{name}.json",
    }


def test_local_job_completed_when_result_exists(tmp_path) -> None:
    job = _local_job(tmp_path)
    out = tmp_path / "out" / "results"
    out.mkdir(parents=True)
    (out / "twin.json").write_text('{"ok": true}')
    record = detect_completed_job(job, {}, {})
    assert record is not None
    assert record["state"] == "completed"
    assert record["executor_kind"] == "local_process"


def test_local_job_pending_without_result(tmp_path) -> None:
    assert detect_completed_job(_local_job(tmp_path), {}, {}) is None


def test_local_job_pending_with_empty_result(tmp_path) -> None:
    job = _local_job(tmp_path)
    out = tmp_path / "out" / "results"
    out.mkdir(parents=True)
    (out / "twin.json").write_text("")
    assert detect_completed_job(job, {}, {}) is None


def test_remote_job_not_matched_by_local_path(tmp_path) -> None:
    job = _local_job(tmp_path)
    job["hosts"] = ["slop-01"]
    job["host"] = "slop-01"
    out = tmp_path / "out" / "results"
    out.mkdir(parents=True)
    (out / "twin.json").write_text('{"ok": true}')
    # Remote jobs keep the remote-report path; a local file must not count.
    assert detect_completed_job(job, {}, {}) is None
