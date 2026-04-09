from __future__ import annotations

from chronohorn.db import ChronohornDB
from chronohorn.fleet.drain import DrainState


def test_drain_state_done_when_no_pending_no_running():
    state = DrainState(
        manifest_path="/fake/manifest.jsonl",
        pending=0, running=0, completed=5, blocked=0, launched=0, pulled=0,
    )
    assert state.is_done is True


def test_drain_state_not_done_when_pending():
    state = DrainState(
        manifest_path="/fake/manifest.jsonl",
        pending=3, running=2, completed=5, blocked=0, launched=0, pulled=0,
    )
    assert state.is_done is False


def test_drain_state_not_done_when_running():
    state = DrainState(
        manifest_path="/fake/manifest.jsonl",
        pending=0, running=2, completed=5, blocked=0, launched=0, pulled=0,
    )
    assert state.is_done is False


def test_drain_state_stalled_detection():
    state = DrainState(
        manifest_path="/fake/manifest.jsonl",
        pending=0, running=0, completed=2, blocked=3, launched=0, pulled=0,
    )
    assert state.is_done is False  # blocked jobs remain


def test_drain_db_tick_filters_db_jobs_by_manifest_identity(tmp_path, monkeypatch):
    from chronohorn.fleet.drain import drain_db_tick

    db = ChronohornDB(tmp_path / "test.db")
    manifest_path = tmp_path / "manifests" / "scan.jsonl"
    manifest_path.parent.mkdir()
    manifest_abs = str(manifest_path.resolve())

    db.record_job(
        "job-a",
        manifest=manifest_abs,
        job_spec={"name": "job-a", "launcher": "managed_command", "backend": "cpu"},
    )
    db.record_job(
        "job-b",
        manifest=str((tmp_path / "manifests" / "other.jsonl").resolve()),
        job_spec={"name": "job-b", "launcher": "managed_command", "backend": "cpu"},
    )

    monkeypatch.setattr("chronohorn.fleet.drain.probe_fleet_state", lambda jobs: {"remote": {}, "local": None})
    monkeypatch.setattr("chronohorn.fleet.drain.collect_performance_samples", lambda globs: [])
    monkeypatch.setattr(
        "chronohorn.fleet.drain.partition_running_jobs",
        lambda jobs, fleet_state: (jobs, [], [], []),
    )

    state = drain_db_tick(db=db, manifests=["scan.jsonl"], dispatch=False)

    assert state.pending == 1
    assert state.running == 0
    assert state.completed == 0
    db.close()
