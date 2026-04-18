from __future__ import annotations

import time

from chronohorn.db import ChronohornDB, IMPORTED_RESULT_MANIFEST
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


def test_drain_db_tick_reconciles_orphaned_running_job_to_pending(tmp_path, monkeypatch):
    from chronohorn.fleet.drain import drain_db_tick

    db = ChronohornDB(tmp_path / "test.db")
    db.record_job(
        "job-a",
        manifest="manual",
        job_spec={"name": "job-a", "launcher": "managed_command", "backend": "cpu"},
    )
    db._write(
        "UPDATE jobs SET state='running', host='slop-01', remote_run='/data/chronohorn/out' WHERE name='job-a'",
        wait=True,
    )

    monkeypatch.setattr("chronohorn.fleet.drain.probe_fleet_state", lambda jobs: {"remote": {}, "local": None})
    monkeypatch.setattr("chronohorn.fleet.drain.collect_performance_samples", lambda globs: [])
    monkeypatch.setattr(
        "chronohorn.fleet.drain.partition_running_jobs",
        lambda jobs, fleet_state: (jobs, [], [], []),
    )

    state = drain_db_tick(db=db, manifests=[], dispatch=False)
    job = db.job_spec("job-a")

    assert state.pending == 1
    assert job is not None
    assert job["state"] == "pending"
    db.close()


def test_drain_db_tick_materializes_manifest_jobs_into_db(tmp_path, monkeypatch):
    from chronohorn.fleet.drain import drain_db_tick

    manifest_path = tmp_path / "scan.jsonl"
    manifest_path.write_text('{"name":"job-a","launcher":"managed_command","backend":"cpu"}\n', encoding="utf-8")
    db = ChronohornDB(tmp_path / "test.db")

    monkeypatch.setattr("chronohorn.fleet.drain.probe_fleet_state", lambda jobs: {"remote": {}, "local": None})
    monkeypatch.setattr("chronohorn.fleet.drain.collect_performance_samples", lambda globs: [])
    monkeypatch.setattr(
        "chronohorn.fleet.drain.partition_running_jobs",
        lambda jobs, fleet_state: (jobs, [], [], []),
    )

    state = drain_db_tick(db=db, manifests=[str(manifest_path)], dispatch=False)
    job = db.job_spec("job-a")

    assert state.pending == 1
    assert job is not None
    assert job["manifest"] == str(manifest_path.resolve())
    assert job["state"] == "pending"
    db.close()


def test_drain_db_tick_normalizes_imported_archive_jobs_out_of_active_queue(tmp_path, monkeypatch):
    from chronohorn.fleet.drain import drain_db_tick

    db = ChronohornDB(tmp_path / "test.db")
    db.record_job(
        "imported-orphan",
        manifest=IMPORTED_RESULT_MANIFEST,
        job_spec={"name": "imported-orphan", "launcher": "managed_command", "backend": "cpu"},
    )
    db.record_result(
        "imported-complete",
        {
            "model": {"test_bpb": 1.9},
            "config": {"train": {"steps": 1000}},
            "training": {"performance": {"steps_completed": 1000, "elapsed_sec": 1}},
        },
        compute_forecast=False,
    )
    db._write("UPDATE jobs SET state='pending' WHERE name='imported-complete'", wait=True)

    monkeypatch.setattr("chronohorn.fleet.drain.probe_fleet_state", lambda jobs: {"remote": {}, "local": None})
    monkeypatch.setattr("chronohorn.fleet.drain.collect_performance_samples", lambda globs: [])
    monkeypatch.setattr(
        "chronohorn.fleet.drain.partition_running_jobs",
        lambda jobs, fleet_state: (jobs, [], [], []),
    )

    state = drain_db_tick(db=db, manifests=[], dispatch=False)

    assert state.pending == 0
    assert db.job_spec("imported-orphan")["state"] == "failed"
    assert db.job_spec("imported-complete")["state"] == "completed"
    db.close()


def test_drain_db_tick_backfills_running_runtime_metadata(tmp_path, monkeypatch):
    from chronohorn.fleet.drain import drain_db_tick

    db = ChronohornDB(tmp_path / "test.db")
    db.record_job(
        "job-a",
        manifest="manual",
        job_spec={
            "name": "job-a",
            "launcher": "k8s_job",
            "backend": "cuda",
            "executor_kind": "k8s_cluster",
            "runtime_namespace": "default",
            "runtime_job_name": "ch-job-a",
        },
    )

    monkeypatch.setattr("chronohorn.fleet.drain.probe_fleet_state", lambda jobs: {"remote": {}, "local": None})
    monkeypatch.setattr("chronohorn.fleet.drain.collect_performance_samples", lambda globs: [])
    monkeypatch.setattr(
        "chronohorn.fleet.drain.partition_running_jobs",
        lambda jobs, fleet_state: (
            [],
            [
                {
                    "name": "job-a",
                    "host": "slop-01",
                    "executor_kind": "k8s_cluster",
                    "executor_name": "slop-cluster",
                    "runtime_namespace": "default",
                    "runtime_job_name": "ch-job-a",
                    "runtime_pod_name": "ch-job-a-abcde",
                    "runtime_node_name": "slop-01",
                }
            ],
            [],
            [],
        ),
    )

    state = drain_db_tick(db=db, manifests=[], dispatch=False)
    job = db.job_spec("job-a")

    assert state.running == 1
    assert job is not None
    assert job["state"] == "running"
    assert job["host"] == "slop-01"
    assert job["runtime_pod_name"] == "ch-job-a-abcde"
    assert job["runtime_node_name"] == "slop-01"
    db.close()


def test_drain_db_tick_marks_completed_jobs_completed_before_result_ingest(tmp_path, monkeypatch):
    from chronohorn.fleet.drain import drain_db_tick

    db = ChronohornDB(tmp_path / "test.db")
    db.record_job(
        "job-a",
        manifest="manual",
        job_spec={
            "name": "job-a",
            "launcher": "k8s_job",
            "backend": "cuda",
            "executor_kind": "k8s_cluster",
            "runtime_namespace": "default",
            "runtime_job_name": "ch-job-a",
        },
    )
    db._write("UPDATE jobs SET state='running', host='slop-01' WHERE name='job-a'", wait=True)

    monkeypatch.setattr("chronohorn.fleet.drain.probe_fleet_state", lambda jobs: {"remote": {}, "local": None})
    monkeypatch.setattr("chronohorn.fleet.drain.collect_performance_samples", lambda globs: [])
    monkeypatch.setattr(
        "chronohorn.fleet.drain.partition_running_jobs",
        lambda jobs, fleet_state: (
            [],
            [],
            [
                {
                    "name": "job-a",
                    "host": "slop-01",
                    "executor_kind": "k8s_cluster",
                    "executor_name": "slop-cluster",
                    "runtime_namespace": "default",
                    "runtime_job_name": "ch-job-a",
                    "runtime_pod_name": "ch-job-a-abcde",
                    "runtime_node_name": "slop-01",
                }
            ],
            [],
        ),
    )
    monkeypatch.setattr("chronohorn.fleet.drain.pull_all_completed_results", lambda *args, **kwargs: [])

    state = drain_db_tick(db=db, manifests=[], dispatch=False)
    job = db.job_spec("job-a")
    events = db.events_recent(10)

    assert state.completed == 1
    assert job is not None
    assert job["state"] == "completed"
    assert job["host"] == "slop-01"
    assert job["runtime_pod_name"] == "ch-job-a-abcde"
    assert any(e["event"] == "reconciled_job_completed" for e in events)
    db.close()


def test_drain_db_tick_records_launch_failure_event(tmp_path, monkeypatch):
    from chronohorn.fleet.drain import drain_db_tick

    db = ChronohornDB(tmp_path / "test.db")
    db.record_job(
        "job-a",
        manifest="manual",
        job_spec={"name": "job-a", "launcher": "managed_command", "backend": "cpu", "host": "slop-01"},
    )

    monkeypatch.setattr("chronohorn.fleet.drain.probe_fleet_state", lambda jobs: {"remote": {}, "local": None})
    monkeypatch.setattr("chronohorn.fleet.drain.collect_performance_samples", lambda globs: [])
    monkeypatch.setattr(
        "chronohorn.fleet.drain.partition_running_jobs",
        lambda jobs, fleet_state: (jobs, [], [], []),
    )
    monkeypatch.setattr(
        "chronohorn.fleet.drain.assign_jobs_best_effort",
        lambda pending, fleet_state, telemetry: (pending, []),
    )
    monkeypatch.setattr("chronohorn.fleet.drain.launch_job", lambda job: (_ for _ in ()).throw(RuntimeError("launch boom")))
    monkeypatch.setattr("chronohorn.fleet.drain.pull_all_completed_results", lambda *args, **kwargs: [])

    state = drain_db_tick(db=db, manifests=[], dispatch=True)
    events = db.events_recent(10)

    assert state.pending == 1
    assert any(e["event"] == "launch_failed" and "launch boom" in str(e["data"]) for e in events)
    db.close()


def test_drain_db_tick_records_stale_warning_event(tmp_path, monkeypatch):
    from chronohorn.fleet.drain import drain_db_tick

    db = ChronohornDB(tmp_path / "test.db")
    db.record_job(
        "job-a",
        manifest="manual",
        job_spec={"name": "job-a", "launcher": "managed_command", "backend": "cpu", "steps": 1, "host": "slop-01"},
    )

    class _Telemetry:
        tokens_per_second = 1000.0

    monkeypatch.setattr("chronohorn.fleet.drain.probe_fleet_state", lambda jobs: {"remote": {}, "local": None})
    monkeypatch.setattr("chronohorn.fleet.drain.collect_performance_samples", lambda globs: [_Telemetry()])
    monkeypatch.setattr(
        "chronohorn.fleet.drain.partition_running_jobs",
        lambda jobs, fleet_state: (
            [],
            [{"name": "job-a", "steps": 1, "host": "slop-01"}],
            [],
            [],
        ),
    )
    monkeypatch.setattr(
        "chronohorn.fleet.drain.load_launch_record",
        lambda name: {"host": "slop-01", "launched_at_unix": time.time() - 100.0},
    )
    monkeypatch.setattr("chronohorn.fleet.drain.pull_all_completed_results", lambda *args, **kwargs: [])

    state = drain_db_tick(db=db, manifests=[], dispatch=False)
    events = db.events_recent(10)

    assert state.stale_warned == 1
    assert any(e["event"] == "stale_warning" for e in events)
    db.close()


def test_drain_db_tick_catches_up_completed_jobs_missing_sharts_export(tmp_path, monkeypatch):
    from chronohorn.fleet.drain import drain_db_tick

    db = ChronohornDB(tmp_path / "test.db")
    db.record_job(
        "job-orphan",
        manifest="manual",
        job_spec={
            "name": "job-orphan",
            "launcher": "k8s_job",
            "backend": "cuda",
            "executor_kind": "k8s_cluster",
            "runtime_namespace": "default",
            "runtime_job_name": "ch-job-orphan",
        },
    )
    # Flip to completed with host+remote_run so runtime_record_for_job populates the record.
    db._write(
        "UPDATE jobs SET state='completed', completed_at=?, host='slop-01', "
        "remote_run='/tmp/chronohorn-runs/job-orphan', runtime_node_name='slop-01' "
        "WHERE name='job-orphan'",
        (time.time(),),
        wait=True,
    )

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    monkeypatch.setattr("chronohorn.fleet.drain.probe_fleet_state", lambda jobs: {"remote": {}, "local": None})
    monkeypatch.setattr("chronohorn.fleet.drain.collect_performance_samples", lambda globs: [])
    monkeypatch.setattr(
        "chronohorn.fleet.drain.partition_running_jobs",
        lambda jobs, fleet_state: ([], [], [], []),
    )
    monkeypatch.setattr("chronohorn.fleet.results._resolve_checkpoint_export_dir", lambda: export_dir)

    captured: list[list[dict]] = []

    def _stub_pull(records, **kwargs):
        captured.append(list(records))
        return []

    monkeypatch.setattr("chronohorn.fleet.drain.pull_all_completed_results", _stub_pull)

    state = drain_db_tick(db=db, manifests=[], dispatch=False)

    # Two calls: the main (no completed jobs) + the catch-up pass.
    assert len(captured) == 2
    assert captured[0] == []  # main pass
    assert state.catchup_attempted == 1
    assert len(captured[1]) == 1
    assert captured[1][0]["name"] == "job-orphan"
    assert captured[1][0]["remote_run"] == "/tmp/chronohorn-runs/job-orphan"
    db.close()


def test_drain_db_tick_catchup_skips_when_sharts_export_exists(tmp_path, monkeypatch):
    from chronohorn.fleet.drain import drain_db_tick

    db = ChronohornDB(tmp_path / "test.db")
    db.record_job(
        "job-exported",
        manifest="manual",
        job_spec={
            "name": "job-exported",
            "launcher": "k8s_job",
            "backend": "cuda",
            "executor_kind": "k8s_cluster",
            "runtime_namespace": "default",
            "runtime_job_name": "ch-job-exported",
        },
    )
    db._write(
        "UPDATE jobs SET state='completed', completed_at=?, host='slop-01', "
        "remote_run='/tmp/chronohorn-runs/job-exported', runtime_node_name='slop-01' "
        "WHERE name='job-exported'",
        (time.time(),),
        wait=True,
    )

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "job-exported.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr("chronohorn.fleet.drain.probe_fleet_state", lambda jobs: {"remote": {}, "local": None})
    monkeypatch.setattr("chronohorn.fleet.drain.collect_performance_samples", lambda globs: [])
    monkeypatch.setattr(
        "chronohorn.fleet.drain.partition_running_jobs",
        lambda jobs, fleet_state: ([], [], [], []),
    )
    monkeypatch.setattr("chronohorn.fleet.results._resolve_checkpoint_export_dir", lambda: export_dir)

    captured: list[list[dict]] = []

    def _stub_pull(records, **kwargs):
        captured.append(list(records))
        return []

    monkeypatch.setattr("chronohorn.fleet.drain.pull_all_completed_results", _stub_pull)

    state = drain_db_tick(db=db, manifests=[], dispatch=False)

    # Only the main pull call runs; the catch-up pass finds nothing eligible.
    assert state.catchup_attempted == 0
    assert captured == [[]]
    db.close()
