from __future__ import annotations

import logging

from chronohorn.fleet.models import PerformanceSample


def _sample(*, execution_backend: str, backend_family: str, tok_s: float) -> PerformanceSample:
    return PerformanceSample(
        source_path="/tmp/sample.json",
        model_family="causal-bank",
        workload_kind="training.frontier",
        execution_backend=execution_backend,
        backend_family=backend_family,
        accelerator_arch="test",
        device_name=None,
        tokens_per_second=tok_s,
        estimated_sustained_tflops=None,
        work_tokens=1000,
    )


def test_estimate_expected_duration_prefers_execution_backend_match():
    from chronohorn.fleet.daemon import _estimate_expected_duration

    job = {"backend": "cuda", "train_tokens": 1000}
    telemetry = [
        _sample(execution_backend="cpu", backend_family="cpu", tok_s=10.0),
        _sample(execution_backend="cuda", backend_family="nvidia", tok_s=100.0),
    ]

    expected = _estimate_expected_duration(job, telemetry)

    assert expected == 10.0


def test_detect_stale_containers_k8s_kill_failure_is_reported_not_masked(monkeypatch):
    from chronohorn.fleet.daemon import detect_stale_containers

    monkeypatch.setattr(
        "chronohorn.fleet.dispatch.load_launch_record",
        lambda name: {
            "launched_at_unix": 1.0,
            "host": "slop-01",
            "executor_kind": "k8s_cluster",
            "runtime_namespace": "chronohorn",
            "runtime_job_name": "ch-job1",
        },
    )
    monkeypatch.setattr(
        "chronohorn.fleet.k8s.infer_executor_kind",
        lambda payload: "k8s_cluster",
    )

    def _boom(record):
        raise RuntimeError("delete failed")

    monkeypatch.setattr("chronohorn.fleet.k8s.delete_k8s_job", _boom)

    reports = detect_stale_containers(
        running_jobs=[{"name": "job1", "host": "slop-01", "backend": "cuda", "train_tokens": 10}],
        telemetry=[_sample(execution_backend="cuda", backend_family="nvidia", tok_s=1.0)],
        kill_stale=True,
        logger=logging.getLogger("test.daemon"),
    )

    assert len(reports) == 1
    assert reports[0]["action"] == "kill"
    assert reports[0]["target"] == "chronohorn/ch-job1"
    assert reports[0]["killed"] is False
    assert "delete failed" in reports[0]["kill_error"]


def test_detect_stale_containers_warn_path_does_not_raise(monkeypatch):
    from chronohorn.fleet.daemon import detect_stale_containers

    monkeypatch.setattr("time.time", lambda: 10.0)
    monkeypatch.setattr(
        "chronohorn.fleet.dispatch.load_launch_record",
        lambda name: {
            "launched_at_unix": 1.0,
            "host": "slop-01",
            "executor_kind": "managed_command",
        },
    )

    reports = detect_stale_containers(
        running_jobs=[{"name": "job1", "host": "slop-01", "backend": "cuda", "train_tokens": 1}],
        telemetry=[_sample(execution_backend="cuda", backend_family="nvidia", tok_s=1.0)],
        warn_multiplier=2.0,
        kill_multiplier=20.0,
        kill_stale=False,
        logger=logging.getLogger("test.daemon"),
    )

    assert len(reports) == 1
    assert reports[0]["action"] == "warn"


def test_run_daemon_uses_db_backed_tick(tmp_path, monkeypatch):
    from chronohorn.fleet.daemon import run_daemon
    from chronohorn.fleet.drain import DrainState

    manifest_path = tmp_path / "scan.jsonl"
    manifest_path.write_text('{"name":"job-a","launcher":"managed_command","backend":"cpu"}\n', encoding="utf-8")

    class _FakeDB:
        def __init__(self) -> None:
            self.closed = False

        def active_jobs(self, manifest=None):
            return []

        def close(self) -> None:
            self.closed = True

    fake_db = _FakeDB()
    seen: dict[str, object] = {}

    monkeypatch.setattr("chronohorn.fleet.daemon.ChronohornDB", lambda path: fake_db)
    monkeypatch.setattr("chronohorn.fleet.daemon.read_pid", lambda pid_path=None: None)
    monkeypatch.setattr("chronohorn.fleet.daemon._write_pid", lambda pid_path: None)
    monkeypatch.setattr("chronohorn.fleet.daemon._remove_pid", lambda pid_path: None)
    monkeypatch.setattr("chronohorn.fleet.daemon.signal.signal", lambda signum, handler: None)
    monkeypatch.setattr("chronohorn.fleet.daemon._setup_logger", lambda path: logging.getLogger("test.daemon.run"))

    def _fake_drain_db_tick(*, db, manifests, job_names, classes, telemetry_globs, result_out_dir, dispatch):
        seen["db"] = db
        seen["manifests"] = list(manifests)
        seen["dispatch"] = dispatch
        return DrainState(
            manifest_path=",".join(manifests),
            pending=0,
            running=0,
            completed=0,
            blocked=0,
            launched=0,
            pulled=0,
        )

    monkeypatch.setattr("chronohorn.fleet.daemon.drain_db_tick", _fake_drain_db_tick)

    rc = run_daemon(
        manifest_path,
        poll_interval=1,
        db_path=tmp_path / "test.db",
        pid_path=tmp_path / "drain.pid",
        log_path=tmp_path / "drain.log",
    )

    assert rc == 0
    assert seen["db"] is fake_db
    assert seen["manifests"] == [str(manifest_path.resolve())]
    assert seen["dispatch"] is True
    assert fake_db.closed is True
