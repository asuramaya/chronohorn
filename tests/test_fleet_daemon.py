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
