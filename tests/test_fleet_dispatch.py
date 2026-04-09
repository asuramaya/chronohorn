from __future__ import annotations

import pytest


def _gpu_host_state(
    *,
    total_gpu_mb: int,
    used_gpu_mb: int = 0,
    schedulable: bool = True,
    allocatable_gpus: int = 1,
    taint_blockers: list[str] | None = None,
) -> dict[str, object]:
    return {
        "execution_backend": "cuda",
        "backend_family": "nvidia",
        "accelerator_arch": "ada",
        "device_name": "test-gpu",
        "nproc": 12,
        "total_mem_bytes": 64 * 1024**3,
        "available_mem_bytes": 48 * 1024**3,
        "gpu_busy": False,
        "gpu_samples": [
            {
                "util_pct": 0,
                "mem_used_mb": used_gpu_mb,
                "mem_total_mb": total_gpu_mb,
            }
        ],
        "class_counts": {"cpu_serial": 0, "cpu_wide": 0, "cuda_gpu": 0, "other": 0},
        "planned_class_counts": {"cpu_serial": 0, "cpu_wide": 0, "cuda_gpu": 0, "other": 0},
        "planned_reserved_cores": 0,
        "planned_jobs": [],
        "containers": [],
        "k8s_node": {
            "schedulable": schedulable,
            "allocatable_gpus": allocatable_gpus,
            "taint_blockers": taint_blockers or [],
        },
    }


def test_parse_vm_stat_bytes_non_darwin_returns_error(monkeypatch):
    from chronohorn.fleet.dispatch import parse_vm_stat_bytes

    monkeypatch.setattr("chronohorn.fleet.dispatch.platform.system", lambda: "Linux")

    called = False

    def _should_not_run(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("vm_stat should not run on non-Darwin platforms")

    monkeypatch.setattr("chronohorn.fleet.dispatch.capture_checked", _should_not_run)

    result = parse_vm_stat_bytes()

    assert result == {"error": "vm_stat only available on macOS"}
    assert called is False


def test_probe_local_host_linux_uses_free(monkeypatch):
    from chronohorn.fleet.dispatch import probe_local_host

    monkeypatch.setattr("chronohorn.fleet.dispatch.platform.system", lambda: "Linux")
    monkeypatch.setattr("chronohorn.fleet.dispatch.platform.machine", lambda: "x86_64")
    monkeypatch.setattr("chronohorn.fleet.dispatch.os.cpu_count", lambda: 16)

    def _fake_capture(argv, **kwargs):
        assert argv == ["free", "-b"]
        return (
            "               total        used        free      shared  buff/cache   available\n"
            "Mem:     17179869184  8589934592  2147483648           0  6442450944  9663676416\n"
            "Swap:              0           0           0\n"
        )

    monkeypatch.setattr("chronohorn.fleet.dispatch.capture_checked", _fake_capture)

    local = probe_local_host()

    assert local["execution_backend"] == "cpu"
    assert local["nproc"] == 16
    assert local["total_mem_bytes"] == 17179869184
    assert local["available_mem_bytes"] == 9663676416
    assert local["page_size"] > 0
    assert "probe_error" not in local


def test_choose_host_prefers_smallest_sufficient_gpu_tier():
    from chronohorn.fleet.planner import choose_host

    job = {
        "name": "cb-s8-test",
        "launcher": "managed_command",
        "backend": "cuda",
        "resource_class": "cuda_gpu",
        "workload_kind": "training.frontier",
        "hosts": ["slop-home", "slop-01"],
        "min_gpu_mem_gb": 7.0,
        "gpu_placement_policy": "smallest_sufficient",
    }
    fleet_state = {
        "remote": {
            "slop-home": _gpu_host_state(total_gpu_mb=8192),
            "slop-01": _gpu_host_state(total_gpu_mb=16384),
        },
        "local": None,
    }

    decision = choose_host(job, fleet_state, [], ("slop-home", "slop-01"))

    assert decision.host == "slop-home"


def test_choose_host_skips_unschedulable_small_gpu_lane():
    from chronohorn.fleet.planner import choose_host

    job = {
        "name": "cb-s8-test",
        "launcher": "managed_command",
        "backend": "cuda",
        "resource_class": "cuda_gpu",
        "workload_kind": "training.frontier",
        "hosts": ["slop-home", "slop-01"],
        "min_gpu_mem_gb": 7.0,
        "gpu_placement_policy": "smallest_sufficient",
    }
    fleet_state = {
        "remote": {
            "slop-home": _gpu_host_state(
                total_gpu_mb=8192,
                schedulable=False,
                taint_blockers=["node.kubernetes.io/disk-pressure:NoSchedule"],
            ),
            "slop-01": _gpu_host_state(total_gpu_mb=16384),
        },
        "local": None,
    }

    decision = choose_host(job, fleet_state, [], ("slop-home", "slop-01"))

    assert decision.host == "slop-01"


def test_choose_host_rejects_zero_allocatable_k8s_gpu():
    from chronohorn.fleet.planner import choose_host

    job = {
        "name": "cb-s8-test",
        "launcher": "managed_command",
        "backend": "cuda",
        "resource_class": "cuda_gpu",
        "workload_kind": "training.frontier",
        "hosts": ["slop-02"],
        "min_gpu_mem_gb": 7.0,
    }
    fleet_state = {
        "remote": {
            "slop-02": _gpu_host_state(total_gpu_mb=16384, allocatable_gpus=0),
        },
        "local": None,
    }

    with pytest.raises(RuntimeError, match="no eligible host"):
        choose_host(job, fleet_state, [], ("slop-02",))


def test_partition_running_jobs_does_not_requeue_stale_k8s_job(monkeypatch):
    from chronohorn.fleet.dispatch import partition_running_jobs

    job = {
        "name": "cb-stale",
        "launcher": "managed_command",
        "backend": "cuda",
        "resource_class": "cuda_gpu",
    }

    monkeypatch.setattr("chronohorn.fleet.dispatch.query_remote_run_states", lambda jobs: {})
    monkeypatch.setattr(
        "chronohorn.fleet.dispatch.query_k8s_run_states",
        lambda jobs: {
            ("default", "cb-stale"): {
                "phase": "failed",
                "executor_name": "slop-cluster",
                "runtime_namespace": "default",
                "runtime_job_name": "ch-cb-stale",
                "runtime_pod_name": "ch-cb-stale-abcde",
                "runtime_node_name": "slop-01",
                "log_last_line": "",
                "log_tail_text": "",
            }
        },
    )
    monkeypatch.setattr(
        "chronohorn.fleet.dispatch.load_launch_record",
        lambda name: {
            "launcher": "k8s_job",
            "runtime_namespace": "default",
            "host": "slop-01",
            "executor_name": "slop-cluster",
        },
    )

    pending, running, completed, stale = partition_running_jobs([job], {"remote": {}, "local": None})

    assert pending == []
    assert running == []
    assert completed == []
    assert len(stale) == 1
    assert stale[0]["name"] == "cb-stale"
    assert stale[0]["phase"] == "failed"
