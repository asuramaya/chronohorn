"""Local GPU-etiquette gate for the fleet-of-one laptop.

The remote busy heuristic (mem_used > 256MB = busy) would refuse this
box forever: the GPU permanently hosts a companion process at ~3GB
resident, 0% util. The local gate instead checks free memory against
the job's requirement and defers on high utilization.
"""
from __future__ import annotations

import pytest

from chronohorn.fleet.dispatch import ensure_local_gpu_capacity


def _fleet_state(samples: list[dict[str, int]] | None) -> dict[str, object]:
    return {"local": {"gpu_samples": samples or []}, "remote": {}}


def _gpu_job(**extra: object) -> dict[str, object]:
    return {"name": "gpu-gate-test", "resource_class": "cuda_gpu", **extra}


def test_passes_with_companion_resident_but_idle() -> None:
    # The real shape of this laptop: 12GB card, companion holds ~3GB at 0%.
    state = _fleet_state([{"util_pct": 0, "mem_used_mb": 3159, "mem_total_mb": 12288}])
    ensure_local_gpu_capacity(_gpu_job(min_gpu_mem_gb=6), state)


def test_refuses_when_no_gpu_visible() -> None:
    with pytest.raises(RuntimeError, match="no GPU visible"):
        ensure_local_gpu_capacity(_gpu_job(), _fleet_state(None))


def test_refuses_when_free_memory_insufficient() -> None:
    state = _fleet_state([{"util_pct": 0, "mem_used_mb": 9000, "mem_total_mb": 12288}])
    with pytest.raises(RuntimeError, match="free="):
        ensure_local_gpu_capacity(_gpu_job(min_gpu_mem_gb=6), state)


def test_defers_when_gpu_busy() -> None:
    # Companion running a heavy capture: plenty of memory, high util.
    state = _fleet_state([{"util_pct": 85, "mem_used_mb": 4000, "mem_total_mb": 12288}])
    with pytest.raises(RuntimeError, match="util=85%"):
        ensure_local_gpu_capacity(_gpu_job(min_gpu_mem_gb=2), state)


def test_util_threshold_overridable() -> None:
    state = _fleet_state([{"util_pct": 85, "mem_used_mb": 4000, "mem_total_mb": 12288}])
    ensure_local_gpu_capacity(_gpu_job(max_gpu_util_pct=90), state)


def test_reads_requirement_from_folded_config() -> None:
    # Manifest normalization folds min_gpu_mem_gb into the config dict.
    state = _fleet_state([{"util_pct": 0, "mem_used_mb": 9000, "mem_total_mb": 12288}])
    job = _gpu_job(config={"min_gpu_mem_gb": 6})
    with pytest.raises(RuntimeError, match="required=6.0GiB"):
        ensure_local_gpu_capacity(job, state)


def test_noop_for_cpu_jobs() -> None:
    job = {"name": "cpu-job", "resource_class": "cpu_serial"}
    ensure_local_gpu_capacity(job, _fleet_state(None))
