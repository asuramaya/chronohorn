from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .models import HostCapability, PerformanceSample, PlacementDecision, WorkloadDemand

DEFAULT_MIN_AVAILABLE_MEM_GB = {
    "cpu_serial": 4.0,
    "cpu_wide": 24.0,
    "metal_gpu": 12.0,
    "cuda_gpu": 12.0,
}


def default_min_available_mem_gb(job: dict[str, Any]) -> float:
    override = job.get("min_available_mem_gb")
    if override is not None:
        return float(override)
    resource_class = str(job.get("resource_class", ""))
    return DEFAULT_MIN_AVAILABLE_MEM_GB.get(resource_class, 4.0)


def placement_cores(job: dict[str, Any]) -> int:
    return max(1, int(job.get("placement_cores", job.get("threads", 1))))


def candidate_hosts_for_job(job: dict[str, Any], default_remote_hosts: Iterable[str]) -> list[str]:
    hosts = job.get("hosts")
    if isinstance(hosts, list) and hosts:
        return [str(host) for host in hosts]
    host = job.get("host")
    if isinstance(host, str) and host and host != "auto":
        return [host]
    launcher = str(job.get("launcher", ""))
    backend = str(job.get("backend", ""))
    if launcher == "local_command":
        return ["local"]
    if launcher == "managed_command" and backend in {"", "adaptive", "auto", "any"}:
        return ["local", *list(default_remote_hosts)]
    if backend == "metal":
        return ["local"]
    return list(default_remote_hosts)


def infer_workload_kind(job: dict[str, Any]) -> str:
    explicit = str(job.get("workload_kind", "")).strip()
    if explicit:
        return explicit
    launcher = str(job.get("launcher", ""))
    name = str(job.get("name", "")).lower()
    resource_class = str(job.get("resource_class", ""))
    if launcher == "slop_oracle_budgeted_build" or "row-stats" in name or "build" in name:
        return "artifact.build"
    if launcher.startswith("slop_") and ("eval" in launcher or "eval_from_table" in launcher) or "fullval" in name or "eval" in name:
        return "evaluation.fullval"
    if "parity" in name:
        return "training.parity"
    if resource_class in {"cuda_gpu", "metal_gpu"} or "train" in name or "frontier" in name:
        return "training.frontier"
    return "unknown"


def infer_work_tokens(job: dict[str, Any]) -> int | None:
    for key in ("work_tokens", "val_tokens", "train_tokens"):
        value = job.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def infer_model_family(job: dict[str, Any]) -> str:
    from chronohorn.families.registry import resolve_family_id
    explicit = str(job.get("family") or job.get("model_family") or "").strip()
    if explicit:
        fid = resolve_family_id(explicit)
        return fid if fid else explicit
    name = str(job.get("name", "")).lower()
    for token in name.replace("-", "_").split("_"):
        fid = resolve_family_id(token)
        if fid is not None:
            return fid
    return "unknown"


def infer_execution_backend(job: dict[str, Any], host: str) -> str:
    backend = str(job.get("backend", "")).strip().lower()
    if backend in {"metal", "cuda", "cpu"}:
        return backend
    if host == "local":
        return "metal"
    if str(job.get("resource_class", "")) == "cuda_gpu":
        return "cuda"
    return "cpu"


def workload_demand_for_job(job: dict[str, Any], host: str) -> WorkloadDemand:
    return WorkloadDemand(
        name=str(job["name"]),
        launcher=str(job.get("launcher", "")),
        execution_backend=infer_execution_backend(job, host),
        resource_class=str(job.get("resource_class", "")),
        model_family=infer_model_family(job),
        workload_kind=infer_workload_kind(job),
        work_tokens=infer_work_tokens(job),
        placement_cores=placement_cores(job),
        min_available_mem_gb=default_min_available_mem_gb(job),
    )


def host_capability_from_state(host: str, state: dict[str, Any]) -> HostCapability:
    return HostCapability(
        host=host,
        execution_backend=str(state.get("execution_backend", "unknown")),
        backend_family=str(state.get("backend_family", "unknown")),
        accelerator_arch=str(state.get("accelerator_arch", "unknown")),
        device_name=state.get("device_name"),
        nproc=int(state.get("nproc", 0)),
        total_mem_bytes=int(state.get("total_mem_bytes", 0)),
        available_mem_bytes=int(state.get("available_mem_bytes", 0)),
        gpu_busy=bool(state.get("gpu_busy", False)),
        class_counts=dict(state.get("class_counts", {})),
        planned_class_counts=dict(state.get("planned_class_counts", {})),
        planned_reserved_cores=int(state.get("planned_reserved_cores", 0)),
        planned_jobs=tuple(state.get("planned_jobs", [])),
        containers=tuple(state.get("containers", [])),
        probe_error=state.get("probe_error"),
    )


def telemetry_match_score(sample: PerformanceSample, demand: WorkloadDemand, host: HostCapability) -> tuple[int, int, int, int]:
    return (
        int(sample.execution_backend == demand.execution_backend),
        int(sample.backend_family == host.backend_family),
        int(sample.accelerator_arch == host.accelerator_arch),
        int(sample.workload_kind == demand.workload_kind) + int(sample.model_family == demand.model_family),
    )


def select_performance_sample(
    samples: list[PerformanceSample], demand: WorkloadDemand, host: HostCapability
) -> PerformanceSample | None:
    ranked: list[tuple[tuple[int, int, int, int, float], PerformanceSample]] = []
    for sample in samples:
        score = telemetry_match_score(sample, demand, host)
        if score[0] == 0:
            continue
        tflops = sample.estimated_sustained_tflops or 0.0
        ranked.append(((*score, tflops), sample))
    if not ranked:
        return None
    ranked.sort(reverse=True)
    return ranked[0][1]


def host_is_eligible(host: HostCapability, demand: WorkloadDemand) -> bool:
    min_available_mem_bytes = int(demand.min_available_mem_gb * 1024**3)
    if host.available_mem_bytes < min_available_mem_bytes:
        return False
    if demand.execution_backend == "cuda" and host.backend_family != "nvidia":
        return False
    if demand.execution_backend == "metal" and host.backend_family != "apple":
        return False
    if demand.resource_class == "cuda_gpu" and host.gpu_busy:
        return False
    return True


def heuristic_fallback_tuple(demand: WorkloadDemand, host: HostCapability) -> tuple[float, ...]:
    free_cores = max(0, host.nproc - host.planned_reserved_cores)
    total_containers = len(host.containers) + len(host.planned_jobs)
    if demand.resource_class == "cpu_wide":
        return (
            float(host.class_counts.get("cpu_wide", 0) + host.planned_class_counts.get("cpu_wide", 0)) * -1000.0,
            float(host.available_mem_bytes),
            float(free_cores),
            float(-total_containers),
        )
    if demand.resource_class == "cuda_gpu":
        return (
            float(host.available_mem_bytes),
            float(free_cores),
            float(-total_containers),
        )
    return (
        float(free_cores),
        float(host.available_mem_bytes),
        float(-total_containers),
    )


def explain_decision(
    demand: WorkloadDemand,
    host: HostCapability,
    sample: PerformanceSample | None,
    predicted_seconds: float | None,
) -> str:
    head = (
        f"{host.host}:{host.backend_family}/{host.accelerator_arch}"
        f" backend={demand.execution_backend} workload={demand.workload_kind}"
    )
    if sample is None:
        return f"{head} fallback=capacity no_telemetry"
    reason = (
        f"{head} telemetry={sample.workload_kind}@{sample.backend_family}/{sample.accelerator_arch}"
        f" tok/s={sample.tokens_per_second:.0f}"
    )
    if sample.estimated_sustained_tflops is not None:
        reason += f" tflops={sample.estimated_sustained_tflops:.3f}"
    if predicted_seconds is not None:
        reason += f" predicted_sec={predicted_seconds:.1f}"
    return reason


def choose_host(
    job: dict[str, Any],
    fleet_state: dict[str, Any],
    samples: list[PerformanceSample],
    default_remote_hosts: Iterable[str],
) -> PlacementDecision:
    candidates = candidate_hosts_for_job(job, default_remote_hosts)
    host_caps: list[HostCapability] = []
    for host_name in candidates:
        if host_name == "local":
            local = fleet_state.get("local")
            if local is None:
                continue
            host_caps.append(host_capability_from_state("local", local))
            continue
        remote = fleet_state.get("remote", {}).get(host_name)
        if remote is not None:
            host_caps.append(host_capability_from_state(host_name, remote))
    if not host_caps:
        raise RuntimeError(f"{job['name']}: no candidate hosts available")

    ranked: list[tuple[tuple[float, ...], PlacementDecision]] = []
    ineligible_details: list[str] = []
    for host in host_caps:
        demand = workload_demand_for_job(job, host.host)
        if not host_is_eligible(host, demand):
            ineligible_details.append(
                f"{host.host}:avail={host.available_mem_bytes/1024**3:.1f}GiB backend={host.backend_family}/{host.accelerator_arch}"
            )
            continue
        sample = select_performance_sample(samples, demand, host)
        predicted_seconds = None
        expected_tokens_per_second = None
        expected_tflops = None
        if sample is not None:
            expected_tokens_per_second = sample.tokens_per_second
            expected_tflops = sample.estimated_sustained_tflops
            if demand.work_tokens is not None and sample.tokens_per_second > 0:
                predicted_seconds = float(demand.work_tokens) / sample.tokens_per_second
        fallback = heuristic_fallback_tuple(demand, host)
        if predicted_seconds is not None:
            score = (-predicted_seconds, expected_tflops or 0.0, *fallback)
        elif expected_tflops is not None:
            score = (expected_tflops, *fallback)
        else:
            score = fallback
        ranked.append(
            (
                score,
                PlacementDecision(
                    host=host.host,
                    predicted_seconds=predicted_seconds,
                    expected_tokens_per_second=expected_tokens_per_second,
                    expected_tflops=expected_tflops,
                    telemetry_basis=sample,
                    reason=explain_decision(demand, host, sample, predicted_seconds),
                ),
            )
        )
    if not ranked:
        details = ", ".join(ineligible_details) if ineligible_details else "no eligible host"
        demand = workload_demand_for_job(job, candidates[0])
        raise RuntimeError(
            f"{job['name']}: no eligible host for {demand.execution_backend}/{demand.resource_class or demand.workload_kind} ({details})"
        )
    ranked.sort(reverse=True)
    return ranked[0][1]
