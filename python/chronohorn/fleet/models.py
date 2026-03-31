from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PerformanceSample:
    source_path: str
    model_family: str
    workload_kind: str
    execution_backend: str
    backend_family: str
    accelerator_arch: str
    device_name: str | None
    tokens_per_second: float
    estimated_sustained_tflops: float | None
    work_tokens: int | None


@dataclass(frozen=True)
class HostCapability:
    host: str
    execution_backend: str
    backend_family: str
    accelerator_arch: str
    device_name: str | None
    nproc: int
    total_mem_bytes: int
    available_mem_bytes: int
    gpu_busy: bool
    class_counts: dict[str, int] = field(default_factory=dict)
    planned_class_counts: dict[str, int] = field(default_factory=dict)
    planned_reserved_cores: int = 0
    planned_jobs: tuple[str, ...] = ()
    containers: tuple[str, ...] = ()
    probe_error: dict[str, object] | None = None


@dataclass(frozen=True)
class WorkloadDemand:
    name: str
    launcher: str
    execution_backend: str
    resource_class: str
    model_family: str
    workload_kind: str
    work_tokens: int | None
    placement_cores: int
    min_available_mem_gb: float


@dataclass(frozen=True)
class PlacementDecision:
    host: str
    predicted_seconds: float | None
    expected_tokens_per_second: float | None
    expected_tflops: float | None
    telemetry_basis: PerformanceSample | None
    reason: str
