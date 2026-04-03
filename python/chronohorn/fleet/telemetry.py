from __future__ import annotations

import glob
import json
import os
import platform
import re
from pathlib import Path
from typing import Any

from .models import PerformanceSample


CHRONOHORN_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TELEMETRY_GLOBS = (
    "/tmp/chronohorn_*.json",
    "/tmp/chronohorn_frontier_artifacts/*.json",
    str(CHRONOHORN_ROOT / "out" / "**" / "*.json"),
)


def normalize_arch_label(raw: str | None, *, default: str = "unknown") -> str:
    if not raw:
        return default
    cleaned = re.sub(r"[^a-z0-9]+", "-", raw.lower()).strip("-")
    return cleaned or default


def infer_backend_family(execution_backend: str, backend_environment: dict[str, Any]) -> str:
    device = str(backend_environment.get("device") or execution_backend or "").lower()
    device_name = str(backend_environment.get("device_name") or "").lower()
    machine = str(backend_environment.get("machine") or platform.machine() or "").lower()
    if execution_backend == "metal" or device in {"mlx", "mps", "metal"}:
        return "apple"
    if execution_backend == "cuda" or device == "cuda" or "nvidia" in device_name:
        return "nvidia"
    if execution_backend == "cpu" or device == "cpu":
        if machine in {"arm64", "aarch64"}:
            return "apple"
        return "cpu"
    return normalize_arch_label(device, default="unknown")


def infer_accelerator_arch(execution_backend: str, backend_environment: dict[str, Any]) -> str:
    device_name = str(backend_environment.get("device_name") or "").strip()
    device = str(backend_environment.get("device") or execution_backend or "").lower()
    if execution_backend == "metal" or device in {"mlx", "mps", "metal"}:
        return "apple-silicon"
    if device_name:
        return normalize_arch_label(device_name)
    machine = str(backend_environment.get("machine") or platform.machine() or "").lower()
    if device == "cpu" and machine:
        if machine in {"arm64", "aarch64"}:
            return "apple-silicon"
        return normalize_arch_label(f"{machine}-cpu")
    return normalize_arch_label(device, default="unknown")


def infer_model_family(payload: dict[str, Any], source_path: str) -> str:
    from chronohorn.families.registry import resolve_family_id
    # Check explicit fields first
    explicit = payload.get("family") or payload.get("model_family")
    if explicit:
        fid = resolve_family_id(str(explicit))
        return fid if fid else str(explicit)
    # Try to infer from title / path tokens
    haystack = f"{payload.get('title', '')} {source_path}".lower()
    for token in haystack.replace("-", "_").replace("/", " ").replace(".", " ").split():
        fid = resolve_family_id(token)
        if fid is not None:
            return fid
    return "unknown"


def infer_workload_kind(payload: dict[str, Any], source_path: str) -> str:
    path = source_path.lower()
    title = str(payload.get("title") or "").lower()
    if "parity" in path or "parity" in title:
        return "training.parity"
    if "frontier" in path or "trainer" in title or "train" in path:
        return "training.frontier"
    if "eval" in path or "fullval" in path:
        return "evaluation.fullval"
    return "unknown"


def infer_execution_backend(payload: dict[str, Any], backend_environment: dict[str, Any]) -> str:
    direct = str(payload.get("backend") or payload.get("device") or "").lower()
    if direct in {"mlx", "metal"}:
        return "metal"
    if direct in {"torch", "cuda"} and str(backend_environment.get("device") or "").lower() == "cuda":
        return "cuda"
    if str(backend_environment.get("device") or "").lower() == "cpu":
        return "cpu"
    device = str(backend_environment.get("device") or "").lower()
    if device in {"mlx", "mps", "metal"}:
        return "metal"
    if device == "cuda":
        return "cuda"
    if device == "cpu":
        return "cpu"
    return direct or "unknown"


def extract_performance_payload(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if isinstance(payload.get("training"), dict):
        training = payload["training"]
        performance = training.get("performance")
        backend_environment = training.get("backend_environment") or {}
        if isinstance(performance, dict):
            return performance, backend_environment if isinstance(backend_environment, dict) else {}
    performance = payload.get("performance")
    backend_environment = payload.get("backend_environment") or {}
    if isinstance(performance, dict):
        return performance, backend_environment if isinstance(backend_environment, dict) else {}
    return None, {}


def iter_telemetry_paths(extra_globs: list[str] | None = None) -> list[str]:
    patterns = list(DEFAULT_TELEMETRY_GLOBS)
    if extra_globs:
        patterns.extend(extra_globs)
    seen: set[str] = set()
    paths: list[str] = []
    for pattern in patterns:
        for candidate in glob.glob(pattern, recursive=True):
            if candidate in seen or not os.path.isfile(candidate):
                continue
            seen.add(candidate)
            paths.append(candidate)
    return sorted(paths)


def collect_performance_samples(extra_globs: list[str] | None = None) -> list[PerformanceSample]:
    samples: list[PerformanceSample] = []
    for path in iter_telemetry_paths(extra_globs):
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        performance, backend_environment = extract_performance_payload(payload)
        if performance is None:
            continue
        try:
            tokens_per_second = float(performance["tokens_per_second"])
        except (KeyError, TypeError, ValueError):
            continue
        tflops_raw = performance.get("estimated_sustained_tflops")
        estimated_sustained_tflops = None
        if tflops_raw is not None:
            try:
                estimated_sustained_tflops = float(tflops_raw)
            except (TypeError, ValueError):
                estimated_sustained_tflops = None
        work_tokens_raw = performance.get("tokens_completed")
        work_tokens = None
        if work_tokens_raw is not None:
            try:
                work_tokens = int(work_tokens_raw)
            except (TypeError, ValueError):
                work_tokens = None
        execution_backend = infer_execution_backend(payload, backend_environment)
        backend_family = infer_backend_family(execution_backend, backend_environment)
        accelerator_arch = infer_accelerator_arch(execution_backend, backend_environment)
        samples.append(
            PerformanceSample(
                source_path=path,
                model_family=infer_model_family(payload, path),
                workload_kind=infer_workload_kind(payload, path),
                execution_backend=execution_backend,
                backend_family=backend_family,
                accelerator_arch=accelerator_arch,
                device_name=str(backend_environment.get("device_name") or "") or None,
                tokens_per_second=tokens_per_second,
                estimated_sustained_tflops=estimated_sustained_tflops,
                work_tokens=work_tokens,
            )
        )
    return samples
