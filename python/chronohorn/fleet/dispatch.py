from __future__ import annotations

import argparse
import base64
import functools
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shlex
import subprocess
import time
from typing import Any, Sequence

from chronohorn.families.registry import resolve_training_adapter
from chronohorn.manifest_normalization import normalize_manifest_payload

# Launcher names that execute on remote hosts through the legacy SSH/Docker path.
_REMOTE_LAUNCHERS = {
    "slop_family_eval_from_table",
    "slop_causal_bank_eval_from_table",  # legacy alias
    "slop_oracle_budgeted_build",
    "slop_docker_command",
}

# Launcher names that dispatch to the family table eval handler.
_FAMILY_EVAL_LAUNCHERS = {
    "slop_family_eval_from_table",
    "slop_causal_bank_eval_from_table",  # legacy alias
}

from .planner import (
    candidate_hosts_for_job,
    choose_host,
    default_min_available_mem_gb,
    placement_cores,
)
from .k8s import (
    DEFAULT_K8S_EXECUTOR_NAME,
    default_runtime_namespace,
    infer_executor_kind,
    query_k8s_run_states,
    submit_k8s_job,
)
from .telemetry import (
    collect_performance_samples,
    infer_accelerator_arch,
    infer_backend_family,
    normalize_arch_label,
)


CHRONOHORN_ROOT = Path(__file__).resolve().parents[3]
MONOREPO_ROOT = CHRONOHORN_ROOT.parent
DEFAULT_OUT_DIR = CHRONOHORN_ROOT / "out" / "fleet"
DEFAULT_REMOTE_HOSTS = ("slop-01", "slop-02")
DEFAULT_SSH_ARGS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=5")
SNAPSHOT_EXCLUDE_DIRS = {".git", "target", "out", "__pycache__"}
SNAPSHOT_EXCLUDE_SUFFIXES = (".pyc", ".pyo", ".swp", ".tmp", ".DS_Store")
DEFAULT_MANIFEST_ENV = {
    "CHRONOHORN_ROOT": str(CHRONOHORN_ROOT),
    "CHRONOHORN_MONOREPO_ROOT": str(MONOREPO_ROOT),
}


def expand_env_vars(value: str) -> str:
    env = {**DEFAULT_MANIFEST_ENV, **os.environ}

    def replace(match: re.Match[str]) -> str:
        key = match.group(1) or match.group(2) or ""
        return env.get(key, match.group(0))

    return re.sub(r"\$(\w+)|\$\{([^}]+)\}", replace, value)


def expand_value(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(expand_env_vars(value))
    if isinstance(value, list):
        return [expand_value(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_value(item) for key, item in value.items()}
    return value


def load_manifest(path: Path) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []

    # Resolve manifest path with fallback candidates
    path_obj = Path(path)
    candidates = [path_obj]

    if not path_obj.exists():
        # Try fallback paths
        filename = path_obj.name
        candidates.extend([
            Path("manifests") / filename,
            Path("chronohorn/manifests") / filename,
        ])

        # Find the first existing candidate
        resolved_path = None
        for candidate in candidates:
            if candidate.exists():
                resolved_path = candidate
                break

        if resolved_path is None:
            candidate_paths = " | ".join(str(c) for c in candidates)
            raise FileNotFoundError(f"Manifest not found. Tried: {candidate_paths}")

        path_obj = resolved_path

    resolved_manifest_path = str(path_obj.expanduser().resolve())
    for line_number, raw_line in enumerate(path_obj.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        payload = expand_value(payload)
        payload = normalize_manifest_payload(payload)
        name = payload.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{path}:{line_number}: missing non-empty job name")
        payload.setdefault("manifest_path", resolved_manifest_path)
        payload.setdefault("run_id", f"{resolved_manifest_path}::{name}")
        jobs.append(payload)
    return jobs


def select_jobs(jobs: list[dict[str, Any]], names: list[str] | None) -> list[dict[str, Any]]:
    if not names:
        return jobs
    wanted = set(names)
    selected = [job for job in jobs if job["name"] in wanted]
    missing = sorted(wanted - {job["name"] for job in selected})
    if missing:
        raise ValueError(f"manifest is missing requested jobs: {', '.join(missing)}")
    return selected


def filter_jobs_by_class(jobs: list[dict[str, Any]], classes: list[str] | None) -> list[dict[str, Any]]:
    if not classes:
        return jobs
    wanted = set(classes)
    selected = [job for job in jobs if str(job.get("resource_class", "")) in wanted]
    if not selected:
        raise ValueError(f"manifest does not contain any jobs in classes: {', '.join(sorted(wanted))}")
    return selected


def shell_join(argv: list[str]) -> str:
    return shlex.join([str(arg) for arg in argv])


def ssh_argv(host: str, remote_command: str) -> list[str]:
    return ["ssh", *DEFAULT_SSH_ARGS, host, remote_command]


@functools.lru_cache(maxsize=None)
def compute_tree_stage_key(root: Path) -> str:
    digest = hashlib.sha256()
    digest.update(str(root.resolve()).encode("utf-8"))
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(
            part in SNAPSHOT_EXCLUDE_DIRS or part.endswith(".egg-info") or part.endswith(".dist-info")
            for part in rel.parts
        ):
            continue
        if path.is_dir():
            continue
        if path.name.endswith(SNAPSHOT_EXCLUDE_SUFFIXES):
            continue
        stat = path.stat()
        digest.update(rel.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()[:16]


def remote_container_name(name: str) -> str:
    return f"chronohorn-{''.join(ch if ch.isalnum() or ch in '._-' else '-' for ch in name)}"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def resolve_snapshot_paths(source_dir: Path, job: dict[str, Any]) -> list[str] | None:
    raw_paths = job.get("snapshot_paths")
    if raw_paths is None:
        return None
    if not isinstance(raw_paths, list) or not raw_paths:
        raise ValueError(f"{job['name']}: snapshot_paths must be a non-empty list when provided")
    resolved: list[str] = []
    source_root = source_dir.resolve()
    for raw in raw_paths:
        rel = str(raw).strip().lstrip("./")
        if not rel:
            raise ValueError(f"{job['name']}: snapshot_paths entries must be non-empty")
        candidate = (source_dir / rel).resolve()
        try:
            candidate.relative_to(source_root)
        except ValueError as exc:
            raise ValueError(
                f"{job['name']}: snapshot path {rel!r} escapes source_dir {source_dir}"
            ) from exc
        if not candidate.exists():
            raise ValueError(f"{job['name']}: snapshot path does not exist: {candidate}")
        resolved.append(rel)
    return resolved


def write_launch_record(name: str, payload: dict[str, Any]) -> Path:
    record_path = DEFAULT_OUT_DIR / f"{name}.launch.json"
    ensure_parent(record_path)
    record_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record_path


def run_checked(argv: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print(f"+ {shell_join(argv)}")
    subprocess.run(argv, cwd=cwd, env=env, check=True)


def capture_checked(argv: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(argv, cwd=cwd, env=env, text=True, capture_output=True, check=True)
    return completed.stdout


def capture_checked_retry(
    argv: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    attempts: int = 4,
    delay_sec: float = 1.5,
) -> str:
    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(1, attempts + 1):
        try:
            return capture_checked(argv, cwd=cwd, env=env)
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if attempt == attempts:
                raise
            time.sleep(delay_sec)
    assert last_error is not None
    raise last_error


def classify_remote_container(name: str) -> str:
    lowered = name.lower()
    if any(token in lowered for token in ("build", "pack", "compiler", "stats")):
        return "cpu_wide"
    if any(token in lowered for token in ("cuda", "gpu", "distill", "trainer")):
        return "cuda_gpu"
    if any(token in lowered for token in ("fullval", "eval", "audit")):
        return "cpu_serial"
    return "other"


def parse_vm_stat_bytes() -> dict[str, Any]:
    output = capture_checked(["vm_stat"])
    lines = output.splitlines()
    if not lines:
        raise RuntimeError("vm_stat returned no output")
    page_match = re.search(r"page size of (\d+) bytes", lines[0])
    if not page_match:
        raise RuntimeError("failed to parse vm_stat page size")
    page_size = int(page_match.group(1))
    page_counts: dict[str, int] = {}
    for line in lines[1:]:
        match = re.match(r'^Pages ([^:]+):\s+(\d+)\.$', line.strip())
        if not match:
            continue
        page_counts[match.group(1).lower().replace(" ", "_")] = int(match.group(2))
    available_pages = sum(
        page_counts.get(key, 0) for key in ("free", "inactive", "speculative", "purgeable")
    )
    return {
        "host": "local",
        "execution_backend": "metal",
        "backend_family": "apple",
        "accelerator_arch": "apple-silicon" if platform.machine().lower() in {"arm64", "aarch64"} else normalize_arch_label(platform.machine()),
        "device_name": "Apple Silicon",
        "page_size": page_size,
        "pages": page_counts,
        "nproc": os.cpu_count() or 0,
        "total_mem_bytes": 0,
        "available_mem_bytes": available_pages * page_size,
        "containers": [],
        "class_counts": {"cpu_serial": 0, "cpu_wide": 0, "cuda_gpu": 0, "other": 0},
        "planned_jobs": [],
        "planned_class_counts": {"cpu_serial": 0, "cpu_wide": 0, "cuda_gpu": 0, "other": 0},
        "planned_reserved_cores": 0,
        "gpu_busy": False,
    }


def split_marked_sections(output: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw_line in output.splitlines():
        line = raw_line.rstrip("\n")
        if line.startswith("__") and line.endswith("__"):
            current = line.strip("_")
            sections[current] = []
            continue
        if current is not None:
            sections[current].append(line)
    return sections


def parse_free_mem_bytes(lines: list[str]) -> tuple[int, int]:
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Mem:"):
            parts = stripped.split()
            if len(parts) >= 7:
                return int(parts[1]), int(parts[6])
    raise RuntimeError("failed to parse free -b output")


def parse_gpu_samples(lines: list[str]) -> list[dict[str, int]]:
    samples: list[dict[str, int]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        parts = [part.strip() for part in stripped.split(",")]
        if len(parts) != 3:
            continue
        try:
            util_pct, mem_used_mb, mem_total_mb = (int(part) for part in parts)
        except ValueError:
            continue
        samples.append(
            {"util_pct": util_pct, "mem_used_mb": mem_used_mb, "mem_total_mb": mem_total_mb}
        )
    return samples


def parse_gpu_info(lines: list[str]) -> tuple[list[str], str | None]:
    names: list[str] = []
    arch: str | None = None
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        parts = [part.strip() for part in stripped.split(",")]
        if not parts:
            continue
        name = parts[0]
        if not name:
            continue
        names.append(name)
        if arch is None:
            arch = normalize_arch_label(name)
    return names, arch


def probe_remote_host(host: str) -> dict[str, Any]:
    remote_payload = """set -euo pipefail
echo "__NPROC__"
nproc
echo "__UNAME__"
uname -m
echo "__FREE__"
free -b
echo "__DOCKER__"
sudo -n docker ps --format "{{.Names}}" 2>/dev/null || docker ps --format "{{.Names}}" 2>/dev/null || true
echo "__GPUINFO__"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits 2>/dev/null || true
echo "__GPU__"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || true
"""
    try:
        output = capture_checked_retry(
            ssh_argv(host, shell_join(["/bin/bash", "-lc", remote_payload])),
            attempts=4,
            delay_sec=2.0,
        )
    except subprocess.CalledProcessError as exc:
        # Keep the control plane alive when the heavyweight host probe flakes.
        return {
            "host": host,
            "execution_backend": "unknown",
            "backend_family": "unknown",
            "accelerator_arch": "unknown",
            "device_name": None,
            "nproc": 0,
            "total_mem_bytes": 0,
            "available_mem_bytes": 0,
            "containers": [],
            "class_counts": {"cpu_serial": 0, "cpu_wide": 0, "cuda_gpu": 0, "other": 0},
            "gpu_samples": [],
            "gpu_busy": False,
            "planned_jobs": [],
            "planned_class_counts": {"cpu_serial": 0, "cpu_wide": 0, "cuda_gpu": 0, "other": 0},
            "planned_reserved_cores": 0,
            "wide_core_reserve": 0,
            "probe_error": {
                "returncode": exc.returncode,
                "stderr": exc.stderr[-4000:] if isinstance(exc.stderr, str) else "",
            },
        }
    sections = split_marked_sections(output)
    nproc_lines = [line.strip() for line in sections.get("NPROC", []) if line.strip()]
    if not nproc_lines:
        raise RuntimeError(f"{host}: missing nproc output")
    nproc = int(nproc_lines[0])
    uname_lines = [line.strip() for line in sections.get("UNAME", []) if line.strip()]
    machine = uname_lines[0] if uname_lines else ""
    total_mem_bytes, available_mem_bytes = parse_free_mem_bytes(sections.get("FREE", []))
    containers = [line.strip() for line in sections.get("DOCKER", []) if line.strip()]
    class_counts = {"cpu_serial": 0, "cpu_wide": 0, "cuda_gpu": 0, "other": 0}
    for name in containers:
        class_counts[classify_remote_container(name)] += 1
    gpu_names, accelerator_arch = parse_gpu_info(sections.get("GPUINFO", []))
    gpu_samples = parse_gpu_samples(sections.get("GPU", []))
    gpu_busy = class_counts["cuda_gpu"] > 0 or any(
        sample["util_pct"] > 5 or sample["mem_used_mb"] > 256 for sample in gpu_samples
    )
    wide_core_reserve = max(8, nproc // 2)
    estimated_reserved_cores = class_counts["cpu_serial"] + class_counts["cpu_wide"] * wide_core_reserve
    execution_backend = "cuda" if gpu_names else "cpu"
    backend_environment = {"device": execution_backend, "device_name": gpu_names[0] if gpu_names else None, "machine": machine}
    backend_family = infer_backend_family(execution_backend, backend_environment)
    return {
        "host": host,
        "execution_backend": execution_backend,
        "backend_family": backend_family,
        "accelerator_arch": accelerator_arch or infer_accelerator_arch(execution_backend, backend_environment),
        "device_name": gpu_names[0] if gpu_names else None,
        "nproc": nproc,
        "total_mem_bytes": total_mem_bytes,
        "available_mem_bytes": available_mem_bytes,
        "containers": containers,
        "class_counts": class_counts,
        "gpu_samples": gpu_samples,
        "gpu_busy": gpu_busy,
        "planned_jobs": [],
        "planned_class_counts": {"cpu_serial": 0, "cpu_wide": 0, "cuda_gpu": 0, "other": 0},
        "planned_reserved_cores": estimated_reserved_cores,
        "wide_core_reserve": wide_core_reserve,
    }


def probe_fleet_state(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    need_remote = any(
        any(host != "local" for host in candidate_hosts_for_job(job, DEFAULT_REMOTE_HOSTS))
        for job in jobs
    )
    need_local = any("local" in candidate_hosts_for_job(job, DEFAULT_REMOTE_HOSTS) for job in jobs)
    remote_hosts = sorted(
        {
            host
            for job in jobs
            for host in (
                job.get("hosts")
                if isinstance(job.get("hosts"), list)
                else [job.get("host")] if job.get("host") not in (None, "", "auto") else list(DEFAULT_REMOTE_HOSTS)
            )
            if isinstance(host, str) and host and host != "local"
        }
    )
    if need_remote and not remote_hosts:
        remote_hosts = list(DEFAULT_REMOTE_HOSTS)
    remote = {host: probe_remote_host(host) for host in remote_hosts} if need_remote else {}
    local = parse_vm_stat_bytes() if need_local else None
    return {"remote": remote, "local": local}


def ensure_local_capacity(job: dict[str, Any], fleet_state: dict[str, Any]) -> None:
    local = fleet_state.get("local")
    if local is None:
        return
    min_available_mem_bytes = int(default_min_available_mem_gb(job) * 1024**3)
    if local["available_mem_bytes"] < min_available_mem_bytes:
        raise RuntimeError(
            f"{job['name']}: local memory gate failed; available={local['available_mem_bytes']/1024**3:.1f}GiB "
            f"required={min_available_mem_bytes/1024**3:.1f}GiB"
        )


def reserve_assignment(job: dict[str, Any], fleet_state: dict[str, Any]) -> None:
    host = str(job.get("host"))
    resource_class = str(job.get("resource_class", ""))
    if host == "local":
        return
    remote_state = fleet_state["remote"][host]
    remote_state["planned_jobs"].append(job["name"])
    reserved_cores = placement_cores(job)
    if resource_class in remote_state["planned_class_counts"]:
        remote_state["planned_class_counts"][resource_class] += 1
    else:
        remote_state["planned_class_counts"]["other"] += 1
    if resource_class == "cpu_wide":
        remote_state["planned_reserved_cores"] += reserved_cores
    elif resource_class == "cpu_serial":
        remote_state["planned_reserved_cores"] += reserved_cores
    elif resource_class == "cuda_gpu":
        remote_state["gpu_busy"] = True


def assign_job(job: dict[str, Any], fleet_state: dict[str, Any], samples: list[Any]) -> dict[str, Any]:
    assigned = dict(job)
    decision = choose_host(assigned, fleet_state, samples, DEFAULT_REMOTE_HOSTS)
    assigned["host"] = decision.host
    assigned["planner"] = {
        "reason": decision.reason,
        "predicted_seconds": decision.predicted_seconds,
        "expected_tokens_per_second": decision.expected_tokens_per_second,
        "expected_tflops": decision.expected_tflops,
        "telemetry_basis": (
            {
                "source_path": decision.telemetry_basis.source_path,
                "workload_kind": decision.telemetry_basis.workload_kind,
                "backend_family": decision.telemetry_basis.backend_family,
                "accelerator_arch": decision.telemetry_basis.accelerator_arch,
                "tokens_per_second": decision.telemetry_basis.tokens_per_second,
                "estimated_sustained_tflops": decision.telemetry_basis.estimated_sustained_tflops,
            }
            if decision.telemetry_basis is not None
            else None
        ),
    }
    if decision.host == "local":
        ensure_local_capacity(assigned, fleet_state)
        return assigned
    reserve_assignment(assigned, fleet_state)
    return assigned


def assign_jobs(jobs: list[dict[str, Any]], fleet_state: dict[str, Any], samples: list[Any]) -> list[dict[str, Any]]:
    return [assign_job(job, fleet_state, samples) for job in jobs]


def assign_jobs_best_effort(
    jobs: list[dict[str, Any]],
    fleet_state: dict[str, Any],
    samples: list[Any],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    assigned: list[dict[str, Any]] = []
    blocked: list[dict[str, str]] = []
    for job in jobs:
        try:
            assigned.append(assign_job(job, fleet_state, samples))
        except Exception as exc:  # noqa: BLE001
            blocked.append({"name": str(job.get("name", "")), "reason": str(exc)})
    return assigned, blocked


def local_job_running_record(name: str) -> dict[str, Any] | None:
    record_path = DEFAULT_OUT_DIR / f"{name}.launch.json"
    if not record_path.exists():
        return None
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    pid = record.get("pid")
    if not isinstance(pid, int):
        return None
    try:
        os.kill(pid, 0)
    except OSError:
        return None
    log_path = record.get("log_path")
    if isinstance(log_path, str) and log_path:
        record["log_tail_text"] = read_log_tail_text(Path(log_path))
        record["log_last_line"] = last_nonempty_line(record["log_tail_text"])
    return record


def read_log_tail_text(path: Path, *, max_lines: int = 64) -> str:
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    if max_lines <= 0:
        return ""
    return "\n".join(lines[-max_lines:]) + ("\n" if lines else "")


def last_nonempty_line(text: str) -> str:
    for line in reversed(text.splitlines()):
        if line.strip():
            return line
    return ""


def load_launch_record(name: str) -> dict[str, Any] | None:
    record_path = DEFAULT_OUT_DIR / f"{name}.launch.json"
    if not record_path.exists():
        return None
    try:
        payload = json.loads(record_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def record_remote_run(record: dict[str, Any], name: str) -> str:
    remote_run = record.get("remote_run")
    if isinstance(remote_run, str) and remote_run:
        return remote_run
    return f"/tmp/chronohorn-runs/{name}"


def query_remote_run_states(
    jobs: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for job in jobs:
        record = load_launch_record(job["name"])
        if record is None:
            continue
        host = record.get("host")
        if not isinstance(host, str) or not host or host == "local":
            continue
        launcher = str(record.get("launcher", ""))
        if launcher not in _REMOTE_LAUNCHERS:
            continue
        grouped.setdefault(host, []).append(
            {
                "name": job["name"],
                "container_name": remote_container_name(job["name"]),
                "remote_run": record_remote_run(record, job["name"]),
            }
        )

    states: dict[tuple[str, str], dict[str, Any]] = {}
    for host, entries in grouped.items():
        lines = "\n".join(
            f"{entry['name']}\t{entry['container_name']}\t{entry['remote_run']}" for entry in entries
        )
        remote_payload = f"""set -euo pipefail
docker_names="$(sudo -n docker ps --format '{{{{.Names}}}}' || true)"
while IFS=$'\\t' read -r name container run; do
  [[ -z "$name" ]] && continue
  running=0
  if grep -Fx "$container" <<<"$docker_names" >/dev/null 2>&1; then
    running=1
  fi
  report="$run/report.txt"
  result_json="$run/results/$name.json"
  log="$run/job.log"
  report_exists=0
  log_exists=0
  report_size=0
  log_size=0
  report_tail=""
  log_tail=""
  log_tail_block=""
  if [[ -e "$report" ]]; then
    report_exists=1
    report_size="$(stat -c %s "$report" 2>/dev/null || echo 0)"
    report_tail="$(tail -n 1 "$report" 2>/dev/null | base64 -w0 || true)"
  elif [[ -e "$result_json" ]]; then
    report_exists=1
    report_size="$(stat -c %s "$result_json" 2>/dev/null || echo 0)"
    report_tail="$(tail -n 1 "$result_json" 2>/dev/null | base64 -w0 || true)"
  fi
  if [[ -e "$log" ]]; then
    log_exists=1
    log_size="$(stat -c %s "$log" 2>/dev/null || echo 0)"
    log_tail="$(tail -n 1 "$log" 2>/dev/null | base64 -w0 || true)"
    log_tail_block="$(tail -n 64 "$log" 2>/dev/null | base64 -w0 || true)"
  fi
  printf '%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n' "$name" "$running" "$report_exists" "$log_exists" "$report_size" "$log_size" "$report_tail" "$log_tail" "$log_tail_block"
done <<'EOF'
{lines}
EOF
"""
        try:
            output = capture_checked_retry(
                ssh_argv(host, shell_join(["/bin/bash", "-lc", remote_payload])),
                attempts=4,
                delay_sec=1.5,
            )
        except subprocess.CalledProcessError:
            # Host-specific run-state polling is best-effort; container probes may still
            # have enough truth to keep status useful.
            continue
        parsed_by_name: dict[str, dict[str, Any]] = {}
        for raw_line in output.splitlines():
            parts = raw_line.split("\t")
            if len(parts) != 9:
                continue
            name, running, report_exists, log_exists, report_size, log_size, report_tail, log_tail, log_tail_block = parts
            parsed_by_name[name] = {
                "host": host,
                "name": name,
                "running": running == "1",
                "report_exists": report_exists == "1",
                "log_exists": log_exists == "1",
                "report_size_bytes": int(report_size or "0"),
                "log_size_bytes": int(log_size or "0"),
                "report_last_line": (
                    base64.b64decode(report_tail).decode("utf-8", errors="replace") if report_tail else ""
                ),
                "log_last_line": base64.b64decode(log_tail).decode("utf-8", errors="replace") if log_tail else "",
                "log_tail_text": (
                    base64.b64decode(log_tail_block).decode("utf-8", errors="replace") if log_tail_block else ""
                ),
            }
        for entry in entries:
            state = parsed_by_name.get(entry["name"])
            if state is None:
                continue
            states[(host, entry["name"])] = state
    return states


def detect_running_job(
    job: dict[str, Any],
    fleet_state: dict[str, Any],
    remote_run_states: dict[tuple[str, str], dict[str, Any]],
    k8s_run_states: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any] | None:
    record = load_launch_record(job["name"]) or {}
    executor_kind = infer_executor_kind(record) or infer_executor_kind(job)
    if executor_kind == "k8s_cluster":
        namespace = str(
            record.get("runtime_namespace")
            or job.get("runtime_namespace")
            or default_runtime_namespace(job)
        )
        state = k8s_run_states.get((namespace, job["name"]))
        if state and state.get("phase") in {"pending", "running"}:
            return {
                "name": job["name"],
                "family": job.get("family"),
                "host": state.get("runtime_node_name") or record.get("host") or job.get("host"),
                "backend": job.get("backend"),
                "resource_class": job.get("resource_class"),
                "launcher": record.get("launcher") or job.get("launcher"),
                "executor_kind": "k8s_cluster",
                "executor_name": state.get("executor_name") or record.get("executor_name"),
                "state": state.get("phase"),
                "runtime_namespace": state.get("runtime_namespace"),
                "runtime_job_name": state.get("runtime_job_name"),
                "runtime_pod_name": state.get("runtime_pod_name"),
                "runtime_node_name": state.get("runtime_node_name"),
                "reason": state.get("reason"),
                "message": state.get("message"),
                "log_last_line": state.get("log_last_line"),
                "log_tail_text": state.get("log_tail_text"),
            }
        return None
    candidates = candidate_hosts_for_job(job, DEFAULT_REMOTE_HOSTS)
    if candidates == ["local"]:
        local_record = local_job_running_record(job["name"])
        if local_record is None:
            return None
        return {
            "name": job["name"],
            "family": job.get("family"),
            "host": "local",
            "backend": job.get("backend"),
            "resource_class": job.get("resource_class"),
            "launcher": job.get("launcher"),
            "executor_kind": "local_process",
            "executor_name": "local",
            "state": "running",
            "record": local_record,
            "log_last_line": local_record.get("log_last_line"),
            "log_tail_text": local_record.get("log_tail_text"),
        }
    expected = remote_container_name(job["name"])
    for host in candidates:
        run_state = remote_run_states.get((host, job["name"]))
        state = fleet_state.get("remote", {}).get(host)
        if state is not None and expected in state["containers"]:
            payload = {
                "name": job["name"],
                "family": job.get("family"),
                "host": host,
                "backend": job.get("backend"),
                "resource_class": job.get("resource_class"),
                "launcher": job.get("launcher"),
                "executor_kind": "docker_host",
                "executor_name": host,
                "state": "running",
                "container_name": expected,
            }
            if run_state:
                payload["log_last_line"] = run_state.get("log_last_line")
                payload["log_size_bytes"] = run_state.get("log_size_bytes")
                payload["log_tail_text"] = run_state.get("log_tail_text")
            return payload
        if run_state and run_state.get("running"):
            return {
                "name": job["name"],
                "family": job.get("family"),
                "host": host,
                "backend": job.get("backend"),
                "resource_class": job.get("resource_class"),
                "launcher": job.get("launcher"),
                "executor_kind": "docker_host",
                "executor_name": host,
                "state": "running",
                "container_name": expected,
                "log_last_line": run_state.get("log_last_line"),
                "log_size_bytes": run_state.get("log_size_bytes"),
                "log_tail_text": run_state.get("log_tail_text"),
            }
    return None


def detect_completed_job(
    job: dict[str, Any],
    remote_run_states: dict[tuple[str, str], dict[str, Any]],
    k8s_run_states: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any] | None:
    record = load_launch_record(job["name"])
    if record is None:
        return None
    executor_kind = infer_executor_kind(record) or infer_executor_kind(job)
    if executor_kind == "k8s_cluster":
        namespace = str(
            record.get("runtime_namespace")
            or job.get("runtime_namespace")
            or default_runtime_namespace(job)
        )
        state = k8s_run_states.get((namespace, job["name"]))
        if not state or state.get("phase") != "succeeded":
            return None
        return {
            "name": job["name"],
            "family": job.get("family"),
            "host": state.get("runtime_node_name") or record.get("host") or job.get("host"),
            "backend": job.get("backend"),
            "resource_class": job.get("resource_class"),
            "launcher": record.get("launcher") or job.get("launcher"),
            "executor_kind": "k8s_cluster",
            "executor_name": state.get("executor_name") or record.get("executor_name"),
            "state": "completed",
            "runtime_namespace": state.get("runtime_namespace"),
            "runtime_job_name": state.get("runtime_job_name"),
            "runtime_pod_name": state.get("runtime_pod_name"),
            "runtime_node_name": state.get("runtime_node_name"),
        }
        return None
    host = record.get("host")
    if not isinstance(host, str) or not host or host == "local":
        return None
    state = remote_run_states.get((host, job["name"]))
    if not state:
        return None
    if state["report_exists"] and state["report_size_bytes"] > 0 and not state["running"]:
        return {
            "name": job["name"],
            "family": job.get("family"),
            "host": host,
            "backend": job.get("backend"),
            "resource_class": job.get("resource_class"),
            "launcher": job.get("launcher"),
            "executor_kind": "docker_host",
            "executor_name": host,
            "state": "completed",
            "report_size_bytes": state["report_size_bytes"],
            "report_last_line": state["report_last_line"],
        }
    return None


def detect_stale_job(
    job: dict[str, Any],
    remote_run_states: dict[tuple[str, str], dict[str, Any]],
    k8s_run_states: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any] | None:
    record = load_launch_record(job["name"])
    if record is None:
        return None
    executor_kind = infer_executor_kind(record) or infer_executor_kind(job)
    if executor_kind == "k8s_cluster":
        namespace = str(
            record.get("runtime_namespace")
            or job.get("runtime_namespace")
            or default_runtime_namespace(job)
        )
        state = k8s_run_states.get((namespace, job["name"]))
        if not state:
            return None
        if state.get("phase") in {"failed", "missing"}:
            return {
                "name": job["name"],
                "family": job.get("family"),
                "host": state.get("runtime_node_name") or record.get("host") or job.get("host"),
                "backend": job.get("backend"),
                "resource_class": job.get("resource_class"),
                "launcher": record.get("launcher") or job.get("launcher"),
                "executor_kind": "k8s_cluster",
                "executor_name": state.get("executor_name") or record.get("executor_name"),
                "state": "stale",
                "runtime_namespace": state.get("runtime_namespace"),
                "runtime_job_name": state.get("runtime_job_name"),
                "runtime_pod_name": state.get("runtime_pod_name"),
                "runtime_node_name": state.get("runtime_node_name"),
                "phase": state.get("phase"),
                "log_last_line": state.get("log_last_line"),
                "log_tail_text": state.get("log_tail_text"),
            }
        return None
    host = record.get("host")
    if not isinstance(host, str) or not host or host == "local":
        return None
    state = remote_run_states.get((host, job["name"]))
    if not state:
        return None
    if not state["running"] and state["log_exists"] and not state["report_exists"]:
        return {
            "name": job["name"],
            "family": job.get("family"),
            "host": host,
            "backend": job.get("backend"),
            "resource_class": job.get("resource_class"),
            "launcher": job.get("launcher"),
            "executor_kind": "docker_host",
            "executor_name": host,
            "state": "stale",
            "log_size_bytes": state["log_size_bytes"],
        }
    return None


def partition_running_jobs(
    jobs: list[dict[str, Any]], fleet_state: dict[str, Any], *, relaunch_completed: bool = False
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    pending: list[dict[str, Any]] = []
    running: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    stale: list[dict[str, Any]] = []
    remote_run_states = query_remote_run_states(jobs)
    k8s_run_states = query_k8s_run_states(jobs)
    for job in jobs:
        running_record = detect_running_job(job, fleet_state, remote_run_states, k8s_run_states)
        if running_record is None:
            completed_record = detect_completed_job(job, remote_run_states, k8s_run_states)
            if completed_record is not None and not relaunch_completed:
                completed.append(completed_record)
                continue
            stale_record = detect_stale_job(job, remote_run_states, k8s_run_states)
            if stale_record is not None:
                stale.append(stale_record)
            pending.append(job)
        else:
            running.append(running_record)
    return pending, running, completed, stale


def fleet_state_summary(fleet_state: dict[str, Any]) -> dict[str, Any]:
    local = fleet_state.get("local")
    summary: dict[str, Any] = {"remote": {}}
    if local is not None:
        summary["local"] = {
            "execution_backend": local["execution_backend"],
            "backend_family": local["backend_family"],
            "accelerator_arch": local["accelerator_arch"],
            "device_name": local["device_name"],
            "available_mem_gib": round(local["available_mem_bytes"] / 1024**3, 3),
            "page_size": local["page_size"],
            "nproc": local["nproc"],
        }
    for host, state in fleet_state.get("remote", {}).items():
        summary["remote"][host] = {
            "execution_backend": state["execution_backend"],
            "backend_family": state["backend_family"],
            "accelerator_arch": state["accelerator_arch"],
            "device_name": state["device_name"],
            "nproc": state["nproc"],
            "available_mem_gib": round(state["available_mem_bytes"] / 1024**3, 3),
            "containers": state["containers"],
            "class_counts": state["class_counts"],
            "planned_class_counts": state["planned_class_counts"],
            "gpu_busy": state["gpu_busy"],
            "planned_reserved_cores": state["planned_reserved_cores"],
            "planned_jobs": state["planned_jobs"],
        }
        if state.get("probe_error") is not None:
            summary["remote"][host]["probe_error"] = state["probe_error"]
    return summary


def local_command_argv(job: dict[str, Any]) -> list[str]:
    if "argv" in job:
        argv = job["argv"]
        if not isinstance(argv, list) or not argv:
            raise ValueError(f"{job['name']}: argv must be a non-empty list")
        return [str(arg) for arg in argv]
    command = job.get("command")
    if not isinstance(command, str) or not command.strip():
        raise ValueError(f"{job['name']}: local_command requires argv or command")
    return ["/bin/zsh", "-lc", command]


def launch_local_command(job: dict[str, Any]) -> dict[str, Any]:
    cwd_raw = job.get("cwd")
    if not isinstance(cwd_raw, str) or not cwd_raw:
        raise ValueError(f"{job['name']}: local_command requires cwd")
    cwd = Path(cwd_raw)
    argv = local_command_argv(job)
    env = os.environ.copy()
    for key, value in job.get("env", {}).items():
        env[str(key)] = str(value)
    log_path = Path(job.get("log_path") or (DEFAULT_OUT_DIR / f"{job['name']}.log"))
    ensure_parent(log_path)
    with log_path.open("ab") as handle:
        proc = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    record = {
        "name": job["name"],
        "run_id": job.get("run_id"),
        "manifest_path": job.get("manifest_path"),
        "family": job.get("family"),
        "backend": job.get("backend"),
        "resource_class": job.get("resource_class"),
        "goal": job.get("goal"),
        "planner": job.get("planner"),
        "requested_launcher": job.get("requested_launcher", job["launcher"]),
        "launcher": job["launcher"],
        "executor_kind": "local_process",
        "executor_name": "local",
        "host": job.get("host", "local"),
        "cwd": str(cwd),
        "argv": argv,
        "pid": proc.pid,
        "log_path": str(log_path),
        "launched_at_unix": time.time(),
    }
    write_launch_record(job["name"], record)
    return record


def launch_managed_command(job: dict[str, Any]) -> dict[str, Any]:
    host = str(job.get("host", ""))
    if host == "local":
        local_job = dict(job)
        local_job["requested_launcher"] = "managed_command"
        source_dir = Path(str(local_job.get("source_dir", CHRONOHORN_ROOT)))
        remote_cwd_rel = str(local_job.get("remote_cwd_rel", ".")).strip()
        local_job["cwd"] = str((source_dir / remote_cwd_rel).resolve())
        local_job["launcher"] = "local_command"
        return launch_local_command(local_job)
    remote_job = dict(job)
    remote_job["requested_launcher"] = "managed_command"
    remote_job["launcher"] = "k8s_job"
    remote_job.setdefault("executor_kind", "k8s_cluster")
    remote_job.setdefault("executor_name", DEFAULT_K8S_EXECUTOR_NAME)
    return launch_k8s_job(remote_job)


def launch_slop_family_eval_from_table(job: dict[str, Any]) -> dict[str, Any]:
    family_id = str(job.get("family") or job.get("model_family") or "").strip()
    if not family_id:
        raise ValueError(f"{job['name']}: slop_family_eval_from_table requires family")
    adapter = resolve_training_adapter(family_id)
    stage_key = compute_tree_stage_key(CHRONOHORN_ROOT)
    argv = adapter.build_table_eval_argv(job=job, chronohorn_root=CHRONOHORN_ROOT)
    env = os.environ.copy()
    env["CHRONOHORN_STAGE_KEY"] = stage_key
    run_checked(argv, cwd=CHRONOHORN_ROOT, env=env)
    record = {
        "name": job["name"],
        "run_id": job.get("run_id"),
        "manifest_path": job.get("manifest_path"),
        "family": job.get("family"),
        "backend": job.get("backend"),
        "resource_class": job.get("resource_class"),
        "goal": job.get("goal"),
        "planner": job.get("planner"),
        "requested_launcher": job.get("requested_launcher", job["launcher"]),
        "launcher": job["launcher"],
        "executor_kind": "ssh_host",
        "executor_name": job["host"],
        "host": job["host"],
        "argv": argv,
        "stage_key": stage_key,
        "launched_at_unix": time.time(),
    }
    write_launch_record(job["name"], record)
    return record


def launch_slop_build_table(job: dict[str, Any]) -> dict[str, Any]:
    script = CHRONOHORN_ROOT / "scripts" / "slop_build_oracle_budgeted_table.zsh"
    argv = [
        "zsh",
        str(script),
        str(job["host"]),
        str(job["name"]),
        str(job["train_tokens"]),
        str(job["profile"]),
        str(job.get("threads", 12)),
        str(job.get("report_every", 5000)),
        str(job.get("oracle_stride", 1)),
    ]
    run_checked(argv, cwd=CHRONOHORN_ROOT)
    record = {
        "name": job["name"],
        "run_id": job.get("run_id"),
        "manifest_path": job.get("manifest_path"),
        "family": job.get("family"),
        "backend": job.get("backend"),
        "resource_class": job.get("resource_class"),
        "goal": job.get("goal"),
        "planner": job.get("planner"),
        "requested_launcher": job.get("requested_launcher", job["launcher"]),
        "launcher": job["launcher"],
        "executor_kind": "ssh_host",
        "executor_name": job["host"],
        "host": job["host"],
        "argv": argv,
        "launched_at_unix": time.time(),
    }
    write_launch_record(job["name"], record)
    return record


def render_remote_exports(env_map: dict[str, str]) -> str:
    exports = [f"export {key}={shlex.quote(value)}" for key, value in sorted(env_map.items())]
    return "\n".join(exports)


def launch_slop_docker_command(job: dict[str, Any]) -> dict[str, Any]:
    host = str(job["host"])
    name = str(job["name"])
    image = str(job["image"])
    source_dir = Path(str(job.get("source_dir", MONOREPO_ROOT)))
    remote_cwd_rel = str(job.get("remote_cwd_rel", ".")).strip()
    command = job.get("command")
    if not isinstance(command, str) or not command.strip():
        raise ValueError(f"{name}: slop_docker_command requires a non-empty command string")
    if not source_dir.exists():
        raise ValueError(f"{name}: source_dir does not exist: {source_dir}")
    snapshot_paths = resolve_snapshot_paths(source_dir, job)

    remote_cache = "/tmp/chronohorn-cache"
    remote_run = f"/tmp/chronohorn-runs/{name}"
    remote_snapshot = f"{remote_run}/snapshot"
    remote_assets = f"{remote_run}/assets"
    container_name = remote_container_name(name)

    run_checked(
        ssh_argv(
            host,
            (
                "sudo -n rm -rf "
                f"{shlex.quote(remote_run)}"
                " >/dev/null 2>&1 || true; "
                "mkdir -p "
                f"{shlex.quote(remote_snapshot)} "
                f"{shlex.quote(remote_assets)} "
                f"{shlex.quote(remote_cache + '/cargo/registry')} "
                f"{shlex.quote(remote_cache + '/cargo/git')} "
                f"{shlex.quote(remote_cache + '/target')}"
            ),
        )
    )
    if snapshot_paths:
        remote_snapshot_dirs = sorted(
            {
                str(PurePosixPath(remote_snapshot) / Path(rel).parent.as_posix())
                for rel in snapshot_paths
            }
        )
        run_checked(
            ssh_argv(
                host,
                "mkdir -p " + " ".join(shlex.quote(path) for path in remote_snapshot_dirs),
            )
        )
    rsync_argv = [
        "rsync",
        "-a",
        "--delete",
        "--exclude",
        ".git",
        "--exclude",
        "target",
        "--exclude",
        "out",
        "--exclude",
        "__pycache__",
        "--exclude",
        ".DS_Store",
        "--exclude",
        "*.egg-info",
        "--exclude",
        "*.dist-info",
    ]
    for pattern in job.get("rsync_excludes", []):
        rsync_argv.extend(["--exclude", str(pattern)])
    rsync_cwd: Path | None = None
    if snapshot_paths:
        rsync_argv.append("--relative")
        for rel in snapshot_paths:
            rsync_argv.append(f"./{rel}")
        rsync_cwd = source_dir
    else:
        rsync_argv.append(f"{source_dir}/")
    rsync_argv.append(f"{host}:{remote_snapshot}/")
    run_checked(rsync_argv, cwd=rsync_cwd)
    for sync_path_raw in job.get("sync_paths", []):
        sync_path = Path(str(sync_path_raw))
        run_checked(["rsync", "-a", str(sync_path), f"{host}:{remote_assets}/"])

    remote_env = {
        "CHRONOHORN_REMOTE_ASSETS": "/assets",
        "CHRONOHORN_REMOTE_RUN": "/run",
    }
    if "threads" in job:
        remote_env["CHRONOHORN_THREADS"] = str(job["threads"])
    for key, value in job.get("env", {}).items():
        remote_env[str(key)] = str(value)

    docker_env_flags = " ".join(
        f"-e {shlex.quote(key)}={shlex.quote(value)}" for key, value in sorted(remote_env.items())
    )
    gpu_flag = "--gpus all" if job.get("gpu", False) else ""
    workdir = f"/snapshot/{remote_cwd_rel}".rstrip("/")
    remote_payload = f"""set -euo pipefail
sudo -n docker rm -f {shlex.quote(container_name)} >/dev/null 2>&1 || true
sudo -n docker image inspect {shlex.quote(image)} >/dev/null 2>&1 || sudo -n docker pull {shlex.quote(image)} >/dev/null
nohup sudo -n docker run --rm --name {shlex.quote(container_name)} {gpu_flag} \\
  {docker_env_flags} \\
  -v {shlex.quote(remote_snapshot)}:/snapshot \\
  -v {shlex.quote(remote_assets)}:/assets \\
  -v {shlex.quote(remote_run)}:/run \\
  -v {shlex.quote(remote_cache)}:/cache \\
  -v /data:/data \\
  {shlex.quote(image)} \\
  bash -lc '
    set -euo pipefail
    cd {shlex.quote(workdir)}
    {render_remote_exports(remote_env)}
    {command}
  ' > {shlex.quote(remote_run + '/job.log')} 2>&1 &
echo {shlex.quote(container_name)}
echo {shlex.quote(remote_run)}
"""
    run_checked(ssh_argv(host, remote_payload))
    record = {
        "name": name,
        "run_id": job.get("run_id"),
        "manifest_path": job.get("manifest_path"),
        "family": job.get("family"),
        "backend": job.get("backend"),
        "resource_class": job.get("resource_class"),
        "goal": job.get("goal"),
        "planner": job.get("planner"),
        "requested_launcher": job.get("requested_launcher", job["launcher"]),
        "launcher": job["launcher"],
        "executor_kind": "docker_host",
        "executor_name": host,
        "host": host,
        "image": image,
        "gpu": bool(job.get("gpu", False)),
        "container_name": container_name,
        "remote_run": remote_run,
        "remote_snapshot": remote_snapshot,
        "remote_assets": remote_assets,
        "remote_cwd_rel": remote_cwd_rel,
        "snapshot_paths": snapshot_paths,
        "command": command,
        "launched_at_unix": time.time(),
    }
    write_launch_record(name, record)
    return record


def launch_k8s_job(job: dict[str, Any]) -> dict[str, Any]:
    remote_job = dict(job)
    remote_job.setdefault("launcher", "k8s_job")
    remote_job.setdefault("executor_kind", "k8s_cluster")
    remote_job.setdefault("executor_name", DEFAULT_K8S_EXECUTOR_NAME)
    record = submit_k8s_job(remote_job)
    write_launch_record(remote_job["name"], record)
    return record


def launch_job(job: dict[str, Any]) -> dict[str, Any]:
    launcher = job.get("launcher")
    if launcher == "local_command":
        return launch_local_command(job)
    if launcher == "managed_command":
        return launch_managed_command(job)
    if launcher == "k8s_job":
        return launch_k8s_job(job)
    if launcher in _FAMILY_EVAL_LAUNCHERS:
        return launch_slop_family_eval_from_table(job)
    if launcher == "slop_oracle_budgeted_build":
        return launch_slop_build_table(job)
    if launcher == "slop_docker_command":
        return launch_slop_docker_command(job)
    raise ValueError(f"{job['name']}: unsupported launcher {launcher}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="chronohorn fleet",
        description=(
            "Manifest-driven Chronohorn fleet launcher and telemetry-backed "
            "runtime planner for local Apple and remote CPU/NVIDIA lanes."
        ),
    )
    parser.add_argument("--manifest", required=True, help="JSONL manifest path")
    parser.add_argument(
        "--job",
        action="append",
        default=[],
        help="Launch only the named job (repeatable). Default: launch all jobs in manifest order.",
    )
    parser.add_argument(
        "--class",
        dest="classes",
        action="append",
        default=[],
        help="Launch only jobs in the given resource_class (repeatable).",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Probe the fleet and print the current state plus planned assignments without launching jobs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve dynamic placement and print the launch plan without starting jobs.",
    )
    parser.add_argument(
        "--relaunch-completed",
        action="store_true",
        help="Launch jobs even when a completed remote report already exists.",
    )
    parser.add_argument(
        "--telemetry-glob",
        action="append",
        default=[],
        help="Additional glob to scan for Chronohorn performance JSONs when planning placement.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest).expanduser().resolve()
    jobs = load_manifest(manifest_path)
    jobs = filter_jobs_by_class(jobs, args.classes or None)
    jobs = select_jobs(jobs, args.job or None)
    fleet_state = probe_fleet_state(jobs)
    telemetry_samples = collect_performance_samples(args.telemetry_glob or None)
    pending_jobs, running_jobs, completed_jobs, stale_jobs = partition_running_jobs(
        jobs, fleet_state, relaunch_completed=args.relaunch_completed
    )
    assigned_jobs, blocked_jobs = assign_jobs_best_effort(pending_jobs, fleet_state, telemetry_samples)
    summary = {
        "manifest": str(manifest_path),
        "fleet": fleet_state_summary(fleet_state),
        "telemetry": {
            "sample_count": len(telemetry_samples),
            "sources": sorted({sample.source_path for sample in telemetry_samples}),
        },
        "already_running": running_jobs,
        "completed": completed_jobs,
        "stale": stale_jobs,
        "blocked": blocked_jobs,
        "planned": assigned_jobs,
    }
    if args.status or args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    results = [launch_job(job) for job in assigned_jobs]
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "fleet": fleet_state_summary(fleet_state),
                "already_running": running_jobs,
                "completed": completed_jobs,
                "stale": stale_jobs,
                "blocked": blocked_jobs,
                "launched": results,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
