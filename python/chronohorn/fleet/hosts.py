from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Any, Sequence

from chronohorn.fleet.dispatch import (
    DEFAULT_REMOTE_HOSTS,
    capture_checked_retry,
    partition_running_jobs,
    probe_fleet_state,
    probe_remote_host,
    shell_join,
    split_marked_sections,
    ssh_argv,
)


DEFAULT_FLEET_HOSTS: tuple[str, ...] = tuple(DEFAULT_REMOTE_HOSTS)


def normalize_hosts(hosts: Sequence[str] | None = None) -> list[str]:
    values = hosts or DEFAULT_FLEET_HOSTS
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in values:
        host = str(raw).strip()
        if not host or host in seen:
            continue
        seen.add(host)
        ordered.append(host)
    return ordered


def _parse_container_rows(lines: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or "|" not in stripped:
            continue
        name, status = stripped.split("|", 1)
        rows.append({"name": name.strip(), "status": status.strip()})
    return rows


def _parse_top_processes(lines: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(None, 5)
        if len(parts) < 6:
            continue
        pid, cpu_pct, mem_pct, elapsed, command, args = parts
        try:
            pid_value: int | None = int(pid)
        except ValueError:
            pid_value = None
        try:
            cpu_value: float | None = float(cpu_pct)
        except ValueError:
            cpu_value = None
        try:
            mem_value: float | None = float(mem_pct)
        except ValueError:
            mem_value = None
        rows.append(
            {
                "pid": pid_value,
                "cpu_pct": cpu_value,
                "mem_pct": mem_value,
                "elapsed": elapsed,
                "command": command,
                "args": args,
            }
        )
    return rows


def _is_probe_noise(row: dict[str, Any]) -> bool:
    command = str(row.get("command") or "")
    args = str(row.get("args") or "")
    if command == "sshd" and "sshd: slop [priv]" in args:
        return True
    if command == "bash" and (
        "chronohorn-runs" in args
        or "ps -eo pid=,%cpu=,%mem=,etime=,comm=,args=" in args
        or "__HOSTNAME__" in args
    ):
        return True
    if command in {"head", "ps"}:
        return True
    return False


def _parse_gpu_apps(lines: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        parts = [part.strip() for part in stripped.split(",")]
        if len(parts) < 3:
            continue
        pid_text, process_name, used_memory_text = parts[:3]
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        used_memory_text = used_memory_text.replace(" MiB", "").strip()
        try:
            used_memory_mb: int | None = int(used_memory_text)
        except ValueError:
            used_memory_mb = None
        rows.append(
            {
                "pid": pid,
                "process_name": process_name,
                "used_memory_mb": used_memory_mb,
            }
        )
    return rows


def _first_text(lines: list[str]) -> str | None:
    for line in lines:
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def _int_text(lines: list[str]) -> int | None:
    text = _first_text(lines)
    if text is None:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _configured_hosts(job: dict[str, Any]) -> list[str]:
    hosts = job.get("hosts")
    if isinstance(hosts, list):
        values = [str(host).strip() for host in hosts if str(host).strip()]
        if values:
            return values
    host = str(job.get("host") or "").strip()
    if host and host != "auto":
        return [host]
    return list(DEFAULT_FLEET_HOSTS)


def _job_matches_hosts(job: dict[str, Any], wanted: set[str]) -> bool:
    if not wanted:
        return True
    return any(host in wanted for host in _configured_hosts(job))


def _job_matches_manifest(job: dict[str, Any], manifest: str | None) -> bool:
    if not manifest:
        return True
    wanted = Path(str(manifest)).name
    job_manifest = Path(str(job.get("manifest") or job.get("manifest_path") or "")).name
    return job_manifest == wanted


def _job_brief(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": str(job.get("name") or ""),
        "family": job.get("family"),
        "state": job.get("state"),
        "executor_kind": job.get("executor_kind"),
        "executor_name": job.get("executor_name"),
        "resource_class": job.get("resource_class"),
        "launcher": job.get("launcher"),
        "backend": job.get("backend"),
        "configured_hosts": _configured_hosts(job),
    }


def probe_host(
    host: str,
    *,
    include_processes: bool = False,
    process_limit: int = 8,
    include_remote_results: bool = False,
    remote_result_dir: str = "/data/chronohorn/out/results",
) -> dict[str, Any]:
    info = dict(probe_remote_host(host))
    info["online"] = info.get("probe_error") is None
    info["container_rows"] = [{"name": name, "status": "running"} for name in info.get("containers", [])]
    if not include_processes and not include_remote_results:
        return info
    if not info["online"]:
        return info

    limit = max(1, int(process_limit))
    remote_dir = shlex.quote(str(remote_result_dir))
    remote_payload = f"""set -euo pipefail
echo "__HOSTNAME__"
hostname
echo "__UPTIME__"
uptime
echo "__DOCKER_STATUS__"
sudo -n docker ps --format "{{{{.Names}}}}|{{{{.Status}}}}" 2>/dev/null || docker ps --format "{{{{.Names}}}}|{{{{.Status}}}}" 2>/dev/null || true
echo "__GPU_APPS__"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true
echo "__WAITERS__"
ps aux | grep -v grep | grep -c "v11_waiter\\|nohup.*bash.*docker" || true
echo "__TOP__"
ps -eo pid=,%cpu=,%mem=,etime=,comm=,args= --sort=-%cpu | head -n {limit}
echo "__RESULTS__"
if [[ -d {remote_dir} ]]; then
  find {remote_dir} -maxdepth 1 -type f -name '*.json' | wc -l
else
  echo 0
fi
"""
    try:
        output = capture_checked_retry(
            ssh_argv(host, shell_join(["/bin/bash", "-lc", remote_payload])),
            attempts=2,
            delay_sec=1.0,
        )
    except subprocess.CalledProcessError as exc:
        info["detail_error"] = {
            "returncode": exc.returncode,
            "stderr": exc.stderr[-4000:] if isinstance(exc.stderr, str) else "",
        }
        return info

    sections = split_marked_sections(output)
    container_rows = _parse_container_rows(sections.get("DOCKER_STATUS", []))
    if container_rows:
        info["container_rows"] = container_rows
        info["containers"] = [row["name"] for row in container_rows]
    info["hostname"] = _first_text(sections.get("HOSTNAME", [])) or host
    info["uptime"] = _first_text(sections.get("UPTIME", []))
    info["waiter_count"] = _int_text(sections.get("WAITERS", []))
    top_processes = _parse_top_processes(sections.get("TOP", [])) if include_processes else []
    info["top_processes"] = [row for row in top_processes if not _is_probe_noise(row)]
    info["gpu_apps"] = _parse_gpu_apps(sections.get("GPU_APPS", []))
    if include_remote_results:
        info["remote_result_count"] = _int_text(sections.get("RESULTS", []))
    return info


def probe_hosts(
    hosts: Sequence[str] | None = None,
    *,
    include_processes: bool = False,
    process_limit: int = 8,
    include_remote_results: bool = False,
    remote_result_dir: str = "/data/chronohorn/out/results",
) -> list[dict[str, Any]]:
    return [
        probe_host(
            host,
            include_processes=include_processes,
            process_limit=process_limit,
            include_remote_results=include_remote_results,
            remote_result_dir=remote_result_dir,
        )
        for host in normalize_hosts(hosts)
    ]


def inspect_remote_runs(
    db,
    *,
    hosts: Sequence[str] | None = None,
    job_names: Sequence[str] | None = None,
    classes: Sequence[str] | None = None,
    manifest: str | None = None,
    include_logs: bool = False,
    relaunch_completed: bool = False,
) -> dict[str, Any]:
    wanted_hosts = set(normalize_hosts(hosts)) if hosts else set()
    wanted_names = {str(name).strip() for name in (job_names or []) if str(name).strip()}
    wanted_classes = {str(value).strip() for value in (classes or []) if str(value).strip()}

    jobs = [
        job
        for job in db.active_jobs()
        if _job_matches_manifest(job, manifest)
        and (not wanted_names or str(job.get("name") or "") in wanted_names)
        and (not wanted_classes or str(job.get("resource_class") or "") in wanted_classes)
        and _job_matches_hosts(job, wanted_hosts)
    ]
    fleet_state = probe_fleet_state(jobs)
    pending, running, completed, stale = partition_running_jobs(
        jobs,
        fleet_state,
        relaunch_completed=relaunch_completed,
    )
    launch_pending = [row for row in running if str(row.get("state") or "") == "pending"]
    running = [row for row in running if str(row.get("state") or "") != "pending"]

    def sanitize(row: dict[str, Any]) -> dict[str, Any]:
        payload = dict(row)
        if not include_logs:
            payload.pop("log_tail_text", None)
        return payload

    host_counts: dict[str, int] = {}
    for row in running + completed + stale:
        host = str(row.get("host") or "")
        if host:
            host_counts[host] = host_counts.get(host, 0) + 1

    return {
        "summary": {
            "pending": len(pending),
            "running": len(running),
            "launch_pending": len(launch_pending),
            "completed": len(completed),
            "stale": len(stale),
            "host_counts": host_counts,
        },
        "pending": [_job_brief(job) for job in pending],
        "launch_pending": [sanitize(row) for row in launch_pending],
        "running": [sanitize(row) for row in running],
        "completed": [sanitize(row) for row in completed],
        "stale": [sanitize(row) for row in stale],
        "fleet": fleet_state,
    }
