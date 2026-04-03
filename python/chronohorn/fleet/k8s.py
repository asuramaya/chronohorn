from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import time
from typing import Any, Mapping, Sequence


DEFAULT_SSH_ARGS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=5")
DEFAULT_K8S_GATEWAY_HOST = "slop-01"
DEFAULT_K8S_EXECUTOR_NAME = "slop-cluster"
DEFAULT_K8S_NAMESPACE = "default"
DEFAULT_REMOTE_RUN_ROOT = "/tmp/chronohorn-runs"
DEFAULT_REMOTE_CACHE_DIR = "/tmp/chronohorn-cache"

_K8S_NAME_RE = re.compile(r"[^a-z0-9-]+")


def ssh_argv(host: str, remote_command: str) -> list[str]:
    return ["ssh", *DEFAULT_SSH_ARGS, host, remote_command]


def shell_join(argv: Sequence[str]) -> str:
    return shlex.join([str(arg) for arg in argv])


def k8s_job_name(name: str) -> str:
    lowered = str(name).strip().lower()
    slug = _K8S_NAME_RE.sub("-", lowered)
    slug = re.sub(r"-{2,}", "-", slug).strip("-") or "run"
    digest = hashlib.sha1(str(name).encode("utf-8")).hexdigest()[:8]
    body_limit = 63 - len("ch-") - 1 - len(digest)
    body = slug[:body_limit].rstrip("-") or "run"
    return f"ch-{body}-{digest}"


def infer_executor_kind(spec: Mapping[str, Any]) -> str | None:
    explicit = str(spec.get("executor_kind") or "").strip()
    if explicit:
        return explicit
    launcher = str(spec.get("launcher") or spec.get("requested_launcher") or "").strip()
    host = str(spec.get("host") or "").strip()
    backend = str(spec.get("backend") or "").strip().lower()
    if host == "local" or launcher == "local_command":
        return "local_process"
    if launcher == "k8s_job":
        return "k8s_cluster"
    if launcher == "slop_docker_command":
        return "docker_host"
    if launcher in {
        "slop_family_eval_from_table",
        "slop_causal_bank_eval_from_table",
        "slop_oracle_budgeted_build",
    }:
        return "ssh_host"
    if launcher == "managed_command":
        if backend == "metal":
            return "local_process"
        if host == "local":
            return "local_process"
        return "k8s_cluster"
    return None


def default_executor_name(spec: Mapping[str, Any], *, executor_kind: str | None = None) -> str | None:
    explicit = str(spec.get("executor_name") or "").strip()
    if explicit:
        return explicit
    kind = executor_kind or infer_executor_kind(spec)
    if kind == "local_process":
        return "local"
    if kind == "k8s_cluster":
        return DEFAULT_K8S_EXECUTOR_NAME
    host = str(spec.get("host") or "").strip()
    if host:
        return host
    return None


def default_remote_source_dir(spec: Mapping[str, Any]) -> str:
    explicit = str(spec.get("remote_source_dir") or "").strip()
    if explicit:
        return explicit
    source_dir = str(spec.get("source_dir") or "").strip()
    if source_dir and Path(source_dir).name == "chronohorn":
        return "/data/chronohorn"
    return "/data"


def default_runtime_namespace(spec: Mapping[str, Any]) -> str:
    return str(
        spec.get("runtime_namespace")
        or spec.get("k8s_namespace")
        or DEFAULT_K8S_NAMESPACE
    ).strip() or DEFAULT_K8S_NAMESPACE


def default_runtime_job_name(spec: Mapping[str, Any]) -> str:
    explicit = str(spec.get("runtime_job_name") or "").strip()
    if explicit:
        return explicit
    return k8s_job_name(str(spec.get("name") or "job"))


def gateway_host(spec: Mapping[str, Any]) -> str:
    return str(spec.get("cluster_gateway_host") or DEFAULT_K8S_GATEWAY_HOST)


def remote_run_path(spec: Mapping[str, Any]) -> str:
    explicit = str(spec.get("remote_run") or "").strip()
    if explicit:
        return explicit
    return f"{DEFAULT_REMOTE_RUN_ROOT}/{spec['name']}"


def _run_kubectl(
    args: Sequence[str],
    *,
    gateway: str,
    input_text: str | None = None,
    timeout: float = 60.0,
) -> subprocess.CompletedProcess[str]:
    remote_command = shell_join(["sudo", "-n", "k3s", "kubectl", *[str(arg) for arg in args]])
    return subprocess.run(
        ssh_argv(gateway, remote_command),
        input=input_text,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=True,
    )


def _capture_kubectl_json(
    args: Sequence[str],
    *,
    gateway: str,
    timeout: float = 60.0,
) -> dict[str, Any]:
    completed = _run_kubectl(args, gateway=gateway, timeout=timeout)
    payload = json.loads(completed.stdout or "{}")
    return payload if isinstance(payload, dict) else {}


def _ensure_namespace(namespace: str, *, gateway: str) -> None:
    remote_command = (
        f"sudo -n k3s kubectl get namespace {shlex.quote(namespace)} >/dev/null 2>&1 "
        f"|| sudo -n k3s kubectl create namespace {shlex.quote(namespace)} >/dev/null"
    )
    subprocess.run(
        ssh_argv(gateway, remote_command),
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    )


def _mounted_workdir(spec: Mapping[str, Any]) -> str:
    source_root = PurePosixPath(default_remote_source_dir(spec))
    remote_cwd_rel = str(spec.get("remote_cwd_rel") or ".").strip() or "."
    if remote_cwd_rel in {".", "./"}:
        return source_root.as_posix()
    return str(source_root.joinpath(remote_cwd_rel))


def _remote_env(spec: Mapping[str, Any]) -> dict[str, str]:
    env_map = {
        "CHRONOHORN_REMOTE_ASSETS": "/run/assets",
        "CHRONOHORN_REMOTE_RUN": "/run",
    }
    if spec.get("threads") is not None:
        env_map["CHRONOHORN_THREADS"] = str(spec["threads"])
    raw_env = spec.get("env")
    if isinstance(raw_env, Mapping):
        for key, value in raw_env.items():
            env_map[str(key)] = str(value)
    return env_map


def _label_value(value: str) -> str:
    text = _K8S_NAME_RE.sub("-", str(value).strip().lower())
    text = re.sub(r"-{2,}", "-", text).strip("-")
    if not text:
        return "unknown"
    return text[:63].rstrip("-") or "unknown"


def build_job_manifest(spec: Mapping[str, Any]) -> dict[str, Any]:
    name = str(spec.get("name") or "")
    image = str(spec.get("image") or "")
    command = str(spec.get("command") or "")
    if not name:
        raise ValueError("k8s_job requires a non-empty name")
    if not image:
        raise ValueError(f"{name}: k8s_job requires an image")
    if not command.strip():
        raise ValueError(f"{name}: k8s_job requires a non-empty command")
    if spec.get("sync_paths"):
        raise ValueError(f"{name}: k8s_job does not support sync_paths; stage assets in cluster storage instead")

    namespace = default_runtime_namespace(spec)
    runtime_job = default_runtime_job_name(spec)
    remote_run = remote_run_path(spec)
    remote_env = _remote_env(spec)
    mounted_workdir = _mounted_workdir(spec)
    gpu = bool(spec.get("gpu")) or str(spec.get("resource_class") or "") == "cuda_gpu"
    host = str(spec.get("host") or "").strip()
    node_selector = {"kubernetes.io/hostname": host} if host and host not in {"auto", "local"} else None
    env_list = [{"name": key, "value": value} for key, value in sorted(remote_env.items())]

    shell_lines = [
        "set -euo pipefail",
        "mkdir -p /run/results /run/assets /cache",
        "exec > >(tee -a /run/job.log) 2>&1",
    ]
    shell_lines.extend(
        f"export {key}={shlex.quote(value)}"
        for key, value in sorted(remote_env.items())
    )
    shell_lines.extend(
        [
            f"cd {shlex.quote(mounted_workdir)}",
            command,
        ]
    )
    shell_command = "\n".join(shell_lines)

    resources: dict[str, Any] = {}
    if gpu:
        resources = {
            "limits": {"nvidia.com/gpu": "1"},
            "requests": {"nvidia.com/gpu": "1"},
        }

    manifest = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": runtime_job,
            "namespace": namespace,
            "labels": {
                "app.kubernetes.io/name": "chronohorn",
                "app.kubernetes.io/managed-by": "chronohorn",
                "chronohorn-name": _label_value(name),
            },
            "annotations": {
                "chronohorn/original-name": name,
                "chronohorn/executor-kind": "k8s_cluster",
            },
        },
        "spec": {
            "backoffLimit": 0,
            "template": {
                "metadata": {
                    "labels": {
                        "app.kubernetes.io/name": "chronohorn",
                        "app.kubernetes.io/managed-by": "chronohorn",
                        "chronohorn-name": _label_value(name),
                    }
                },
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [
                        {
                            "name": "runner",
                            "image": image,
                            "imagePullPolicy": "IfNotPresent",
                            "command": ["bash", "-lc", shell_command],
                            "workingDir": mounted_workdir,
                            "env": env_list,
                            "resources": resources,
                            "volumeMounts": [
                                {"name": "data", "mountPath": "/data"},
                                {"name": "cache", "mountPath": "/cache"},
                                {"name": "run", "mountPath": "/run"},
                            ],
                        }
                    ],
                    "volumes": [
                        {"name": "data", "hostPath": {"path": "/data", "type": "Directory"}},
                        {
                            "name": "cache",
                            "hostPath": {
                                "path": DEFAULT_REMOTE_CACHE_DIR,
                                "type": "DirectoryOrCreate",
                            },
                        },
                        {
                            "name": "run",
                            "hostPath": {
                                "path": remote_run,
                                "type": "DirectoryOrCreate",
                            },
                        },
                    ],
                },
            },
        },
    }
    if node_selector:
        manifest["spec"]["template"]["spec"]["nodeSelector"] = node_selector
    return manifest


def submit_k8s_job(spec: Mapping[str, Any]) -> dict[str, Any]:
    gateway = gateway_host(spec)
    namespace = default_runtime_namespace(spec)
    runtime_job = default_runtime_job_name(spec)
    manifest = build_job_manifest(spec)
    _ensure_namespace(namespace, gateway=gateway)
    _run_kubectl(["apply", "-f", "-"], gateway=gateway, input_text=json.dumps(manifest), timeout=90.0)
    host = str(spec.get("host") or "").strip()
    return {
        "name": str(spec["name"]),
        "run_id": spec.get("run_id"),
        "manifest_path": spec.get("manifest_path"),
        "family": spec.get("family"),
        "backend": spec.get("backend"),
        "resource_class": spec.get("resource_class"),
        "goal": spec.get("goal"),
        "planner": spec.get("planner"),
        "requested_launcher": spec.get("requested_launcher", spec.get("launcher")),
        "launcher": "k8s_job",
        "executor_kind": "k8s_cluster",
        "executor_name": str(spec.get("executor_name") or DEFAULT_K8S_EXECUTOR_NAME),
        "cluster_gateway_host": gateway,
        "host": host,
        "image": str(spec.get("image") or ""),
        "gpu": bool(spec.get("gpu", False)),
        "remote_run": remote_run_path(spec),
        "remote_source_dir": default_remote_source_dir(spec),
        "remote_cwd_rel": str(spec.get("remote_cwd_rel") or "."),
        "command": str(spec.get("command") or ""),
        "runtime_namespace": namespace,
        "runtime_job_name": runtime_job,
        "runtime_pod_name": "",
        "runtime_node_name": host,
        "launched_at_unix": time.time(),
    }


def _job_identity(spec: Mapping[str, Any]) -> tuple[str, str, str, str]:
    namespace = default_runtime_namespace(spec)
    runtime_job = default_runtime_job_name(spec)
    return (
        gateway_host(spec),
        namespace,
        runtime_job,
        str(spec.get("name") or ""),
    )


def _choose_latest_pod(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not items:
        return None
    items = sorted(
        items,
        key=lambda item: str(item.get("metadata", {}).get("creationTimestamp") or ""),
    )
    return items[-1]


def _last_nonempty_line(text: str) -> str:
    for line in reversed(text.splitlines()):
        if line.strip():
            return line
    return ""


def get_k8s_logs(
    *,
    gateway: str,
    namespace: str,
    runtime_job_name: str,
    runtime_pod_name: str | None = None,
    tail_lines: int = 64,
) -> str:
    args: list[str] = ["logs", "-n", namespace, "--tail", str(max(1, int(tail_lines)))]
    if runtime_pod_name:
        args.append(runtime_pod_name)
    else:
        args.append(f"job/{runtime_job_name}")
    try:
        completed = _run_kubectl(args, gateway=gateway, timeout=45.0)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        if "not found" in stderr.lower():
            return ""
        raise
    return completed.stdout or ""


def get_k8s_job_logs(
    *,
    gateway: str,
    namespace: str,
    runtime_job_name: str,
    runtime_pod_name: str | None = None,
    tail_lines: int = 64,
) -> str:
    return get_k8s_logs(
        gateway=gateway,
        namespace=namespace,
        runtime_job_name=runtime_job_name,
        runtime_pod_name=runtime_pod_name,
        tail_lines=tail_lines,
    )


def query_k8s_run_states(
    jobs: Sequence[Mapping[str, Any]],
    *,
    include_logs: bool = False,
    tail_lines: int = 64,
) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[tuple[str, str]]] = {}
    name_map: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    for job in jobs:
        if infer_executor_kind(job) != "k8s_cluster":
            continue
        gateway, namespace, runtime_job, name = _job_identity(job)
        grouped.setdefault((gateway, namespace), []).append((runtime_job, name))
        name_map[(gateway, namespace, runtime_job, name)] = job

    states: dict[tuple[str, str], dict[str, Any]] = {}
    for (gateway, namespace), names in grouped.items():
        try:
            payload = _capture_kubectl_json(
                [
                    "get",
                    "jobs.batch,pods",
                    "-n",
                    namespace,
                    "-l",
                    "app.kubernetes.io/name=chronohorn",
                    "-o",
                    "json",
                ],
                gateway=gateway,
                timeout=45.0,
            )
        except subprocess.CalledProcessError:
            continue
        items = payload.get("items")
        if not isinstance(items, list):
            continue
        jobs_by_name: dict[str, dict[str, Any]] = {}
        pods_by_job: dict[str, list[dict[str, Any]]] = {}
        for item in items:
            if not isinstance(item, Mapping):
                continue
            kind = str(item.get("kind") or "")
            metadata = item.get("metadata", {})
            runtime_job = str(metadata.get("name") or "")
            if not runtime_job:
                continue
            if kind == "Job":
                jobs_by_name[runtime_job] = dict(item)
                continue
            if kind == "Pod":
                labels = metadata.get("labels", {})
                owner_job = str(labels.get("job-name") or "")
                if owner_job:
                    pods_by_job.setdefault(owner_job, []).append(dict(item))

        for runtime_job, name in names:
            job_item = jobs_by_name.get(runtime_job)
            latest_pod = _choose_latest_pod(pods_by_job.get(runtime_job, []))
            pod_metadata = latest_pod.get("metadata", {}) if latest_pod else {}
            pod_spec = latest_pod.get("spec", {}) if latest_pod else {}
            pod_status = latest_pod.get("status", {}) if latest_pod else {}
            job_status = job_item.get("status", {}) if job_item else {}
            job_conditions = job_status.get("conditions") or []
            pod_conditions = pod_status.get("conditions") or []
            active = int(job_status.get("active") or 0)
            succeeded = int(job_status.get("succeeded") or 0)
            failed = int(job_status.get("failed") or 0)
            pod_phase = str(pod_status.get("phase") or "")
            if succeeded > 0:
                phase = "succeeded"
            elif failed > 0:
                phase = "failed"
            elif pod_phase == "Pending":
                phase = "pending"
            elif active > 0 or pod_phase == "Running":
                phase = "running"
            elif job_item is None:
                phase = "missing"
            else:
                phase = "pending"
            runtime_pod_name = str(pod_metadata.get("name") or "")
            runtime_node_name = str(pod_spec.get("nodeName") or "")
            reason = str(pod_status.get("reason") or "")
            message = str(pod_status.get("message") or "")
            if not reason:
                for condition in reversed(pod_conditions):
                    if not isinstance(condition, Mapping):
                        continue
                    if str(condition.get("status") or "") == "False":
                        reason = str(condition.get("reason") or "")
                        message = str(condition.get("message") or "")
                        break
            if not reason:
                for condition in reversed(job_conditions):
                    if not isinstance(condition, Mapping):
                        continue
                    reason = str(condition.get("reason") or "")
                    message = str(condition.get("message") or "")
                    if reason or message:
                        break
            log_tail_text = ""
            if include_logs and phase in {"running", "failed", "succeeded"}:
                try:
                    log_tail_text = get_k8s_logs(
                        gateway=gateway,
                        namespace=namespace,
                        runtime_job_name=runtime_job,
                        runtime_pod_name=runtime_pod_name or None,
                        tail_lines=tail_lines,
                    )
                except subprocess.CalledProcessError:
                    log_tail_text = ""
            states[(namespace, name)] = {
                "name": name,
                "executor_kind": "k8s_cluster",
                "executor_name": str(
                    name_map[(gateway, namespace, runtime_job, name)].get("executor_name")
                    or DEFAULT_K8S_EXECUTOR_NAME
                ),
                "cluster_gateway_host": gateway,
                "runtime_namespace": namespace,
                "runtime_job_name": runtime_job,
                "runtime_pod_name": runtime_pod_name,
                "runtime_node_name": runtime_node_name,
                "phase": phase,
                "running": phase == "running",
                "active": active,
                "succeeded": succeeded,
                "failed": failed,
                "pod_phase": pod_phase,
                "reason": reason,
                "message": message,
                "start_time": job_status.get("startTime") or pod_status.get("startTime"),
                "completion_time": job_status.get("completionTime"),
                "log_tail_text": log_tail_text,
                "log_last_line": _last_nonempty_line(log_tail_text),
            }
    return states


def get_k8s_job_status(spec: Mapping[str, Any], *, include_logs: bool = False, tail_lines: int = 64) -> dict[str, Any]:
    states = query_k8s_run_states([spec], include_logs=include_logs, tail_lines=tail_lines)
    namespace = default_runtime_namespace(spec)
    name = str(spec.get("name") or "")
    state = states.get((namespace, name))
    if state is None:
        return {
            "name": name,
            "executor_kind": "k8s_cluster",
            "executor_name": str(spec.get("executor_name") or DEFAULT_K8S_EXECUTOR_NAME),
            "cluster_gateway_host": gateway_host(spec),
            "runtime_namespace": namespace,
            "runtime_job_name": default_runtime_job_name(spec),
            "phase": "missing",
        }
    return state


def delete_k8s_job(spec: Mapping[str, Any]) -> dict[str, Any]:
    gateway = gateway_host(spec)
    namespace = default_runtime_namespace(spec)
    runtime_job = default_runtime_job_name(spec)
    try:
        _run_kubectl(
            [
                "delete",
                "job",
                runtime_job,
                "-n",
                namespace,
                "--ignore-not-found=true",
                "--wait=false",
            ],
            gateway=gateway,
            timeout=45.0,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(stderr or f"failed to delete k8s job {runtime_job}") from exc
    return {
        "name": str(spec.get("name") or ""),
        "executor_kind": "k8s_cluster",
        "executor_name": str(spec.get("executor_name") or DEFAULT_K8S_EXECUTOR_NAME),
        "cluster_gateway_host": gateway,
        "runtime_namespace": namespace,
        "runtime_job_name": runtime_job,
        "status": "delete_requested",
    }
