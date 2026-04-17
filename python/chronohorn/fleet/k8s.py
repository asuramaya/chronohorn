from __future__ import annotations

import hashlib
import json
import re
import shlex
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .validation import (
    validate_env_key,
    validate_job_name,
    validate_posix_path_within_root,
    validate_relative_posix_subpath,
)

DEFAULT_SSH_ARGS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=5")
DEFAULT_K8S_GATEWAY_HOST = "slop-01"
DEFAULT_K8S_EXECUTOR_NAME = "slop-cluster"
DEFAULT_K8S_NAMESPACE = "default"
DEFAULT_REMOTE_RUN_ROOT = "/tmp/chronohorn-runs"
DEFAULT_REMOTE_CACHE_DIR = "/tmp/chronohorn-cache"
_DEFAULT_TOLERATED_TAINTS = {
    ("chronohorn-gpu", "reserved", "NoSchedule"),
}

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
        return validate_posix_path_within_root(
            explicit,
            root=DEFAULT_REMOTE_RUN_ROOT,
            field_name="remote_run",
        )
    return validate_posix_path_within_root(
        validate_job_name(str(spec["name"])),
        root=DEFAULT_REMOTE_RUN_ROOT,
        field_name="remote_run",
    )


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
    remote_cwd_rel = validate_relative_posix_subpath(
        str(spec.get("remote_cwd_rel") or "."),
        field_name="remote_cwd_rel",
    )
    if remote_cwd_rel == ".":
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
            env_map[validate_env_key(str(key))] = str(value)
    return env_map


def _label_value(value: str) -> str:
    text = _K8S_NAME_RE.sub("-", str(value).strip().lower())
    text = re.sub(r"-{2,}", "-", text).strip("-")
    if not text:
        return "unknown"
    return text[:63].rstrip("-") or "unknown"


def _resource_quantity_int(raw: Any) -> int:
    text = str(raw or "").strip()
    if not text:
        return 0
    match = re.match(r"^([0-9]+)", text)
    if match is None:
        return 0
    return int(match.group(1))


def _condition_map(payload: Mapping[str, Any]) -> dict[str, str]:
    rows = payload.get("status", {}).get("conditions") or []
    result: dict[str, str] = {}
    if not isinstance(rows, list):
        return result
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        name = str(row.get("type") or "").strip()
        if not name:
            continue
        result[name] = str(row.get("status") or "").strip()
    return result


def _blocking_taints(payload: Mapping[str, Any]) -> list[str]:
    taints = payload.get("spec", {}).get("taints") or []
    blockers: list[str] = []
    if not isinstance(taints, list):
        return blockers
    for raw_taint in taints:
        if not isinstance(raw_taint, Mapping):
            continue
        key = str(raw_taint.get("key") or "").strip()
        value = str(raw_taint.get("value") or "").strip()
        effect = str(raw_taint.get("effect") or "").strip()
        if effect != "NoSchedule":
            continue
        if (key, value, effect) in _DEFAULT_TOLERATED_TAINTS:
            continue
        blockers.append(f"{key}={value}:{effect}" if value else f"{key}:{effect}")
    return blockers


def probe_k8s_gpu_pods(*, gateway: str | None = None, timeout: float = 15.0) -> dict[str, int]:
    """Count GPU-requesting pods (Pending + Running) per node.

    Returns {node_name: count}. A count > 0 means the GPU is claimed
    even if nvidia-smi shows 0% utilization (pod still starting).
    """
    gateway_host_name = gateway or DEFAULT_K8S_GATEWAY_HOST
    try:
        payload = _capture_kubectl_json(
            ["get", "pods", "--all-namespaces", "-o", "json"],
            gateway=gateway_host_name,
            timeout=timeout,
        )
    except Exception:
        return {}
    counts: dict[str, int] = {}
    for item in payload.get("items", []):
        phase = (item.get("status") or {}).get("phase", "")
        if phase not in ("Pending", "Running"):
            continue
        # Only count chronohorn pods — other workloads on the cluster
        # may be Pending for GPUs they'll never get.
        pod_name = (item.get("metadata") or {}).get("name", "")
        if not pod_name.startswith("ch-"):
            continue
        node = (item.get("spec") or {}).get("nodeName") or ""
        has_gpu = False
        for container in (item.get("spec") or {}).get("containers", []):
            requests = (container.get("resources") or {}).get("requests") or {}
            limits = (container.get("resources") or {}).get("limits") or {}
            if requests.get("nvidia.com/gpu") or limits.get("nvidia.com/gpu"):
                has_gpu = True
                break
        if not has_gpu:
            continue
        # Resolve the target node: actual node if assigned, nodeSelector if pending
        if not node:
            selector = (item.get("spec") or {}).get("nodeSelector") or {}
            node = selector.get("kubernetes.io/hostname", "")
        if node:
            counts[node] = counts.get(node, 0) + 1
    return counts


def probe_k8s_node(host: str, *, gateway: str | None = None, timeout: float = 20.0) -> dict[str, Any]:
    gateway_host_name = gateway or DEFAULT_K8S_GATEWAY_HOST
    try:
        payload = _capture_kubectl_json(
            ["get", "node", host, "-o", "json"],
            gateway=gateway_host_name,
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "host": host,
            "schedulable": None,
            "probe_error": str(exc),
        }

    conditions = _condition_map(payload)
    ready = conditions.get("Ready") == "True"
    unschedulable = bool(payload.get("spec", {}).get("unschedulable", False))
    taint_blockers = _blocking_taints(payload)
    allocatable = payload.get("status", {}).get("allocatable") or {}
    capacity = payload.get("status", {}).get("capacity") or {}
    return {
        "host": host,
        "ready": ready,
        "unschedulable": unschedulable,
        "schedulable": ready and not unschedulable and not taint_blockers,
        "allocatable_gpus": _resource_quantity_int(allocatable.get("nvidia.com/gpu")),
        "capacity_gpus": _resource_quantity_int(capacity.get("nvidia.com/gpu")),
        "taint_blockers": taint_blockers,
    }


_DATA_PROVISION_SCRIPT = r'''
import os, shutil, sys
from pathlib import Path

REPO = "willdepueoai/parameter-golf"
SHARD_SUB = "datasets/datasets/fineweb10B_sp1024"
TOK_SUB = "datasets/tokenizers"
DATA_ROOT = Path(os.environ.get("CHRONOHORN_DATA_ROOT", "/data/chronohorn/fineweb10B_sp1024"))
TOK_DIR = Path(os.environ.get("CHRONOHORN_TOKENIZER_DIR", "/data/chronohorn/tokenizers"))
TRAIN_SHARDS = int(os.environ.get("CHRONOHORN_TRAIN_SHARDS", "80"))

def need(d, names):
    return [n for n in names if not (d / n).exists() or (d / n).stat().st_size == 0]

shards = [f"fineweb_train_{i:06d}.bin" for i in range(TRAIN_SHARDS)] + ["fineweb_val_000000.bin"]
toks = ["fineweb_1024_bpe.model", "fineweb_1024_bpe.vocab"]
missing_s = need(DATA_ROOT, shards)
missing_t = need(TOK_DIR, toks)

if not missing_s and not missing_t:
    print(f"data-provision: {len(shards)} shards + tokenizer present", flush=True)
    sys.exit(0)

print(f"data-provision: downloading {len(missing_s)} shard(s) + {len(missing_t)} tokenizer file(s)", flush=True)
from huggingface_hub import hf_hub_download
DATA_ROOT.mkdir(parents=True, exist_ok=True)
TOK_DIR.mkdir(parents=True, exist_ok=True)
for name in missing_s:
    print(f"  {name}", flush=True)
    c = Path(hf_hub_download(REPO, name, subfolder=SHARD_SUB, repo_type="dataset")).resolve()
    shutil.copy2(c, DATA_ROOT / name)
for name in missing_t:
    print(f"  {name}", flush=True)
    c = Path(hf_hub_download(REPO, name, subfolder=TOK_SUB, repo_type="dataset")).resolve()
    shutil.copy2(c, TOK_DIR / name)
still = need(DATA_ROOT, shards)
if still:
    print(f"data-provision: FAIL — {len(still)} shard(s) still missing", file=sys.stderr, flush=True)
    sys.exit(1)
print("data-provision: ok", flush=True)
'''


# Per-host memory limits (Gi). Leave ~8GB headroom per node for k3s, containerd,
# kernel page cache of mmap'd shards, and system services. Measured: idle k3s-agent
# uses ~2GB, shard page cache ~4-12GB for byte data. 56Gi leaves enough room on
# the 64GB hosts for the big byte-seqlen4096 scan-at-seq=4096 workload (empirically
# needs ~46GB RSS under compile). slop-home at 24Gi remains fleet-safe for 32GB.
_HOST_MEMORY_LIMIT_GI = {
    "slop-01": 56,
    "slop-02": 56,
    "slop-home": 24,
}
_DEFAULT_MEMORY_LIMIT_GI = 24  # fleet-safe fallback for unknown hosts


def _build_data_init_container(
    image: str,
    data_root: str | None,
) -> dict[str, Any] | None:
    """Build a k8s init container that provisions training data from HuggingFace.

    Returns None if no data_root is detected (non-training jobs).
    """
    if not data_root:
        return None
    env = [{"name": "CHRONOHORN_DATA_ROOT", "value": data_root}]
    install_and_run = (
        "pip install -q huggingface_hub 2>/dev/null; "
        "python -u -c " + shlex.quote(_DATA_PROVISION_SCRIPT)
    )
    return {
        "name": "data-provision",
        "image": image,
        "imagePullPolicy": "IfNotPresent",
        "command": ["bash", "-c", install_and_run],
        "env": env,
        "volumeMounts": [
            {"name": "data", "mountPath": "/data"},
            {"name": "cache", "mountPath": "/cache"},
        ],
    }


def _extract_data_root(command: str) -> str | None:
    """Extract --data-root value from a shell command string."""
    match = re.search(r"--data-root\s+(\S+)", command)
    return match.group(1) if match else None


def build_job_manifest(spec: Mapping[str, Any]) -> dict[str, Any]:
    name = validate_job_name(str(spec.get("name") or ""))
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
        [
            f"cd {shlex.quote(mounted_workdir)}",
            command,
            # Persist ALL results to durable host storage before pod exits.
            # This survives k8s TTL cleanup (ttlSecondsAfterFinished).
            # The drain loop pulls from here, not /tmp/chronohorn-runs/.
            "mkdir -p /data/chronohorn/checkpoints",
            "cp -f /run/results/*.checkpoint.pt /data/chronohorn/checkpoints/ 2>/dev/null || true",
            "cp -f /run/results/*.training_state.pt /data/chronohorn/checkpoints/ 2>/dev/null || true",
            "cp -f /run/results/*.json /data/chronohorn/checkpoints/ 2>/dev/null || true",
            "cp -f /run/results/*.probes.jsonl /data/chronohorn/checkpoints/ 2>/dev/null || true",
        ]
    )
    shell_command = "\n".join(shell_lines)

    # Host-level memory ceiling: container OOMs cleanly instead of triggering
    # SystemOOM on the node (which takes the kubelet down with it and cascades
    # NotReady across the fleet). Sized per-host so A4000 nodes (64GB) can use
    # their headroom while the Quadro host (32GB) stays safe. Unknown hosts
    # fall back to the smallest limit so a mis-scheduled Job never crashes a
    # smaller node.
    memory_limit_gi = _HOST_MEMORY_LIMIT_GI.get(host, _DEFAULT_MEMORY_LIMIT_GI)
    resources: dict[str, Any] = {
        "requests": {"memory": "4Gi"},
        "limits": {"memory": f"{memory_limit_gi}Gi"},
    }
    if gpu:
        resources["limits"]["nvidia.com/gpu"] = "1"
        resources["requests"]["nvidia.com/gpu"] = "1"

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
            "ttlSecondsAfterFinished": 3600,
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
                    "tolerations": [
                        {
                            "key": "chronohorn-gpu",
                            "operator": "Equal",
                            "value": "reserved",
                            "effect": "NoSchedule",
                        },
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
    data_root = _extract_data_root(command)
    init_container = _build_data_init_container(image, data_root)
    if init_container is not None:
        manifest["spec"]["template"]["spec"]["initContainers"] = [init_container]
    if node_selector:
        manifest["spec"]["template"]["spec"]["nodeSelector"] = node_selector
    if gpu:
        manifest["spec"]["template"]["spec"]["runtimeClassName"] = "nvidia"
    return manifest


def _sync_source_to_host(spec: Mapping[str, Any], host: str) -> None:
    """Rsync snapshot_paths from local source_dir to the remote host before k8s job launch."""
    source_dir = str(spec.get("source_dir") or "").strip()
    snapshot_paths = spec.get("snapshot_paths")
    remote_source = default_remote_source_dir(spec)
    if not source_dir or not snapshot_paths or not host:
        return
    rsync_argv = [
        "rsync", "-rlz",
        "--exclude", "__pycache__", "--exclude", "*.pyc",
        "--exclude", "*.egg-info", "--exclude", "*.dist-info",
        "--relative",
    ]
    for rel in snapshot_paths:
        rsync_argv.append(f"./{rel}")
    rsync_argv.append(f"{host}:{remote_source}/")
    try:
        subprocess.run(rsync_argv, cwd=source_dir, capture_output=True, text=True, timeout=120, check=True)
    except Exception as exc:
        import sys
        print(f"chronohorn k8s: source sync to {host} failed: {exc}", file=sys.stderr)


def submit_k8s_job(spec: Mapping[str, Any]) -> dict[str, Any]:
    gateway = gateway_host(spec)
    namespace = default_runtime_namespace(spec)
    runtime_job = default_runtime_job_name(spec)
    manifest = build_job_manifest(spec)
    _ensure_namespace(namespace, gateway=gateway)
    # Sync source code to target host before launching
    host = str(spec.get("host") or "").strip()
    if host:
        _sync_source_to_host(spec, host)
    # Delete any existing job with the same name synchronously before creating.
    # k8s job names are deterministic, so re-dispatching after a stop_run reuses
    # the same name.  An async delete (--wait=false) could race with the apply
    # and kill the new job, so we wait for the old one to be gone.
    try:
        _run_kubectl(
            ["delete", "job", runtime_job, "-n", namespace,
             "--ignore-not-found=true", "--wait=true"],
            gateway=gateway, timeout=60.0,
        )
    except subprocess.CalledProcessError:
        pass  # best-effort — if the old job is already gone, that's fine
    _run_kubectl(["apply", "-f", "-"], gateway=gateway, input_text=json.dumps(manifest), timeout=90.0)
    # Build the record immediately after apply — if this fails, clean up the orphan
    host = str(spec.get("host") or "").strip()
    try:
        record = _build_submit_record(spec, gateway, namespace, runtime_job, host)
    except Exception:
        # Rollback: delete the k8s job we just applied to prevent orphans
        try:
            _run_kubectl(
                ["delete", "job", runtime_job, "-n", namespace, "--wait=false"],
                gateway=gateway, timeout=30.0,
            )
        except Exception as cleanup_exc:
            import sys
            print(f"chronohorn k8s: orphan cleanup also failed: {cleanup_exc}", file=sys.stderr)
        raise
    return record


def _build_submit_record(
    spec: Mapping[str, Any], gateway: str, namespace: str,
    runtime_job: str, host: str,
) -> dict[str, Any]:
    return {
        "name": validate_job_name(str(spec["name"])),
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
        "remote_cwd_rel": validate_relative_posix_subpath(
            str(spec.get("remote_cwd_rel") or "."),
            field_name="remote_cwd_rel",
        ),
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
        validate_job_name(str(spec.get("name") or "")),
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
                "--wait=true",
            ],
            gateway=gateway,
            timeout=90.0,
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
