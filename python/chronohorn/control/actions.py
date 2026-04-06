from __future__ import annotations

import os
import shlex
import signal
import time
from collections.abc import Sequence
from typing import Any

from chronohorn.control.models import ControlAction
from chronohorn.fleet.dispatch import (
    launch_job,
    load_launch_record,
    remote_container_name,
    run_checked,
    ssh_argv,
    write_launch_record,
)
from chronohorn.fleet.k8s import delete_k8s_job, infer_executor_kind


def stop_job(action: ControlAction) -> dict[str, Any]:
    name = str(action.target_name or "")
    if not name:
        raise ValueError("stop action is missing target name")
    record = load_launch_record(name)
    runtime_state = action.metadata.get("runtime_state") if isinstance(action.metadata, dict) else None
    runtime_state = runtime_state if isinstance(runtime_state, dict) else {}
    if record is None:
        record = {
            "name": name,
            "host": action.host,
            "launcher": action.launcher,
            "executor_kind": action.metadata.get("executor_kind") if isinstance(action.metadata, dict) else None,
            "executor_name": action.metadata.get("executor_name") if isinstance(action.metadata, dict) else None,
            "cluster_gateway_host": action.metadata.get("cluster_gateway_host") if isinstance(action.metadata, dict) else None,
            "runtime_namespace": action.metadata.get("runtime_namespace") if isinstance(action.metadata, dict) else None,
            "runtime_job_name": action.metadata.get("runtime_job_name") if isinstance(action.metadata, dict) else None,
            "container_name": runtime_state.get("container_name"),
            "pid": runtime_state.get("pid"),
        }
    host = str(record.get("host") or "")
    launcher = str(record.get("launcher") or "")
    if host == "local":
        pid = record.get("pid")
        if not isinstance(pid, int):
            raise ValueError(f"{name}: local launch record is missing pid")
        try:
            os.kill(pid, signal.SIGTERM)
            status = "stop_requested"
        except ProcessLookupError:
            status = "already_stopped"
        record["last_control_action"] = {"action": "stop", "status": status, "at_unix": time.time()}
        write_launch_record(name, record)
        return {"name": name, "host": host, "launcher": launcher, "status": status, "pid": pid}
    executor_kind = infer_executor_kind(record) or infer_executor_kind(
        {
            "launcher": launcher,
            "host": host,
        }
    )
    if executor_kind == "k8s_cluster":
        result = delete_k8s_job(
            {
                "name": name,
                "executor_name": record.get("executor_name"),
                "cluster_gateway_host": record.get("cluster_gateway_host"),
                "runtime_namespace": record.get("runtime_namespace"),
                "runtime_job_name": record.get("runtime_job_name"),
            }
        )
        record["last_control_action"] = {
            "action": "stop",
            "status": "stop_requested",
            "at_unix": time.time(),
            "runtime_namespace": result.get("runtime_namespace"),
            "runtime_job_name": result.get("runtime_job_name"),
        }
        write_launch_record(name, record)
        return {
            "name": name,
            "host": host,
            "launcher": launcher,
            "executor_kind": "k8s_cluster",
            "status": "stop_requested",
            "runtime_namespace": result.get("runtime_namespace"),
            "runtime_job_name": result.get("runtime_job_name"),
        }
    if not host:
        raise ValueError(f"{name}: launch record is missing host")
    container_name = str(
        record.get("container_name")
        or runtime_state.get("container_name")
        or remote_container_name(name)
    )
    quoted_container = shlex.quote(container_name)
    remote_payload = f"sudo -n docker rm -f {quoted_container} >/dev/null 2>&1 || docker rm -f {quoted_container} >/dev/null 2>&1 || true"
    run_checked(ssh_argv(host, remote_payload))
    record["last_control_action"] = {
        "action": "stop",
        "status": "stop_requested",
        "at_unix": time.time(),
        "container_name": container_name,
    }
    write_launch_record(name, record)
    return {
        "name": name,
        "host": host,
        "launcher": launcher,
        "status": "stop_requested",
        "container_name": container_name,
    }


def execute_control_actions(
    actions: Sequence[ControlAction],
    *,
    allow_stop: bool = False,
    max_launches: int | None = None,
) -> list[dict[str, Any]]:
    executed: list[dict[str, Any]] = []
    launches = 0
    for action in actions:
        if action.action == "launch_job":
            if max_launches is not None and launches >= max(0, max_launches):
                executed.append(
                    {
                        "action": action.action,
                        "target_name": action.target_name,
                        "status": "skipped",
                        "reason": "launch limit reached",
                    }
                )
                continue
            assigned_job = action.metadata.get("assigned_job")
            if not isinstance(assigned_job, dict):
                executed.append(
                    {
                        "action": action.action,
                        "target_name": action.target_name,
                        "status": "skipped",
                        "reason": "missing assigned job payload",
                    }
                )
                continue
            record = launch_job(assigned_job)
            executed.append(
                {
                    "action": action.action,
                    "target_name": action.target_name,
                    "status": "launched",
                    "record": record,
                }
            )
            launches += 1
            continue
        if action.action == "stop_run":
            if not allow_stop:
                executed.append(
                    {
                        "action": action.action,
                        "target_name": action.target_name,
                        "status": "skipped",
                        "reason": "stop actions require --allow-stop",
                    }
                )
                continue
            if not action.target_name:
                executed.append(
                    {
                        "action": action.action,
                        "target_name": None,
                        "status": "skipped",
                        "reason": "missing target name",
                    }
                )
                continue
            record = stop_job(action)
            executed.append(
                {
                    "action": action.action,
                    "target_name": action.target_name,
                    "status": "stop_requested",
                    "record": record,
                }
            )
            continue
        executed.append(
            {
                "action": action.action,
                "target_name": action.target_name,
                "status": "skipped",
                "reason": "recommendation only",
            }
        )
    return executed
