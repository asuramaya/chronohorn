"""Pre-launch validation for fleet jobs.

Catches configuration errors and missing resources *before* a job is
submitted to k8s or Docker, so we fail fast with a clear message instead
of burning a GPU slot on an instant crash.
"""

from __future__ import annotations

import re
import shlex
import subprocess
from collections.abc import Mapping
from typing import Any

from .k8s import DEFAULT_SSH_ARGS, infer_executor_kind

# ---------------------------------------------------------------------------
# Safety gates: (flag_value, override_flag) pairs.
#
# If the command contains ``--<flag_name> <flag_value>`` but does NOT contain
# ``--<override_flag>``, the job is rejected before launch.
# ---------------------------------------------------------------------------

_COMMAND_SAFETY_GATES: list[tuple[str, str, str]] = [
    # flag_name, gated_value, required_override
    ("linear-readout-kind", "tied_recursive", "allow-experimental-recursive-readout"),
]


class PreflightError(RuntimeError):
    """Raised when a pre-launch check fails."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FLAG_VALUE_RE: dict[str, re.Pattern[str]] = {}


def _flag_value_pattern(flag_name: str) -> re.Pattern[str]:
    """Compile and cache a regex that extracts ``--flag-name <value>``."""
    if flag_name not in _FLAG_VALUE_RE:
        escaped = re.escape(flag_name)
        _FLAG_VALUE_RE[flag_name] = re.compile(
            rf"--{escaped}\s+(\S+)"
        )
    return _FLAG_VALUE_RE[flag_name]


def _extract_flag_value(command: str, flag_name: str) -> str | None:
    """Extract the value of ``--flag-name`` from a shell command string."""
    match = _flag_value_pattern(flag_name).search(command)
    return match.group(1) if match else None


def _has_flag(command: str, flag_name: str) -> bool:
    """Check if ``--flag-name`` appears anywhere in the command."""
    return f"--{flag_name}" in command


# ---------------------------------------------------------------------------
# Check: command-level safety gates
# ---------------------------------------------------------------------------

def _check_command_safety_gates(job: Mapping[str, Any]) -> list[str]:
    """Validate the command string against known safety gates.

    Returns a list of error messages (empty = all clear).
    """
    command = str(job.get("command") or "")
    if not command:
        return []
    errors: list[str] = []
    for flag_name, gated_value, override_flag in _COMMAND_SAFETY_GATES:
        actual = _extract_flag_value(command, flag_name)
        if actual == gated_value and not _has_flag(command, override_flag):
            errors.append(
                f"--{flag_name} {gated_value} requires --{override_flag} "
                f"(add it to the manifest command or config)"
            )
    return errors


# ---------------------------------------------------------------------------
# Check: remote data path exists on target node
# ---------------------------------------------------------------------------

def _check_remote_data_path(job: Mapping[str, Any]) -> list[str]:
    """SSH to the target host and verify --data-root exists.

    Only runs for remote jobs (k8s, docker). Returns a list of error
    messages (empty = all clear).
    """
    command = str(job.get("command") or "")
    host = str(job.get("host") or "").strip()
    if not host or host in {"local", "auto", ""}:
        return []

    data_root = _extract_flag_value(command, "data-root")
    if not data_root:
        return []

    # Probe the target node directly via SSH
    check_cmd = f"test -d {shlex.quote(data_root)}"
    try:
        result = subprocess.run(
            ["ssh", *DEFAULT_SSH_ARGS, host, check_cmd],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return [f"cannot verify data path on {host}: {exc}"]

    if result.returncode != 0:
        return [
            f"data-root {data_root} does not exist on {host} — "
            f"check volume mounts or pick a different host"
        ]
    return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preflight_check(job: Mapping[str, Any]) -> None:
    """Run all pre-launch checks on a job spec.

    Raises ``PreflightError`` with a combined message if any check fails.
    Prints warnings to stderr for non-fatal issues.
    """
    name = str(job.get("name", "<unnamed>"))
    errors: list[str] = []

    errors.extend(_check_command_safety_gates(job))

    executor = infer_executor_kind(job)
    if executor in {"k8s_cluster", "docker_host"}:
        errors.extend(_check_remote_data_path(job))

    if errors:
        detail = "; ".join(errors)
        raise PreflightError(f"{name}: preflight failed — {detail}")
