"""Robust drain daemon — replaces ad-hoc bash loops.

Runs drain_tick() on a configurable interval with:
  - PID file at out/drain.pid
  - Rotating log at out/drain.log (3 x 10 MB)
  - Graceful SIGTERM/SIGINT handling (finish current tick, exit)
  - Auto-restart on unhandled exceptions with exponential backoff
  - Per-tick status reporting (pending/running/completed/blocked)
  - Optional stale container detection and kill
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import os
import signal
import sys
import time
from pathlib import Path
from typing import Sequence

from chronohorn.fleet.drain import drain_tick, DrainState


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEFAULT_OUT_DIR = Path("out")
PID_FILE = DEFAULT_OUT_DIR / "drain.pid"
LOG_FILE = DEFAULT_OUT_DIR / "drain.log"

# ---------------------------------------------------------------------------
# Backoff schedule (seconds) for auto-restart after unhandled exceptions
# ---------------------------------------------------------------------------

BACKOFF_SCHEDULE = [5, 10, 30, 60]


def _backoff_seconds(consecutive_failures: int) -> int:
    idx = min(consecutive_failures, len(BACKOFF_SCHEDULE) - 1)
    return BACKOFF_SCHEDULE[idx]


# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------

def _setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("chronohorn.drain_daemon")
    logger.setLevel(logging.DEBUG)
    # Clear existing handlers (relevant if re-entered after restart)
    logger.handlers.clear()

    # Rotating file handler: 10 MB per file, keep 3 backups
    fh = logging.handlers.RotatingFileHandler(
        str(log_path), maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)

    # Also log to stderr so interactive users can see what happens
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# PID file helpers
# ---------------------------------------------------------------------------

def _write_pid(pid_path: Path) -> None:
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()) + "\n", encoding="utf-8")


def _remove_pid(pid_path: Path) -> None:
    try:
        pid_path.unlink(missing_ok=True)
    except OSError:
        pass


def read_pid(pid_path: Path = PID_FILE) -> int | None:
    """Read PID from file. Returns None if missing or not a running process."""
    if not pid_path.exists():
        return None
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return None
    # Check if the process is actually alive
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        # Process exists but we can't signal it — still report it
        return pid
    return pid


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

class _ShutdownRequested(Exception):
    """Raised when SIGTERM or SIGINT is received."""


_shutdown_flag = False


def _signal_handler(signum: int, frame: object) -> None:
    global _shutdown_flag
    _shutdown_flag = True


# ---------------------------------------------------------------------------
# Stale container detection helpers
# ---------------------------------------------------------------------------

def _estimate_expected_duration(job: dict, telemetry: list) -> float | None:
    """Estimate expected wall-clock seconds for a job from telemetry.

    Uses tokens_per_second from telemetry samples matching the same backend
    family, combined with the job's train_tokens count.  Falls back to a
    conservative 2-hour default if we cannot estimate.
    """
    train_tokens = job.get("train_tokens")
    if not train_tokens:
        # Try to infer from steps (rough: 1 step ~ 2048 tokens)
        steps = job.get("steps") or job.get("train_steps")
        if steps:
            train_tokens = int(steps) * 2048
    if not train_tokens:
        return None

    # Gather tokens_per_second from telemetry for matching backend
    backend = (job.get("backend") or "").lower()
    speeds = []
    for sample in telemetry:
        if backend and hasattr(sample, "backend_family"):
            if sample.backend_family.lower() != backend:
                continue
        if hasattr(sample, "tokens_per_second") and sample.tokens_per_second > 0:
            speeds.append(sample.tokens_per_second)

    if not speeds:
        # Use all telemetry if backend filtering left nothing
        for sample in telemetry:
            if hasattr(sample, "tokens_per_second") and sample.tokens_per_second > 0:
                speeds.append(sample.tokens_per_second)

    if not speeds:
        return None

    median_speed = sorted(speeds)[len(speeds) // 2]
    return float(train_tokens) / median_speed


def detect_stale_containers(
    running_jobs: list[dict],
    telemetry: list,
    *,
    warn_multiplier: float = 2.0,
    kill_multiplier: float = 4.0,
    kill_stale: bool = False,
    logger: logging.Logger | None = None,
) -> list[dict]:
    """Detect running containers that exceed expected duration.

    Returns list of dicts with keys: name, host, elapsed, expected, action.
    """
    import time as _time

    from chronohorn.fleet.dispatch import (
        load_launch_record,
        remote_container_name,
        ssh_argv,
    )
    from chronohorn.fleet.k8s import delete_k8s_job, infer_executor_kind

    stale_reports: list[dict] = []
    now = _time.time()

    for job in running_jobs:
        name = job.get("name", "")
        record = load_launch_record(name)
        if not record:
            continue
        launched_at = record.get("launched_at_unix")
        if not launched_at:
            continue
        elapsed = now - launched_at

        expected = _estimate_expected_duration(job, telemetry)
        if expected is None:
            # Conservative fallback: 2 hours
            expected = 7200.0

        host = job.get("host") or record.get("host", "local")
        report = {
            "name": name,
            "host": host,
            "elapsed_sec": round(elapsed, 1),
            "expected_sec": round(expected, 1),
        }

        if elapsed > kill_multiplier * expected:
            report["action"] = "kill" if kill_stale else "warn_critical"
            if logger:
                logger.warning(
                    "STALE (%.1fx expected): %s on %s — elapsed %.0fs, expected %.0fs",
                    elapsed / expected, name, host, elapsed, expected,
                )
            if kill_stale and host != "local":
                try:
                    executor_kind = infer_executor_kind(record) or infer_executor_kind(job)
                    if executor_kind == "k8s_cluster":
                        delete_k8s_job(record)
                        if logger:
                            logger.warning(
                                "DELETED stale k8s job %s in %s",
                                record.get("runtime_job_name") or name,
                                record.get("runtime_namespace") or "chronohorn",
                            )
                    else:
                        container = remote_container_name(name)
                        import subprocess
                        subprocess.run(
                            ssh_argv(host, f"sudo -n docker rm -f {container} >/dev/null 2>&1 || true"),
                            capture_output=True,
                            timeout=15,
                        )
                        if logger:
                            logger.warning("KILLED stale container %s on %s", container, host)
                    report["killed"] = True
                except Exception as exc:
                    if logger:
                        logger.error("Failed to kill %s on %s: %s", container, host, exc)
                    report["killed"] = False
            stale_reports.append(report)

        elif elapsed > warn_multiplier * expected:
            report["action"] = "warn"
            if logger:
                logger.warning(
                    "SLOW (%.1fx expected): %s on %s — elapsed %.0fs, expected %.0fs",
                    elapsed / expected, name, host, elapsed, expected,
                )
            stale_reports.append(report)

    return stale_reports


# ---------------------------------------------------------------------------
# Core daemon loop
# ---------------------------------------------------------------------------

def run_daemon(
    manifest_path: str | Path,
    *,
    poll_interval: int = 30,
    job_names: Sequence[str] = (),
    classes: Sequence[str] = (),
    telemetry_globs: Sequence[str] | None = None,
    result_out_dir: Path | None = None,
    kill_stale: bool = False,
    pid_path: Path = PID_FILE,
    log_path: Path = LOG_FILE,
) -> int:
    """Run the drain daemon. Returns exit code (0 = clean, 1 = error)."""
    global _shutdown_flag
    _shutdown_flag = False

    logger = _setup_logger(log_path)

    # Check for already-running daemon
    existing_pid = read_pid(pid_path)
    if existing_pid is not None:
        logger.error("Daemon already running (pid %d). Stop it first.", existing_pid)
        return 1

    # Install signal handlers
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    _write_pid(pid_path)
    logger.info(
        "drain daemon started (pid=%d, manifest=%s, interval=%ds, kill_stale=%s)",
        os.getpid(), manifest_path, poll_interval, kill_stale,
    )

    consecutive_failures = 0
    tick = 0
    exit_code = 0

    try:
        while not _shutdown_flag:
            tick += 1
            try:
                state = drain_tick(
                    manifest_path,
                    job_names=job_names,
                    classes=classes,
                    telemetry_globs=telemetry_globs,
                    result_out_dir=result_out_dir,
                )
                consecutive_failures = 0  # reset on success

                logger.info(
                    "[tick %d] pending=%d running=%d completed=%d blocked=%d launched=%d pulled=%d",
                    tick, state.pending, state.running, state.completed,
                    state.blocked, state.launched, state.pulled,
                )

                # Stale container detection
                if state.running > 0:
                    try:
                        from chronohorn.fleet.dispatch import (
                            load_manifest,
                            select_jobs,
                            filter_jobs_by_class,
                            probe_fleet_state,
                            partition_running_jobs,
                        )
                        from chronohorn.fleet.telemetry import collect_performance_samples

                        jobs = load_manifest(Path(manifest_path))
                        if job_names:
                            jobs = select_jobs(jobs, list(job_names))
                        if classes:
                            jobs = filter_jobs_by_class(jobs, list(classes))
                        fleet_state = probe_fleet_state(jobs)
                        telemetry = collect_performance_samples(telemetry_globs)
                        _, running_jobs, _, _ = partition_running_jobs(jobs, fleet_state)

                        stale_reports = detect_stale_containers(
                            running_jobs, telemetry,
                            kill_stale=kill_stale, logger=logger,
                        )
                        if stale_reports:
                            logger.info(
                                "stale check: %d containers flagged", len(stale_reports),
                            )
                    except Exception as exc:
                        logger.debug("stale detection error: %s", exc)

                # Done?
                if state.is_done:
                    logger.info("drain complete: all jobs finished")
                    break

                if state.pending == 0 and state.running == 0 and state.blocked > 0:
                    logger.warning(
                        "drain stalled: %d jobs blocked, none running", state.blocked,
                    )
                    # Don't exit — keep polling in case fleet comes back
                    pass

            except Exception as exc:
                consecutive_failures += 1
                backoff = _backoff_seconds(consecutive_failures - 1)
                logger.error(
                    "tick %d failed (%d consecutive): %s — retrying in %ds",
                    tick, consecutive_failures, exc, backoff,
                )
                # Sleep the backoff period, but respect shutdown signal
                deadline = time.time() + backoff
                while time.time() < deadline and not _shutdown_flag:
                    time.sleep(1)
                continue

            # Sleep until next tick, waking every second to check shutdown flag
            deadline = time.time() + poll_interval
            while time.time() < deadline and not _shutdown_flag:
                time.sleep(1)

    except _ShutdownRequested:
        logger.info("shutdown requested")
    finally:
        _remove_pid(pid_path)
        logger.info("drain daemon stopped (pid=%d, ticks=%d)", os.getpid(), tick)

    return exit_code
