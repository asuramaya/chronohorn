"""Fleet drain loop: poll, re-dispatch, pull results until manifest is done."""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from chronohorn.fleet.dispatch import (
    load_manifest,
    select_jobs,
    filter_jobs_by_class,
    probe_fleet_state,
    partition_running_jobs,
    assign_jobs_best_effort,
    launch_job,
    write_launch_record,
    load_launch_record,
)
from chronohorn.fleet.telemetry import collect_performance_samples
from chronohorn.fleet.results import pull_all_completed_results


@dataclass(frozen=True)
class DrainState:
    manifest_path: str
    pending: int
    running: int
    completed: int
    blocked: int
    launched: int
    pulled: int
    stale_warned: int = 0

    @property
    def is_done(self) -> bool:
        return self.pending == 0 and self.running == 0 and self.blocked == 0


def drain_db_tick(
    *,
    db,
    manifests: Sequence[str] = (),
    job_names: Sequence[str] = (),
    classes: Sequence[str] = (),
    telemetry_globs: Sequence[str] | None = None,
    result_out_dir: Path | None = None,
) -> DrainState:
    """Run one dispatch+pull cycle from DB-backed job specs."""
    manifest_names = {Path(manifest).name for manifest in manifests if str(manifest).strip()}
    jobs = db.active_jobs()
    if manifest_names:
        jobs = [job for job in jobs if str(job.get("manifest") or "") in manifest_names]
    if job_names:
        jobs = select_jobs(jobs, list(job_names))
    if classes:
        jobs = filter_jobs_by_class(jobs, list(classes))

    fleet_state = probe_fleet_state(jobs)
    telemetry = collect_performance_samples(telemetry_globs)
    pending, running, completed, stale = partition_running_jobs(jobs, fleet_state)

    assigned, blocked = assign_jobs_best_effort(pending, fleet_state, telemetry)
    launched_count = 0
    for assigned_job in assigned:
        try:
            record = launch_job(assigned_job)
            write_launch_record(assigned_job["name"], record)
            db.record_launch(
                assigned_job["name"],
                host=record.get("host", "local"),
                executor_kind=record.get("executor_kind", ""),
                executor_name=record.get("executor_name", ""),
                launcher=record.get("launcher", ""),
                container=record.get("container_name", ""),
                remote_run=record.get("remote_run", ""),
                runtime_namespace=record.get("runtime_namespace", ""),
                runtime_job_name=record.get("runtime_job_name", ""),
                runtime_pod_name=record.get("runtime_pod_name", ""),
                runtime_node_name=record.get("runtime_node_name", ""),
            )
            db.record_event("launched", name=assigned_job["name"], host=assigned_job.get("host", "local"))
            launched_count += 1
            print(f"  launched {assigned_job['name']} -> {assigned_job.get('host', 'local')}", file=sys.stderr)
        except Exception as exc:
            print(f"  FAILED to launch {assigned_job['name']}: {exc}", file=sys.stderr)

    completed_records = []
    for job in completed:
        launch_rec = load_launch_record(job["name"])
        if launch_rec:
            completed_records.append(launch_rec)
    pull_results = pull_all_completed_results(completed_records, local_out_dir=result_out_dir, db=db)
    pulled_count = sum(1 for row in pull_results if row.success and not row.skipped)
    stale_warned = _detect_stale_running(running, telemetry)
    scope = ",".join(sorted(manifest_names)) if manifest_names else "__db__"

    return DrainState(
        manifest_path=scope,
        pending=len(pending) - launched_count,
        running=len(running) + launched_count,
        completed=len(completed),
        blocked=len(blocked),
        launched=launched_count,
        pulled=pulled_count,
        stale_warned=stale_warned,
    )


def drain_tick(
    manifest_path: str | Path,
    *,
    job_names: Sequence[str] = (),
    classes: Sequence[str] = (),
    telemetry_globs: Sequence[str] | None = None,
    result_out_dir: Path | None = None,
    db=None,
) -> DrainState:
    """Run one dispatch+pull cycle. Returns the current drain state."""
    manifest_path = Path(manifest_path)
    jobs = load_manifest(manifest_path)
    if job_names:
        jobs = select_jobs(jobs, list(job_names))
    if classes:
        jobs = filter_jobs_by_class(jobs, list(classes))

    fleet_state = probe_fleet_state(jobs)
    telemetry = collect_performance_samples(telemetry_globs)
    pending, running, completed, stale = partition_running_jobs(jobs, fleet_state)

    # Try to launch pending jobs
    assigned, blocked = assign_jobs_best_effort(pending, fleet_state, telemetry)
    launched_count = 0
    for assigned_job in assigned:
        try:
            record = launch_job(assigned_job)
            write_launch_record(assigned_job["name"], record)
            if db is not None:
                db.record_launch(
                    assigned_job["name"],
                    host=assigned_job.get("host", "local"),
                    executor_kind=record.get("executor_kind", ""),
                    executor_name=record.get("executor_name", ""),
                    launcher=record.get("launcher", ""),
                    container=record.get("container_name", ""),
                    remote_run=record.get("remote_run", ""),
                    runtime_namespace=record.get("runtime_namespace", ""),
                    runtime_job_name=record.get("runtime_job_name", ""),
                    runtime_pod_name=record.get("runtime_pod_name", ""),
                    runtime_node_name=record.get("runtime_node_name", ""),
                )
                db.record_event("launched", name=assigned_job["name"], host=assigned_job.get("host", "local"))
            launched_count += 1
            print(f"  launched {assigned_job['name']} -> {assigned_job.get('host', 'local')}", file=sys.stderr)
        except Exception as exc:
            print(f"  FAILED to launch {assigned_job['name']}: {exc}", file=sys.stderr)

    # Pull results from completed jobs
    completed_records = []
    for job in completed:
        launch_rec = load_launch_record(job["name"])
        if launch_rec:
            completed_records.append(launch_rec)
    pull_results = pull_all_completed_results(completed_records, local_out_dir=result_out_dir, db=db)
    pulled_count = sum(1 for r in pull_results if r.success and not r.skipped)

    # Stale container detection on running jobs
    stale_warned = _detect_stale_running(running, telemetry)

    return DrainState(
        manifest_path=str(manifest_path),
        pending=len(pending) - launched_count,
        running=len(running) + launched_count,
        completed=len(completed),
        blocked=len(blocked),
        launched=launched_count,
        pulled=pulled_count,
        stale_warned=stale_warned,
    )


def _detect_stale_running(
    running_jobs: list[dict], telemetry: list, *, warn_multiplier: float = 2.0,
) -> int:
    """Log warnings for running jobs that exceed warn_multiplier * expected duration.

    Returns the number of jobs flagged as stale.
    """
    warned = 0
    now = time.time()
    for job in running_jobs:
        name = job.get("name", "")
        record = load_launch_record(name)
        if not record:
            continue
        launched_at = record.get("launched_at_unix")
        if not launched_at:
            continue
        elapsed = now - launched_at

        # Estimate expected duration from telemetry token throughput
        expected = _estimate_duration_from_telemetry(job, telemetry)
        if expected is None:
            continue

        if elapsed > warn_multiplier * expected:
            host = job.get("host", record.get("host", "local"))
            ratio = elapsed / expected
            print(
                f"  STALE WARNING: {name} on {host} — "
                f"elapsed {elapsed:.0f}s ({ratio:.1f}x expected {expected:.0f}s)",
                file=sys.stderr,
            )
            warned += 1
    return warned


def _estimate_duration_from_telemetry(job: dict, telemetry: list) -> float | None:
    """Rough wall-clock estimate for a job, using telemetry throughput."""
    train_tokens = job.get("train_tokens")
    if not train_tokens:
        steps = job.get("steps") or job.get("train_steps")
        if steps:
            train_tokens = int(steps) * 2048
    if not train_tokens:
        return None

    speeds = [s.tokens_per_second for s in telemetry if s.tokens_per_second > 0]
    if not speeds:
        return None
    median_speed = sorted(speeds)[len(speeds) // 2]
    return float(train_tokens) / median_speed


def drain_loop(
    manifest_path: str | Path,
    *,
    poll_interval: int = 60,
    job_names: Sequence[str] = (),
    classes: Sequence[str] = (),
    telemetry_globs: Sequence[str] | None = None,
    result_out_dir: Path | None = None,
    max_ticks: int | None = None,
    db=None,
) -> DrainState:
    """Poll until all manifest jobs are completed or blocked."""
    tick = 0
    while True:
        tick += 1
        state = drain_tick(
            manifest_path,
            job_names=job_names,
            classes=classes,
            telemetry_globs=telemetry_globs,
            result_out_dir=result_out_dir,
            db=db,
        )

        status_line = (
            f"[tick {tick}] pending={state.pending} running={state.running} "
            f"completed={state.completed} blocked={state.blocked} "
            f"launched={state.launched} pulled={state.pulled}"
        )
        print(status_line, file=sys.stderr)

        if state.is_done:
            print("drain complete: all jobs finished", file=sys.stderr)
            return state

        if state.pending == 0 and state.running == 0 and state.blocked > 0:
            print(f"drain stalled: {state.blocked} jobs blocked, none running", file=sys.stderr)
            return state

        if max_ticks is not None and tick >= max_ticks:
            print(f"drain stopped: reached max ticks ({max_ticks})", file=sys.stderr)
            return state

        time.sleep(poll_interval)
