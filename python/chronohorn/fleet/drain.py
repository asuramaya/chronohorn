"""Fleet drain loop: poll, re-dispatch, pull results until manifest is done."""
from __future__ import annotations

import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chronohorn.db import IMPORTED_RESULT_MANIFEST
from chronohorn.fleet.dispatch import (
    assign_jobs_best_effort,
    filter_jobs_by_class,
    launch_job,
    load_launch_record,
    load_manifest,
    partition_running_jobs,
    probe_fleet_state,
    runtime_record_for_job,
    select_jobs,
    write_launch_record,
)
from chronohorn.fleet.results import pull_all_completed_results
from chronohorn.fleet.telemetry import collect_performance_samples
from chronohorn.fleet.validation import (
    validate_job_name,
    validate_posix_path_within_any_root,
    validate_posix_path_within_root,
)
from chronohorn.manifest_paths import manifest_matches
from chronohorn.metrics import probe_bpb_from_row
from chronohorn.service_log import service_log

_ALLOWED_REMOTE_RESULT_ROOTS = ("/tmp/chronohorn-runs", "/data/chronohorn/out")


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


def _pull_running_probes(running_jobs: list[dict[str, Any]], *, db) -> int:
    """Pull incremental probe JSONL from running jobs and ingest into DB."""
    from chronohorn.fleet.results import _ssh_cat_file
    ingested = 0
    for job in running_jobs:
        name = str(job.get("name") or "")
        host = str(job.get("host") or "")
        remote_run = str(job.get("remote_run") or "")
        if not name or not host or not remote_run:
            continue
        safe_name = validate_job_name(name)
        safe_remote_run = validate_posix_path_within_any_root(
            remote_run,
            roots=_ALLOWED_REMOTE_RESULT_ROOTS,
            field_name="remote_run",
        )
        probes_remote = f"{safe_remote_run}/results/{safe_name}.probes.jsonl"
        try:
            text = _ssh_cat_file(host, probes_remote)
        except RuntimeError:
            continue  # probes file may not exist yet
        existing = {r["step"] for r in db.query("SELECT step FROM probes WHERE name = ?", (safe_name,))}
        for line in text.strip().splitlines():
            try:
                p = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = p.get("step")
            pbpb = probe_bpb_from_row(p)
            if step and step not in existing and pbpb:
                db.record_probe(
                    safe_name,
                    step,
                    pbpb,
                    loss=p.get("loss", p.get("eval_loss")),
                    elapsed_sec=p.get("elapsed_sec"),
                )
                ingested += 1
    if ingested:
        service_log("fleet.drain", "probes ingested", ingested=ingested, running_jobs=len(running_jobs))
    return ingested


def _drain_log(message: str, *, level: str = "info", **fields: Any) -> None:
    service_log("fleet.drain", message, level=level, **fields)


def _materialize_manifest_jobs(
    *,
    db,
    manifest_paths: Sequence[str],
    job_names: Sequence[str] = (),
    classes: Sequence[str] = (),
) -> None:
    seen_names: set[str] = set()
    for manifest_path in manifest_paths:
        try:
            jobs = load_manifest(Path(manifest_path))
        except FileNotFoundError:
            continue
        if job_names:
            jobs = select_jobs(jobs, list(job_names))
        if classes:
            jobs = filter_jobs_by_class(jobs, list(classes))
        for job in jobs:
            name = str(job.get("name") or "").strip()
            if not name or name in seen_names:
                continue
            seen_names.add(name)
            config = job.get("config")
            db.record_job(
                name,
                manifest=str(job.get("manifest_path") or manifest_path),
                parent=str(job.get("parent") or ""),
                family=str(job.get("family") or "") or None,
                config=dict(config) if isinstance(config, Mapping) else None,
                steps=job.get("steps"),
                seed=job.get("seed"),
                lr=job.get("learning_rate", job.get("lr")),
                batch_size=job.get("batch_size"),
                command=str(job.get("command") or ""),
                job_spec=job,
            )


def _normalize_archive_only_jobs(*, db) -> None:
    active_rows = db.query(
        """
        SELECT j.name,
               EXISTS(SELECT 1 FROM results r WHERE r.name = j.name) AS has_result
        FROM jobs j
        WHERE j.manifest = ?
          AND j.state IN ('pending', 'dispatched', 'running')
        """,
        (IMPORTED_RESULT_MANIFEST,),
    )
    if not active_rows:
        return

    now = time.time()
    ops: list[tuple[str, tuple[Any, ...]]] = []
    for row in active_rows:
        name = str(row.get("name") or "")
        if not name:
            continue
        if bool(row.get("has_result")):
            ops.append(
                (
                    "UPDATE jobs SET state = 'completed', completed_at = COALESCE(completed_at, ?) WHERE name = ?",
                    (now, name),
                )
            )
            db.record_event("normalized_imported_result_completed", name=name)
        else:
            ops.append(
                (
                    "UPDATE jobs SET state = 'failed', completed_at = COALESCE(completed_at, ?) WHERE name = ?",
                    (now, name),
                )
            )
            db.record_event("normalized_imported_result_failed", name=name)
    if ops:
        db._write_many(ops, wait=True)


def _reconcile_db_active_states(
    *,
    db,
    jobs: list[dict[str, Any]],
    running: list[dict[str, Any]],
    completed: list[dict[str, Any]],
    stale: list[dict[str, Any]],
    pull_results,
) -> None:
    running_names = {str(job.get("name") or "") for job in running}
    stale_names = {str(job.get("name") or "") for job in stale}
    running_by_name = {
        str(job.get("name") or ""): job
        for job in running
        if str(job.get("name") or "")
    }
    stale_by_name = {
        str(job.get("name") or ""): job
        for job in stale
        if str(job.get("name") or "")
    }
    completed_by_name = {
        str(job.get("name") or ""): job
        for job in completed
        if str(job.get("name") or "")
    }
    completed_names = set(completed_by_name)
    completed_names.update(
        {
        str(row.job_name)
        for row in (pull_results or [])
        if bool(getattr(row, "success", False)) and bool(getattr(row, "ingested", False))
        }
    )

    ops: list[tuple[str, tuple[Any, ...]]] = []
    now = time.time()
    for job in jobs:
        name = str(job.get("name") or "")
        if not name:
            continue
        current_state = str(job.get("state") or "").lower()
        if name in completed_names:
            desired_state = "completed"
        else:
            desired_state = "running" if name in running_names else "failed" if name in stale_names else "pending"
        if current_state == desired_state:
            continue
        if desired_state == "completed":
            completed_record = completed_by_name.get(name) or {}
            host = completed_record.get("host") or job.get("host")
            executor_kind = completed_record.get("executor_kind") or job.get("executor_kind")
            executor_name = completed_record.get("executor_name") or job.get("executor_name")
            runtime_namespace = completed_record.get("runtime_namespace") or job.get("runtime_namespace")
            runtime_job_name = completed_record.get("runtime_job_name") or job.get("runtime_job_name")
            runtime_pod_name = completed_record.get("runtime_pod_name") or job.get("runtime_pod_name")
            runtime_node_name = completed_record.get("runtime_node_name") or job.get("runtime_node_name")
            ops.append(
                (
                    """
                    UPDATE jobs
                    SET state = 'completed',
                        completed_at = COALESCE(completed_at, ?),
                        host = ?,
                        executor_kind = ?,
                        executor_name = ?,
                        runtime_namespace = ?,
                        runtime_job_name = ?,
                        runtime_pod_name = ?,
                        runtime_node_name = ?
                    WHERE name = ?
                    """,
                    (
                        now,
                        host,
                        executor_kind,
                        executor_name,
                        runtime_namespace,
                        runtime_job_name,
                        runtime_pod_name,
                        runtime_node_name,
                        name,
                    ),
                )
            )
            db.record_event("reconciled_job_completed", name=name, previous_state=current_state)
        elif desired_state == "failed":
            stale_record = stale_by_name.get(name) or {}
            host = stale_record.get("host") or job.get("host")
            executor_kind = stale_record.get("executor_kind") or job.get("executor_kind")
            executor_name = stale_record.get("executor_name") or job.get("executor_name")
            runtime_namespace = stale_record.get("runtime_namespace") or job.get("runtime_namespace")
            runtime_job_name = stale_record.get("runtime_job_name") or job.get("runtime_job_name")
            runtime_pod_name = stale_record.get("runtime_pod_name") or job.get("runtime_pod_name")
            runtime_node_name = stale_record.get("runtime_node_name") or job.get("runtime_node_name")
            # Capture failure diagnostics from the stale/failed record
            failure_reason = stale_record.get("reason") or stale_record.get("log_last_line") or ""
            failure_log = stale_record.get("log_tail_text") or ""
            ops.append(
                (
                    """
                    UPDATE jobs
                    SET state = 'failed',
                        completed_at = ?,
                        host = ?,
                        executor_kind = ?,
                        executor_name = ?,
                        runtime_namespace = ?,
                        runtime_job_name = ?,
                        runtime_pod_name = ?,
                        runtime_node_name = ?,
                        failure_reason = ?,
                        failure_log = ?
                    WHERE name = ?
                    """,
                    (
                        now,
                        host,
                        executor_kind,
                        executor_name,
                        runtime_namespace,
                        runtime_job_name,
                        runtime_pod_name,
                        runtime_node_name,
                        failure_reason[:500] if failure_reason else None,
                        failure_log[:4000] if failure_log else None,
                        name,
                    ),
                )
            )
            db.record_event("reconciled_job_failed", name=name, previous_state=current_state)
        elif desired_state == "running":
            running_record = running_by_name.get(name) or {}
            host = running_record.get("host") or job.get("host")
            executor_kind = running_record.get("executor_kind") or job.get("executor_kind")
            executor_name = running_record.get("executor_name") or job.get("executor_name")
            runtime_namespace = running_record.get("runtime_namespace") or job.get("runtime_namespace")
            runtime_job_name = running_record.get("runtime_job_name") or job.get("runtime_job_name")
            runtime_pod_name = running_record.get("runtime_pod_name") or job.get("runtime_pod_name")
            runtime_node_name = running_record.get("runtime_node_name") or job.get("runtime_node_name")
            ops.append(
                (
                    """
                    UPDATE jobs
                    SET state = 'running',
                        host = ?,
                        executor_kind = ?,
                        executor_name = ?,
                        runtime_namespace = ?,
                        runtime_job_name = ?,
                        runtime_pod_name = ?,
                        runtime_node_name = ?
                    WHERE name = ?
                    """,
                    (
                        host,
                        executor_kind,
                        executor_name,
                        runtime_namespace,
                        runtime_job_name,
                        runtime_pod_name,
                        runtime_node_name,
                        name,
                    ),
                )
            )
            db.record_event("reconciled_job_running", name=name, previous_state=current_state)
        else:
            ops.append(("UPDATE jobs SET state = 'pending' WHERE name = ?", (name,)))
            db.record_event("reconciled_job_pending", name=name, previous_state=current_state)
    if ops:
        db._write_many(ops, wait=True)


def drain_db_tick(
    *,
    db,
    manifests: Sequence[str] = (),
    job_names: Sequence[str] = (),
    classes: Sequence[str] = (),
    telemetry_globs: Sequence[str] | None = None,
    result_out_dir: Path | None = None,
    dispatch: bool = True,
) -> DrainState:
    """Run one dispatch+pull cycle from DB-backed job specs."""
    if manifests:
        _materialize_manifest_jobs(
            db=db,
            manifest_paths=manifests,
            job_names=job_names,
            classes=classes,
        )
    _normalize_archive_only_jobs(db=db)

    jobs = db.active_jobs()
    manifest_filters = [str(value or "").strip() for value in manifests if str(value or "").strip()]
    if manifest_filters:
        jobs = [
            job
            for job in jobs
            if manifest_matches(str(job.get("manifest") or ""), manifest_filters)
        ]
    if job_names:
        jobs = select_jobs(jobs, list(job_names))
    if classes:
        jobs = filter_jobs_by_class(jobs, list(classes))

    fleet_state = probe_fleet_state(jobs)
    telemetry = collect_performance_samples(telemetry_globs)
    pending, running, completed, stale = partition_running_jobs(jobs, fleet_state)

    launched_count = 0
    if not dispatch:
        assigned, blocked = [], []
    else:
        assigned, blocked = assign_jobs_best_effort(pending, fleet_state, telemetry)
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
            _drain_log("job launched", name=assigned_job["name"], host=assigned_job.get("host", "local"))
        except Exception as exc:
            db.record_event(
                "launch_failed",
                name=assigned_job.get("name"),
                host=assigned_job.get("host", "local"),
                error=str(exc)[:500],
            )
            _drain_log(
                "launch failed",
                level="error",
                name=assigned_job.get("name"),
                host=assigned_job.get("host", "local"),
                error=str(exc),
            )

    completed_records = []
    for job in completed:
        launch_rec = runtime_record_for_job(job)
        if launch_rec:
            completed_records.append(launch_rec)
    pull_results = pull_all_completed_results(completed_records, local_out_dir=result_out_dir, db=db)
    pulled_count = sum(1 for row in pull_results if row.success and not row.skipped)

    # Pull probes from running jobs — feeds the dashboard with live learning curves
    # Use DB state, not fleet probe — manually registered jobs may not be discovered by fleet
    if db is not None:
        db_running = [j for j in jobs if str(j.get("state") or "").lower() == "running"]
        _pull_running_probes(db_running or running, db=db)
        _reconcile_db_active_states(
            db=db,
            jobs=jobs,
            running=running,
            completed=completed,
            stale=stale,
            pull_results=pull_results,
        )

    stale_warned = _detect_stale_running(running, telemetry, db=db)
    scope = ",".join(manifest_filters) if manifest_filters else "__db__"

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
    if db is not None:
        _materialize_manifest_jobs(
            db=db,
            manifest_paths=[str(manifest_path)],
            job_names=job_names,
            classes=classes,
        )
        _normalize_archive_only_jobs(db=db)
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
            _drain_log("job launched", name=assigned_job["name"], host=assigned_job.get("host", "local"))
        except Exception as exc:
            if db is not None:
                db.record_event(
                    "launch_failed",
                    name=assigned_job.get("name"),
                    host=assigned_job.get("host", "local"),
                    error=str(exc)[:500],
                )
            _drain_log(
                "launch failed",
                level="error",
                name=assigned_job.get("name"),
                host=assigned_job.get("host", "local"),
                error=str(exc),
            )

    # Pull results from completed jobs
    completed_records = []
    for job in completed:
        launch_rec = runtime_record_for_job(job)
        if launch_rec:
            completed_records.append(launch_rec)
    pull_results = pull_all_completed_results(completed_records, local_out_dir=result_out_dir, db=db)
    pulled_count = sum(1 for r in pull_results if r.success and not r.skipped)

    # Stale container detection on running jobs
    stale_warned = _detect_stale_running(running, telemetry, db=db)

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
    running_jobs: list[dict], telemetry: list, *, warn_multiplier: float = 2.0, db=None,
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
            _drain_log(
                "stale warning",
                level="warning",
                name=name,
                host=host,
                elapsed_sec=round(elapsed, 3),
                expected_sec=round(expected, 3),
                ratio=round(ratio, 3),
            )
            if db is not None:
                db.record_event(
                    "stale_warning",
                    name=name,
                    host=host,
                    elapsed_sec=round(elapsed, 3),
                    expected_sec=round(expected, 3),
                    ratio=round(ratio, 3),
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

        _drain_log(
            "tick",
            tick=tick,
            pending=state.pending,
            running=state.running,
            completed=state.completed,
            blocked=state.blocked,
            launched=state.launched,
            pulled=state.pulled,
        )

        if state.is_done:
            _drain_log("complete", pending=state.pending, running=state.running, blocked=state.blocked)
            return state

        if state.pending == 0 and state.running == 0 and state.blocked > 0:
            _drain_log("stalled", level="warning", blocked=state.blocked)
            return state

        if max_ticks is not None and tick >= max_ticks:
            _drain_log("stopped at max ticks", level="warning", max_ticks=max_ticks)
            return state

        time.sleep(poll_interval)
