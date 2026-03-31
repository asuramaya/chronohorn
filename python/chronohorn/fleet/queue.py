from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

from .dispatch import (
    DEFAULT_OUT_DIR,
    assign_job,
    collect_performance_samples,
    filter_jobs_by_class,
    fleet_state_summary,
    launch_job,
    load_manifest,
    partition_running_jobs,
    probe_fleet_state,
    select_jobs,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="chronohorn fleet queue",
        description="Queue manifest-driven Chronohorn work and keep feeding eligible lanes as hardware frees up.",
    )
    parser.add_argument("--manifest", required=True, help="JSONL manifest path")
    parser.add_argument("--job", action="append", default=[], help="Queue only the named job (repeatable).")
    parser.add_argument(
        "--class",
        dest="classes",
        action="append",
        default=[],
        help="Queue only jobs in the given resource_class (repeatable).",
    )
    parser.add_argument(
        "--poll-sec",
        type=float,
        default=60.0,
        help="Seconds between queue polls when no new jobs can launch.",
    )
    parser.add_argument(
        "--max-launches-per-pass",
        type=int,
        default=2,
        help="Maximum number of new jobs to launch per queue poll.",
    )
    parser.add_argument(
        "--telemetry-glob",
        action="append",
        default=[],
        help="Additional glob to scan for Chronohorn performance JSONs when planning placement.",
    )
    parser.add_argument(
        "--relaunch-completed",
        action="store_true",
        help="Queue jobs even when a completed remote report already exists.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single queue cycle and exit.",
    )
    return parser.parse_args(argv)


def queue_once(
    manifest_path: Path,
    *,
    job_names: list[str],
    classes: list[str],
    relaunch_completed: bool,
    max_launches_per_pass: int,
    telemetry_globs: list[str],
) -> dict[str, object]:
    jobs = load_manifest(manifest_path)
    jobs = filter_jobs_by_class(jobs, classes or None)
    jobs = select_jobs(jobs, job_names or None)
    fleet_state = probe_fleet_state(jobs)
    telemetry_samples = collect_performance_samples(telemetry_globs or None)
    pending_jobs, running_jobs, completed_jobs, stale_jobs = partition_running_jobs(
        jobs, fleet_state, relaunch_completed=relaunch_completed
    )

    launched: list[dict[str, object]] = []
    blocked: list[dict[str, object]] = []
    planned_count = 0
    for job in pending_jobs:
        if planned_count >= max_launches_per_pass:
            break
        try:
            assigned = assign_job(job, fleet_state, telemetry_samples)
        except Exception as exc:  # noqa: BLE001
            blocked.append({"name": job["name"], "reason": str(exc)})
            continue
        try:
            launched.append(launch_job(assigned))
        except Exception as exc:  # noqa: BLE001
            blocked.append({"name": job["name"], "reason": f"launch failed: {exc}"})
            continue
        planned_count += 1

    remaining_pending = max(len(pending_jobs) - len(launched), 0)
    return {
        "manifest": str(manifest_path),
        "fleet": fleet_state_summary(fleet_state),
        "telemetry_sample_count": len(telemetry_samples),
        "already_running": running_jobs,
        "completed": completed_jobs,
        "stale": stale_jobs,
        "launched": launched,
        "blocked": blocked,
        "pending_count": remaining_pending,
        "running_count": len(running_jobs),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest).expanduser().resolve()

    while True:
        cycle = queue_once(
            manifest_path,
            job_names=args.job or [],
            classes=args.classes or [],
            relaunch_completed=args.relaunch_completed,
            max_launches_per_pass=max(1, args.max_launches_per_pass),
            telemetry_globs=args.telemetry_glob or [],
        )
        print(json.dumps(cycle, indent=2, sort_keys=True), flush=True)
        if args.once:
            return 0

        if cycle["pending_count"] == 0 and cycle["running_count"] == 0:
            return 0
        time.sleep(max(args.poll_sec, 1.0))
