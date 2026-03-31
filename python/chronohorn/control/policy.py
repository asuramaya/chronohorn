from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from chronohorn.control.models import ControlAction, ControlPlan
from chronohorn.control.ranker import (
    control_rank_score,
    current_metric,
    dominates,
    forecast_metric,
    marginal_gain_per_hour,
    metric_is_lower_better,
    remaining_wallclock_sec,
    run_metric_name,
)
from chronohorn.fleet.dispatch import (
    assign_job,
    collect_performance_samples,
    filter_jobs_by_class,
    fleet_state_summary,
    load_manifest,
    partition_running_jobs,
    probe_fleet_state,
    select_jobs,
)
from chronohorn.families.registry import resolve_training_adapter
from chronohorn.pipeline import build_runtime_store, normalize_runtime_config
from chronohorn.store import RunSnapshot, RunStore


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_jobs(manifest_paths: Sequence[str], *, job_names: Sequence[str], classes: Sequence[str]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for manifest_path in manifest_paths:
        jobs.extend(load_manifest(Path(manifest_path).expanduser().resolve()))
    jobs = filter_jobs_by_class(jobs, list(classes) or None)
    jobs = select_jobs(jobs, list(job_names) or None)
    return jobs


def _filter_loaded_jobs(
    jobs: list[dict[str, Any]],
    *,
    job_names: Sequence[str],
    classes: Sequence[str],
) -> list[dict[str, Any]]:
    selected = filter_jobs_by_class(list(jobs), list(classes) or None)
    selected = select_jobs(selected, list(job_names) or None)
    return selected


def _pending_launch_actions(
    pending_jobs: list[dict[str, Any]],
    fleet_state: dict[str, Any],
    telemetry_samples: list[Any],
    *,
    completed_runs: list[RunSnapshot],
    max_launches: int,
) -> tuple[list[ControlAction], list[dict[str, Any]]]:
    actions: list[ControlAction] = []
    blocked: list[dict[str, Any]] = []
    planned_count = 0
    ranked_pending_jobs = sorted(
        pending_jobs,
        key=lambda job: (
            -_job_priority(job, completed_runs=completed_runs),
            str(job.get("name") or ""),
        ),
    )
    for job in ranked_pending_jobs:
        if planned_count >= max(1, max_launches):
            break
        try:
            assigned = assign_job(job, fleet_state, telemetry_samples)
        except Exception as exc:  # noqa: BLE001
            blocked.append({"name": job["name"], "reason": str(exc)})
            continue
        planner = assigned.get("planner", {}) if isinstance(assigned.get("planner"), dict) else {}
        predicted_seconds = _safe_float(planner.get("predicted_seconds"))
        expected_tflops = _safe_float(planner.get("expected_tflops"))
        job_priority = _job_priority(job, completed_runs=completed_runs)
        rationale_bits = [str(planner.get("reason") or "eligible next manifest row"), f"frontier_score={job_priority:.3f}"]
        if predicted_seconds is not None:
            rationale_bits.append(f"predicted {predicted_seconds/3600.0:.2f}h")
        if expected_tflops is not None:
            rationale_bits.append(f"{expected_tflops:.3f} TF/s")
        actions.append(
            ControlAction(
                action="launch_job",
                target_name=str(assigned["name"]),
                family=str(assigned.get("family") or assigned.get("model_family") or ""),
                priority=90.0 - planned_count,
                rationale="; ".join(rationale_bits),
                state="pending",
                host=str(assigned.get("host") or "") or None,
                launcher=str(assigned.get("launcher") or "") or None,
                metadata={"assigned_job": assigned, "planner": planner},
            )
        )
        planned_count += 1
    return actions, blocked


def _job_priority(job: dict[str, Any], *, completed_runs: list[RunSnapshot]) -> float:
    family_id = str(job.get("family") or job.get("model_family") or "").strip()
    base_score = 0.0
    if family_id:
        family_peers = [run for run in completed_runs if run.family == family_id]
        if family_peers:
            best_peer = min(family_peers, key=control_rank_score)
            best_metric = forecast_metric(best_peer)
            if best_metric is not None:
                base_score += max(0.0, 5.0 - float(best_metric))
        try:
            adapter = resolve_training_adapter(family_id)
        except Exception:  # noqa: BLE001
            return base_score
        scorer = getattr(adapter, "score_frontier_job", None)
        if callable(scorer):
            try:
                base_score += float(scorer(job=job, completed_runs=completed_runs))
            except Exception:  # noqa: BLE001
                return base_score
    return base_score


def _promotion_kind(run: RunSnapshot) -> str:
    if run.artifact_viable:
        return "export_candidate"
    return "confirm_or_compress"


def _promotion_actions(completed_runs: list[RunSnapshot], *, top_completed: int) -> list[ControlAction]:
    scored = [run for run in completed_runs if forecast_metric(run) is not None]
    scored.sort(key=control_rank_score)
    actions: list[ControlAction] = []
    for idx, run in enumerate(scored[: max(0, top_completed)]):
        metric_name = run_metric_name(run)
        metric_value = forecast_metric(run)
        if metric_value is None:
            continue
        rationale = f"top completed {run.family or 'unknown'} candidate"
        if metric_name:
            rationale += f"; forecast {metric_name}={metric_value:.4f}"
        gain_per_hour = marginal_gain_per_hour(run)
        if gain_per_hour is not None:
            rationale += f"; gain/hour={gain_per_hour:.4f}"
        actions.append(
            ControlAction(
                action="promote_candidate",
                target_name=run.name,
                family=run.family,
                priority=80.0 - idx,
                rationale=rationale,
                state=run.state,
                host=run.host,
                launcher=run.launcher,
                metadata={
                    "promotion_kind": _promotion_kind(run),
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "control_rank_score": control_rank_score(run),
                },
            )
        )
    return actions


def _stop_actions(
    running_runs: list[RunSnapshot],
    completed_runs: list[RunSnapshot],
    *,
    stop_margin: float,
    min_gain_per_hour: float,
) -> tuple[list[ControlAction], list[ControlAction]]:
    stop: list[ControlAction] = []
    keep: list[ControlAction] = []
    peer_pool = [run for run in completed_runs + running_runs if forecast_metric(run) is not None]
    for run in running_runs:
        best_peer: RunSnapshot | None = None
        for peer in peer_pool:
            if peer.name == run.name:
                continue
            if peer.family != run.family:
                continue
            if not dominates(peer, run, margin=stop_margin):
                continue
            if best_peer is None or control_rank_score(peer) < control_rank_score(best_peer):
                best_peer = peer
        gain_per_hour = marginal_gain_per_hour(run)
        if best_peer is not None and gain_per_hour is not None and gain_per_hour < min_gain_per_hour:
            rationale = (
                f"dominated by {best_peer.name}; "
                f"forecast {run_metric_name(best_peer)}={forecast_metric(best_peer):.4f} "
                f"beats running pessimistic bound and gain/hour={gain_per_hour:.4f}"
            )
            stop.append(
                ControlAction(
                    action="stop_run",
                    target_name=run.name,
                    family=run.family,
                    priority=95.0,
                    rationale=rationale,
                    state=run.state,
                    host=run.host,
                    launcher=run.launcher,
                    metadata={
                        "dominated_by": best_peer.name,
                        "gain_per_hour": gain_per_hour,
                        "remaining_wallclock_sec": remaining_wallclock_sec(run),
                        "runtime_state": run.metadata.get("runtime_state"),
                    },
                )
            )
            continue
        rationale = "active frontier lane"
        if gain_per_hour is not None:
            rationale += f"; gain/hour={gain_per_hour:.4f}"
        keep.append(
            ControlAction(
                action="continue_run",
                target_name=run.name,
                family=run.family,
                priority=20.0,
                rationale=rationale,
                state=run.state,
                host=run.host,
                launcher=run.launcher,
                metadata={
                    "metric_name": run_metric_name(run),
                    "current_metric_value": current_metric(run),
                    "forecast_metric_value": forecast_metric(run),
                    "remaining_wallclock_sec": remaining_wallclock_sec(run),
                },
            )
        )
    return stop, keep


def _blocked_actions(blocked: list[dict[str, Any]]) -> list[ControlAction]:
    actions: list[ControlAction] = []
    for row in blocked:
        actions.append(
            ControlAction(
                action="blocked_launch",
                target_name=str(row.get("name") or ""),
                family=None,
                priority=5.0,
                rationale=str(row.get("reason") or "blocked"),
                state="blocked",
            )
        )
    return actions


def build_control_plan(
    config: dict[str, Any],
    *,
    job_names: Sequence[str] = (),
    classes: Sequence[str] = (),
    telemetry_globs: Sequence[str] = (),
    relaunch_completed: bool = False,
    max_launches: int = 2,
    stop_margin: float = 0.01,
    min_gain_per_hour: float = 0.01,
    top_completed: int = 3,
) -> ControlPlan:
    normalized = normalize_runtime_config(config)
    normalized["probe_runtime"] = bool(normalized.get("probe_runtime", False)) or bool(normalized.get("manifest_paths"))
    manifest_paths = list(normalized.get("manifest_paths") or [])
    all_jobs = _load_jobs(manifest_paths, job_names=(), classes=())
    if all_jobs:
        normalized["_manifest_jobs_cache"] = list(all_jobs)
        fleet_state = probe_fleet_state(all_jobs)
        normalized["_fleet_state_cache"] = fleet_state
        pending_all, running_all, completed_all, stale_all = partition_running_jobs(
            all_jobs,
            fleet_state,
            relaunch_completed=relaunch_completed,
        )
        normalized["_pending_jobs_cache"] = pending_all
        normalized["_running_jobs_cache"] = running_all
        normalized["_completed_jobs_cache"] = completed_all
        normalized["_stale_jobs_cache"] = stale_all
    else:
        fleet_state = {"remote": {}, "local": None}
    store, stages_run = build_runtime_store(normalized)
    jobs = _filter_loaded_jobs(all_jobs, job_names=job_names, classes=classes)
    telemetry_samples = collect_performance_samples(list(telemetry_globs) or None)
    pending_jobs, running_jobs_raw, completed_jobs_raw, stale_jobs = partition_running_jobs(
        jobs,
        fleet_state,
        relaunch_completed=relaunch_completed,
    ) if jobs else ([], [], [], [])

    runs = store.runs()
    completed_runs = [run for run in runs if run.state == "completed"]
    running_runs = [run for run in runs if run.state == "running"]
    launch_actions, blocked_rows = _pending_launch_actions(
        pending_jobs,
        fleet_state,
        telemetry_samples,
        completed_runs=completed_runs,
        max_launches=max_launches,
    )
    stop_actions, keep_actions = _stop_actions(
        running_runs,
        completed_runs,
        stop_margin=stop_margin,
        min_gain_per_hour=min_gain_per_hour,
    )
    promotion_actions = _promotion_actions(completed_runs, top_completed=top_completed)
    blocked_actions = _blocked_actions(blocked_rows)

    actions = stop_actions + launch_actions + promotion_actions + keep_actions + blocked_actions
    actions.sort(key=lambda row: (-row.priority, row.target_name or ""))

    best_completed = None
    scored_completed = [run for run in completed_runs if forecast_metric(run) is not None]
    if scored_completed:
        best_completed = min(scored_completed, key=control_rank_score)

    summary = {
        "stages_run": stages_run,
        "store": store.summary(),
        "fleet": fleet_state_summary(fleet_state),
        "telemetry_sample_count": len(telemetry_samples),
        "pending_count": len(pending_jobs),
        "running_count": len(running_jobs_raw),
        "completed_count": len(completed_jobs_raw),
        "stale_count": len(stale_jobs),
        "action_counts": {
            "launch": sum(1 for row in actions if row.action == "launch_job"),
            "stop": sum(1 for row in actions if row.action == "stop_run"),
            "promote": sum(1 for row in actions if row.action == "promote_candidate"),
            "continue": sum(1 for row in actions if row.action == "continue_run"),
            "blocked": sum(1 for row in actions if row.action == "blocked_launch"),
        },
        "best_completed": (
            {
                "name": best_completed.name,
                "family": best_completed.family,
                "metric_name": run_metric_name(best_completed),
                "metric_value": forecast_metric(best_completed),
                "state": best_completed.state,
                "artifact_viable": best_completed.artifact_viable,
            }
            if best_completed is not None
            else None
        ),
    }

    return ControlPlan(
        summary=summary,
        actions=tuple(actions),
        runs=tuple(run.as_dict() for run in runs[:10]),
    )
