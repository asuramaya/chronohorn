from __future__ import annotations

import glob
import json
from pathlib import Path
import re
import shlex
import subprocess
from typing import Any, Protocol, Sequence

from chronohorn.engine.budgets import CompetitionBudget, DEFAULT_GOLF_V1_BUDGET, resolve_competition_budget
from chronohorn.engine.forecasting import build_result_forecast
from chronohorn.engine.results import extract_result_metric, extract_result_summary, load_result_json
from chronohorn.fleet.dispatch import (
    capture_checked_retry,
    load_manifest,
    partition_running_jobs,
    probe_fleet_state,
    ssh_argv,
)
from chronohorn.fleet.forecast_results import build_forecast_row, collect_result_paths

from .store import RunRecord, RunStore

CHRONOHORN_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LAUNCH_GLOBS = [str(CHRONOHORN_ROOT / "out" / "fleet" / "*.launch.json")]
_PROGRESS_RE = re.compile(
    r"^\s*(?P<step>\d+)\s+\|\s+loss\s+(?P<loss>[-+0-9.eE]+)\s+\|\s+best\s+(?P<best>[-+0-9.eE]+)\s+\|\s+"
    r"(?P<tokens_per_second>[-+0-9.eE]+)\s+tok/s\s+\|\s+train\s+(?P<train_tflops>[-+0-9.eE]+)\s+TF/s\s+\|\s+"
    r"total\s+(?P<total_tflops>[-+0-9.eE]+)\s+TF/s\s*$"
)
_PROBE_RE = re.compile(
    r"^\s*probe\s+(?P<step>\d+)\s+\|\s+test\s+loss\s+(?P<test_loss>[-+0-9.eE]+)\s+\|\s+"
    r"bpt\s+(?P<bpt>[-+0-9.eE]+)\s+\|\s+bpb\s+(?P<bpb>[-+0-9.eE]+)\s*$"
)


def _coerce_paths(values: Sequence[str] | None) -> list[Path]:
    return [Path(value).expanduser().resolve() for value in (values or [])]


def _infer_family(*values: Any) -> str:
    haystack = " ".join(str(value) for value in values if value is not None).lower()
    if "causal-bank" in haystack or "causal_bank" in haystack or "conker" in haystack:
        return "causal-bank"
    if "oracle" in haystack:
        return "oracle"
    return "unknown"


def _manifest_metadata(job: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        "family": job.get("family") or job.get("model_family"),
        "backend": job.get("backend"),
        "resource_class": job.get("resource_class"),
        "launcher": job.get("launcher"),
        "goal": job.get("goal"),
        "work_tokens": job.get("work_tokens"),
        "host": job.get("host"),
        "run_id": job.get("run_id"),
        "manifest_path": job.get("manifest_path"),
    }
    for key in (
        "variant",
        "scale",
        "steps",
        "seq_len",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "local_window",
        "local_scale_override",
        "oscillatory_frac",
        "oscillatory_period_min",
        "oscillatory_period_max",
        "linear_readout_kind",
        "linear_readout_num_experts",
        "linear_half_life_max",
        "static_bank_gate",
        "bank_gate_span",
        "seed",
        "profile",
    ):
        if key in job:
            metadata[key] = job.get(key)
    return metadata


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric


def _result_name(path: Path) -> str:
    return path.stem


def _remote_result_json_path(remote_run: str, name: str) -> str:
    return str(Path(remote_run).joinpath("results", f"{name}.json"))


def _budget_from_config(config: dict[str, Any]) -> CompetitionBudget:
    budget_name = str(config.get("budget_name") or DEFAULT_GOLF_V1_BUDGET.name)
    base = resolve_competition_budget(budget_name)
    train_tflops_budget = float(config.get("train_tflops_budget") or base.train_tflops_budget)
    artifact_limit_mb = float(config.get("artifact_limit_mb") or base.artifact_limit_mb)
    return CompetitionBudget(
        name=budget_name,
        train_tflops_budget=train_tflops_budget,
        artifact_limit_mb=artifact_limit_mb,
        primary_metric_name=base.primary_metric_name,
    )


def _fetch_remote_result_payload(*, host: str, remote_run: str, name: str) -> dict[str, Any] | None:
    result_path = _remote_result_json_path(remote_run, name)
    remote_payload = f"""set -euo pipefail
result={shlex.quote(result_path)}
if [[ ! -s "$result" ]]; then
  exit 3
fi
cat "$result"
"""
    try:
        output = capture_checked_retry(
            ssh_argv(host, shlex.join(["/bin/bash", "-lc", remote_payload])),
            attempts=2,
            delay_sec=1.0,
        )
    except subprocess.CalledProcessError:
        return None
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _add_result_payload_records(
    store: RunStore,
    payload: dict[str, Any],
    *,
    source: str,
    name: str,
    run_id: str | None = None,
    path: str | None = None,
) -> None:
    summary = extract_result_summary(payload, path=path)
    metric = extract_result_metric(payload)
    model = payload.get("model") if isinstance(payload.get("model"), dict) else {}
    training = payload.get("training") if isinstance(payload.get("training"), dict) else {}
    family = _infer_family(payload.get("family"), name, model.get("preset"), payload.get("title"))
    store.add(
        RunRecord(
            kind="result",
            source=source,
            family=family,
            name=name,
            run_id=run_id,
            status="completed",
            path=path,
            metric_name=metric.name,
            metric_value=metric.value,
            metadata={
                "family": payload.get("family") or model.get("model_family"),
                "title": payload.get("title"),
                "summary": summary.as_dict(),
                "backend": training.get("backend"),
                "device": training.get("device"),
            },
        )
    )
    for row in training.get("performance_log") or []:
        if not isinstance(row, dict):
            continue
        store.add(
            RunRecord(
                kind="progress",
                source=source,
                family=family,
                name=name,
                run_id=run_id,
                status="completed",
                metric_name="train_loss",
                metric_value=_safe_float(row.get("loss")),
                metadata={
                    "step": row.get("step"),
                    "train_loss": _safe_float(row.get("loss")),
                    "best_loss": _safe_float(row.get("best_loss")),
                    "tokens_per_second": _safe_float(row.get("tokens_per_second")),
                    "train_tflops": _safe_float(row.get("estimated_sustained_tflops")),
                    "total_tflops": _safe_float(row.get("estimated_total_tflops")),
                },
            )
        )
    for row in training.get("probes") or []:
        if not isinstance(row, dict):
            continue
        metric_name = "bpb" if row.get("bpb") is not None else "bits_per_token"
        metric_value = _safe_float(row.get("bpb"))
        if metric_value is None:
            metric_value = _safe_float(row.get("bits_per_token"))
        store.add(
            RunRecord(
                kind="probe",
                source=source,
                family=family,
                name=name,
                run_id=run_id,
                status="completed",
                metric_name=metric_name,
                metric_value=metric_value,
                metadata={
                    "step": row.get("step"),
                    "test_loss": _safe_float(row.get("test_loss")),
                    "bits_per_token": _safe_float(row.get("bits_per_token")),
                    "bpb": _safe_float(row.get("bpb")),
                },
            )
        )


def _collect_result_payload_rows(store: RunStore, config: dict[str, Any]) -> list[dict[str, Any]]:
    cached = config.get("_result_payload_rows_cache")
    if isinstance(cached, list):
        return cached
    rows: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    result_paths = collect_result_paths(
        list(config.get("result_paths") or []),
        list(config.get("result_globs") or []),
    )
    result_name_counts: dict[str, int] = {}
    for result_path in result_paths:
        result_name_counts[_result_name(result_path)] = result_name_counts.get(_result_name(result_path), 0) + 1
    for result_path in result_paths:
        name = _result_name(result_path)
        run_id = name if result_name_counts.get(name, 0) <= 1 else str(result_path)
        rows.append(
            {
                "name": name,
                "run_id": run_id,
                "source": str(result_path),
                "path": str(result_path),
                "payload": load_result_json(result_path),
            }
        )
        seen_names.add(name)
    if not bool(config.get("discover_remote_results", True)):
        config["_result_payload_rows_cache"] = rows
        return rows
    for run in store.runs():
        if run.state != "completed" or run.name in seen_names:
            continue
        launch = run.metadata.get("launch")
        if not isinstance(launch, dict):
            continue
        host = str(launch.get("host") or run.host or "")
        remote_run = str(launch.get("remote_run") or "")
        if not host or host == "local" or not remote_run:
            continue
        payload = _fetch_remote_result_payload(host=host, remote_run=remote_run, name=run.name)
        if payload is None:
            continue
        remote_result_path = _remote_result_json_path(remote_run, run.name)
        rows.append(
            {
                "name": run.name,
                "run_id": run.run_id or run.name,
                "source": f"ssh://{host}{remote_result_path}",
                "path": f"ssh://{host}{remote_result_path}",
                "payload": payload,
            }
        )
        seen_names.add(run.name)
    config["_result_payload_rows_cache"] = rows
    return rows


def normalize_runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        "manifest_paths": list(config.get("manifest_paths") or []),
        "launch_globs": list(config.get("launch_globs") or DEFAULT_LAUNCH_GLOBS),
        "result_paths": list(config.get("result_paths") or []),
        "result_globs": list(config.get("result_globs") or []),
        "probe_runtime": bool(config.get("probe_runtime", False)),
        "relaunch_completed": bool(config.get("relaunch_completed", False)),
        "budget_name": str(config.get("budget_name") or DEFAULT_GOLF_V1_BUDGET.name),
        "train_tflops_budget": float(config.get("train_tflops_budget") or DEFAULT_GOLF_V1_BUDGET.train_tflops_budget),
        "artifact_limit_mb": float(config.get("artifact_limit_mb") or DEFAULT_GOLF_V1_BUDGET.artifact_limit_mb),
    }
    for key, value in config.items():
        if key.startswith("_"):
            normalized[key] = value
    return normalized


def _manifest_jobs(config: dict[str, Any]) -> list[dict[str, Any]]:
    cached = config.get("_manifest_jobs_cache")
    if isinstance(cached, list):
        return cached
    jobs: list[dict[str, Any]] = []
    for manifest_path in _coerce_paths(config.get("manifest_paths")):
        jobs.extend(load_manifest(manifest_path))
    config["_manifest_jobs_cache"] = jobs
    return jobs


def _manifest_job_names(config: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for job in _manifest_jobs(config):
        names.add(str(job["name"]))
    return names


def _manifest_jobs_by_name(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cached = config.get("_manifest_jobs_by_name_cache")
    if isinstance(cached, dict):
        return cached
    mapping = {str(job["name"]): job for job in _manifest_jobs(config)}
    config["_manifest_jobs_by_name_cache"] = mapping
    return mapping


class Stage(Protocol):
    name: str

    def run(self, store: RunStore, config: dict[str, Any]) -> None: ...


class Pipeline:
    def __init__(self, stages: Sequence[Stage]) -> None:
        self._stages = list(stages)
        self.stages_run: list[str] = []

    def run(self, config: dict[str, Any], store: RunStore | None = None) -> RunStore:
        active_store = RunStore() if store is None else store
        self.stages_run = []
        for stage in self._stages:
            stage.run(active_store, config)
            self.stages_run.append(stage.name)
        return active_store


class ManifestStage:
    name = "manifest"

    def run(self, store: RunStore, config: dict[str, Any]) -> None:
        for manifest_path in _coerce_paths(config.get("manifest_paths")):
            for job in load_manifest(manifest_path):
                store.add(
                    RunRecord(
                        kind="manifest",
                        source=str(manifest_path),
                        family=_infer_family(job.get("family"), job.get("model_family"), job.get("name"), job.get("goal"), job),
                        name=str(job["name"]),
                        run_id=(None if job.get("run_id") in {None, ""} else str(job.get("run_id"))),
                        status="declared",
                        path=str(manifest_path),
                        metadata=_manifest_metadata(job),
                    )
                )


class RuntimeStateStage:
    name = "runtime_state"

    def run(self, store: RunStore, config: dict[str, Any]) -> None:
        if not bool(config.get("probe_runtime")):
            return
        jobs = _manifest_jobs(config)
        jobs_by_name = _manifest_jobs_by_name(config)
        if not jobs:
            return
        fleet_state = config.get("_fleet_state_cache")
        if not isinstance(fleet_state, dict):
            fleet_state = probe_fleet_state(jobs)
            config["_fleet_state_cache"] = fleet_state
        pending = config.get("_pending_jobs_cache")
        running = config.get("_running_jobs_cache")
        completed = config.get("_completed_jobs_cache")
        stale = config.get("_stale_jobs_cache")
        if not all(isinstance(rows, list) for rows in (pending, running, completed, stale)):
            pending, running, completed, stale = partition_running_jobs(
                jobs,
                fleet_state,
                relaunch_completed=bool(config.get("relaunch_completed")),
            )
            config["_pending_jobs_cache"] = pending
            config["_running_jobs_cache"] = running
            config["_completed_jobs_cache"] = completed
            config["_stale_jobs_cache"] = stale
        for job in pending:
            store.add(
                RunRecord(
                    kind="runtime_state",
                    source="probe_fleet_state",
                    family=_infer_family(job.get("family"), job.get("name"), job.get("goal"), job),
                    name=str(job["name"]),
                    run_id=(None if job.get("run_id") in {None, ""} else str(job.get("run_id"))),
                    status="pending",
                    metadata=_manifest_metadata(job),
                )
            )
        for row in running:
            manifest_job = jobs_by_name.get(str(row["name"]))
            store.add(
                RunRecord(
                    kind="runtime_state",
                    source="probe_fleet_state",
                    family=_infer_family(row.get("family"), row.get("name"), row.get("launcher"), row),
                    name=str(row["name"]),
                    run_id=(
                        None
                        if (row.get("run_id") or (manifest_job or {}).get("run_id")) in {None, ""}
                        else str(row.get("run_id") or (manifest_job or {}).get("run_id"))
                    ),
                    status="running",
                    metadata=dict(row),
                )
            )
        for row in completed:
            manifest_job = jobs_by_name.get(str(row["name"]))
            store.add(
                RunRecord(
                    kind="runtime_state",
                    source="probe_fleet_state",
                    family=_infer_family(row.get("family"), row.get("name"), row.get("launcher"), row),
                    name=str(row["name"]),
                    run_id=(
                        None
                        if (row.get("run_id") or (manifest_job or {}).get("run_id")) in {None, ""}
                        else str(row.get("run_id") or (manifest_job or {}).get("run_id"))
                    ),
                    status="completed",
                    metadata=dict(row),
                )
            )
        for row in stale:
            manifest_job = jobs_by_name.get(str(row["name"]))
            store.add(
                RunRecord(
                    kind="runtime_state",
                    source="probe_fleet_state",
                    family=_infer_family(row.get("family"), row.get("name"), row.get("launcher"), row),
                    name=str(row["name"]),
                    run_id=(
                        None
                        if (row.get("run_id") or (manifest_job or {}).get("run_id")) in {None, ""}
                        else str(row.get("run_id") or (manifest_job or {}).get("run_id"))
                    ),
                    status="stale",
                    metadata=dict(row),
                )
            )


class LiveLogStage:
    name = "live_log"

    def run(self, store: RunStore, config: dict[str, Any]) -> None:
        if not bool(config.get("probe_runtime")):
            return
        for row in list(store.filter(kind="runtime_state")):
            log_tail_text = str(row.metadata.get("log_tail_text") or "")
            if not log_tail_text.strip():
                continue
            latest_progress: RunRecord | None = None
            latest_probe: RunRecord | None = None
            for raw_line in log_tail_text.splitlines():
                progress_match = _PROGRESS_RE.match(raw_line)
                if progress_match:
                    step = int(progress_match.group("step"))
                    latest_progress = RunRecord(
                        kind="progress",
                        source=row.source,
                        family=row.family,
                        name=row.name,
                        run_id=row.run_id,
                        status=row.status,
                        metric_name="train_loss",
                        metric_value=_safe_float(progress_match.group("loss")),
                        metadata={
                            "step": step,
                            "train_loss": _safe_float(progress_match.group("loss")),
                            "best_loss": _safe_float(progress_match.group("best")),
                            "tokens_per_second": _safe_float(progress_match.group("tokens_per_second")),
                            "train_tflops": _safe_float(progress_match.group("train_tflops")),
                            "total_tflops": _safe_float(progress_match.group("total_tflops")),
                            "raw_line": raw_line,
                        },
                    )
                    continue
                probe_match = _PROBE_RE.match(raw_line)
                if probe_match:
                    step = int(probe_match.group("step"))
                    latest_probe = RunRecord(
                        kind="probe",
                        source=row.source,
                        family=row.family,
                        name=row.name,
                        run_id=row.run_id,
                        status=row.status,
                        metric_name="bpb",
                        metric_value=_safe_float(probe_match.group("bpb")),
                        metadata={
                            "step": step,
                            "test_loss": _safe_float(probe_match.group("test_loss")),
                            "bits_per_token": _safe_float(probe_match.group("bpt")),
                            "bpb": _safe_float(probe_match.group("bpb")),
                            "raw_line": raw_line,
                        },
                    )
            if latest_progress is not None:
                store.add(latest_progress)
            if latest_probe is not None:
                store.add(latest_probe)


class LaunchStage:
    name = "launch"

    def run(self, store: RunStore, config: dict[str, Any]) -> None:
        launch_globs = list(config.get("launch_globs") or DEFAULT_LAUNCH_GLOBS)
        manifest_names = _manifest_job_names(config)
        jobs_by_name = _manifest_jobs_by_name(config)
        launch_paths: set[Path] = set()
        for raw_glob in launch_globs:
            for match in glob.glob(raw_glob, recursive=True):
                path = Path(match).expanduser()
                if path.is_file() and path.suffix == ".json":
                    launch_paths.add(path.resolve())
        for launch_path in sorted(launch_paths):
            payload = _load_json_object(launch_path)
            name = str(payload.get("name") or launch_path.stem)
            if manifest_names and name not in manifest_names:
                continue
            manifest_job = jobs_by_name.get(name)
            store.add(
                RunRecord(
                    kind="launch",
                    source=str(launch_path),
                    family=_infer_family(
                        payload.get("family"),
                        name,
                        payload.get("launcher"),
                        payload.get("goal"),
                        payload,
                    ),
                    name=name,
                    run_id=(
                        None
                        if (payload.get("run_id") or (manifest_job or {}).get("run_id")) in {None, ""}
                        else str(payload.get("run_id") or (manifest_job or {}).get("run_id"))
                    ),
                    status="launched",
                    path=str(launch_path),
                    metadata={
                        "family": payload.get("family"),
                        "host": payload.get("host"),
                        "launcher": payload.get("launcher"),
                        "backend": payload.get("backend"),
                        "resource_class": payload.get("resource_class"),
                        "remote_run": payload.get("remote_run"),
                        "launched_at_unix": payload.get("launched_at_unix"),
                    },
                )
            )


class ResultStage:
    name = "result"

    def run(self, store: RunStore, config: dict[str, Any]) -> None:
        for row in _collect_result_payload_rows(store, config):
            payload = row.get("payload")
            name = row.get("name")
            source = row.get("source")
            if not isinstance(payload, dict) or not isinstance(name, str) or not isinstance(source, str):
                continue
            path = row.get("path")
            _add_result_payload_records(
                store,
                payload,
                source=source,
                name=name,
                run_id=(None if row.get("run_id") in {None, ""} else str(row.get("run_id"))),
                path=path if isinstance(path, str) else None,
            )


class ForecastStage:
    name = "forecast"

    def run(self, store: RunStore, config: dict[str, Any]) -> None:
        budget = _budget_from_config(config)
        for row in _collect_result_payload_rows(store, config):
            payload = row.get("payload")
            name = row.get("name")
            path = row.get("path")
            source = row.get("source")
            if not isinstance(payload, dict) or not isinstance(name, str):
                continue
            forecast = build_result_forecast(payload, budget=budget)
            forecast_row = build_forecast_row(path or source or name, forecast)
            store.add(
                RunRecord(
                    kind="forecast",
                    source=str(source or path or name),
                    family=_infer_family(payload.get("family"), name, payload.get("title")),
                    name=name,
                    run_id=(None if row.get("run_id") in {None, ""} else str(row.get("run_id"))),
                    status=str(forecast_row.get("decision_signal") or "unknown"),
                    path=str(path) if isinstance(path, str) else None,
                    metric_name=str(forecast_row.get("metric_name") or ""),
                    metric_value=forecast_row.get("forecast_metric_at_budget"),
                    metadata={
                        "artifact_viable": forecast_row.get("artifact_viable"),
                        "decision": forecast_row.get("decision"),
                        "decision_signal": forecast_row.get("decision_signal"),
                        "current_metric_value": forecast_row.get("current_metric_value"),
                        "forecast_metric_at_budget": forecast_row.get("forecast_metric_at_budget"),
                        "forecast_delta_from_current": forecast_row.get("forecast_delta_from_current"),
                        "estimated_sustained_tflops": forecast_row.get("estimated_sustained_tflops"),
                        "estimated_sustained_total_tflops": forecast_row.get("estimated_sustained_total_tflops"),
                        "tokens_per_second": forecast_row.get("tokens_per_second"),
                        "compute_utilization": forecast_row.get("compute_utilization"),
                        "forecast_confidence": forecast_row.get("forecast_confidence"),
                        "compute_axis": forecast_row.get("compute_axis"),
                        "probe_overhead": forecast_row.get("probe_overhead"),
                        "uncertainty": forecast_row.get("uncertainty"),
                        "marginal_gain_per_tflop": (
                            None
                            if forecast_row.get("forecast", {}).get("projection", {}).get("dbpb_dtotal_tflop") is None
                            else -float(forecast_row["forecast"]["projection"]["dbpb_dtotal_tflop"])
                        ),
                        "marginal_gain_per_train_tflop": (
                            None
                            if forecast_row.get("forecast", {}).get("projection", {}).get("dbpb_dtrain_tflop") is None
                            else -float(forecast_row["forecast"]["projection"]["dbpb_dtrain_tflop"])
                        ),
                    },
                )
            )


def build_runtime_store(config: dict[str, Any], *, stages: Sequence[Stage] | None = None) -> tuple[RunStore, list[str]]:
    config = normalize_runtime_config(config)
    pipeline = Pipeline(
        stages
        or (
            ManifestStage(),
            RuntimeStateStage(),
            LiveLogStage(),
            LaunchStage(),
            ResultStage(),
            ForecastStage(),
        )
    )
    store = pipeline.run(config)
    return store, list(pipeline.stages_run)


def build_store_payload(
    store: RunStore,
    *,
    stages_run: Sequence[str] | None = None,
    top_k: int = 10,
    include_records: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "summary": store.summary(),
        "stages_run": list(stages_run or []),
        "runs": [run.as_dict() for run in store.best_runs(top_k)],
    }
    if include_records:
        payload["records"] = [record.as_dict() for record in store]
    return payload
