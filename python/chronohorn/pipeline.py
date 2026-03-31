from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any, Protocol, Sequence

from chronohorn.engine.budgets import CompetitionBudget, DEFAULT_GOLF_V1_BUDGET, resolve_competition_budget
from chronohorn.engine.forecasting import build_result_forecast
from chronohorn.engine.results import extract_result_metric, extract_result_summary, load_result_json
from chronohorn.fleet.dispatch import load_manifest, partition_running_jobs, probe_fleet_state
from chronohorn.fleet.forecast_results import build_forecast_row, collect_result_paths

from .store import RunRecord, RunStore

CHRONOHORN_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LAUNCH_GLOBS = [str(CHRONOHORN_ROOT / "out" / "fleet" / "*.launch.json")]


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
    return {
        "backend": job.get("backend"),
        "resource_class": job.get("resource_class"),
        "launcher": job.get("launcher"),
        "goal": job.get("goal"),
        "work_tokens": job.get("work_tokens"),
        "host": job.get("host"),
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def _result_name(path: Path) -> str:
    return path.stem


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


def normalize_runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
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


def _manifest_job_names(config: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for manifest_path in _coerce_paths(config.get("manifest_paths")):
        for job in load_manifest(manifest_path):
            names.add(str(job["name"]))
    return names


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
        jobs: list[dict[str, Any]] = []
        manifest_paths = _coerce_paths(config.get("manifest_paths"))
        for manifest_path in manifest_paths:
            jobs.extend(load_manifest(manifest_path))
        if not jobs:
            return
        fleet_state = probe_fleet_state(jobs)
        pending, running, completed, stale = partition_running_jobs(
            jobs,
            fleet_state,
            relaunch_completed=bool(config.get("relaunch_completed")),
        )
        for job in pending:
            store.add(
                RunRecord(
                    kind="runtime_state",
                    source="probe_fleet_state",
                    family=_infer_family(job.get("name"), job.get("goal"), job),
                    name=str(job["name"]),
                    status="pending",
                    metadata=_manifest_metadata(job),
                )
            )
        for row in running:
            store.add(
                RunRecord(
                    kind="runtime_state",
                    source="probe_fleet_state",
                    family=_infer_family(row.get("name"), row.get("launcher"), row),
                    name=str(row["name"]),
                    status="running",
                    metadata=dict(row),
                )
            )
        for row in completed:
            store.add(
                RunRecord(
                    kind="runtime_state",
                    source="probe_fleet_state",
                    family=_infer_family(row.get("name"), row.get("launcher"), row),
                    name=str(row["name"]),
                    status="completed",
                    metadata=dict(row),
                )
            )
        for row in stale:
            store.add(
                RunRecord(
                    kind="runtime_state",
                    source="probe_fleet_state",
                    family=_infer_family(row.get("name"), row.get("launcher"), row),
                    name=str(row["name"]),
                    status="stale",
                    metadata=dict(row),
                )
            )


class LaunchStage:
    name = "launch"

    def run(self, store: RunStore, config: dict[str, Any]) -> None:
        launch_globs = list(config.get("launch_globs") or DEFAULT_LAUNCH_GLOBS)
        manifest_names = _manifest_job_names(config)
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
            store.add(
                RunRecord(
                    kind="launch",
                    source=str(launch_path),
                    family=_infer_family(name, payload.get("launcher"), payload.get("goal"), payload),
                    name=name,
                    status="launched",
                    path=str(launch_path),
                    metadata={
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
        result_paths = collect_result_paths(
            list(config.get("result_paths") or []),
            list(config.get("result_globs") or []),
        )
        for result_path in result_paths:
            payload = load_result_json(result_path)
            summary = extract_result_summary(payload, path=result_path)
            metric = extract_result_metric(payload)
            model = payload.get("model") if isinstance(payload.get("model"), dict) else {}
            training = payload.get("training") if isinstance(payload.get("training"), dict) else {}
            store.add(
                RunRecord(
                    kind="result",
                    source=str(result_path),
                    family=_infer_family(result_path.stem, model.get("preset"), payload.get("title")),
                    name=_result_name(result_path),
                    status="completed",
                    path=str(result_path),
                    metric_name=metric.name,
                    metric_value=metric.value,
                    metadata={
                        "title": payload.get("title"),
                        "summary": summary.as_dict(),
                        "backend": training.get("backend"),
                        "device": training.get("device"),
                    },
                )
            )


class ForecastStage:
    name = "forecast"

    def run(self, store: RunStore, config: dict[str, Any]) -> None:
        budget = _budget_from_config(config)
        result_paths = collect_result_paths(
            list(config.get("result_paths") or []),
            list(config.get("result_globs") or []),
        )
        for result_path in result_paths:
            payload = load_result_json(result_path)
            forecast = build_result_forecast(payload, budget=budget)
            row = build_forecast_row(result_path, forecast)
            store.add(
                RunRecord(
                    kind="forecast",
                    source=str(result_path),
                    family=_infer_family(result_path.stem, payload.get("title")),
                    name=_result_name(result_path),
                    status=str(row.get("decision_signal") or "unknown"),
                    path=str(result_path),
                    metric_name=str(row.get("metric_name") or ""),
                    metric_value=row.get("forecast_metric_at_budget"),
                    metadata={
                        "artifact_viable": row.get("artifact_viable"),
                        "decision": row.get("decision"),
                        "forecast_confidence": row.get("forecast_confidence"),
                        "compute_axis": row.get("compute_axis"),
                        "probe_overhead": row.get("probe_overhead"),
                        "uncertainty": row.get("uncertainty"),
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
