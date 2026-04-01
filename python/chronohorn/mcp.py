"""MCP tool server for Chronohorn runtime observation, forecasting, and control."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from chronohorn.control.actions import execute_control_actions
from chronohorn.control.models import ControlAction
from chronohorn.control.policy import build_control_plan
from chronohorn.engine.budgets import DEFAULT_GOLF_V1_BUDGET, CompetitionBudget, resolve_competition_budget
from chronohorn.engine.forecasting import build_result_forecast
from chronohorn.engine.results import load_result_json
from chronohorn.fleet.forecast_results import build_forecast_row, collect_result_paths
from chronohorn.pipeline import (
    ForecastStage,
    LaunchStage,
    ManifestStage,
    ResultStage,
    RuntimeStateStage,
    build_runtime_store,
    build_store_payload,
    normalize_runtime_config,
)
from chronohorn.store import RunStore

TOOLS = {
    "chronohorn_manifests": {
        "description": "Ingest manifest JSONL files into the current Chronohorn runtime store.",
        "parameters": {
            "manifest_paths": {"type": "array", "description": "Manifest JSONL paths", "required": True},
        },
    },
    "chronohorn_runtime_status": {
        "description": "Probe live runtime state from manifest-defined jobs and record pending/running/completed/stale status.",
        "parameters": {
            "manifest_paths": {"type": "array", "description": "Manifest JSONL paths", "required": True},
        },
    },
    "chronohorn_launches": {
        "description": "Ingest Chronohorn launch-record JSONs into the current runtime store.",
        "parameters": {
            "launch_globs": {"type": "array", "description": "Launch-record globs"},
        },
    },
    "chronohorn_results": {
        "description": "Ingest Chronohorn result JSONs into the current runtime store.",
        "parameters": {
            "result_paths": {"type": "array", "description": "Result JSON paths or directories"},
            "result_globs": {"type": "array", "description": "Result JSON globs"},
        },
    },
    "chronohorn_forecast": {
        "description": "Project Chronohorn result JSONs onto the active competition budget.",
        "parameters": {
            "result_paths": {"type": "array", "description": "Result JSON paths or directories"},
            "result_globs": {"type": "array", "description": "Result JSON globs"},
            "budget_name": {"type": "string", "description": "Competition budget label"},
            "train_tflops_budget": {"type": "number", "description": "Training budget in TFLOPs"},
            "artifact_limit_mb": {"type": "number", "description": "Artifact-size budget in MB"},
        },
    },
    "chronohorn_records": {
        "description": "Query filtered runtime records from the current Chronohorn store.",
        "parameters": {
            "kind": {"type": "string", "description": "Filter by record kind"},
            "source": {"type": "string", "description": "Filter by record source"},
            "family": {"type": "string", "description": "Filter by family"},
            "name": {"type": "string", "description": "Filter by run name"},
            "status": {"type": "string", "description": "Filter by record status"},
            "top_k": {"type": "integer", "description": "Maximum number of records to return"},
        },
    },
    "chronohorn_status": {
        "description": "Return a compact summary of the current Chronohorn runtime store.",
        "parameters": {
            "top_k": {"type": "integer", "description": "Maximum number of merged runs to return"},
        },
    },
    "chronohorn_frontier": {
        "description": "Return the best raw and artifact-feasible frontier rows from the current Chronohorn runtime store.",
        "parameters": {
            "top_k": {"type": "integer", "description": "Maximum number of rows per leaderboard"},
        },
    },
    "chronohorn_pipeline": {
        "description": "Run the Chronohorn observer pipeline end to end and replace the current runtime store.",
        "parameters": {
            "manifest_paths": {"type": "array", "description": "Manifest JSONL paths"},
            "state_paths": {"type": "array", "description": "Tracked state JSON paths"},
            "launch_globs": {"type": "array", "description": "Launch-record globs"},
            "result_paths": {"type": "array", "description": "Result JSON paths or directories"},
            "result_globs": {"type": "array", "description": "Result JSON globs"},
            "probe_runtime": {"type": "boolean", "description": "Probe live runtime state from manifests"},
            "budget_name": {"type": "string", "description": "Competition budget label"},
            "train_tflops_budget": {"type": "number", "description": "Training budget in TFLOPs"},
            "artifact_limit_mb": {"type": "number", "description": "Artifact-size budget in MB"},
            "top_k": {"type": "integer", "description": "Maximum number of merged runs to return"},
            "include_records": {"type": "boolean", "description": "Include raw records in the response"},
        },
    },
    "chronohorn_control_recommend": {
        "description": "Build a closed-loop Chronohorn control plan with launch, stop, and promotion recommendations.",
        "parameters": {
            "manifest_paths": {"type": "array", "description": "Manifest JSONL paths"},
            "launch_globs": {"type": "array", "description": "Launch-record globs"},
            "result_paths": {"type": "array", "description": "Result JSON paths or directories"},
            "result_globs": {"type": "array", "description": "Result JSON globs"},
            "probe_runtime": {"type": "boolean", "description": "Probe live runtime state from manifests"},
            "budget_name": {"type": "string", "description": "Competition budget label"},
            "train_tflops_budget": {"type": "number", "description": "Training budget in TFLOPs"},
            "artifact_limit_mb": {"type": "number", "description": "Artifact-size budget in MB"},
            "job_names": {"type": "array", "description": "Restrict to named jobs"},
            "classes": {"type": "array", "description": "Restrict to resource classes"},
            "telemetry_globs": {"type": "array", "description": "Additional telemetry globs"},
            "relaunch_completed": {"type": "boolean", "description": "Treat completed jobs as relaunch-eligible"},
            "max_launches": {"type": "integer", "description": "Maximum pending launches to recommend"},
            "stop_margin": {"type": "number", "description": "Metric margin required for domination"},
            "min_gain_per_hour": {"type": "number", "description": "Minimum marginal improvement per hour before stop"},
            "top_completed": {"type": "integer", "description": "Top completed runs to flag for promotion"},
        },
    },
    "chronohorn_control_act": {
        "description": "Execute recommended Chronohorn launch actions and optional stop actions.",
        "parameters": {
            "manifest_paths": {"type": "array", "description": "Manifest JSONL paths"},
            "launch_globs": {"type": "array", "description": "Launch-record globs"},
            "result_paths": {"type": "array", "description": "Result JSON paths or directories"},
            "result_globs": {"type": "array", "description": "Result JSON globs"},
            "probe_runtime": {"type": "boolean", "description": "Probe live runtime state from manifests"},
            "budget_name": {"type": "string", "description": "Competition budget label"},
            "train_tflops_budget": {"type": "number", "description": "Training budget in TFLOPs"},
            "artifact_limit_mb": {"type": "number", "description": "Artifact-size budget in MB"},
            "job_names": {"type": "array", "description": "Restrict to named jobs"},
            "classes": {"type": "array", "description": "Restrict to resource classes"},
            "telemetry_globs": {"type": "array", "description": "Additional telemetry globs"},
            "relaunch_completed": {"type": "boolean", "description": "Treat completed jobs as relaunch-eligible"},
            "max_launches": {"type": "integer", "description": "Maximum pending launches to execute"},
            "stop_margin": {"type": "number", "description": "Metric margin required for domination"},
            "min_gain_per_hour": {"type": "number", "description": "Minimum marginal improvement per hour before stop"},
            "top_completed": {"type": "integer", "description": "Top completed runs to flag for promotion"},
            "allow_stop": {"type": "boolean", "description": "Actually stop dominated running jobs"},
        },
    },
    "chronohorn_reset": {
        "description": "Reset the in-memory Chronohorn runtime store.",
        "parameters": {},
    },
    "chronohorn_fleet_dispatch": {
        "description": "Dispatch pending jobs from a manifest to the fleet. Returns launched, blocked, and running jobs.",
        "parameters": {
            "manifest_path": {"type": "string", "description": "Manifest JSONL path", "required": True},
            "job_names": {"type": "array", "description": "Restrict to named jobs"},
            "classes": {"type": "array", "description": "Restrict to resource classes"},
            "dry_run": {"type": "boolean", "description": "Plan only, do not launch"},
        },
    },
    "chronohorn_fleet_drain_tick": {
        "description": "Run one drain cycle: dispatch pending jobs, pull completed results. Call repeatedly to drain a manifest.",
        "parameters": {
            "manifest_path": {"type": "string", "description": "Manifest JSONL path", "required": True},
            "job_names": {"type": "array", "description": "Restrict to named jobs"},
            "classes": {"type": "array", "description": "Restrict to resource classes"},
        },
    },
    "chronohorn_fleet_status": {
        "description": "Check fleet placement and job status for a manifest without launching.",
        "parameters": {
            "manifest_path": {"type": "string", "description": "Manifest JSONL path", "required": True},
        },
    },
}


def _budget_from_args(args: dict[str, Any]) -> CompetitionBudget:
    budget_name = str(args.get("budget_name") or DEFAULT_GOLF_V1_BUDGET.name)
    base = resolve_competition_budget(budget_name)
    return CompetitionBudget(
        name=budget_name,
        train_tflops_budget=float(args.get("train_tflops_budget") or base.train_tflops_budget),
        artifact_limit_mb=float(args.get("artifact_limit_mb") or base.artifact_limit_mb),
        primary_metric_name=base.primary_metric_name,
    )


def _pipeline_config(args: dict[str, Any]) -> dict[str, Any]:
    return normalize_runtime_config(
        {
            "manifest_paths": list(args.get("manifest_paths") or []),
            "state_paths": list(args.get("state_paths") or []),
            "launch_globs": list(args.get("launch_globs") or []),
            "result_paths": list(args.get("result_paths") or []),
            "result_globs": list(args.get("result_globs") or []),
            "probe_runtime": bool(args.get("probe_runtime", False)),
            "budget_name": str(args.get("budget_name") or DEFAULT_GOLF_V1_BUDGET.name),
            "train_tflops_budget": float(args.get("train_tflops_budget") or DEFAULT_GOLF_V1_BUDGET.train_tflops_budget),
            "artifact_limit_mb": float(args.get("artifact_limit_mb") or DEFAULT_GOLF_V1_BUDGET.artifact_limit_mb),
        }
    )


class ToolServer:
    def __init__(self) -> None:
        self._store = RunStore()
        self._stages_run: list[str] = []

    @property
    def store(self) -> RunStore:
        return self._store

    def list_tools(self) -> list[dict[str, Any]]:
        return [{"name": name, **definition} for name, definition in TOOLS.items()]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "chronohorn_manifests":
            return self._do_manifests(arguments)
        if name == "chronohorn_runtime_status":
            return self._do_runtime_status(arguments)
        if name == "chronohorn_launches":
            return self._do_launches(arguments)
        if name == "chronohorn_results":
            return self._do_results(arguments)
        if name == "chronohorn_forecast":
            return self._do_forecast(arguments)
        if name == "chronohorn_records":
            return self._do_records(arguments)
        if name == "chronohorn_status":
            return self._do_status(arguments)
        if name == "chronohorn_frontier":
            return self._do_frontier(arguments)
        if name == "chronohorn_pipeline":
            return self._do_pipeline(arguments)
        if name == "chronohorn_control_recommend":
            return self._do_control_recommend(arguments)
        if name == "chronohorn_control_act":
            return self._do_control_act(arguments)
        if name == "chronohorn_reset":
            return self._do_reset(arguments)
        if name == "chronohorn_fleet_dispatch":
            return self._do_fleet_dispatch(arguments)
        if name == "chronohorn_fleet_drain_tick":
            return self._do_fleet_drain_tick(arguments)
        if name == "chronohorn_fleet_status":
            return self._do_fleet_status(arguments)
        return {"error": f"Unknown tool: {name}"}

    def _run_stage(self, stage: Any, config: dict[str, Any]) -> dict[str, Any]:
        stage.run(self._store, config)
        if stage.name not in self._stages_run:
            self._stages_run.append(stage.name)
        return build_store_payload(self._store, stages_run=self._stages_run, top_k=10, include_records=False)

    def _do_manifests(self, args: dict[str, Any]) -> dict[str, Any]:
        return self._run_stage(ManifestStage(), {"manifest_paths": list(args.get("manifest_paths") or [])})

    def _do_runtime_status(self, args: dict[str, Any]) -> dict[str, Any]:
        return self._run_stage(
            RuntimeStateStage(),
            {
                "manifest_paths": list(args.get("manifest_paths") or []),
                "probe_runtime": True,
            },
        )

    def _do_launches(self, args: dict[str, Any]) -> dict[str, Any]:
        return self._run_stage(LaunchStage(), {"launch_globs": list(args.get("launch_globs") or [])})

    def _do_results(self, args: dict[str, Any]) -> dict[str, Any]:
        return self._run_stage(
            ResultStage(),
            {
                "result_paths": list(args.get("result_paths") or []),
                "result_globs": list(args.get("result_globs") or []),
            },
        )

    def _do_forecast(self, args: dict[str, Any]) -> dict[str, Any]:
        budget = _budget_from_args(args)
        rows: list[dict[str, Any]] = []
        for path in collect_result_paths(list(args.get("result_paths") or []), list(args.get("result_globs") or [])):
            payload = load_result_json(path)
            forecast = build_result_forecast(payload, budget=budget)
            rows.append(build_forecast_row(path, forecast))
        if rows:
            self._run_stage(
                ForecastStage(),
                {
                    "result_paths": list(args.get("result_paths") or []),
                    "result_globs": list(args.get("result_globs") or []),
                    "budget_name": budget.name,
                    "train_tflops_budget": budget.train_tflops_budget,
                    "artifact_limit_mb": budget.artifact_limit_mb,
                    "discover_remote_results": False,
                },
            )
        return {
            "budget": {
                "name": budget.name,
                "train_tflops_budget": budget.train_tflops_budget,
                "artifact_limit_mb": budget.artifact_limit_mb,
            },
            "rows": rows,
        }

    def _do_records(self, args: dict[str, Any]) -> dict[str, Any]:
        top_k = int(args.get("top_k") or 50)
        rows = [
            record.as_dict()
            for record in self._store.filter(
                kind=args.get("kind"),
                source=args.get("source"),
                family=args.get("family"),
                name=args.get("name"),
                status=args.get("status"),
            )[: max(top_k, 0)]
        ]
        return {"count": len(rows), "records": rows}

    def _do_status(self, args: dict[str, Any]) -> dict[str, Any]:
        top_k = int(args.get("top_k") or 10)
        return build_store_payload(self._store, stages_run=self._stages_run, top_k=top_k, include_records=False)

    def _do_frontier(self, args: dict[str, Any]) -> dict[str, Any]:
        top_k = int(args.get("top_k") or 10)
        payload = build_store_payload(self._store, stages_run=self._stages_run, top_k=top_k, include_records=False)
        return {
            "summary": payload.get("summary", {}),
            "stages_run": payload.get("stages_run", []),
            "frontier": payload.get("frontier", {}),
        }

    def _do_pipeline(self, args: dict[str, Any]) -> dict[str, Any]:
        top_k = int(args.get("top_k") or 10)
        include_records = bool(args.get("include_records", False))
        self._store, self._stages_run = build_runtime_store(_pipeline_config(args))
        return build_store_payload(self._store, stages_run=self._stages_run, top_k=top_k, include_records=include_records)

    def _do_control_recommend(self, args: dict[str, Any]) -> dict[str, Any]:
        plan = build_control_plan(
            _pipeline_config(args),
            job_names=list(args.get("job_names") or []),
            classes=list(args.get("classes") or []),
            telemetry_globs=list(args.get("telemetry_globs") or []),
            relaunch_completed=bool(args.get("relaunch_completed", False)),
            max_launches=int(args.get("max_launches") or 2),
            stop_margin=float(args.get("stop_margin") or 0.01),
            min_gain_per_hour=float(args.get("min_gain_per_hour") or 0.01),
            top_completed=int(args.get("top_completed") or 3),
        )
        return plan.as_dict()

    def _do_control_act(self, args: dict[str, Any]) -> dict[str, Any]:
        plan = build_control_plan(
            _pipeline_config(args),
            job_names=list(args.get("job_names") or []),
            classes=list(args.get("classes") or []),
            telemetry_globs=list(args.get("telemetry_globs") or []),
            relaunch_completed=bool(args.get("relaunch_completed", False)),
            max_launches=int(args.get("max_launches") or 2),
            stop_margin=float(args.get("stop_margin") or 0.01),
            min_gain_per_hour=float(args.get("min_gain_per_hour") or 0.01),
            top_completed=int(args.get("top_completed") or 3),
        )
        actions = [ControlAction(**row) for row in plan.as_dict().get("actions", [])]
        executed = execute_control_actions(
            actions,
            allow_stop=bool(args.get("allow_stop", False)),
            max_launches=int(args.get("max_launches") or 2),
        )
        return {"plan": plan.as_dict(), "executed": executed}

    def _do_reset(self, args: dict[str, Any]) -> dict[str, Any]:
        self._store = RunStore()
        self._stages_run = []
        return {"status": "ok", "message": "Chronohorn runtime store reset"}

    def _do_fleet_dispatch(self, args: dict[str, Any]) -> dict[str, Any]:
        from chronohorn.fleet.dispatch import main as fleet_main
        import io
        import contextlib

        argv = ["--manifest", str(args["manifest_path"])]
        for name in (args.get("job_names") or []):
            argv.extend(["--job", name])
        for cls in (args.get("classes") or []):
            argv.extend(["--class", cls])
        if args.get("dry_run"):
            argv.append("--dry-run")

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fleet_main(argv)
        try:
            return json.loads(buf.getvalue())
        except (json.JSONDecodeError, ValueError):
            return {"raw_output": buf.getvalue()}

    def _do_fleet_drain_tick(self, args: dict[str, Any]) -> dict[str, Any]:
        from chronohorn.fleet.drain import drain_tick

        state = drain_tick(
            args["manifest_path"],
            job_names=list(args.get("job_names") or []),
            classes=list(args.get("classes") or []),
        )
        return {
            "pending": state.pending,
            "running": state.running,
            "completed": state.completed,
            "blocked": state.blocked,
            "launched": state.launched,
            "pulled": state.pulled,
            "done": state.is_done,
        }

    def _do_fleet_status(self, args: dict[str, Any]) -> dict[str, Any]:
        return self._do_fleet_dispatch({**args, "dry_run": True})
