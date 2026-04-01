"""MCP tool server for Chronohorn runtime observation, forecasting, and control."""

from __future__ import annotations

import glob as _glob
import json
from pathlib import Path
from typing import Any


def _is_illegal_result(payload: dict[str, Any], name: str = "") -> bool:
    """Detect results that leak future information (illegal for golf)."""
    cfg = payload.get("config", {})
    train = cfg.get("train", {}) if isinstance(cfg.get("train"), dict) else cfg
    patch_size = train.get("patch_size", 1)
    decoder = train.get("patch_causal_decoder", "NOT_SET")
    if patch_size > 1 and decoder in ("none", "NOT_SET"):
        return True
    # Heuristic for results that don't store patch config
    if "patch" in name and "cpatch" not in name:
        bpb = (payload.get("model") or {}).get("test_bpb", 99)
        steps = train.get("steps", 0)
        if bpb < 1.0 and steps <= 5000:
            return True
    return False

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
    "chronohorn_learning_curve": {
        "description": "Return the learning curve (probe data) for a named run as step/bpb/tflops triples.",
        "parameters": {
            "name": {"type": "string", "description": "Run name", "required": True},
            "result_dir": {"type": "string", "description": "Result directory (default out/results)"},
        },
    },
    "chronohorn_compare": {
        "description": "Compare learning curves of multiple runs side by side.",
        "parameters": {
            "names": {"type": "array", "description": "Run names to compare", "required": True},
            "result_dir": {"type": "string", "description": "Result directory"},
        },
    },
    "chronohorn_marginal_rank": {
        "description": "Rank completed runs by marginal bpb gain per TFLOP (compute efficiency).",
        "parameters": {
            "result_dir": {"type": "string", "description": "Result directory"},
            "top_k": {"type": "integer", "description": "Maximum results to return"},
        },
    },
    "chronohorn_auto_deepen": {
        "description": "Pick top runs by marginal gain, generate and optionally dispatch a deepening manifest.",
        "parameters": {
            "source_manifest": {"type": "string", "description": "Source manifest path", "required": True},
            "top_n": {"type": "integer", "description": "Number of top runs to deepen (default 4)"},
            "target_steps": {"type": "integer", "description": "Step count for deepened runs (default 10000)"},
            "dispatch": {"type": "boolean", "description": "Dispatch immediately after generating"},
            "result_dir": {"type": "string", "description": "Result directory"},
        },
    },
    "chronohorn_artifact_check": {
        "description": "Check artifact size and 16MB viability for a named run.",
        "parameters": {
            "name": {"type": "string", "description": "Run name", "required": True},
            "result_dir": {"type": "string", "description": "Result directory"},
        },
    },
    "chronohorn_subscribe": {
        "description": "Return runs that changed state since the last subscribe call.",
        "parameters": {
            "result_dir": {"type": "string", "description": "Result directory"},
        },
    },
    "chronohorn_query": {
        "description": "Run a raw SQL query against the ChronohornDB. Returns rows as dicts.",
        "parameters": {
            "sql": {"type": "string", "description": "SQL query", "required": True},
            "db_path": {"type": "string", "description": "Database path (default out/chronohorn.db)"},
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
    def __init__(self, *, db=None) -> None:
        self._store = RunStore()
        self._shared_db = db  # ChronohornDB reference from runtime
        self._stages_run: list[str] = []
        self._last_seen_results: set[str] = set()

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
        if name == "chronohorn_learning_curve":
            return self._do_learning_curve(arguments)
        if name == "chronohorn_compare":
            return self._do_compare(arguments)
        if name == "chronohorn_marginal_rank":
            return self._do_marginal_rank(arguments)
        if name == "chronohorn_auto_deepen":
            return self._do_auto_deepen(arguments)
        if name == "chronohorn_artifact_check":
            return self._do_artifact_check(arguments)
        if name == "chronohorn_subscribe":
            return self._do_subscribe(arguments)
        if name == "chronohorn_query":
            return self._do_query(arguments)
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

    # -- learning curve helpers ------------------------------------------------

    @staticmethod
    def _load_learning_curve(name: str, result_dir: str = "out/results") -> dict[str, Any]:
        path = Path(result_dir) / f"{name}.json"
        if not path.is_file():
            return {"error": f"Result not found: {path}"}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            return {"error": str(exc)}
        probes = (payload.get("training") or {}).get("probes") or []
        perf = (payload.get("training") or {}).get("performance") or {}
        tflops_per_step = perf.get("train_step_flops_per_step_est")
        if tflops_per_step is not None:
            tflops_per_step = float(tflops_per_step) / 1e12  # convert flops to tflops
        points: list[dict[str, Any]] = []
        for p in probes:
            step = p.get("step")
            bpb = p.get("bpb") if p.get("bpb") is not None else p.get("test_bpb")
            tflops = round(step * tflops_per_step, 4) if (step and tflops_per_step) else None
            points.append({"step": step, "bpb": bpb, "tflops": tflops})
        return {"name": name, "points": points}

    def _do_learning_curve(self, args: dict[str, Any]) -> dict[str, Any]:
        name = str(args["name"])
        if self._shared_db:
            points = self._shared_db.learning_curve(name)
            return {"name": name, "points": points}
        result_dir = str(args.get("result_dir") or "out/results")
        return self._load_learning_curve(name, result_dir)

    def _do_compare(self, args: dict[str, Any]) -> dict[str, Any]:
        names = list(args["names"])
        if self._shared_db:
            curves = self._shared_db.compare_curves(names)
            return {"runs": [{"name": n, "points": p} for n, p in curves.items()]}
        result_dir = str(args.get("result_dir") or "out/results")
        runs = [self._load_learning_curve(n, result_dir) for n in names]
        return {"runs": runs}

    def _do_marginal_rank(self, args: dict[str, Any]) -> dict[str, Any]:
        top_k = int(args.get("top_k") or 50)
        if self._shared_db:
            return {"ranked": self._shared_db.marginal_rank(top_k)}
        result_dir = str(args.get("result_dir") or "out/results")
        ranked: list[dict[str, Any]] = []
        for p in sorted(_glob.glob(f"{result_dir}/*.json")):
            try:
                payload = json.loads(Path(p).read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            name = Path(p).stem
            probes = (payload.get("training") or {}).get("probes") or []
            perf = (payload.get("training") or {}).get("performance") or {}
            tflops_per_step = perf.get("train_step_flops_per_step_est")
            if tflops_per_step is not None:
                tflops_per_step = float(tflops_per_step) / 1e12
            # Need at least 2 probes to compute marginal
            if len(probes) < 2 or tflops_per_step is None:
                continue
            last = probes[-1]
            prev = probes[-2]
            last_step = last.get("step") or 0
            prev_step = prev.get("step") or 0
            last_bpb = last.get("bpb") if last.get("bpb") is not None else last.get("test_bpb")
            prev_bpb = prev.get("bpb") if prev.get("bpb") is not None else prev.get("test_bpb")
            if last_bpb is None or prev_bpb is None or last_step <= prev_step:
                continue
            delta_tflops = (last_step - prev_step) * tflops_per_step
            if delta_tflops <= 0:
                continue
            delta_bpb = prev_bpb - last_bpb  # positive = improvement
            marginal = delta_bpb / delta_tflops
            ranked.append({
                "name": name,
                "bpb": last_bpb,
                "marginal_per_tflop": round(marginal, 8),
                "steps": last_step,
                "illegal": _is_illegal_result(payload, name=name),
            })
        ranked.sort(key=lambda r: -r["marginal_per_tflop"])
        return {"ranked": ranked[:top_k]}

    def _do_auto_deepen(self, args: dict[str, Any]) -> dict[str, Any]:
        from chronohorn.fleet.manifest_transform import load_and_transform

        source_manifest = str(args["source_manifest"])
        top_n = int(args.get("top_n") or 4)
        target_steps = int(args.get("target_steps") or 10000)
        result_dir = str(args.get("result_dir") or "out/results")
        do_dispatch = bool(args.get("dispatch", False))

        # Get the ranked runs
        rank_result = self._do_marginal_rank({"result_dir": result_dir, "top_k": top_n})
        winners = rank_result.get("ranked", [])
        if not winners:
            return {"error": "No runs available to deepen", "ranked": []}

        winner_names = [w["name"] for w in winners]

        # Generate deepening manifest from the source
        output_path = Path(result_dir).parent / "manifests" / "deepen_auto.jsonl"
        rows = load_and_transform(
            Path(source_manifest),
            steps=target_steps,
            output_path=output_path,
        )
        # Filter to only winner names (match by prefix before the step suffix)
        deepened = [r for r in rows if any(wn in r.get("name", "") for wn in winner_names)]
        if not deepened:
            deepened = rows[:top_n]

        result: dict[str, Any] = {
            "winners": winner_names,
            "target_steps": target_steps,
            "manifest_rows": len(deepened),
            "output_path": str(output_path),
        }

        if do_dispatch and output_path.is_file():
            drain_result = self._do_fleet_drain_tick({"manifest_path": str(output_path)})
            result["dispatch"] = drain_result

        return result

    def _do_artifact_check(self, args: dict[str, Any]) -> dict[str, Any]:
        name = str(args["name"])
        result_dir = str(args.get("result_dir") or "out/results")
        path = Path(result_dir) / f"{name}.json"
        if not path.is_file():
            return {"error": f"Result not found: {path}"}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            return {"error": str(exc)}
        model = payload.get("model") or {}
        params = model.get("params")
        payload_mb_est = model.get("payload_mb_est")
        # Estimate int6 size: params * 6 bits / 8 bits per byte / 1e6 bytes per MB
        int6_mb = None
        if params is not None:
            int6_mb = round(float(params) * 6 / 8 / 1e6, 3)
        fits_16mb = None
        if int6_mb is not None:
            fits_16mb = int6_mb <= 16.0
        config = payload.get("config") or {}
        config_summary = {}
        train_config = config.get("train") or config.get("profile") or {}
        if isinstance(train_config, dict):
            for k in ("steps", "seq_len", "batch_size", "learning_rate"):
                if k in train_config:
                    config_summary[k] = train_config[k]
        return {
            "name": name,
            "params": params,
            "payload_mb_est": payload_mb_est,
            "int6_mb": int6_mb,
            "fits_16mb": fits_16mb,
            "config_summary": config_summary,
        }

    def _do_subscribe(self, args: dict[str, Any]) -> dict[str, Any]:
        result_dir = str(args.get("result_dir") or "out/results")
        current: set[str] = set()
        rdir = Path(result_dir)
        if rdir.is_dir():
            current = {p.name for p in rdir.glob("*.json")}
        new_files = sorted(current - self._last_seen_results)
        removed_files = sorted(self._last_seen_results - current)
        self._last_seen_results = current
        return {
            "new": new_files,
            "removed": removed_files,
            "total": len(current),
        }

    def _do_query(self, args: dict[str, Any]) -> dict[str, Any]:
        try:
            sql = str(args["sql"])
            if not sql.strip().upper().startswith("SELECT"):
                return {"error": "Only SELECT queries are allowed"}

            if self._shared_db:
                rows = self._shared_db.query(sql)
            else:
                from chronohorn.db import ChronohornDB
                # Use read_only=True if supported; fall back to plain open with a comment
                db = ChronohornDB(args.get("db_path", "out/chronohorn.db"))  # TODO: open read_only
                try:
                    rows = db.query(sql)
                finally:
                    db.close()
            return {"rows": rows, "count": len(rows)}
        except Exception as exc:
            return {"error": str(exc)}
