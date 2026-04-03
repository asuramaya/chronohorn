"""MCP tool server for Chronohorn runtime observation, forecasting, and control.

All tools read from and write to a ChronohornDB instance — the single source of truth.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from chronohorn.engine.budgets import DEFAULT_GOLF_V1_BUDGET, CompetitionBudget, resolve_competition_budget
from chronohorn.engine.results import load_result_json
from chronohorn.fleet.forecast_results import collect_result_paths

TOOLS = {
    "chronohorn_manifests": {
        "description": "Ingest manifest JSONL files into the Chronohorn database.",
        "parameters": {
            "manifest_paths": {"type": "array", "description": "Manifest JSONL paths", "required": True},
        },
    },
    "chronohorn_runtime_status": {
        "description": "Return runtime summary from the Chronohorn database plus fleet status.",
        "parameters": {},
    },
    "chronohorn_launches": {
        "description": "Return recent launches (dispatched/running jobs) from the Chronohorn database.",
        "parameters": {
            "top_k": {"type": "integer", "description": "Maximum rows (default 20)"},
        },
    },
    "chronohorn_results": {
        "description": "Ingest result JSON files into the Chronohorn database.",
        "parameters": {
            "result_paths": {"type": "array", "description": "Result JSON paths or directories"},
            "result_globs": {"type": "array", "description": "Result JSON globs"},
        },
    },
    "chronohorn_forecast": {
        "description": "Compute and store forecasts for named results (or all unforecasted).",
        "parameters": {
            "names": {"type": "array", "description": "Run names to forecast (empty = auto-forecast all)"},
        },
    },
    "chronohorn_records": {
        "description": "Query filtered result records from the Chronohorn database.",
        "parameters": {
            "kind": {"type": "string", "description": "Filter: 'results', 'jobs', 'probes'"},
            "family": {"type": "string", "description": "Filter by family"},
            "name": {"type": "string", "description": "Filter by run name"},
            "status": {"type": "string", "description": "Filter by job state"},
            "top_k": {"type": "integer", "description": "Maximum records (default 50)"},
        },
    },
    "chronohorn_status": {
        "description": "Return a compact summary of the Chronohorn database: counts, best bpb, pending/running jobs.",
        "parameters": {},
    },
    "chronohorn_frontier": {
        "description": "Return the best non-illegal runs from the Chronohorn database, ranked by bpb.",
        "parameters": {
            "top_k": {"type": "integer", "description": "Maximum rows (default 10)"},
            "family": {"type": "string", "description": "Filter by family"},
            "format": {"type": "string", "description": "'text' for ASCII table, omit for JSON"},
        },
    },
    "chronohorn_control_recommend": {
        "description": "Build a closed-loop control plan with launch, stop, and promotion recommendations.",
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
        "description": "Execute recommended launch actions and optional stop actions.",
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
        "description": "No-op. The Chronohorn database is persistent; use chronohorn_query for ad-hoc cleanup.",
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
        "description": "Return the learning curve (probe data) for a named run from the Chronohorn database.",
        "parameters": {
            "name": {"type": "string", "description": "Run name", "required": True},
            "format": {"type": "string", "description": "'text' for ASCII plot, omit for JSON"},
        },
    },
    "chronohorn_compare": {
        "description": "Compare learning curves of multiple runs side by side from the Chronohorn database.",
        "parameters": {
            "names": {"type": "array", "description": "Run names to compare", "required": True},
        },
    },
    "chronohorn_marginal_rank": {
        "description": "Rank completed runs by marginal bpb gain (slope) from the Chronohorn database.",
        "parameters": {
            "top_k": {"type": "integer", "description": "Maximum results (default 50)"},
            "family": {"type": "string", "description": "Filter by family"},
        },
    },
    "chronohorn_saturation": {
        "description": (
            "Analyze saturation state of a run's learning curve from the Chronohorn database. "
            "Returns: status (learning/decelerating/saturated/overfitting), asymptote estimate, "
            "headroom remaining, doubling gains, learning rate per 1K steps, "
            "estimated saturation step."
        ),
        "parameters": {
            "name": {"type": "string", "description": "Run name to analyze", "required": True},
        },
    },
    "chronohorn_saturation_frontier": {
        "description": (
            "Rank all experiments by forecast asymptote (lowest = most potential) from the Chronohorn database."
        ),
        "parameters": {
            "top_k": {"type": "integer", "description": "Maximum results (default 20)"},
            "family": {"type": "string", "description": "Filter by family"},
        },
    },
    "chronohorn_auto_deepen": {
        "description": "Pick top runs by learning slope, generate deepening jobs in the DB. Optionally dispatch.",
        "parameters": {
            "top_n": {"type": "integer", "description": "Number of top runs to deepen (default 4)"},
            "target_steps": {"type": "integer", "description": "Step count for deepened runs (default 10000)"},
            "dry_run": {"type": "boolean", "description": "If true (default), only preview without creating jobs"},
        },
    },
    "chronohorn_artifact_check": {
        "description": "Check artifact size and 16MB viability for a named run from the Chronohorn database.",
        "parameters": {
            "name": {"type": "string", "description": "Run name", "required": True},
        },
    },
    "chronohorn_subscribe": {
        "description": "Return runs that changed since the last subscribe call (diff tracking via the database).",
        "parameters": {},
    },
    "chronohorn_query": {
        "description": "Run a raw SQL SELECT query against the Chronohorn database. Returns rows as dicts.",
        "parameters": {
            "sql": {"type": "string", "description": "SQL query", "required": True},
        },
    },
    "chronohorn_build_table": {
        "description": "Build an n-gram lookup table from training data shards. Returns the table path and statistics.",
        "parameters": {
            "data_path": {"type": "string", "description": "Path to training data shard (.bin file)", "required": True},
            "output_path": {"type": "string", "description": "Output path for the table (.npz)"},
            "vocab_size": {"type": "integer", "description": "Vocabulary size (default 1024)"},
            "max_order": {"type": "integer", "description": "Maximum n-gram order (default 4)"},
            "bucket_count": {"type": "integer", "description": "Hash bucket count for trigram+ (default 8192)"},
        },
    },
    "chronohorn_events": {
        "description": "Return recent events from the Chronohorn database event log.",
        "parameters": {
            "limit": {"type": "integer", "description": "Maximum events to return (default 30)"},
        },
    },
    "chronohorn_drain_status": {
        "description": "Return drain status: pending, running, and completed job counts from the database.",
        "parameters": {},
    },
    "chronohorn_list_manifests": {
        "description": "Return distinct manifest names from the jobs table.",
        "parameters": {},
    },
    "chronohorn_flag_illegal": {
        "description": "Manually flag a result as illegal or legal. Use when auto-detection missed something.",
        "parameters": {
            "name": {"type": "string", "description": "Result name", "required": True},
            "illegal": {"type": "boolean", "description": "True to flag as illegal, False to clear", "required": True},
        },
    },
    "chronohorn_config_diff": {
        "description": "Compare configs of two runs. Shows what changed, what's unique to each, and metric differences.",
        "parameters": {
            "name1": {"type": "string", "required": True},
            "name2": {"type": "string", "required": True},
        },
    },
    "chronohorn_what_varied": {
        "description": "Find config keys that differ across a set of runs. Shows what was varied in experiments.",
        "parameters": {
            "names": {"type": "array", "description": "Run names to compare (optional, defaults to frontier)"},
            "family": {"type": "string", "description": "Filter by family"},
            "limit": {"type": "integer", "description": "Max runs to analyze (default 50)"},
        },
    },
    "chronohorn_cost": {
        "description": "GPU cost tracking. Total GPU-hours, per-family, per-run, and ROI analysis.",
        "parameters": {
            "name": {"type": "string", "description": "Specific run name (optional, omit for summary)"},
        },
    },
    "chronohorn_terminal_dashboard": {
        "description": "Return a pre-formatted text dashboard with frontier table, top curves, and status. Designed for CLI agents.",
        "parameters": {
            "top_k": {"type": "integer", "description": "Number of frontier entries (default 15)"},
        },
    },
    "chronohorn_changelog": {
        "description": "Show what changed: new results, frontier movement, since a time or N hours ago.",
        "parameters": {
            "hours": {"type": "number", "description": "Hours to look back (default 1)"},
        },
    },
    "chronohorn_journal_write": {
        "description": "Record a hypothesis, conclusion, observation, or decision in the experiment journal.",
        "parameters": {
            "kind": {"type": "string", "description": "hypothesis|conclusion|observation|decision", "required": True},
            "content": {"type": "string", "description": "The journal entry text", "required": True},
            "run_name": {"type": "string", "description": "Optional run name to link to"},
            "tags": {"type": "array", "description": "Optional tags"},
        },
    },
    "chronohorn_journal_read": {
        "description": "Read experiment journal entries.",
        "parameters": {
            "kind": {"type": "string", "description": "Filter by kind"},
            "run_name": {"type": "string", "description": "Filter by run name"},
            "limit": {"type": "integer", "description": "Max entries (default 50)"},
        },
    },
    "chronohorn_predict": {
        "description": "Predict bpb at a target step count using power-law extrapolation from existing probes.",
        "parameters": {
            "name": {"type": "string", "required": True},
            "steps": {"type": "integer", "required": True},
        },
    },
    "chronohorn_prediction_audit": {
        "description": "Audit past predictions against actual results. Shows prediction accuracy.",
        "parameters": {},
    },
    "chronohorn_emit_matrix": {
        "description": "Expand an experiment sweep spec into a manifest. Returns the list of experiments.",
        "parameters": {
            "name_template": {"type": "string", "required": True},
            "base": {"type": "object", "description": "Base config for all experiments", "required": True},
            "sweep": {"type": "object", "description": "Axes to sweep (each key maps to a list of values)", "required": True},
            "output": {"type": "string", "description": "Output manifest JSONL path (optional)"},
        },
    },
    "chronohorn_seed_analysis": {
        "description": "Find runs with different seeds but same config, report mean/std/CI.",
        "parameters": {},
    },
    "chronohorn_interpret": {
        "description": "Interpret a run: why is it good/bad, compared to neighbors, what to try next.",
        "parameters": {
            "name": {"type": "string", "required": True},
        },
    },
    "chronohorn_frontier_velocity": {
        "description": "Frontier improvement rate and trend.",
        "parameters": {},
    },
    "chronohorn_branch_health": {
        "description": "Check if an architecture branch is dead.",
        "parameters": {
            "prefix": {"type": "string", "required": True},
        },
    },
    "chronohorn_experiment_groups": {
        "description": "Detect and summarize experiment groups.",
        "parameters": {},
    },
    "chronohorn_suggest_next": {
        "description": "Suggest the highest-value next experiments based on frontier state, axis exhaustion, and untested axes.",
        "parameters": {},
    },
    "chronohorn_axis_analysis": {
        "description": "Analyze diminishing returns across all experimental axes. Shows which axes are alive, diminishing, or exhausted.",
        "parameters": {},
    },
    "chronohorn_architecture_boundary": {
        "description": "Detect if the current architecture class has hit its ceiling relative to the competition.",
        "parameters": {},
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


class ToolServer:
    def __init__(self, *, db=None) -> None:
        self._shared_db = db or self._auto_open_db()
        self._last_seen_results: set[str] = set()

    @staticmethod
    def _auto_open_db():
        from chronohorn.db import ChronohornDB
        db_path = Path("out/chronohorn.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return ChronohornDB(str(db_path))

    @property
    def db(self):
        return self._shared_db

    def list_tools(self) -> list[dict[str, Any]]:
        return [{"name": name, **definition} for name, definition in TOOLS.items()]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        dispatch = {
            "chronohorn_manifests": self._do_manifests,
            "chronohorn_runtime_status": self._do_runtime_status,
            "chronohorn_launches": self._do_launches,
            "chronohorn_results": self._do_results,
            "chronohorn_forecast": self._do_forecast,
            "chronohorn_records": self._do_records,
            "chronohorn_status": self._do_status,
            "chronohorn_frontier": self._do_frontier,
            "chronohorn_control_recommend": self._do_control_recommend,
            "chronohorn_control_act": self._do_control_act,
            "chronohorn_reset": self._do_reset,
            "chronohorn_fleet_dispatch": self._do_fleet_dispatch,
            "chronohorn_fleet_drain_tick": self._do_fleet_drain_tick,
            "chronohorn_fleet_status": self._do_fleet_status,
            "chronohorn_learning_curve": self._do_learning_curve,
            "chronohorn_compare": self._do_compare,
            "chronohorn_marginal_rank": self._do_marginal_rank,
            "chronohorn_saturation": self._do_saturation,
            "chronohorn_saturation_frontier": self._do_saturation_frontier,
            "chronohorn_auto_deepen": self._do_auto_deepen,
            "chronohorn_artifact_check": self._do_artifact_check,
            "chronohorn_subscribe": self._do_subscribe,
            "chronohorn_query": self._do_query,
            "chronohorn_build_table": self._do_build_table,
            "chronohorn_events": self._do_events,
            "chronohorn_drain_status": self._do_drain_status,
            "chronohorn_list_manifests": self._do_list_manifests,
            "chronohorn_flag_illegal": self._do_flag_illegal,
            "chronohorn_config_diff": self._do_config_diff,
            "chronohorn_what_varied": self._do_what_varied,
            "chronohorn_cost": self._do_cost,
            "chronohorn_terminal_dashboard": self._do_terminal_dashboard,
            "chronohorn_changelog": self._do_changelog,
            "chronohorn_journal_write": self._do_journal_write,
            "chronohorn_journal_read": self._do_journal_read,
            "chronohorn_predict": self._do_predict,
            "chronohorn_prediction_audit": self._do_prediction_audit,
            "chronohorn_emit_matrix": self._do_emit_matrix,
            "chronohorn_seed_analysis": self._do_seed_analysis,
            "chronohorn_interpret": self._do_interpret,
            "chronohorn_frontier_velocity": self._do_frontier_velocity,
            "chronohorn_branch_health": self._do_branch_health,
            "chronohorn_experiment_groups": self._do_experiment_groups,
            "chronohorn_suggest_next": self._do_suggest_next,
            "chronohorn_axis_analysis": self._do_axis_analysis,
            "chronohorn_architecture_boundary": self._do_architecture_boundary,
        }
        handler = dispatch.get(name)
        if handler is None:
            return {"error": f"Unknown tool: {name}"}
        return handler(arguments)

    # -- tool implementations --------------------------------------------------

    def _do_manifests(self, args: dict[str, Any]) -> dict[str, Any]:
        paths = list(args.get("manifest_paths") or [])
        total = 0
        errors = []
        for path in paths:
            try:
                total += self._shared_db.ingest_manifest(path)
            except FileNotFoundError:
                errors.append(f"not found: {path}")
            except Exception as exc:
                errors.append(f"{path}: {exc}")
        result = {"ingested": total, "manifests": len(paths)}
        if errors:
            result["errors"] = errors
        return result

    def _do_results(self, args: dict[str, Any]) -> dict[str, Any]:
        count = 0
        for path in collect_result_paths(list(args.get("result_paths") or []), list(args.get("result_globs") or [])):
            try:
                payload = load_result_json(path)
                name = Path(path).stem
                self._shared_db.record_result(name, payload, json_archive=str(path))
                count += 1
            except Exception:
                pass
        return {"ingested": count}

    def _do_status(self, args: dict[str, Any]) -> dict[str, Any]:
        db = self._shared_db
        result_count = db.result_count()
        best = db.best_bpb()
        pending = db.pending_jobs()
        running = db.running_jobs()
        return {
            "result_count": result_count,
            "best_bpb": best,
            "pending_jobs": len(pending),
            "running_jobs": len(running),
        }

    def _do_frontier(self, args: dict[str, Any]) -> dict[str, Any]:
        top_k = int(args["top_k"]) if args.get("top_k") is not None else 10
        family = args.get("family")
        if family:
            rows = self._shared_db.query("""
                SELECT r.name, r.bpb, r.family, r.train_bpb, r.overfit_pct,
                       r.tok_s, r.wall_sec, r.slope, r.illegal,
                       COALESCE(r.params, c.params) as params,
                       COALESCE(r.int6_mb, c.int6_mb) as int6_mb,
                       COALESCE(r.steps, j.steps) as steps,
                       COALESCE(r.seq_len, c.seq_len) as seq_len,
                       COALESCE(r.tflops, r.total_tflops) as tflops,
                       COALESCE(f.asymptote, f.forecast_bpb) as fc_bpb,
                       COALESCE(f.asymptote_r2, f.r2) as fc_r2,
                       f.headroom, f.saturation_status,
                       c.json_blob as config_json
                FROM results r
                LEFT JOIN configs c ON r.config_id = c.id
                LEFT JOIN forecasts f ON r.name = f.name
                LEFT JOIN jobs j ON r.name = j.name
                WHERE NOT r.illegal AND r.family = ?
                ORDER BY r.bpb LIMIT ?
            """, (family, top_k))
        else:
            rows = self._shared_db.frontier(top_k)
        if args.get("format") == "text":
            from chronohorn.observe.terminal import ascii_frontier_table
            return {"text": ascii_frontier_table(rows, top_k=top_k)}
        return {"frontier": rows, "count": len(rows)}

    def _do_learning_curve(self, args: dict[str, Any]) -> dict[str, Any]:
        name = str(args["name"])
        points = self._shared_db.learning_curve(name)
        if args.get("format") == "text":
            from chronohorn.observe.terminal import ascii_learning_curve
            return {"name": name, "text": ascii_learning_curve(points)}
        return {"name": name, "points": points}

    def _do_compare(self, args: dict[str, Any]) -> dict[str, Any]:
        names = list(args["names"])
        curves = self._shared_db.compare_curves(names)
        return {"runs": [{"name": n, "points": p} for n, p in curves.items()]}

    def _do_marginal_rank(self, args: dict[str, Any]) -> dict[str, Any]:
        top_k = int(args.get("top_k") or 50)
        family = args.get("family")
        if family:
            rows = self._shared_db.query("""
                SELECT r.name, r.bpb, r.family, r.slope,
                       COALESCE(r.tflops, r.total_tflops) as total_tf,
                       r.steps, r.illegal,
                       COALESCE(f.asymptote, f.forecast_bpb) as fc_bpb
                FROM results r
                LEFT JOIN forecasts f ON r.name = f.name
                WHERE NOT r.illegal AND r.slope IS NOT NULL AND r.slope > 0
                      AND r.family = ?
                ORDER BY r.slope DESC LIMIT ?
            """, (family, top_k))
            result = []
            for r in rows:
                d = dict(r)
                d["marginal"] = d["slope"]
                d["slope_alive"] = (d["slope"] or 0) > 0.001
                result.append(d)
            return {"ranked": result}
        return {"ranked": self._shared_db.marginal_rank(top_k)}

    def _do_saturation(self, args: dict[str, Any]) -> dict[str, Any]:
        from chronohorn.engine.saturation import format_saturation_summary
        name = str(args["name"])
        analysis = self._shared_db.saturation(name)
        analysis["summary"] = format_saturation_summary(analysis)
        return analysis

    def _do_saturation_frontier(self, args: dict[str, Any]) -> dict[str, Any]:
        top_k = int(args.get("top_k") or 20)
        family = args.get("family")
        if family:
            rows = self._shared_db.query("""
                SELECT f.name, f.forecast_bpb, f.asymptote, f.headroom,
                       f.last_doubling_gain, f.last_rate_per_1k, f.saturation_status,
                       f.saturation_step, f.asymptote_stability,
                       r.bpb, r.total_tflops
                FROM forecasts f
                JOIN results r ON f.name = r.name
                WHERE f.asymptote IS NOT NULL AND NOT r.illegal
                      AND r.family = ?
                ORDER BY f.asymptote ASC LIMIT ?
            """, (family, top_k))
        else:
            rows = self._shared_db.saturation_frontier(top_k)
        return {"frontier": rows}

    def _do_auto_deepen(self, args: dict[str, Any]) -> dict[str, Any]:
        import re
        db = self._shared_db
        top_n = int(args["top_n"]) if args.get("top_n") is not None else 4
        target_steps = int(args.get("target_steps") or 10000)
        dry_run = bool(args.get("dry_run", True))

        if top_n <= 0:
            return {"deepened": [], "count": 0, "target_steps": target_steps, "dry_run": dry_run}

        # Get top runs by marginal gain that haven't been deepened yet
        candidates = db.query("""
            SELECT r.name, r.bpb, r.slope, r.steps, j.command, j.config_id
            FROM results r
            LEFT JOIN jobs j ON r.name = j.name
            WHERE r.slope > 0.005 AND r.steps IS NOT NULL AND r.steps < ?
                AND NOT r.illegal
                AND r.name NOT IN (
                    SELECT parent FROM jobs WHERE parent IS NOT NULL AND parent != ''
                )
            ORDER BY r.slope DESC
            LIMIT ?
        """, (target_steps, top_n))

        deepened = []
        for row in candidates:
            parent_name = row["name"]
            child_name = f"{parent_name}-s{target_steps}"

            # Skip if child already exists
            existing = db.query("SELECT name FROM jobs WHERE name = ?", (child_name,))
            if existing:
                continue

            parent_cmd = row.get("command") or ""
            if not parent_cmd:
                deepened.append({"name": child_name, "parent": parent_name, "error": "no parent command"})
                continue

            # Update command: replace --steps and --json
            new_cmd = re.sub(r"(?<!\w)--steps\s+\d+", f"--steps {target_steps}", parent_cmd)
            new_cmd = re.sub(r'--json\s+(?:"[^"]+"|\\S+)', f"--json /run/results/{child_name}.json", new_cmd)

            entry = {
                "name": child_name,
                "parent": parent_name,
                "parent_bpb": row["bpb"],
                "parent_slope": row["slope"],
                "target_steps": target_steps,
                "command": new_cmd,
            }

            if not dry_run:
                try:
                    # Get parent config
                    parent_cfg = {}
                    if row.get("config_id"):
                        cfg_rows = db.query("SELECT json_blob FROM configs WHERE id = ?", (row["config_id"],))
                        if cfg_rows and cfg_rows[0].get("json_blob"):
                            parent_cfg = json.loads(cfg_rows[0]["json_blob"])

                    db.record_job(
                        child_name,
                        parent=parent_name,
                        config=parent_cfg,
                        steps=target_steps,
                        command=new_cmd,
                    )
                    entry["status"] = "created"
                except Exception as exc:
                    entry["status"] = f"error: {exc}"
            else:
                entry["status"] = "dry_run"

            deepened.append(entry)

        return {
            "deepened": deepened,
            "count": len(deepened),
            "target_steps": target_steps,
            "dry_run": dry_run,
        }

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
        try:
            return self._do_fleet_dispatch({**args, "dry_run": True})
        except Exception as exc:
            return {"error": str(exc)}

    def _do_forecast(self, args: dict[str, Any]) -> dict[str, Any]:
        names = list(args.get("names") or [])
        if not names:
            # Auto-forecast all results that lack a forecast
            rows = self._shared_db.query("""
                SELECT r.name FROM results r
                LEFT JOIN forecasts f ON r.name = f.name
                WHERE f.name IS NULL
            """)
            names = [r["name"] for r in rows]
        forecasted = []
        skipped = []
        errors = []
        for name in names:
            try:
                self._shared_db.compute_and_store_forecast(name)
                # Check if forecast was actually created
                stored = self._shared_db.query("SELECT name FROM forecasts WHERE name = ?", (name,))
                if stored:
                    forecasted.append(name)
                else:
                    skipped.append(name)
            except Exception as exc:
                errors.append({"name": name, "error": str(exc)})
        return {"forecasted": forecasted, "skipped": skipped, "errors": errors, "count": len(forecasted)}

    def _do_records(self, args: dict[str, Any]) -> dict[str, Any]:
        top_k = int(args.get("top_k") or 50)
        kind = args.get("kind", "results")
        family = args.get("family")
        name = args.get("name")
        status = args.get("status")

        if kind == "jobs":
            clauses, params = ["1=1"], []
            if name:
                clauses.append("name LIKE ?")
                params.append(f"%{name}%")
            if status:
                clauses.append("state = ?")
                params.append(status)
            params.append(top_k)
            rows = self._shared_db.query(
                f"SELECT * FROM jobs WHERE {' AND '.join(clauses)} ORDER BY rowid DESC LIMIT ?",
                tuple(params),
            )
        elif kind == "probes":
            if not name:
                return {"error": "name is required for probes"}
            rows = self._shared_db.query(
                "SELECT * FROM probes WHERE name = ? ORDER BY step LIMIT ?",
                (name, top_k),
            )
        else:
            # Default: results
            clauses, params = ["1=1"], []
            if family:
                clauses.append("r.family = ?")
                params.append(family)
            if name:
                clauses.append("r.name LIKE ?")
                params.append(f"%{name}%")
            params.append(top_k)
            rows = self._shared_db.query(
                f"SELECT r.* FROM results r WHERE {' AND '.join(clauses)} ORDER BY r.bpb LIMIT ?",
                tuple(params),
            )
        return {"count": len(rows), "records": rows}

    def _do_control_recommend(self, args: dict[str, Any]) -> dict[str, Any]:
        try:
            from chronohorn.control.policy import build_control_plan
            from chronohorn.pipeline import normalize_runtime_config
            config = normalize_runtime_config({
                "manifest_paths": list(args.get("manifest_paths") or []),
                "state_paths": list(args.get("state_paths") or []),
                "launch_globs": list(args.get("launch_globs") or []),
                "result_paths": list(args.get("result_paths") or []),
                "result_globs": list(args.get("result_globs") or []),
                "probe_runtime": bool(args.get("probe_runtime", False)),
                "budget_name": str(args.get("budget_name") or DEFAULT_GOLF_V1_BUDGET.name),
                "train_tflops_budget": float(args.get("train_tflops_budget") or DEFAULT_GOLF_V1_BUDGET.train_tflops_budget),
                "artifact_limit_mb": float(args.get("artifact_limit_mb") or DEFAULT_GOLF_V1_BUDGET.artifact_limit_mb),
            })
            plan = build_control_plan(
                config,
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
        except ImportError:
            return {
                "error": "Pipeline dependencies not available. Use chronohorn_frontier and chronohorn_auto_deepen instead.",
                "frontier": self._shared_db.frontier(10),
            }
        except Exception as exc:
            return {"error": f"control_recommend failed: {exc}"}

    def _do_control_act(self, args: dict[str, Any]) -> dict[str, Any]:
        try:
            from chronohorn.control.actions import execute_control_actions
            from chronohorn.control.models import ControlAction
            from chronohorn.control.policy import build_control_plan
            from chronohorn.pipeline import normalize_runtime_config
            config = normalize_runtime_config({
                "manifest_paths": list(args.get("manifest_paths") or []),
                "state_paths": list(args.get("state_paths") or []),
                "launch_globs": list(args.get("launch_globs") or []),
                "result_paths": list(args.get("result_paths") or []),
                "result_globs": list(args.get("result_globs") or []),
                "probe_runtime": bool(args.get("probe_runtime", False)),
                "budget_name": str(args.get("budget_name") or DEFAULT_GOLF_V1_BUDGET.name),
                "train_tflops_budget": float(args.get("train_tflops_budget") or DEFAULT_GOLF_V1_BUDGET.train_tflops_budget),
                "artifact_limit_mb": float(args.get("artifact_limit_mb") or DEFAULT_GOLF_V1_BUDGET.artifact_limit_mb),
            })
            plan = build_control_plan(
                config,
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
        except ImportError:
            return {
                "error": "Pipeline dependencies not available. Use chronohorn_frontier and chronohorn_auto_deepen instead.",
                "frontier": self._shared_db.frontier(10),
            }
        except Exception as exc:
            return {"error": f"control_act failed: {exc}"}

    def _do_reset(self, args: dict[str, Any]) -> dict[str, Any]:
        return {"status": "no-op", "message": "Database is persistent. Use chronohorn_query for ad-hoc cleanup."}

    def _do_artifact_check(self, args: dict[str, Any]) -> dict[str, Any]:
        name = str(args["name"])
        row = self._shared_db.query(
            "SELECT r.*, c.params, c.int6_mb, c.scale, c.readout"
            " FROM results r LEFT JOIN configs c ON r.config_id = c.id"
            " WHERE r.name = ?",
            (name,),
        )
        if row:
            r = row[0]
            params = r.get("params")
            int6_mb = r.get("int6_mb")
            if int6_mb is None and params:
                int6_mb = round(float(params) * 6 / 8 / 1024 / 1024, 2)
            return {
                "name": name,
                "params": params,
                "int6_mb": int6_mb,
                "fits_16mb": (int6_mb or 99) <= 16,
                "bpb": r.get("bpb"),
                "scale": r.get("scale"),
            }
        return {"error": f"Run not found in database: {name}"}

    def _do_subscribe(self, args: dict[str, Any]) -> dict[str, Any]:
        current = set(r["name"] for r in self._shared_db.query("SELECT name FROM results"))
        new = sorted(current - self._last_seen_results)
        removed = sorted(self._last_seen_results - current)
        self._last_seen_results = current
        return {"new": new, "removed": removed, "total": len(current)}

    def _do_query(self, args: dict[str, Any]) -> dict[str, Any]:
        try:
            sql = str(args["sql"])
            if not sql.strip().upper().startswith("SELECT"):
                return {"error": "Only SELECT queries are allowed"}
            rows = self._shared_db.query(sql)
            return {"rows": rows, "count": len(rows)}
        except Exception as exc:
            return {"error": str(exc)}

    def _do_build_table(self, args: dict[str, Any]) -> dict[str, Any]:
        import numpy as np
        from chronohorn.models.ngram_table import NgramTable

        data_path = str(args["data_path"])
        output_path = str(args.get("output_path", "out/ngram_table.npz"))
        vocab_size = int(args.get("vocab_size", 1024))
        max_order = int(args.get("max_order", 4))
        bucket_count = int(args.get("bucket_count", 8192))

        try:
            tokens = np.fromfile(data_path, dtype=np.uint16)
            table = NgramTable(vocab_size=vocab_size, max_order=max_order, bucket_count=bucket_count)
            table.build_from_tokens(tokens)

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            table.save(output_path)

            size_mb = Path(output_path).stat().st_size / 1024 / 1024

            return {
                "path": output_path,
                "tokens": len(tokens),
                "size_mb": round(size_mb, 2),
                "unigram_nonzero": int((table.unigram > 0).sum()),
                "bigram_nonzero": int((table.bigram > 0).sum()),
                "trigram_nonzero": int((table.trigram > 0).sum()),
            }
        except Exception as exc:
            return {"error": str(exc)}

    def _do_runtime_status(self, args: dict[str, Any]) -> dict[str, Any]:
        db = self._shared_db
        status = self._do_status(args)
        try:
            fleet = db.fleet_latest()
        except Exception:
            fleet = {}
        status["fleet"] = fleet
        return status

    def _do_launches(self, args: dict[str, Any]) -> dict[str, Any]:
        top_k = int(args.get("top_k") or 20)
        rows = self._shared_db.query(
            "SELECT * FROM jobs WHERE state IN ('dispatched', 'running') ORDER BY launched_at DESC LIMIT ?",
            (top_k,),
        )
        return {"launches": rows, "count": len(rows)}

    # -- new tools -------------------------------------------------------------

    def _do_events(self, args: dict[str, Any]) -> dict[str, Any]:
        limit = int(args["limit"]) if args.get("limit") is not None else 30
        events = self._shared_db.events_recent(limit)
        return {"events": events, "count": len(events)}

    def _do_drain_status(self, args: dict[str, Any]) -> dict[str, Any]:
        pending = self._shared_db.pending_jobs()
        running = self._shared_db.running_jobs()
        completed = self._shared_db.query(
            "SELECT COUNT(*) as cnt FROM jobs WHERE state = 'completed'"
        )
        completed_count = completed[0]["cnt"] if completed else 0
        return {
            "pending": len(pending),
            "running": len(running),
            "completed": completed_count,
            "total": len(pending) + len(running) + completed_count,
        }

    def _do_list_manifests(self, args: dict[str, Any]) -> dict[str, Any]:
        rows = self._shared_db.query(
            "SELECT DISTINCT manifest, COUNT(*) as job_count FROM jobs GROUP BY manifest ORDER BY manifest"
        )
        return {"manifests": rows}

    def _do_flag_illegal(self, args: dict[str, Any]) -> dict[str, Any]:
        name = str(args["name"])
        illegal = bool(args["illegal"])
        self._shared_db._write(
            "UPDATE results SET illegal = ? WHERE name = ?",
            (int(illegal), name),
            wait=True,
        )
        return {"name": name, "illegal": illegal, "status": "updated"}

    def _do_config_diff(self, args: dict[str, Any]) -> dict[str, Any]:
        return self._shared_db.config_diff(str(args["name1"]), str(args["name2"]))

    def _do_what_varied(self, args: dict[str, Any]) -> dict[str, Any]:
        names = list(args["names"]) if args.get("names") else None
        family = args.get("family")
        limit = int(args["limit"]) if args.get("limit") is not None else 50
        return self._shared_db.what_varied(names=names, family=family, limit=limit)

    def _do_cost(self, args: dict[str, Any]) -> dict[str, Any]:
        name = args.get("name")
        if name:
            return self._shared_db.cost_per_run(str(name))
        return self._shared_db.cost_summary()

    def _do_terminal_dashboard(self, args: dict[str, Any]) -> dict[str, Any]:
        from chronohorn.observe.terminal import ascii_frontier_table, ascii_status, ascii_sparkline
        top_k = int(args["top_k"]) if args.get("top_k") is not None else 15

        summary_data = {
            "result_count": self._shared_db.result_count(),
            "best_bpb": self._shared_db.best_bpb(),
        }
        fam_rows = self._shared_db.query(
            "SELECT family, COUNT(*) as c FROM results GROUP BY family"
        )
        summary_data["families"] = {r["family"]: r["c"] for r in fam_rows} if fam_rows else {}

        board = self._shared_db.frontier(top_k)
        text = ascii_status(summary_data, board)
        return {"text": text}

    # -- W7: changelog ---------------------------------------------------------

    def _do_changelog(self, args: dict[str, Any]) -> dict[str, Any]:
        hours = float(args["hours"]) if args.get("hours") is not None else 1
        return self._shared_db.changelog(since_hours=hours)

    # -- W12: experiment journal -----------------------------------------------

    def _do_journal_write(self, args: dict[str, Any]) -> dict[str, Any]:
        kind = str(args["kind"])
        content = str(args["content"])
        run_name = args.get("run_name")
        tags = list(args["tags"]) if args.get("tags") else None
        self._shared_db.record_journal(kind, content, run_name=run_name, tags=tags)
        return {"status": "recorded", "kind": kind}

    def _do_journal_read(self, args: dict[str, Any]) -> dict[str, Any]:
        kind = args.get("kind")
        run_name = args.get("run_name")
        limit = int(args["limit"]) if args.get("limit") is not None else 50
        entries = self._shared_db.journal_entries(kind=kind, run_name=run_name, limit=limit)
        return {"entries": entries, "count": len(entries)}

    # -- W8: prediction + audit ------------------------------------------------

    def _do_predict(self, args: dict[str, Any]) -> dict[str, Any]:
        name = str(args["name"])
        steps = int(args["steps"])
        return self._shared_db.predict_at_steps(name, steps)

    def _do_prediction_audit(self, args: dict[str, Any]) -> dict[str, Any]:
        results = self._shared_db.audit_predictions()
        return {"predictions": results, "count": len(results)}

    # -- W3: experiment matrix ---------------------------------------------------

    def _do_emit_matrix(self, args: dict[str, Any]) -> dict[str, Any]:
        from chronohorn.fleet.experiment_matrix import expand_matrix, matrix_to_commands, write_manifest, estimate_cost

        spec = {
            "name_template": str(args["name_template"]),
            "base": dict(args.get("base") or {}),
            "sweep": dict(args.get("sweep") or {}),
        }
        experiments = expand_matrix(spec)
        commands = matrix_to_commands(experiments)
        cost = estimate_cost(experiments)

        # Collect warnings from checked experiments
        all_warnings = {exp["name"]: exp["_warnings"] for exp in experiments if exp.get("_warnings")}

        output = args.get("output")
        if output:
            from pathlib import Path
            count = write_manifest(commands, Path(output))
            return {"experiments": commands, "count": len(commands), "written": count, "path": output,
                    "cost": cost, "warnings": all_warnings}

        return {"experiments": commands, "count": len(commands), "cost": cost, "warnings": all_warnings}

    # -- W13: seed analysis / error bars -----------------------------------------

    def _do_seed_analysis(self, args: dict[str, Any]) -> dict[str, Any]:
        groups = self._shared_db.seed_groups()
        return {"groups": groups, "count": len(groups)}

    # -- W16: interpretive layer -------------------------------------------------

    def _do_interpret(self, args: dict[str, Any]) -> dict[str, Any]:
        name = str(args["name"])
        db = self._shared_db

        # Get this run's data
        run_rows = db.query("SELECT * FROM results WHERE name = ?", (name,))
        if not run_rows:
            return {"error": f"run not found: {name}"}
        run = run_rows[0]

        # Get frontier context
        frontier = db.frontier(20)
        rank = next((i + 1 for i, r in enumerate(frontier) if r["name"] == name), None)

        # Get nearest neighbor by bpb
        neighbors = db.query(
            "SELECT name, bpb, family, params, steps, tok_s FROM results "
            "WHERE NOT illegal AND name != ? ORDER BY ABS(bpb - ?) LIMIT 3",
            (name, run["bpb"]),
        )

        # Config diff with best neighbor
        diff = None
        if neighbors:
            diff = db.config_diff(name, neighbors[0]["name"])

        # Saturation
        sat = {}
        if hasattr(db, "saturation"):
            try:
                sat = db.saturation(name)
            except Exception:
                pass

        # Cost
        cost = db.cost_per_run(name)

        # Learning curve shape
        probes = db.learning_curve(name)
        slope_str = (
            "improving"
            if run.get("slope") and run["slope"] > 0.005
            else "plateau"
            if run.get("slope")
            else "unknown"
        )

        # Build interpretation
        interpretation = {
            "name": name,
            "bpb": run["bpb"],
            "rank": rank,
            "rank_total": len(frontier),
            "family": run.get("family", "unknown"),
            "slope_status": slope_str,
            "saturation": sat.get("status", "unknown"),
            "cost": cost,
            "neighbors": [
                {"name": n["name"], "bpb": n["bpb"], "delta": round(n["bpb"] - run["bpb"], 4)}
                for n in neighbors
            ],
            "config_diff_with_nearest": diff.get("changed", {}) if diff else {},
            "probes": len(probes),
        }

        # Check if asymptote is unreliable
        if sat.get("asymptote") and not sat.get("asymptote_reliable", True):
            interpretation["headroom_note"] = "asymptote unreliable (curve still descending)"

        # Generate suggestions
        suggestions = []
        if slope_str == "improving" and (run.get("steps") or 0) < 50000:
            suggestions.append(
                f"Still learning (slope={run.get('slope', 0):.3f}). "
                f"Train longer to {(run.get('steps', 10000) or 10000) * 2} steps."
            )
        if sat.get("headroom") and sat["headroom"] > 0.05 and sat.get("asymptote_reliable", True):
            suggestions.append(f"Headroom to asymptote: {sat['headroom']:.3f} bpb. Worth deepening.")
        elif sat.get("headroom") and sat["headroom"] > 0.05:
            suggestions.append(f"Headroom to asymptote: {sat['headroom']:.3f} bpb (but asymptote unreliable).")
        if diff and diff.get("changed"):
            changed_keys = list(diff["changed"].keys())[:3]
            suggestions.append(f"Differs from nearest neighbor in: {', '.join(changed_keys)}")

        interpretation["suggestions"] = suggestions
        return interpretation

    # -- monitors ---------------------------------------------------------------

    def _do_frontier_velocity(self, args: dict[str, Any]) -> dict[str, Any]:
        return self._shared_db.frontier_velocity()

    def _do_branch_health(self, args: dict[str, Any]) -> dict[str, Any]:
        prefix = str(args["prefix"])
        return self._shared_db.branch_health(prefix)

    def _do_experiment_groups(self, args: dict[str, Any]) -> dict[str, Any]:
        groups = self._shared_db.detect_groups()
        return {"groups": groups, "count": len(groups)}

    # -- advisors ---------------------------------------------------------------

    def _do_suggest_next(self, args: dict[str, Any]) -> dict[str, Any]:
        from chronohorn.engine.advisor import suggest_next, format_suggestions
        suggestions = suggest_next(self._shared_db)
        return {"suggestions": suggestions, "text": format_suggestions(suggestions)}

    def _do_axis_analysis(self, args: dict[str, Any]) -> dict[str, Any]:
        from chronohorn.engine.axis_analysis import analyze_axes
        results = self._shared_db.query("""
            SELECT r.name, r.bpb, r.steps, c.json_blob as config
            FROM results r LEFT JOIN configs c ON r.config_id = c.id
            WHERE NOT r.illegal AND r.bpb < 3
        """)
        for r in results:
            try:
                r["config"] = json.loads(r["config"]) if r["config"] else {}
            except Exception:
                r["config"] = {}
        return {"axes": analyze_axes(results)}

    def _do_architecture_boundary(self, args: dict[str, Any]) -> dict[str, Any]:
        from chronohorn.engine.advisor import architecture_boundary
        return architecture_boundary(self._shared_db)
