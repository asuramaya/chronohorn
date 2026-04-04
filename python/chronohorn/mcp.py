"""MCP tool server for Chronohorn runtime observation, forecasting, and control.

All tools read from and write to a ChronohornDB instance — the single source of truth.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Lazy imports — these are only needed inside tool handler methods, not at module load.
# from chronohorn.engine.budgets import DEFAULT_GOLF_V1_BUDGET, CompetitionBudget, resolve_competition_budget
# from chronohorn.engine.results import load_result_json
# from chronohorn.fleet.forecast_results import collect_result_paths

RESULT_POPULATIONS = {"controlled", "imported_archive", "unknown", "all"}
RESULT_LEGALITY = {"legal", "illegal", "all"}
RESULT_TRUST = {"admissible", "provisional", "quarantined", "all"}

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
        "description": "Return a compact summary of the Chronohorn database, including population counts, best bpb, and trust-state counts.",
        "parameters": {},
    },
    "chronohorn_frontier": {
        "description": "Return the best runs from a selected result population, ranked by bpb, with explicit trust annotations. Defaults to legal controlled runs.",
        "parameters": {
            "top_k": {"type": "integer", "description": "Maximum rows (default 10)"},
            "family": {"type": "string", "description": "Filter by family"},
            "population": {"type": "string", "description": "'controlled', 'imported_archive', 'unknown', or 'all'"},
            "legality": {"type": "string", "description": "'legal' (default), 'illegal', or 'all'"},
            "trust": {"type": "string", "description": "'admissible', 'provisional', 'quarantined', or 'all' (default)"},
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
        "description": "Probe remote fleet hosts: GPU utilization, docker containers, running processes, remote result counts. No manifest required.",
        "parameters": {
            "hosts": {"type": "array", "description": "Optional host list (default slop fleet hosts)"},
        },
    },
    "chronohorn_fleet_converge": {
        "description": "Plan convergence training: find the best config, predict bpb at target steps, estimate GPU-hours. Advisory only — does not launch.",
        "parameters": {
            "name": {"type": "string", "description": "Run name to converge (default: current best)"},
            "steps": {"type": "integer", "description": "Target steps (default 200000)"},
        },
    },
    "chronohorn_fleet_hosts": {
        "description": "Probe remote fleet hosts through the canonical host-state layer. Returns host liveness, GPU status, docker containers, waiters, and optional top host processes.",
        "parameters": {
            "hosts": {"type": "array", "description": "Optional host list (default slop fleet hosts)"},
            "include_processes": {"type": "boolean", "description": "Include top host processes"},
            "process_limit": {"type": "integer", "description": "Maximum top processes per host (default 8)"},
            "include_remote_results": {"type": "boolean", "description": "Include remote result archive counts"},
            "remote_result_dir": {"type": "string", "description": "Remote result directory for archive counting"},
        },
    },
    "chronohorn_remote_runs": {
        "description": "Inspect DB-backed Chronohorn jobs against live executor state. Replaces ad hoc ssh status checks for running, completed, stale, and pending remote jobs across legacy Docker hosts and Kubernetes jobs.",
        "parameters": {
            "hosts": {"type": "array", "description": "Optional host filter"},
            "job_names": {"type": "array", "description": "Optional job-name filter"},
            "classes": {"type": "array", "description": "Optional resource-class filter"},
            "manifest": {"type": "string", "description": "Optional manifest name or path filter"},
            "include_logs": {"type": "boolean", "description": "Include log tail text for running jobs"},
            "relaunch_completed": {"type": "boolean", "description": "Treat completed jobs as relaunch-eligible while inspecting"},
        },
    },
    "chronohorn_k8s_submit_job": {
        "description": "Submit a DB-backed job through the Kubernetes executor and persist k8s runtime identity into the DB.",
        "parameters": {
            "name": {"type": "string", "description": "Run name", "required": True},
        },
    },
    "chronohorn_k8s_job_status": {
        "description": "Return Kubernetes runtime status for a DB-backed Chronohorn job.",
        "parameters": {
            "name": {"type": "string", "description": "Run name", "required": True},
            "include_logs": {"type": "boolean", "description": "Include pod log tail text"},
            "tail_lines": {"type": "integer", "description": "Maximum log lines to tail (default 64)"},
        },
    },
    "chronohorn_k8s_job_logs": {
        "description": "Return Kubernetes pod logs for a DB-backed Chronohorn job.",
        "parameters": {
            "name": {"type": "string", "description": "Run name", "required": True},
            "tail_lines": {"type": "integer", "description": "Maximum log lines to tail (default 64)"},
        },
    },
    "chronohorn_k8s_delete_job": {
        "description": "Delete a Kubernetes-backed Chronohorn job through the control plane and mark the DB state explicitly.",
        "parameters": {
            "name": {"type": "string", "description": "Run name", "required": True},
            "post_state": {"type": "string", "description": "'failed' (default) or 'pending' after delete"},
        },
    },
    "chronohorn_stop_run": {
        "description": "Stop a running Chronohorn job through the control plane using DB and launch-record identity, then mark its DB state explicitly.",
        "parameters": {
            "name": {"type": "string", "description": "Run name", "required": True},
            "post_state": {"type": "string", "description": "'failed' (default) or 'pending' after stop"},
        },
    },
    "chronohorn_fleet_pull": {
        "description": "Pull new result JSONs from remote fleet hosts via SSH/scp and ingest into the DB. Returns pulled count and new results.",
        "parameters": {
            "hosts": {"type": "array", "description": "Optional host list (default slop fleet hosts)"},
            "remote_dir": {"type": "string", "description": "Remote result directory (default /data/chronohorn/out/results)"},
        },
    },
    "chronohorn_fleet_sync": {
        "description": "Full fleet sync: pull results, probe hosts, show frontier and changelog. Equivalent to `fleet sync` CLI.",
        "parameters": {
            "hosts": {"type": "array", "description": "Optional host list (default slop fleet hosts)"},
        },
    },
    "chronohorn_fleet_launch": {
        "description": "Launch training on a remote GPU host. Syncs code, builds docker/k8s container, runs trainer with pass-through args.",
        "parameters": {
            "script": {"type": "string", "description": "Training script path (e.g. scripts/train_polyhash.py)", "required": True},
            "host": {"type": "string", "description": "Remote host (e.g. slop-01). Omit for auto-placement."},
            "arch": {"type": "string", "description": "Architecture version (e.g. v12)", "required": True},
            "name": {"type": "string", "description": "Result name (required for single-seed)"},
            "steps": {"type": "integer", "description": "Training steps (default 10000)"},
            "seed": {"type": "integer", "description": "Seed for single run (default 42)"},
            "single": {"type": "boolean", "description": "Force single-seed run"},
            "fp16": {"type": "boolean", "description": "Mixed precision training"},
            "lr": {"type": "number", "description": "Learning rate (default 0.005)"},
            "backend": {"type": "string", "description": "'docker' (default) or 'k8s'"},
            "parallel": {"type": "boolean", "description": "Fan out multi-seed across hosts"},
            "extra_args": {"type": "array", "description": "Extra trainer args (e.g. ['--hidden-dim', '320', '--num-layers', '6'])"},
            "dry_run": {"type": "boolean", "description": "Print plan without launching"},
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
        "description": "Build an n-gram lookup table from training data shards (polyhash family). Returns the table path and statistics.",
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
        "description": "Return a pre-formatted text dashboard with controlled-frontier status and population counts. Designed for CLI agents.",
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
        "description": "Find runs with different seeds but same config, report mean/std/CI. Defaults to legal controlled runs.",
        "parameters": {
            "population": {"type": "string", "description": "'controlled', 'imported_archive', 'unknown', or 'all'"},
            "legality": {"type": "string", "description": "'legal' (default), 'illegal', or 'all'"},
        },
    },
    "chronohorn_interpret": {
        "description": "Interpret a run: why is it good/bad, compared to neighbors, what to try next.",
        "parameters": {
            "name": {"type": "string", "required": True},
        },
    },
    "chronohorn_frontier_velocity": {
        "description": "Frontier improvement rate and trend for a selected result population. Defaults to legal controlled runs.",
        "parameters": {
            "population": {"type": "string", "description": "'controlled', 'imported_archive', 'unknown', or 'all'"},
            "legality": {"type": "string", "description": "'legal' (default), 'illegal', or 'all'"},
        },
    },
    "chronohorn_branch_health": {
        "description": "Check if an architecture branch is dead within a selected result population. Defaults to legal controlled runs.",
        "parameters": {
            "prefix": {"type": "string", "required": True},
            "population": {"type": "string", "description": "'controlled', 'imported_archive', 'unknown', or 'all'"},
            "legality": {"type": "string", "description": "'legal' (default), 'illegal', or 'all'"},
        },
    },
    "chronohorn_experiment_groups": {
        "description": "Detect and summarize experiment groups.",
        "parameters": {},
    },
    "chronohorn_suggest_next": {
        "description": "Suggest the highest-value next experiments based on the admissible legal controlled frontier, axis exhaustion, and untested axes.",
        "parameters": {},
    },
    "chronohorn_axis_analysis": {
        "description": "Analyze diminishing returns across admissible legal controlled runs. Shows which axes are alive, diminishing, or exhausted.",
        "parameters": {},
    },
    "chronohorn_architecture_boundary": {
        "description": "Detect if the current architecture class has hit its ceiling relative to the competition, using admissible legal controlled evidence.",
        "parameters": {},
    },
    "chronohorn_architecture_audit": {
        "description": "Audit architecture families across endpoint rank, matched-step envelopes, matched-compute envelopes, seed support, and trust state. Defaults to legal controlled runs.",
        "parameters": {
            "population": {"type": "string", "description": "'controlled', 'imported_archive', 'unknown', or 'all'"},
            "legality": {"type": "string", "description": "'legal' (default), 'illegal', or 'all'"},
            "trust": {"type": "string", "description": "'admissible', 'provisional', 'quarantined', or 'all' (default)"},
            "families": {"type": "array", "description": "Optional family filter"},
        },
    },
    "chronohorn_evidence_matrix": {
        "description": "Build a DB-only evidence matrix for runs and manifests, separating intent, execution, observation, interpretation, and admissibility without trusting scalar scores.",
        "parameters": {
            "top_k": {"type": "integer", "description": "Maximum run rows to return (default 50)"},
            "family": {"type": "string", "description": "Optional family filter"},
            "manifest": {"type": "string", "description": "Optional manifest filter"},
            "state": {"type": "string", "description": "Optional job-state filter"},
            "population": {"type": "string", "description": "'controlled', 'imported_archive', 'unknown', or 'all' (default)"},
            "legality": {"type": "string", "description": "'legal', 'illegal', or 'all' (default)"},
            "trust": {"type": "string", "description": "'admissible', 'provisional', 'quarantined', or 'all' (default)"},
            "format": {"type": "string", "description": "'text' for ASCII tables, omit for JSON"},
        },
    },
}


def _budget_from_args(args: dict[str, Any]):
    from chronohorn.engine.budgets import DEFAULT_GOLF_V1_BUDGET, CompetitionBudget, resolve_competition_budget
    budget_name = str(args.get("budget_name") or DEFAULT_GOLF_V1_BUDGET.name)
    base = resolve_competition_budget(budget_name)
    return CompetitionBudget(
        name=budget_name,
        train_tflops_budget=_float_arg(args, "train_tflops_budget", base.train_tflops_budget),
        artifact_limit_mb=_float_arg(args, "artifact_limit_mb", base.artifact_limit_mb),
        primary_metric_name=base.primary_metric_name,
    )


def _format_tool_failure(prefix: str, exc: Exception) -> dict[str, str]:
    detail = str(exc).strip() or repr(exc)
    return {"error": f"{prefix} failed: {detail}"}


def _required(args: dict[str, Any], key: str) -> Any:
    value = args.get(key)
    if value is None:
        raise ValueError(f"required parameter '{key}' is missing")
    return value


def _arg_value(args: dict[str, Any], key: str, default: Any) -> Any:
    value = args.get(key, default)
    return default if value is None else value


def _int_arg(args: dict[str, Any], key: str, default: int) -> int:
    return int(_arg_value(args, key, default))


def _float_arg(args: dict[str, Any], key: str, default: float) -> float:
    return float(_arg_value(args, key, default))


def _enum_arg(args: dict[str, Any], key: str, default: str, allowed: set[str]) -> str:
    value = str(_arg_value(args, key, default))
    if value not in allowed:
        raise ValueError(f"{key} must be one of {sorted(allowed)}")
    return value


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
            "chronohorn_fleet_converge": self._do_fleet_converge,
            "chronohorn_fleet_hosts": self._do_fleet_hosts,
            "chronohorn_fleet_pull": self._do_fleet_pull,
            "chronohorn_fleet_sync": self._do_fleet_sync,
            "chronohorn_fleet_launch": self._do_fleet_launch,
            "chronohorn_remote_runs": self._do_remote_runs,
            "chronohorn_k8s_submit_job": self._do_k8s_submit_job,
            "chronohorn_k8s_job_status": self._do_k8s_job_status,
            "chronohorn_k8s_job_logs": self._do_k8s_job_logs,
            "chronohorn_k8s_delete_job": self._do_k8s_delete_job,
            "chronohorn_stop_run": self._do_stop_run,
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
            "chronohorn_architecture_audit": self._do_architecture_audit,
            "chronohorn_evidence_matrix": self._do_evidence_matrix,
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
        from chronohorn.engine.results import load_result_json
        from chronohorn.fleet.forecast_results import collect_result_paths
        count = 0
        errors: list[str] = []
        for path in collect_result_paths(list(args.get("result_paths") or []), list(args.get("result_globs") or [])):
            try:
                payload = load_result_json(path)
                name = Path(path).stem
                self._shared_db.record_result(name, payload, json_archive=str(path))
                count += 1
            except Exception as exc:
                errors.append(f"{path}: {exc}")
        result = {"ingested": count}
        if errors:
            result["errors"] = errors
        return result

    def _do_status(self, args: dict[str, Any]) -> dict[str, Any]:
        db = self._shared_db
        population_summary = db.population_summary()
        trust_summary = db.trust_summary(population="all", legality="all")
        controlled_legal_trust = db.trust_summary()
        admissible_frontier = db.frontier(3, trust="admissible")
        provisional_frontier = db.frontier(3, trust="provisional")
        result_count = population_summary["result_count"]
        best_any = population_summary.get("populations", {}).get("controlled", {}).get("best_bpb")
        admissible_best = admissible_frontier[0]["bpb"] if admissible_frontier else None
        provisional_best = provisional_frontier[0]["bpb"] if provisional_frontier else None
        best = admissible_best if admissible_best is not None else best_any
        pending = db.pending_jobs()
        running = db.running_jobs()
        return {
            "result_count": result_count,
            "best_bpb": best,
            "best_bpb_any": best_any,
            "admissible_best_bpb": admissible_best,
            "provisional_best_bpb": provisional_best,
            "legal_results": population_summary["legal_count"],
            "illegal_results": population_summary["illegal_count"],
            "populations": population_summary["populations"],
            "trust": trust_summary,
            "controlled_legal_trust": controlled_legal_trust,
            "frontier_heads": {
                "admissible": admissible_frontier,
                "provisional": provisional_frontier,
            },
            "pending_jobs": len(pending),
            "running_jobs": len(running),
        }

    def _do_frontier(self, args: dict[str, Any]) -> dict[str, Any]:
        top_k = int(args["top_k"]) if args.get("top_k") is not None else 10
        family = args.get("family")
        population = _enum_arg(args, "population", "controlled", RESULT_POPULATIONS)
        legality = _enum_arg(args, "legality", "legal", RESULT_LEGALITY)
        trust = _enum_arg(args, "trust", "all", RESULT_TRUST)
        rows = self._shared_db.frontier(top_k, family=family, population=population, legality=legality, trust=trust)
        if args.get("format") == "text":
            from chronohorn.observe.terminal import ascii_frontier_table
            return {"text": ascii_frontier_table(rows, top_k=top_k), "population": population, "legality": legality, "trust": trust}
        return {"frontier": rows, "count": len(rows), "population": population, "legality": legality, "trust": trust}

    def _do_learning_curve(self, args: dict[str, Any]) -> dict[str, Any]:
        name = str(_required(args, "name"))
        points = self._shared_db.learning_curve(name)
        if args.get("format") == "text":
            from chronohorn.observe.terminal import ascii_learning_curve
            return {"name": name, "text": ascii_learning_curve(points)}
        return {"name": name, "points": points}

    def _do_compare(self, args: dict[str, Any]) -> dict[str, Any]:
        names = list(_required(args, "names"))
        curves = self._shared_db.compare_curves(names)
        return {"runs": [{"name": n, "points": p} for n, p in curves.items()]}

    def _do_marginal_rank(self, args: dict[str, Any]) -> dict[str, Any]:
        top_k = _int_arg(args, "top_k", 50)
        family = args.get("family")
        ranked = self._shared_db.marginal_rank(top_k, family=family, population="controlled", legality="legal")
        return {"ranked": ranked, "population": "controlled", "legality": "legal"}

    def _do_saturation(self, args: dict[str, Any]) -> dict[str, Any]:
        from chronohorn.engine.saturation import format_saturation_summary
        name = str(_required(args, "name"))
        analysis = self._shared_db.saturation(name)
        analysis["summary"] = format_saturation_summary(analysis)
        return analysis

    def _do_saturation_frontier(self, args: dict[str, Any]) -> dict[str, Any]:
        top_k = _int_arg(args, "top_k", 20)
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
        target_steps = _int_arg(args, "target_steps", 10000)
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
                            try:
                                parent_cfg = json.loads(cfg_rows[0]["json_blob"])
                            except (json.JSONDecodeError, TypeError) as exc:
                                import sys
                                print(f"chronohorn: auto-deepen parent config decode failed for {parent_name}: {exc}", file=sys.stderr)
                    parent_job = db.job_spec(parent_name) or {}
                    child_job = dict(parent_job)
                    child_job["name"] = child_name
                    child_job["parent"] = parent_name
                    child_job["command"] = new_cmd
                    child_job["steps"] = target_steps
                    child_job["state"] = "pending"
                    child_job["run_id"] = (
                        f"{child_job.get('manifest_path', '')}::{child_name}"
                        if child_job.get("manifest_path")
                        else child_name
                    )
                    child_job["generated_by"] = "auto_deepen"

                    db.record_job(
                        child_name,
                        manifest=str(parent_job.get("manifest") or ""),
                        parent=parent_name,
                        config=parent_cfg,
                        steps=target_steps,
                        command=new_cmd,
                        job_spec=child_job,
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

        argv = ["--manifest", str(_required(args, "manifest_path"))]
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
        from chronohorn.fleet.hosts import probe_hosts
        from chronohorn.observe.serve import FLEET_HOSTS

        hosts = list(args.get("hosts") or FLEET_HOSTS)
        fleet = probe_hosts(
            hosts,
            include_processes=True,
            process_limit=4,
            include_remote_results=True,
        )
        result = []
        for info in fleet:
            gpu_samples = info.get("gpu_samples") or []
            gpu_summary = [
                {"util_pct": s.get("util_pct", 0), "mem_used_mb": s.get("mem_used_mb", 0),
                 "mem_total_mb": s.get("mem_total_mb", 0)}
                for s in gpu_samples
            ]
            result.append({
                "host": info.get("host"),
                "online": info.get("online", False),
                "gpu": gpu_summary,
                "gpu_busy": info.get("gpu_busy", False),
                "containers": [
                    {"name": r.get("name"), "status": r.get("status")}
                    for r in (info.get("container_rows") or [])
                ],
                "waiter_count": info.get("waiter_count", 0),
                "remote_result_count": info.get("remote_result_count"),
            })
        return {"hosts": result}

    def _do_fleet_converge(self, args: dict[str, Any]) -> dict[str, Any]:
        name = args.get("name")
        steps = int(args.get("steps") or 200000)

        if name:
            base = self._shared_db.query("SELECT * FROM results WHERE name = ?", (name,))
            if not base:
                return {"error": f"Run not found: {name}"}
            base = base[0]
        else:
            frontier = self._shared_db.frontier(1, trust="admissible")
            if not frontier:
                frontier = self._shared_db.frontier(1, trust="provisional")
            if not frontier:
                return {"error": "No results in DB"}
            base = frontier[0]

        tok_s = base.get("tok_s") or 350000
        est_seconds = steps * 64 * 512 / tok_s

        result = {
            "base_name": base["name"],
            "base_bpb": round(base["bpb"], 4),
            "base_params": base.get("params"),
            "target_steps": steps,
            "est_gpu_hours": round(est_seconds / 3600, 1),
            "est_minutes": round(est_seconds / 60),
        }

        pred = self._shared_db.predict_at_steps(base["name"], steps)
        if "predicted_bpb" in pred:
            result["predicted_bpb"] = round(pred["predicted_bpb"], 4)
            result["r2"] = pred.get("r2")

        return result

    def _do_fleet_hosts(self, args: dict[str, Any]) -> dict[str, Any]:
        from chronohorn.fleet.hosts import probe_hosts

        hosts = list(args.get("hosts") or [])
        rows = probe_hosts(
            hosts,
            include_processes=bool(args.get("include_processes", False)),
            process_limit=_int_arg(args, "process_limit", 8),
            include_remote_results=bool(args.get("include_remote_results", False)),
            remote_result_dir=str(args.get("remote_result_dir") or "/data/chronohorn/out/results"),
        )
        return {"hosts": rows, "count": len(rows)}

    def _do_fleet_pull(self, args: dict[str, Any]) -> dict[str, Any]:
        from chronohorn.fleet.cli import _do_one_pull
        from chronohorn.observe.serve import FLEET_HOSTS
        from pathlib import Path

        hosts = list(args.get("hosts") or FLEET_HOSTS)
        remote_dir = str(args.get("remote_dir") or "/data/chronohorn/out/results")
        result_dir = Path("out/results")
        result_dir.mkdir(parents=True, exist_ok=True)

        pulled, ingested = _do_one_pull(hosts, remote_dir, result_dir, self._shared_db)
        total = self._shared_db._read_one("SELECT COUNT(*) FROM results")
        best = self._shared_db._read_one("SELECT MIN(bpb) FROM results WHERE bpb > 0")
        return {
            "pulled": pulled,
            "ingested": ingested,
            "total_results": total[0] if total else 0,
            "best_bpb": round(best[0], 4) if best and best[0] else None,
        }

    def _do_fleet_sync(self, args: dict[str, Any]) -> dict[str, Any]:
        from chronohorn.fleet.hosts import probe_hosts
        from chronohorn.observe.serve import FLEET_HOSTS

        hosts = list(args.get("hosts") or FLEET_HOSTS)

        # Pull results
        pull_result = self._do_fleet_pull({"hosts": hosts})

        # Probe hosts
        host_rows = probe_hosts(
            hosts,
            include_processes=True,
            process_limit=3,
            include_remote_results=True,
        )
        host_summary = []
        for h in host_rows:
            host_summary.append({
                "host": h.get("host"),
                "online": h.get("online", False),
                "gpu_busy": h.get("gpu_busy", False),
                "containers": h.get("containers", []),
                "remote_result_count": h.get("remote_result_count", 0),
            })

        # Frontier
        frontier = self._shared_db.frontier(5)
        frontier_rows = [
            {"name": r["name"], "bpb": round(r["bpb"], 4), "params": r.get("params")}
            for r in frontier
        ] if frontier else []

        return {
            "pull": pull_result,
            "hosts": host_summary,
            "frontier": frontier_rows,
        }

    def _do_fleet_launch(self, args: dict[str, Any]) -> dict[str, Any]:
        from chronohorn.fleet.cli import _launch_main
        import io, contextlib

        argv = [
            "--script", str(_required(args, "script")),
            "--arch", str(_required(args, "arch")),
        ]
        if args.get("host"):
            argv.extend(["--host", str(args["host"])])
        if args.get("name"):
            argv.extend(["--name", str(args["name"])])
        if args.get("steps"):
            argv.extend(["--steps", str(args["steps"])])
        if args.get("seed"):
            argv.extend(["--seed", str(args["seed"])])
        if args.get("single"):
            argv.append("--single")
        if args.get("fp16"):
            argv.append("--fp16")
        if args.get("lr"):
            argv.extend(["--lr", str(args["lr"])])
        if args.get("backend"):
            argv.extend(["--backend", str(args["backend"])])
        if args.get("parallel"):
            argv.append("--parallel")
        if args.get("dry_run"):
            argv.append("--dry-run")
        # Pass-through extra args
        extra = list(args.get("extra_args") or [])
        if extra:
            argv.append("--")
            argv.extend(extra)

        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                rc = _launch_main(argv)
            return {"success": rc == 0, "output": buf.getvalue()}
        except SystemExit as e:
            return {"success": False, "output": buf.getvalue(), "exit_code": e.code}
        except Exception as exc:
            return {"success": False, "output": buf.getvalue(), "error": str(exc)}

    def _do_remote_runs(self, args: dict[str, Any]) -> dict[str, Any]:
        from chronohorn.fleet.hosts import inspect_remote_runs

        result = inspect_remote_runs(
            self._shared_db,
            hosts=list(args.get("hosts") or []),
            job_names=list(args.get("job_names") or []),
            classes=list(args.get("classes") or []),
            manifest=str(args["manifest"]) if args.get("manifest") else None,
            include_logs=bool(args.get("include_logs", False)),
            relaunch_completed=bool(args.get("relaunch_completed", False)),
        )
        result["count"] = (
            int(result.get("summary", {}).get("pending", 0))
            + int(result.get("summary", {}).get("running", 0))
            + int(result.get("summary", {}).get("completed", 0))
            + int(result.get("summary", {}).get("stale", 0))
        )
        return result

    def _k8s_job_spec(self, name: str) -> dict[str, Any]:
        job = self._shared_db.job_spec(name)
        if not isinstance(job, dict):
            raise ValueError(f"{name}: no DB job found")
        return job

    def _do_k8s_submit_job(self, args: dict[str, Any]) -> dict[str, Any]:
        from chronohorn.fleet.dispatch import launch_job, write_launch_record

        name = str(_required(args, "name"))
        job = self._k8s_job_spec(name)
        job = dict(job)
        job["launcher"] = "k8s_job"
        job.setdefault("executor_kind", "k8s_cluster")
        record = launch_job(job)
        write_launch_record(name, record)
        self._shared_db.record_launch(
            name,
            host=record.get("host", ""),
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
        self._shared_db.record_event(
            "manual_k8s_submit",
            name=name,
            executor_name=record.get("executor_name"),
            runtime_namespace=record.get("runtime_namespace"),
            runtime_job_name=record.get("runtime_job_name"),
        )
        return {"name": name, "record": record}

    def _do_k8s_job_status(self, args: dict[str, Any]) -> dict[str, Any]:
        from chronohorn.fleet.k8s import get_k8s_job_status

        name = str(_required(args, "name"))
        job = self._k8s_job_spec(name)
        status = get_k8s_job_status(
            job,
            include_logs=bool(args.get("include_logs", False)),
            tail_lines=_int_arg(args, "tail_lines", 64),
        )
        return {"name": name, "status": status}

    def _do_k8s_job_logs(self, args: dict[str, Any]) -> dict[str, Any]:
        from chronohorn.fleet.k8s import get_k8s_job_logs, get_k8s_job_status

        name = str(_required(args, "name"))
        job = self._k8s_job_spec(name)
        status = get_k8s_job_status(job, include_logs=False)
        logs = get_k8s_job_logs(
            gateway=str(status.get("cluster_gateway_host") or job.get("cluster_gateway_host") or ""),
            namespace=str(status.get("runtime_namespace") or job.get("runtime_namespace") or ""),
            runtime_job_name=str(status.get("runtime_job_name") or job.get("runtime_job_name") or ""),
            runtime_pod_name=str(status.get("runtime_pod_name") or "") or None,
            tail_lines=_int_arg(args, "tail_lines", 64),
        )
        return {"name": name, "logs": logs}

    def _do_k8s_delete_job(self, args: dict[str, Any]) -> dict[str, Any]:
        from chronohorn.control.actions import stop_job
        from chronohorn.control.models import ControlAction
        from chronohorn.fleet.dispatch import load_launch_record

        post_state = _enum_arg(args, "post_state", "failed", {"failed", "pending"})
        name = str(_required(args, "name"))
        record = load_launch_record(name) or {}
        job = self._k8s_job_spec(name)
        action = ControlAction(
            action="stop_run",
            target_name=name,
            family=job.get("family"),
            priority=100.0,
            rationale="manual MCP k8s delete",
            state=str(job.get("state") or "running"),
            host=str(record.get("host") or job.get("host") or ""),
            launcher=str(record.get("launcher") or "k8s_job"),
            metadata={
                "executor_kind": record.get("executor_kind") or job.get("executor_kind"),
                "executor_name": record.get("executor_name") or job.get("executor_name"),
                "cluster_gateway_host": record.get("cluster_gateway_host") or job.get("cluster_gateway_host"),
                "runtime_namespace": record.get("runtime_namespace") or job.get("runtime_namespace"),
                "runtime_job_name": record.get("runtime_job_name") or job.get("runtime_job_name"),
            },
        )
        result = stop_job(action)
        self._shared_db._write(
            "UPDATE jobs SET state = ? WHERE name = ?",
            (post_state, name),
            wait=True,
        )
        self._shared_db.record_event(
            "manual_k8s_delete",
            name=name,
            executor_name=result.get("executor_name"),
            runtime_namespace=result.get("runtime_namespace"),
            runtime_job_name=result.get("runtime_job_name"),
            post_state=post_state,
        )
        return {"name": name, "post_state": post_state, "record": result}

    def _do_stop_run(self, args: dict[str, Any]) -> dict[str, Any]:
        from chronohorn.control.actions import stop_job
        from chronohorn.control.models import ControlAction
        from chronohorn.fleet.dispatch import load_launch_record

        post_state = _enum_arg(args, "post_state", "failed", {"failed", "pending"})
        name = str(_required(args, "name"))
        record = load_launch_record(name) or {}
        job = self._shared_db.job_spec(name) or {}
        action = ControlAction(
            action="stop_run",
            target_name=name,
            family=job.get("family"),
            priority=100.0,
            rationale="manual MCP stop",
            state=str(job.get("state") or "running"),
            host=str(record.get("host") or job.get("host") or ""),
            launcher=str(record.get("launcher") or job.get("launcher") or ""),
            metadata={
                "executor_kind": record.get("executor_kind") or job.get("executor_kind"),
                "executor_name": record.get("executor_name") or job.get("executor_name"),
                "cluster_gateway_host": record.get("cluster_gateway_host") or job.get("cluster_gateway_host"),
                "runtime_namespace": record.get("runtime_namespace") or job.get("runtime_namespace"),
                "runtime_job_name": record.get("runtime_job_name") or job.get("runtime_job_name"),
            },
        )
        result = stop_job(action)
        self._shared_db._write(
            "UPDATE jobs SET state = ? WHERE name = ?",
            (post_state, name),
            wait=True,
        )
        self._shared_db.record_event(
            "manual_stop",
            name=name,
            host=result.get("host"),
            launcher=result.get("launcher"),
            post_state=post_state,
        )
        return {"name": name, "post_state": post_state, "record": result}

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
        top_k = _int_arg(args, "top_k", 50)
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
            config = {
                "manifest_paths": list(args.get("manifest_paths") or []),
                "probe_runtime": bool(args.get("probe_runtime", False)),
                "db_path": str(self._shared_db._path),
            }
            plan = build_control_plan(
                config,
                job_names=list(args.get("job_names") or []),
                classes=list(args.get("classes") or []),
                telemetry_globs=list(args.get("telemetry_globs") or []),
                relaunch_completed=bool(args.get("relaunch_completed", False)),
                max_launches=_int_arg(args, "max_launches", 2),
                stop_margin=_float_arg(args, "stop_margin", 0.01),
                min_gain_per_hour=_float_arg(args, "min_gain_per_hour", 0.01),
                top_completed=_int_arg(args, "top_completed", 3),
            )
            return plan.as_dict()
        except ImportError:
            return {
                "error": "Pipeline dependencies not available. Use chronohorn_frontier and chronohorn_auto_deepen instead.",
                "frontier": self._shared_db.frontier(10, trust="admissible"),
            }
        except Exception as exc:
            return _format_tool_failure("control_recommend", exc)

    def _do_control_act(self, args: dict[str, Any]) -> dict[str, Any]:
        try:
            from chronohorn.control.actions import execute_control_actions
            from chronohorn.control.models import ControlAction
            from chronohorn.control.policy import build_control_plan
            config = {
                "manifest_paths": list(args.get("manifest_paths") or []),
                "probe_runtime": bool(args.get("probe_runtime", False)),
                "db_path": str(self._shared_db._path),
            }
            plan = build_control_plan(
                config,
                job_names=list(args.get("job_names") or []),
                classes=list(args.get("classes") or []),
                telemetry_globs=list(args.get("telemetry_globs") or []),
                relaunch_completed=bool(args.get("relaunch_completed", False)),
                max_launches=_int_arg(args, "max_launches", 2),
                stop_margin=_float_arg(args, "stop_margin", 0.01),
                min_gain_per_hour=_float_arg(args, "min_gain_per_hour", 0.01),
                top_completed=_int_arg(args, "top_completed", 3),
            )
            actions = [ControlAction(**row) for row in plan.as_dict().get("actions", [])]
            executed = execute_control_actions(
                actions,
                allow_stop=bool(args.get("allow_stop", False)),
                max_launches=_int_arg(args, "max_launches", 2),
            )
            return {"plan": plan.as_dict(), "executed": executed}
        except ImportError:
            return {
                "error": "Pipeline dependencies not available. Use chronohorn_frontier and chronohorn_auto_deepen instead.",
                "frontier": self._shared_db.frontier(10, trust="admissible"),
            }
        except Exception as exc:
            return _format_tool_failure("control_act", exc)

    def _do_reset(self, args: dict[str, Any]) -> dict[str, Any]:
        return {"status": "no-op", "message": "Database is persistent. Use chronohorn_query for ad-hoc cleanup."}

    def _do_artifact_check(self, args: dict[str, Any]) -> dict[str, Any]:
        name = str(_required(args, "name"))
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
            sql = str(_required(args, "sql"))
            if not sql.strip().upper().startswith("SELECT"):
                return {"error": "Only SELECT queries are allowed"}
            rows = self._shared_db.query(sql)
            return {"rows": rows, "count": len(rows)}
        except Exception as exc:
            return {"error": str(exc)}

    def _do_build_table(self, args: dict[str, Any]) -> dict[str, Any]:
        import importlib
        import numpy as np

        data_path = str(_required(args, "data_path"))
        output_path = str(args.get("output_path", "out/ngram_table.npz"))
        vocab_size = int(args.get("vocab_size", 1024))
        max_order = int(args.get("max_order", 4))
        bucket_count = int(args.get("bucket_count", 8192))

        try:
            # Load via importlib to avoid direct family import in core infra
            ngram_mod = importlib.import_module("chronohorn.families.polyhash.models.ngram_table")
            NgramTable = ngram_mod.NgramTable

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
        except Exception as exc:
            fleet = {}
            status["fleet_error"] = str(exc)
        status["fleet"] = fleet
        return status

    def _do_launches(self, args: dict[str, Any]) -> dict[str, Any]:
        top_k = _int_arg(args, "top_k", 20)
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
        name = str(_required(args, "name"))
        illegal = bool(_required(args, "illegal"))
        self._shared_db._write(
            "UPDATE results SET illegal = ? WHERE name = ?",
            (int(illegal), name),
            wait=True,
        )
        return {"name": name, "illegal": illegal, "status": "updated"}

    def _do_config_diff(self, args: dict[str, Any]) -> dict[str, Any]:
        return self._shared_db.config_diff(str(_required(args, "name1")), str(_required(args, "name2")))

    def _do_what_varied(self, args: dict[str, Any]) -> dict[str, Any]:
        names = list(_required(args, "names")) if args.get("names") else None
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

        summary_data = self._shared_db.summary()
        fam_rows = self._shared_db.query(
            "SELECT family, COUNT(*) as c FROM results GROUP BY family"
        )
        summary_data["families"] = {r["family"]: r["c"] for r in fam_rows} if fam_rows else {}

        board = self._shared_db.frontier(top_k, trust="admissible")
        text = ascii_status(summary_data, board)
        return {"text": text}

    # -- W7: changelog ---------------------------------------------------------

    def _do_changelog(self, args: dict[str, Any]) -> dict[str, Any]:
        hours = float(args["hours"]) if args.get("hours") is not None else 1
        return self._shared_db.changelog(since_hours=hours)

    # -- W12: experiment journal -----------------------------------------------

    def _do_journal_write(self, args: dict[str, Any]) -> dict[str, Any]:
        kind = str(_required(args, "kind"))
        content = str(_required(args, "content"))
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
        name = str(_required(args, "name"))
        steps = int(_required(args, "steps"))
        return self._shared_db.predict_at_steps(name, steps)

    def _do_prediction_audit(self, args: dict[str, Any]) -> dict[str, Any]:
        results = self._shared_db.audit_predictions()
        return {"predictions": results, "count": len(results)}

    # -- W3: experiment matrix ---------------------------------------------------

    def _do_emit_matrix(self, args: dict[str, Any]) -> dict[str, Any]:
        from chronohorn.fleet.experiment_matrix import expand_matrix, matrix_to_commands, write_manifest, estimate_cost

        spec = {
            "name_template": str(_required(args, "name_template")),
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
        population = _enum_arg(args, "population", "controlled", RESULT_POPULATIONS)
        legality = _enum_arg(args, "legality", "legal", RESULT_LEGALITY)
        groups = self._shared_db.seed_groups(population=population, legality=legality)
        return {"groups": groups, "count": len(groups), "population": population, "legality": legality}

    # -- W16: interpretive layer -------------------------------------------------

    def _do_interpret(self, args: dict[str, Any]) -> dict[str, Any]:
        name = str(_required(args, "name"))
        db = self._shared_db

        # Get this run's data
        run_rows = db.query("SELECT * FROM results WHERE name = ?", (name,))
        if not run_rows:
            return {"error": f"run not found: {name}"}
        run = run_rows[0]

        # Get frontier context
        frontier = db.frontier(20, trust="admissible")
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
        sat_error = None
        if hasattr(db, "saturation"):
            try:
                sat = db.saturation(name)
            except Exception as exc:
                sat_error = str(exc)

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
        if sat_error:
            interpretation["saturation_error"] = sat_error

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
        population = _enum_arg(args, "population", "controlled", RESULT_POPULATIONS)
        legality = _enum_arg(args, "legality", "legal", RESULT_LEGALITY)
        return self._shared_db.frontier_velocity(population=population, legality=legality)

    def _do_branch_health(self, args: dict[str, Any]) -> dict[str, Any]:
        prefix = str(_required(args, "prefix"))
        population = _enum_arg(args, "population", "controlled", RESULT_POPULATIONS)
        legality = _enum_arg(args, "legality", "legal", RESULT_LEGALITY)
        return self._shared_db.branch_health(prefix, population=population, legality=legality)

    def _do_experiment_groups(self, args: dict[str, Any]) -> dict[str, Any]:
        groups = self._shared_db.detect_groups(population="controlled", legality="legal")
        return {"groups": groups, "count": len(groups), "population": "controlled", "legality": "legal"}

    # -- advisors ---------------------------------------------------------------

    def _do_suggest_next(self, args: dict[str, Any]) -> dict[str, Any]:
        from chronohorn.engine.advisor import suggest_next, format_suggestions
        suggestions = suggest_next(self._shared_db)
        return {"suggestions": suggestions, "text": format_suggestions(suggestions), "population": "controlled", "legality": "legal", "trust": "admissible"}

    def _do_axis_analysis(self, args: dict[str, Any]) -> dict[str, Any]:
        from chronohorn.engine.axis_analysis import analyze_axes
        results = self._shared_db.analysis_rows(max_bpb=3.0, controlled_only=True, trust="admissible")
        return {"axes": analyze_axes(results), "population": "controlled", "legality": "legal", "trust": "admissible"}

    def _do_architecture_boundary(self, args: dict[str, Any]) -> dict[str, Any]:
        from chronohorn.engine.advisor import architecture_boundary
        result = architecture_boundary(self._shared_db)
        result["population"] = "controlled"
        result["legality"] = "legal"
        result["trust"] = "admissible"
        return result

    def _do_architecture_audit(self, args: dict[str, Any]) -> dict[str, Any]:
        population = _enum_arg(args, "population", "controlled", RESULT_POPULATIONS)
        legality = _enum_arg(args, "legality", "legal", RESULT_LEGALITY)
        trust = _enum_arg(args, "trust", "all", RESULT_TRUST)
        families = [str(value) for value in (args.get("families") or []) if value]
        return self._shared_db.architecture_audit(
            population=population,
            legality=legality,
            trust=trust,
            families=families or None,
        )

    def _do_evidence_matrix(self, args: dict[str, Any]) -> dict[str, Any]:
        population = _enum_arg(args, "population", "all", RESULT_POPULATIONS)
        legality = _enum_arg(args, "legality", "all", RESULT_LEGALITY)
        trust = _enum_arg(args, "trust", "all", RESULT_TRUST)
        matrix = self._shared_db.evidence_matrix(
            top_k=_int_arg(args, "top_k", 50),
            family=str(args.get("family") or "").strip() or None,
            manifest=str(args.get("manifest") or "").strip() or None,
            population=population,
            legality=legality,
            trust=trust,
            state=str(args.get("state") or "").strip() or None,
        )
        if args.get("format") == "text":
            from chronohorn.observe.terminal import ascii_evidence_matrix

            return {"text": ascii_evidence_matrix(matrix, top_k=_int_arg(args, "top_k", 20))}
        return matrix
