from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from typing import Any

from chronohorn.control.actions import execute_control_actions
from chronohorn.control.policy import build_control_plan
from chronohorn.engine.budgets import DEFAULT_GOLF_V1_BUDGET


def _add_runtime_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", action="append", default=[], help="Manifest JSONL path (repeatable).")
    parser.add_argument("--launch-glob", action="append", default=[], help="Launch-record glob.")
    parser.add_argument("--result-path", action="append", default=[], help="Result JSON path or directory.")
    parser.add_argument("--result-glob", action="append", default=[], help="Result JSON glob.")
    parser.add_argument(
        "--probe-runtime",
        action="store_true",
        help="Probe live runtime state from manifests and remote jobs.",
    )
    parser.add_argument(
        "--budget-name",
        default=DEFAULT_GOLF_V1_BUDGET.name,
        help="Budget label used for forecast records.",
    )
    parser.add_argument(
        "--train-tflops-budget",
        type=float,
        default=DEFAULT_GOLF_V1_BUDGET.train_tflops_budget,
        help="Training budget in TFLOPs used for forecast records.",
    )
    parser.add_argument(
        "--artifact-limit-mb",
        type=float,
        default=DEFAULT_GOLF_V1_BUDGET.artifact_limit_mb,
        help="Artifact-size budget in MB used for forecast records.",
    )
    parser.add_argument("--job", action="append", default=[], help="Restrict to named job(s).")
    parser.add_argument(
        "--class",
        dest="classes",
        action="append",
        default=[],
        help="Restrict to resource_class values.",
    )
    parser.add_argument("--telemetry-glob", action="append", default=[], help="Additional telemetry globs.")
    parser.add_argument(
        "--relaunch-completed",
        action="store_true",
        help="Treat completed jobs as eligible for relaunch planning.",
    )


def _config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "manifest_paths": list(args.manifest or []),
        "launch_globs": list(args.launch_glob or []),
        "result_paths": list(args.result_path or []),
        "result_globs": list(args.result_glob or []),
        "probe_runtime": bool(args.probe_runtime),
        "budget_name": args.budget_name,
        "train_tflops_budget": args.train_tflops_budget,
        "artifact_limit_mb": args.artifact_limit_mb,
        "relaunch_completed": bool(args.relaunch_completed),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="chronohorn control",
        description=(
            "Closed-loop frontier controller for Chronohorn. "
            "Builds recommendations and can execute safe runtime actions."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    recommend = subparsers.add_parser("recommend", help="Build a control plan from the current runtime state.")
    _add_runtime_inputs(recommend)
    recommend.add_argument("--max-launches", type=int, default=2, help="Maximum pending launches to recommend.")
    recommend.add_argument("--stop-margin", type=float, default=0.01, help="Metric margin required for domination.")
    recommend.add_argument(
        "--min-gain-per-hour",
        type=float,
        default=0.01,
        help="Minimum marginal improvement per hour before a dominated run becomes a stop candidate.",
    )
    recommend.add_argument("--top-completed", type=int, default=3, help="Top completed runs to flag for promotion.")
    recommend.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    act = subparsers.add_parser("act", help="Execute recommended launch actions and optional stop actions.")
    _add_runtime_inputs(act)
    act.add_argument("--max-launches", type=int, default=2, help="Maximum pending launches to execute.")
    act.add_argument("--stop-margin", type=float, default=0.01, help="Metric margin required for domination.")
    act.add_argument(
        "--min-gain-per-hour",
        type=float,
        default=0.01,
        help="Minimum marginal improvement per hour before a dominated run becomes a stop candidate.",
    )
    act.add_argument("--top-completed", type=int, default=3, help="Top completed runs to flag for promotion.")
    act.add_argument(
        "--allow-stop",
        action="store_true",
        help="Actually stop dominated running jobs. Without this, stop actions are only reported.",
    )
    act.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser.parse_args(argv)


def _print_text(payload: dict[str, Any]) -> None:
    print("priority action target state host rationale")
    for row in payload.get("actions", []):
        print(
            " ".join(
                [
                    f"{float(row.get('priority', 0.0)):.1f}",
                    str(row.get("action") or "-"),
                    str(row.get("target_name") or "-"),
                    str(row.get("state") or "-"),
                    str(row.get("host") or "-"),
                    str(row.get("rationale") or "-"),
                ]
            )
        )
    print(json.dumps(payload.get("summary", {}), indent=2, sort_keys=True))


def _plan_from_args(args: argparse.Namespace) -> dict[str, Any]:
    plan = build_control_plan(
        _config_from_args(args),
        job_names=args.job or [],
        classes=args.classes or [],
        telemetry_globs=args.telemetry_glob or [],
        relaunch_completed=bool(args.relaunch_completed),
        max_launches=max(1, args.max_launches),
        stop_margin=float(args.stop_margin),
        min_gain_per_hour=float(args.min_gain_per_hour),
        top_completed=max(0, args.top_completed),
    )
    return plan.as_dict()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command is None:
        raise SystemExit("chronohorn control requires a subcommand; run with --help")

    plan_payload = _plan_from_args(args)
    if args.command == "recommend":
        if args.json:
            print(json.dumps(plan_payload, indent=2, sort_keys=True))
        else:
            _print_text(plan_payload)
        return 0

    if args.command == "act":
        from chronohorn.control.models import ControlAction

        actions = [ControlAction(**row) for row in plan_payload.get("actions", [])]
        executed = execute_control_actions(
            actions,
            allow_stop=bool(args.allow_stop),
            max_launches=max(1, args.max_launches),
        )
        payload = {"plan": plan_payload, "executed": executed}
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            _print_text(plan_payload)
            print(json.dumps({"executed": executed}, indent=2, sort_keys=True))
        return 0

    raise SystemExit(f"unknown chronohorn control subcommand: {args.command}")
