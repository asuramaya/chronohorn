from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from chronohorn.engine.budgets import DEFAULT_GOLF_V1_BUDGET
from chronohorn.pipeline import build_runtime_store, build_store_payload, normalize_runtime_config
from chronohorn.store import RunStore


def _add_runtime_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", action="append", default=[], help="Manifest JSONL path (repeatable).")
    parser.add_argument("--state-path", action="append", default=[], help="Tracked state JSON path (repeatable).")
    parser.add_argument(
        "--launch-glob",
        action="append",
        default=[],
        help="Launch-record glob. Defaults to out/fleet/*.launch.json.",
    )
    parser.add_argument("--result-path", action="append", default=[], help="Result JSON path or directory (repeatable).")
    parser.add_argument("--result-glob", action="append", default=[], help="Result JSON glob (repeatable).")
    parser.add_argument(
        "--probe-runtime",
        action="store_true",
        help="Probe the current fleet state from manifests and store runtime_state records.",
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


def _runtime_config(args: argparse.Namespace) -> dict[str, Any]:
    return normalize_runtime_config(
        {
        "manifest_paths": list(args.manifest or []),
        "state_paths": list(args.state_path or []),
        "launch_globs": list(args.launch_glob or []),
        "result_paths": list(args.result_path or []),
        "result_globs": list(args.result_glob or []),
        "probe_runtime": bool(getattr(args, "probe_runtime", False)),
        "budget_name": args.budget_name,
        "train_tflops_budget": args.train_tflops_budget,
        "artifact_limit_mb": args.artifact_limit_mb,
        }
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="chronohorn observe",
        description=(
            "Normalize manifests, launch records, result JSONs, and budget forecasts "
            "into a single agent-facing Chronohorn run store."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Build a runtime store from manifests, launch records, result files, and forecasts.",
    )
    _add_runtime_inputs(pipeline_parser)
    pipeline_parser.add_argument("--write-store", help="Write the normalized run store to JSON.")
    pipeline_parser.add_argument("--top", type=int, default=10, help="Number of merged runs to include.")
    pipeline_parser.add_argument("--include-records", action="store_true", help="Include raw records in the payload.")
    pipeline_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    status_parser = subparsers.add_parser(
        "status",
        help="Show a compact summary of a stored run ledger or a freshly built runtime store.",
    )
    _add_runtime_inputs(status_parser)
    status_parser.add_argument("--store", help="Path to a saved run-store JSON.")
    status_parser.add_argument("--top", type=int, default=10, help="Number of merged runs to include.")
    status_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    query_parser = subparsers.add_parser(
        "query-records",
        help="Filter raw runtime records from a saved or freshly built run store.",
    )
    _add_runtime_inputs(query_parser)
    query_parser.add_argument("--store", help="Path to a saved run-store JSON.")
    query_parser.add_argument("--kind", help="Filter by record kind.")
    query_parser.add_argument("--source", help="Filter by record source.")
    query_parser.add_argument("--family", help="Filter by family name.")
    query_parser.add_argument("--name", help="Filter by run name.")
    query_parser.add_argument("--status", help="Filter by record status.")
    query_parser.add_argument("--top", type=int, default=50, help="Maximum number of records to print.")
    query_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    frontier_parser = subparsers.add_parser(
        "frontier",
        help="Show the best raw and artifact-feasible frontier rows from the shared Chronohorn runtime store.",
    )
    _add_runtime_inputs(frontier_parser)
    frontier_parser.add_argument("--store", help="Path to a saved run-store JSON.")
    frontier_parser.add_argument("--top", type=int, default=10, help="Maximum number of rows per frontier leaderboard.")
    frontier_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    serve_parser = subparsers.add_parser(
        "serve",
        help="Launch a lightweight HTTP visualization server for runtime state.",
    )
    serve_parser.add_argument("--port", type=int, default=7070, help="HTTP port to listen on.")
    serve_parser.add_argument("--result-dir", default="out/results", help="Result directory to serve.")

    return parser.parse_args(argv)


def _load_or_build_store(args: argparse.Namespace) -> tuple[RunStore, list[str]]:
    if getattr(args, "store", None):
        return RunStore.load(args.store), []
    config = _runtime_config(args)
    return build_runtime_store(config)


def _print_status_text(payload: dict[str, Any]) -> None:
    print("runs state decision metric forecast artifact host name")
    for row in payload.get("runs", []):
        print(
            " ".join(
                [
                    str(row.get("state") or "-"),
                    str(row.get("decision") or "-"),
                    str(row.get("metric_name") or "-"),
                    f"{row.get('metric_value', '-')}",
                    f"{row.get('forecast_metric_value', '-')}",
                    "yes" if row.get("artifact_viable") else "no",
                    str(row.get("host") or "-"),
                    str(row.get("name") or "-"),
                ]
            )
        )
    print(json.dumps(payload.get("summary", {}), indent=2, sort_keys=True))


def _print_query_text(records: list[dict[str, Any]]) -> None:
    print("kind status family metric path name")
    for row in records:
        print(
            " ".join(
                [
                    str(row.get("kind") or "-"),
                    str(row.get("status") or "-"),
                    str(row.get("family") or "-"),
                    str(row.get("metric_value") if row.get("metric_value") is not None else "-"),
                    str(row.get("path") or "-"),
                    str(row.get("name") or "-"),
                ]
            )
        )


def _print_frontier_text(payload: dict[str, Any]) -> None:
    frontier = payload.get("frontier", {})
    print("best ranked")
    for row in frontier.get("best_ranked", []):
        metric = row.get("forecast_metric_value")
        if metric is None:
            metric = row.get("metric_value")
        print(
            " ".join(
                [
                    str(row.get("state") or "-"),
                    "yes" if row.get("artifact_viable") else "no",
                    str(metric if metric is not None else "-"),
                    str(row.get("name") or "-"),
                ]
            )
        )
    print("\nbest raw")
    for row in frontier.get("best_raw", []):
        metric = row.get("metric_value")
        print(
            " ".join(
                [
                    str(row.get("state") or "-"),
                    "yes" if row.get("artifact_viable") else "no",
                    str(metric if metric is not None else "-"),
                    str(row.get("name") or "-"),
                ]
            )
        )
    print("\nbest feasible")
    for row in frontier.get("best_feasible", []):
        metric = row.get("metric_value")
        print(
            " ".join(
                [
                    str(row.get("state") or "-"),
                    "yes" if row.get("artifact_viable") else "no",
                    str(metric if metric is not None else "-"),
                    str(row.get("name") or "-"),
                ]
            )
        )
    notes = frontier.get("notes") or []
    if notes:
        print("\nnotes")
        for row in notes:
            text = ((row.get("metadata") or {}).get("text") or "").strip()
            if text:
                print(f"- {text}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command is None:
        raise SystemExit("chronohorn observe requires a subcommand; run with --help")

    if args.command == "pipeline":
        store, stages_run = build_runtime_store(_runtime_config(args))
        if args.write_store:
            store.save(Path(args.write_store))
        payload = build_store_payload(
            store,
            stages_run=stages_run,
            top_k=args.top,
            include_records=args.include_records,
        )
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            _print_status_text(payload)
        return 0

    if args.command == "status":
        store, stages_run = _load_or_build_store(args)
        payload = build_store_payload(store, stages_run=stages_run, top_k=args.top, include_records=False)
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            _print_status_text(payload)
        return 0

    if args.command == "query-records":
        store, _ = _load_or_build_store(args)
        records = [
            record.as_dict()
            for record in store.filter(
                kind=args.kind,
                source=args.source,
                family=args.family,
                name=args.name,
                status=args.status,
            )[: max(args.top, 0)]
        ]
        payload = {"count": len(records), "records": records}
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            _print_query_text(records)
        return 0

    if args.command == "frontier":
        store, stages_run = _load_or_build_store(args)
        payload = build_store_payload(store, stages_run=stages_run, top_k=args.top, include_records=False)
        frontier = {"summary": payload.get("summary", {}), "stages_run": payload.get("stages_run", []), "frontier": payload.get("frontier", {})}
        if args.json:
            print(json.dumps(frontier, indent=2, sort_keys=True))
        else:
            _print_frontier_text(frontier)
        return 0

    if args.command == "serve":
        from chronohorn.observe.serve import main as serve_main
        return serve_main(["--port", str(args.port), "--result-dir", args.result_dir])

    raise SystemExit(f"unknown chronohorn observe subcommand: {args.command}")
