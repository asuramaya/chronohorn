from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from chronohorn.engine.budgets import DEFAULT_GOLF_V1_BUDGET


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
    parser.add_argument("--result-dir", default="out/results", help="Result directory for DB rebuild.")
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


def _get_db(args: argparse.Namespace):
    from chronohorn.db import ChronohornDB

    result_dir = getattr(args, "result_dir", "out/results")
    db_path = Path(result_dir).parent / "chronohorn.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = ChronohornDB(str(db_path))
    if db.result_count() == 0:
        db.rebuild_from_archive(result_dir)
    return db


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
        help="Show a compact summary of the Chronohorn database.",
    )
    _add_runtime_inputs(status_parser)
    status_parser.add_argument("--store", help="(deprecated, ignored) Path to a saved run-store JSON.")
    status_parser.add_argument("--top", type=int, default=10, help="Number of merged runs to include.")
    status_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    query_parser = subparsers.add_parser(
        "query-records",
        help="Filter raw records from the Chronohorn database.",
    )
    _add_runtime_inputs(query_parser)
    query_parser.add_argument("--store", help="(deprecated, ignored) Path to a saved run-store JSON.")
    query_parser.add_argument("--kind", help="Filter by record kind.")
    query_parser.add_argument("--source", help="Filter by record source.")
    query_parser.add_argument("--family", help="Filter by family name.")
    query_parser.add_argument("--name", help="Filter by run name.")
    query_parser.add_argument("--status", help="Filter by record status.")
    query_parser.add_argument("--top", type=int, default=50, help="Maximum number of records to print.")
    query_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    frontier_parser = subparsers.add_parser(
        "frontier",
        help="Show the best non-illegal frontier rows from the Chronohorn database.",
    )
    _add_runtime_inputs(frontier_parser)
    frontier_parser.add_argument("--store", help="(deprecated, ignored) Path to a saved run-store JSON.")
    frontier_parser.add_argument("--top", type=int, default=10, help="Maximum number of rows per frontier leaderboard.")
    frontier_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    serve_parser = subparsers.add_parser(
        "serve",
        help="Launch a lightweight HTTP visualization server for runtime state.",
    )
    serve_parser.add_argument("--port", type=int, default=7070, help="HTTP port to listen on.")
    serve_parser.add_argument("--result-dir", default="out/results", help="Result directory to serve.")

    return parser.parse_args(argv)


def _print_status_text(payload: dict[str, Any]) -> None:
    summary = payload.get("summary", {})
    frontier = payload.get("frontier", [])
    print(f"results: {summary.get('result_count', 0)}  "
          f"best_bpb: {summary.get('best_bpb', '-')}  "
          f"pending: {summary.get('pending_jobs', 0)}  "
          f"running: {summary.get('running_jobs', 0)}")
    if frontier:
        print("\nfrontier:")
        print("  bpb   slope   steps  name")
        for row in frontier:
            bpb = row.get("bpb", "-")
            slope = row.get("slope", "-")
            steps = row.get("steps", "-")
            name = row.get("name", "-")
            bpb_s = f"{bpb:.4f}" if isinstance(bpb, (int, float)) else str(bpb)
            slope_s = f"{slope:.5f}" if isinstance(slope, (int, float)) else str(slope)
            print(f"  {bpb_s:>7}  {slope_s:>8}  {str(steps):>5}  {name}")


def _print_query_text(records: list[dict[str, Any]]) -> None:
    if not records:
        print("(no records)")
        return
    # Detect record type from available columns
    sample = records[0] if records else {}
    if "bpb" in sample:
        # Result records
        print(f"{'name':<40} {'family':<12} {'bpb':>8} {'steps':>7} {'illegal':>7} {'params':>10}")
        for row in records:
            name = str(row.get("name") or "-")
            family = str(row.get("family") or "-")
            bpb = row.get("bpb")
            bpb_s = f"{bpb:.4f}" if isinstance(bpb, (int, float)) else "-"
            steps = str(row.get("steps") or "-")
            illegal = "YES" if row.get("illegal") else "no"
            params = row.get("params")
            params_s = f"{params:,}" if isinstance(params, (int, float)) and params else "-"
            print(f"{name:<40} {family:<12} {bpb_s:>8} {steps:>7} {illegal:>7} {params_s:>10}")
    elif "state" in sample:
        # Job records
        print(f"{'name':<40} {'state':<12} {'manifest':<20} {'steps':>7}")
        for row in records:
            name = str(row.get("name") or "-")
            state = str(row.get("state") or "-")
            manifest = str(row.get("manifest") or "-")
            steps = str(row.get("steps") or "-")
            print(f"{name:<40} {state:<12} {manifest:<20} {steps:>7}")
    elif "step" in sample:
        # Probe records
        print(f"{'name':<40} {'step':>7} {'bpb':>8} {'tflops':>8}")
        for row in records:
            name = str(row.get("name") or "-")
            step = str(row.get("step") or "-")
            bpb = row.get("bpb")
            bpb_s = f"{bpb:.4f}" if isinstance(bpb, (int, float)) else "-"
            tflops = row.get("tflops")
            tf_s = f"{tflops:.4f}" if isinstance(tflops, (int, float)) else "-"
            print(f"{name:<40} {step:>7} {bpb_s:>8} {tf_s:>8}")
    else:
        # Fallback: print all keys
        for row in records:
            print(" ".join(f"{k}={v}" for k, v in row.items() if v is not None))


def _print_frontier_text(payload: dict[str, Any]) -> None:
    frontier = payload.get("frontier", [])
    if not frontier:
        print("No frontier data available.")
        return
    print("best ranked (by bpb)")
    print("  bpb       feasible  slope     name")
    for row in frontier:
        bpb = row.get("bpb", "-")
        int6_mb = row.get("int6_mb")
        feasible = "yes" if (int6_mb is not None and int6_mb <= 16) else "no"
        slope = row.get("slope", "-")
        name = row.get("name", "-")
        bpb_s = f"{bpb:.4f}" if isinstance(bpb, (int, float)) else str(bpb)
        slope_s = f"{slope:.5f}" if isinstance(slope, (int, float)) else str(slope)
        print(f"  {bpb_s:>8}  {feasible:>8}  {slope_s:>8}  {name}")
    summary = payload.get("summary", {})
    if summary:
        print(f"\nsummary: {json.dumps(summary, indent=2, sort_keys=True)}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command is None:
        raise SystemExit("chronohorn observe requires a subcommand; run with --help")

    if args.command == "pipeline":
        # DB-first path: always use the database as the primary data source.
        # Ingest any provided manifests/results, then query from the DB.
        # Only fall back to the legacy RunStore path when --write-store is
        # explicitly requested and pipeline deps are available.
        use_legacy_store = bool(args.write_store)
        if use_legacy_store:
            try:
                from chronohorn.pipeline import build_runtime_store, build_store_payload, normalize_runtime_config
                config = normalize_runtime_config({
                    "manifest_paths": list(args.manifest or []),
                    "state_paths": list(args.state_path or []),
                    "launch_globs": list(args.launch_glob or []),
                    "result_paths": list(args.result_path or []),
                    "result_globs": list(args.result_glob or []),
                    "probe_runtime": bool(getattr(args, "probe_runtime", False)),
                    "budget_name": args.budget_name,
                    "train_tflops_budget": args.train_tflops_budget,
                    "artifact_limit_mb": args.artifact_limit_mb,
                })
                store, stages_run = build_runtime_store(config)
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
                    try:
                        _print_status_text(payload)
                    except (KeyError, TypeError, AttributeError):
                        print(json.dumps(payload, indent=2, sort_keys=True))
                return 0
            except ImportError:
                pass  # Fall through to DB path below

        # Primary path: use the database
        db = _get_db(args)
        # Ingest any provided manifests and results into the DB
        for m in (args.manifest or []):
            try:
                db.ingest_manifest(m)
            except Exception:
                pass
        from chronohorn.fleet.forecast_results import collect_result_paths
        from chronohorn.engine.results import load_result_json
        for path in collect_result_paths(list(args.result_path or []), list(args.result_glob or [])):
            try:
                result_payload = load_result_json(path)
                db.record_result(Path(path).stem, result_payload, json_archive=str(path))
            except Exception:
                pass
        payload = {"summary": db.summary(), "frontier": db.frontier(args.top)}
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            try:
                _print_status_text(payload)
            except (KeyError, TypeError, AttributeError):
                print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.command == "status":
        db = _get_db(args)
        summary = db.summary()
        frontier = db.frontier(args.top)
        payload = {"summary": summary, "frontier": frontier}
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            _print_status_text(payload)
        return 0

    if args.command == "query-records":
        db = _get_db(args)
        kind = args.kind or "results"
        name = args.name
        family = args.family
        status = args.status
        top_k = max(args.top, 0)

        if kind == "jobs":
            clauses, params = ["1=1"], []
            if name:
                clauses.append("name LIKE ?")
                params.append(f"%{name}%")
            if status:
                clauses.append("state = ?")
                params.append(status)
            params.append(top_k)
            records = db.query(
                f"SELECT * FROM jobs WHERE {' AND '.join(clauses)} ORDER BY rowid DESC LIMIT ?",
                tuple(params),
            )
        elif kind == "probes":
            if not name:
                raise SystemExit("--name is required for kind=probes")
            records = db.query(
                "SELECT * FROM probes WHERE name = ? ORDER BY step LIMIT ?",
                (name, top_k),
            )
        else:
            clauses, params = ["1=1"], []
            if family:
                clauses.append("r.family = ?")
                params.append(family)
            if name:
                clauses.append("r.name LIKE ?")
                params.append(f"%{name}%")
            params.append(top_k)
            records = db.query(
                f"SELECT r.* FROM results r WHERE {' AND '.join(clauses)} ORDER BY r.bpb LIMIT ?",
                tuple(params),
            )

        payload = {"count": len(records), "records": records}
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            _print_query_text(records)
        return 0

    if args.command == "frontier":
        db = _get_db(args)
        frontier = db.frontier(args.top)
        summary = db.summary()
        payload = {"summary": summary, "frontier": frontier}
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            _print_frontier_text(payload)
        return 0

    if args.command == "serve":
        from chronohorn.observe.serve import main as serve_main
        return serve_main(["--port", str(args.port), "--result-dir", args.result_dir])

    raise SystemExit(f"unknown chronohorn observe subcommand: {args.command}")
