from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .causal_bank_matrix import main as emit_causal_bank_matrix_main
from .dispatch import main as dispatch_main
from .family_matrix import main as emit_family_matrix_main
from .forecast_results import main as forecast_results_main
from .queue import main as queue_main


def _print_help() -> None:
    print(
        "\n".join(
            [
                "usage: chronohorn fleet <subcommand> [args]",
                "",
                "subcommands:",
                "  dispatch               manifest-driven launch and status surface",
                "  queue                  keep feeding eligible hardware lanes from a manifest",
                "  forecast-results       project result JSONs with compute, probe, and decision signals",
                "  emit-family-matrix     emit a frontier manifest through the family registry",
                "  emit-causal-bank-matrix  emit the current causal-bank ablation manifest",
                "  transform              filter and mutate a manifest without editing scan code",
                "  drain                 poll and re-dispatch until manifest is complete",
                "",
                "notes:",
                "  omitting the subcommand defaults to `dispatch` for compatibility",
                "  run `chronohorn fleet <subcommand> --help` for subcommand-specific flags",
            ]
        )
    )


def _transform_main(argv: Sequence[str]) -> int:
    from .manifest_transform import load_and_transform

    parser = argparse.ArgumentParser(
        prog="chronohorn fleet transform",
        description="Filter and mutate a manifest without editing scan code.",
    )
    parser.add_argument("--manifest", required=True, type=Path, help="Source manifest path")
    parser.add_argument("--filter", dest="name_pattern", default=None, metavar="GLOB",
                        help="Filter rows by name glob pattern (e.g. 'ex-a-*')")
    parser.add_argument("--steps", type=int, default=None, help="Override step count")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate")
    parser.add_argument("--output", required=True, type=Path, help="Output manifest path")

    args = parser.parse_args(argv)

    rows = load_and_transform(
        args.manifest,
        name_pattern=args.name_pattern,
        steps=args.steps,
        seed=args.seed,
        learning_rate=args.learning_rate,
        output_path=args.output,
    )
    print(f"Wrote {len(rows)} row(s) to {args.output}")
    return 0


def _drain_main(argv: Sequence[str]) -> int:
    import json as _json

    parser = argparse.ArgumentParser(
        prog="chronohorn fleet drain",
        description="Poll and re-dispatch until a manifest is complete.",
    )
    parser.add_argument("--manifest", required=True, type=Path, help="Manifest JSONL path")
    parser.add_argument("--job", action="append", default=[], help="Restrict to named jobs")
    parser.add_argument("--class", dest="classes", action="append", default=[], help="Restrict to resource classes")
    parser.add_argument("--poll-interval", type=int, default=60, help="Seconds between polls (default 60)")
    parser.add_argument("--result-dir", type=Path, default=None, help="Local directory for pulled results")
    parser.add_argument("--telemetry-glob", action="append", default=[], help="Extra telemetry globs")
    parser.add_argument("--max-ticks", type=int, default=None, help="Maximum poll cycles")

    args = parser.parse_args(argv)

    from .drain import drain_loop

    state = drain_loop(
        args.manifest,
        poll_interval=args.poll_interval,
        job_names=args.job,
        classes=args.classes,
        telemetry_globs=args.telemetry_glob or None,
        result_out_dir=args.result_dir,
        max_ticks=args.max_ticks,
    )

    print(_json.dumps({
        "manifest": state.manifest_path,
        "pending": state.pending,
        "running": state.running,
        "completed": state.completed,
        "blocked": state.blocked,
        "done": state.is_done,
    }, indent=2))
    return 0 if state.is_done else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in {"-h", "--help", "help"}:
        _print_help()
        return 0
    if args and args[0] == "queue":
        return queue_main(args[1:])
    if args and args[0] == "emit-family-matrix":
        return emit_family_matrix_main(args[1:])
    if args and args[0] == "emit-causal-bank-matrix":
        return emit_causal_bank_matrix_main(args[1:])
    if args and args[0] == "forecast-results":
        return forecast_results_main(args[1:])
    if args and args[0] == "drain":
        return _drain_main(args[1:])
    if args and args[0] == "transform":
        return _transform_main(args[1:])
    if args and args[0] == "dispatch":
        return dispatch_main(args[1:])
    return dispatch_main(args)
