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
    if args and args[0] == "transform":
        return _transform_main(args[1:])
    if args and args[0] == "dispatch":
        return dispatch_main(args[1:])
    return dispatch_main(args)
