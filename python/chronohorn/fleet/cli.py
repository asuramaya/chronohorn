from __future__ import annotations

import sys
from typing import Sequence

from .causal_bank_matrix import main as emit_causal_bank_matrix_main
from .dispatch import main as dispatch_main
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
                "  emit-causal-bank-matrix  emit the current causal-bank ablation manifest",
                "",
                "notes:",
                "  omitting the subcommand defaults to `dispatch` for compatibility",
                "  run `chronohorn fleet <subcommand> --help` for subcommand-specific flags",
            ]
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in {"-h", "--help", "help"}:
        _print_help()
        return 0
    if args and args[0] == "queue":
        return queue_main(args[1:])
    if args and args[0] == "emit-causal-bank-matrix":
        return emit_causal_bank_matrix_main(args[1:])
    if args and args[0] == "forecast-results":
        return forecast_results_main(args[1:])
    if args and args[0] == "dispatch":
        return dispatch_main(args[1:])
    return dispatch_main(args)
