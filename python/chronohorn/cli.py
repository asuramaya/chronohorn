from __future__ import annotations

import argparse
import sys
from typing import Sequence

from ._entrypoints import dispatch_module

_EXPORT_MODULE = "chronohorn.families.causal_bank.export"
_FLEET_MODULE = "chronohorn.fleet"
_CONTROL_MODULE = "chronohorn.control"
_OBSERVE_MODULE = "chronohorn.observe"
_TRAIN_MODULE = "chronohorn.train"
_MCP_MODULE = "chronohorn.mcp_transport"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chronohorn",
        description="Chronohorn package surface for descendant training, runtime observation, fleet, export, and MCP entrypoints.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser(
        "export",
        help="dispatch the Chronohorn export surface",
    )
    subparsers.add_parser(
        "pull",
        help="pull new results from remote GPU hosts and ingest into DB",
    )
    subparsers.add_parser(
        "fleet",
        help="dispatch the Chronohorn fleet launch/status surface",
    )
    subparsers.add_parser(
        "control",
        help="dispatch the Chronohorn closed-loop frontier control surface",
    )
    subparsers.add_parser(
        "observe",
        help="dispatch the Chronohorn observer/store surface",
    )

    subparsers.add_parser(
        "train",
        help="dispatch the Chronohorn train surface",
    )
    subparsers.add_parser(
        "sync",
        help="pull + status + changelog + monitors in one command",
    )
    subparsers.add_parser(
        "converge",
        help="plan convergence training on the best config",
    )
    subparsers.add_parser(
        "mcp",
        help="run the Chronohorn MCP stdio server",
    )
    subparsers.add_parser(
        "runtime",
        help="unified runtime: drain + fleet probe + visualization in one process",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not args:
        parser.print_help(sys.stderr)
        return 0
    if args[0] in {"-h", "--help"}:
        parser.print_help(sys.stdout)
        return 0
    if args[0] == "export":
        return dispatch_module(_EXPORT_MODULE, args[1:])
    if args[0] == "pull":
        from chronohorn.fleet.cli import _pull_main
        return _pull_main(args[1:])
    if args[0] == "sync":
        from chronohorn.fleet.cli import _sync_main
        return _sync_main(args[1:])
    if args[0] == "launch":
        from chronohorn.fleet.cli import _launch_main
        return _launch_main(args[1:])
    if args[0] == "converge":
        from chronohorn.fleet.cli import _converge_main
        return _converge_main(args[1:])
    if args[0] == "fleet":
        return dispatch_module(_FLEET_MODULE, args[1:])
    if args[0] == "control":
        return dispatch_module(_CONTROL_MODULE, args[1:])
    if args[0] == "observe":
        return dispatch_module(_OBSERVE_MODULE, args[1:])
    if args[0] == "mcp":
        return dispatch_module(_MCP_MODULE, args[1:])
    if args[0] == "runtime":
        from chronohorn.runtime import main as runtime_main
        return runtime_main(args[1:])
    if args[0] != "train":
        parser.error("only the 'export', 'fleet', 'pull', 'sync', 'converge', 'control', 'observe', 'mcp', and 'train' surfaces are exposed right now")
    return dispatch_module(_TRAIN_MODULE, args[1:])
