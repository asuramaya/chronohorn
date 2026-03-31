from __future__ import annotations

import argparse
import sys
from typing import Sequence

from ._entrypoints import dispatch_module

_EXPORT_MODULE = "chronohorn.export"
_FLEET_MODULE = "chronohorn.fleet"
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
        "fleet",
        help="dispatch the Chronohorn fleet launch/status surface",
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
        "mcp",
        help="run the Chronohorn MCP stdio server",
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
    if args[0] == "fleet":
        return dispatch_module(_FLEET_MODULE, args[1:])
    if args[0] == "observe":
        return dispatch_module(_OBSERVE_MODULE, args[1:])
    if args[0] == "mcp":
        return dispatch_module(_MCP_MODULE, args[1:])
    if args[0] != "train":
        parser.error("only the 'export', 'fleet', 'observe', 'mcp', and 'train' surfaces are exposed right now")
    return dispatch_module(_TRAIN_MODULE, args[1:])
