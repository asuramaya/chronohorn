from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass

from chronohorn._entrypoints import dispatch_module


@dataclass(frozen=True)
class TrainEntrypoint:
    module: str
    help: str


def _discover_entrypoints() -> dict[str, TrainEntrypoint]:
    """Discover training entrypoints from all registered family adapters."""
    from chronohorn.families.registry import available_family_ids, resolve_training_adapter
    entrypoints: dict[str, TrainEntrypoint] = {}
    for fid in available_family_ids():
        try:
            adapter = resolve_training_adapter(fid)
            for name, (module, help_text) in adapter.training_entrypoints().items():
                entrypoints[name] = TrainEntrypoint(module=module, help=help_text)
        except (KeyError, ImportError, AttributeError):
            pass
    return entrypoints


def build_parser() -> argparse.ArgumentParser:
    entrypoints = _discover_entrypoints()
    entrypoint_lines = [f"- {name}: {ep.help}" for name, ep in sorted(entrypoints.items())]
    parser = argparse.ArgumentParser(
        prog="chronohorn train",
        description=(
            "Chronohorn train surface for descendant-family training, parity, sweep, and queue commands.\n\n"
            "Available entrypoints:\n"
            + "\n".join(entrypoint_lines)
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "entrypoint",
        choices=sorted(entrypoints),
        nargs="?",
        help="which Chronohorn train command to launch",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    entrypoints = _discover_entrypoints()
    args = list(argv or [])
    if not args or args[0] in {"-h", "--help"}:
        build_parser().print_help()
        return 0
    entrypoint = args[0]
    if entrypoint not in entrypoints:
        build_parser().error(f"unknown train entrypoint: {entrypoint}")
    return dispatch_module(entrypoints[entrypoint].module, args[1:], require_installed=False)
