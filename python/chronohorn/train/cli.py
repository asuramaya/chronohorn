from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Sequence

from chronohorn._entrypoints import dispatch_module


@dataclass(frozen=True)
class TrainEntrypoint:
    module: str
    help: str


_CANONICAL_ENTRYPOINTS: dict[str, TrainEntrypoint] = {
    "train-causal-bank-mlx": TrainEntrypoint(
        module="chronohorn.train.train_causal_bank_mlx",
        help="MLX/Metal causal-bank training on token shards",
    ),
    "train-causal-bank-torch": TrainEntrypoint(
        module="chronohorn.train.train_causal_bank_torch",
        help="Torch/CUDA causal-bank training on token shards",
    ),
    "measure-backend-parity": TrainEntrypoint(
        module="chronohorn.train.measure_backend_parity",
        help="backend parity measurement on a deterministic fixed batch",
    ),
    "sweep-static-bank-gate": TrainEntrypoint(
        module="chronohorn.train.sweep_static_bank_gate",
        help="restartable static-bank-gate plateau sweep",
    ),
    "queue-static-bank-gate": TrainEntrypoint(
        module="chronohorn.train.queue_static_bank_gate",
        help="local static-bank-gate training queue with lock and log handling",
    ),
}

def build_parser() -> argparse.ArgumentParser:
    entrypoint_lines = []
    for canonical_name, entrypoint in _CANONICAL_ENTRYPOINTS.items():
        entrypoint_lines.append(f"- {canonical_name}: {entrypoint.help}")
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
        choices=sorted(_CANONICAL_ENTRYPOINTS),
        nargs="?",
        help="which Chronohorn train command to launch",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = list(argv or [])
    if not args:
        parser.print_help()
        return 0
    if args[0] in {"-h", "--help"}:
        parser.print_help()
        return 0
    entrypoint = args[0]
    if entrypoint not in _CANONICAL_ENTRYPOINTS:
        parser.error(f"unknown train entrypoint: {entrypoint}")
    module_name = _CANONICAL_ENTRYPOINTS[entrypoint].module
    return dispatch_module(module_name, args[1:], require_installed=False)
