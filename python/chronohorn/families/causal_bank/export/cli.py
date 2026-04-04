from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chronohorn export",
        description="Chronohorn export surface for manifest-first chronohorn-export bundles.",
    )
    subparsers = parser.add_subparsers(dest="command")

    smoke = subparsers.add_parser(
        "write-smoke-bundle",
        help="write a tiny synthetic chronohorn-export bundle for CLI/runtime smoke tests",
    )
    smoke.add_argument("--output-dir", required=True)
    smoke.add_argument("--family", required=True)
    smoke.add_argument("--variant", default="smoke")
    smoke.add_argument("--kernel-version", default="open-predictive-coder-smoke")
    smoke.add_argument("--tokenizer-id", default="smoke-tokenizer")
    smoke.add_argument("--data-root-id", default="smoke-data-root")
    smoke.add_argument("--artifact-role", default="replay")
    smoke.add_argument("--note", default="synthetic smoke bundle")

    return parser


def _write_smoke_bundle(args: argparse.Namespace) -> int:
    import numpy as np

    from .bundle import write_opc_export_bundle

    output_dir = Path(args.output_dir)
    write_opc_export_bundle(
        output_dir,
        model_family_id=args.family,
        model_variant_id=args.variant,
        kernel_version=args.kernel_version,
        tokenizer_id=args.tokenizer_id,
        data_root_id=args.data_root_id,
        artifact_role=args.artifact_role,
        deterministic_substrate={
            "substrate_family": args.family,
            "layer_count": 0,
            "hidden_size": 4,
            "readout_kind": "smoke",
            "readout_shape": {"kind": "smoke"},
            "routing_kind": "none",
            "routing_shape": {"experts": 0},
            "activation_kind": "identity",
            "memory_kind": "none",
            "feature_view_kind": "smoke",
        },
        learned_state={
            "smoke.weight": np.zeros((2, 4), dtype=np.float32),
            "smoke.bias": np.zeros((2,), dtype=np.float32),
        },
        notes={"notes": args.note},
    )
    print(output_dir / "manifest.json")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "write-smoke-bundle":
        return _write_smoke_bundle(args)
    parser.error(f"unknown export command: {args.command}")
