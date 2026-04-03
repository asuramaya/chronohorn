from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from chronohorn.families import FrontierTopology, available_family_ids, resolve_frontier_emitter


def _parse_env_pairs(values: Sequence[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for raw in values:
        key, sep, value = raw.partition("=")
        if not key or not sep:
            raise ValueError(f"invalid --env entry {raw!r}; expected KEY=VALUE")
        env[key] = value
    return env


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="chronohorn fleet emit-family-matrix",
        description="Emit a family-specific frontier manifest through the generic Chronohorn family registry.",
    )
    parser.add_argument("--family", required=True, choices=available_family_ids(), help="Family to emit.")
    parser.add_argument("--regime", default="current", help="Family-defined frontier regime name.")
    parser.add_argument("--output", default=None, help="Output JSONL manifest path.")
    parser.add_argument("--host", action="append", default=[], help="Eligible host for emitted jobs (repeatable).")
    parser.add_argument("--image", default=None, help="Container image for emitted jobs.")
    parser.add_argument("--source-dir", default=None, help="Source tree root to snapshot.")
    parser.add_argument("--remote-cwd-rel", default=None, help="Working directory inside the remote snapshot.")
    parser.add_argument(
        "--snapshot-path",
        action="append",
        default=[],
        help="Relative path to include in the remote snapshot (repeatable).",
    )
    parser.add_argument(
        "--data-root-remote",
        default=None,
        help="Remote data-root path passed to the family trainer inside the container.",
    )
    parser.add_argument("--env", action="append", default=[], help="Extra environment variable in KEY=VALUE form.")
    return parser.parse_args(argv)


def _topology_from_args(args: argparse.Namespace) -> FrontierTopology:
    base = FrontierTopology(
        source_dir=str(Path(__file__).resolve().parents[4]),
    )
    env = dict(base.env)
    env.update(_parse_env_pairs(args.env or []))
    snapshot_paths = tuple(args.snapshot_path) if args.snapshot_path else base.snapshot_paths
    hosts = tuple(args.host) if args.host else base.hosts
    return FrontierTopology(
        source_dir=str(Path(args.source_dir or base.source_dir).expanduser().resolve()),
        remote_cwd_rel=args.remote_cwd_rel or base.remote_cwd_rel,
        hosts=hosts,
        image=args.image or base.image,
        snapshot_paths=snapshot_paths,
        env=env,
        remote_data_root=args.data_root_remote or base.remote_data_root,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    emitter = resolve_frontier_emitter(args.family)
    topology = _topology_from_args(args)
    rows = emitter.build_scan_rows(regime=args.regime, topology=topology)
    output = Path(args.output or emitter.default_output_path(regime=args.regime)).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    header = f"# {args.family} frontier regime: {args.regime}.\n"
    with output.open("w", encoding="utf-8") as handle:
        handle.write(header)
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"family": args.family, "job_count": len(rows), "output": str(output), "regime": args.regime}, indent=2, sort_keys=True))
    return 0
