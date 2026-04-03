from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

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
                "  transform              filter and mutate a manifest without editing scan code",
                "  launch                 launch training on a remote GPU (no bash, no docker)",
                "  pull                   pull new results from remote hosts",
                "  status                 show fleet pipeline: containers, progress, waiters",
                "  drain                 poll and re-dispatch until manifest is complete",
                "  sync                  pull + status + changelog + monitors in one command",
                "  converge              plan convergence training on the best config",
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


def _do_one_pull(hosts: list[str], remote_dir: str, result_dir: Path, db) -> tuple[int, int]:
    """Run a single pull cycle. Returns (pulled, ingested) counts."""
    import subprocess

    total_pulled = 0
    total_ingested = 0
    for host in hosts:
        # List remote results
        try:
            out = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host,
                 f"ls {remote_dir}/*.json 2>/dev/null"],
                capture_output=True, text=True, timeout=10
            )
            remote_files = [f.strip() for f in out.stdout.strip().splitlines() if f.strip()]
        except Exception:
            print(f"  {host}: offline", file=sys.stderr)
            continue

        pulled = 0
        for remote_path in remote_files:
            name = Path(remote_path).stem
            local_path = result_dir / f"{name}.json"
            if local_path.exists():
                continue
            # Pull via scp
            try:
                subprocess.run(
                    ["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
                     f"{host}:{remote_path}", str(local_path)],
                    capture_output=True, timeout=30, check=True
                )
                pulled += 1
                # Ingest into DB
                try:
                    import json
                    payload = json.loads(local_path.read_text())
                    if isinstance(payload, dict) and payload.get("model", {}).get("test_bpb"):
                        db.record_result(name, payload, json_archive=str(local_path))
                        total_ingested += 1
                except Exception:
                    pass
            except Exception:
                pass

        total_pulled += pulled
        print(f"  {host}: {pulled} new results ({len(remote_files)} total on host)")

    return total_pulled, total_ingested


def _pull_main(argv: Sequence[str]) -> int:
    """Pull new results from remote GPU hosts and ingest into DB."""
    from chronohorn.db import ChronohornDB
    from chronohorn.observe.serve import FLEET_HOSTS

    parser = argparse.ArgumentParser(prog="chronohorn fleet pull")
    parser.add_argument("--hosts", default=",".join(FLEET_HOSTS))
    parser.add_argument("--remote-dir", default="/data/chronohorn/out/results")
    parser.add_argument("--result-dir", type=Path, default=Path("out/results"))
    parser.add_argument("--db", default=None)
    parser.add_argument("--watch", action="store_true", help="Continuously poll for new results")
    parser.add_argument("--poll-interval", type=int, default=60, help="Seconds between polls in watch mode")
    args = parser.parse_args(argv)

    hosts = [h.strip() for h in args.hosts.split(",")]
    args.result_dir.mkdir(parents=True, exist_ok=True)
    db_path = args.db or str(args.result_dir.parent / "chronohorn.db")
    db = ChronohornDB(db_path)

    total_pulled, total_ingested = _do_one_pull(hosts, args.remote_dir, args.result_dir, db)

    # Print summary
    if total_pulled:
        summary = db.summary()
        best = summary.get("best_bpb")
        print(f"\nPulled {total_pulled}, ingested {total_ingested}. DB: {summary['result_count']} results, best={best:.4f}")
        db.record_event("pull_completed", pulled=total_pulled, ingested=total_ingested)
    else:
        print("No new results.")

    if args.watch:
        import time
        print(f"\nWatching for new results every {args.poll_interval}s... (Ctrl+C to stop)")
        try:
            while True:
                time.sleep(args.poll_interval)
                pulled, ingested = _do_one_pull(hosts, args.remote_dir, args.result_dir, db)
                if pulled:
                    summary = db.summary()
                    best = summary.get("best_bpb")
                    print(f"\nPulled {pulled}, ingested {ingested}. DB: {summary['result_count']} results, best={best:.4f}")
                else:
                    print(".", end="", flush=True)
        except KeyboardInterrupt:
            print("\nStopped watching.")

    db.close()
    return 0


def _status_main(argv: Sequence[str]) -> int:
    """Show full fleet pipeline status: containers, progress, waiters, unpulled results."""
    import subprocess
    from chronohorn.observe.serve import FLEET_HOSTS

    parser = argparse.ArgumentParser(prog="chronohorn fleet status")
    parser.add_argument("--hosts", default=",".join(FLEET_HOSTS))
    parser.add_argument("--remote-dir", default="/data/chronohorn/out/results")
    parser.add_argument("--result-dir", type=Path, default=Path("out/results"))
    args = parser.parse_args(argv)

    hosts = [h.strip() for h in args.hosts.split(",")]
    local_results = set(p.stem for p in args.result_dir.glob("*.json")) if args.result_dir.exists() else set()

    for host in hosts:
        print(f"=== {host} ===")

        # Container + GPU
        try:
            r = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host,
                 'sudo docker ps --format "{{.Names}} {{.Status}}" 2>/dev/null; '
                 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null'],
                capture_output=True, text=True, timeout=10)
            lines = r.stdout.strip().splitlines()
            if lines:
                for line in lines:
                    print(f"  {line}")
            else:
                print("  idle (no container)")
        except Exception:
            print("  offline")
            continue

        # Current job progress (probes + completed jobs)
        try:
            r = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host,
                 'sudo docker logs --tail 100 $(sudo docker ps -q) 2>/dev/null | '
                 'grep -E "^echo |PROBE|FINAL|Saved" | tail -10'],
                capture_output=True, text=True, timeout=10)
            progress = r.stdout.strip()
            if progress:
                # Count completed and show current
                lines = progress.splitlines()
                completed = sum(1 for l in lines if "Saved" in l)
                last_probe = [l for l in lines if "PROBE" in l]
                current_job = [l for l in lines if l.startswith("echo ") or l.startswith("v")]
                if current_job:
                    print(f"  current: {current_job[-1].replace('echo ', '')}")
                if last_probe:
                    print(f"  {last_probe[-1].strip()}")
                print(f"  jobs done in this container: {completed}")
        except Exception:
            pass

        # Waiter processes
        try:
            r = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host,
                 'ps aux | grep -v grep | grep -c "v11_waiter\\|nohup.*bash.*docker"'],
                capture_output=True, text=True, timeout=10)
            n = int(r.stdout.strip() or 0)
            print(f"  waiters queued: {n}")
        except Exception:
            print(f"  waiters: unknown")

        # Unpulled results
        try:
            r = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host,
                 f'ls {args.remote_dir}/*.json 2>/dev/null'],
                capture_output=True, text=True, timeout=10)
            remote = set(Path(f.strip()).stem for f in r.stdout.strip().splitlines() if f.strip())
            unpulled = remote - local_results
            print(f"  remote: {len(remote)} results ({len(unpulled)} unpulled)")
            if unpulled:
                for name in sorted(unpulled)[:5]:
                    print(f"    new: {name}")
                if len(unpulled) > 5:
                    print(f"    ... and {len(unpulled) - 5} more")
        except Exception:
            pass

        print()

    return 0


def _sync_main(argv: Sequence[str]) -> int:
    """Pull + status + changelog + monitors in one command."""
    from chronohorn.db import ChronohornDB
    from chronohorn.observe.serve import FLEET_HOSTS
    from chronohorn.observe.terminal import ascii_frontier_table

    parser = argparse.ArgumentParser(prog="chronohorn sync")
    parser.add_argument("--hosts", default=",".join(FLEET_HOSTS))
    parser.add_argument("--remote-dir", default="/data/chronohorn/out/results")
    parser.add_argument("--result-dir", type=Path, default=Path("out/results"))
    parser.add_argument("--db", default=None)
    args = parser.parse_args(argv)

    hosts = [h.strip() for h in args.hosts.split(",")]
    args.result_dir.mkdir(parents=True, exist_ok=True)
    db_path = args.db or str(args.result_dir.parent / "chronohorn.db")
    db = ChronohornDB(db_path)

    # 1. Pull
    pulled, ingested = _do_one_pull(hosts, args.remote_dir, args.result_dir, db)
    if pulled:
        db.record_event("pull_completed", pulled=pulled, ingested=ingested)

    # 2. Fleet status (inline, compact)
    import subprocess
    for host in hosts:
        try:
            r = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host,
                 'nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null; '
                 'sudo docker ps --format "{{.Names}}" 2>/dev/null | head -1'],
                capture_output=True, text=True, timeout=10)
            lines = r.stdout.strip().splitlines()
            gpu = lines[0].strip() if lines else "?"
            container = lines[1].strip() if len(lines) > 1 else "idle"
            print(f"  {host}: gpu={gpu}, container={container}")
        except Exception:
            print(f"  {host}: offline")

    # 3. Summary
    summary = db.summary()
    print(f"\n  {summary['result_count']} results, best={summary['best_bpb']:.4f} bpb")

    # 4. Changelog
    changelog = db.changelog()
    new = changelog.get("new_results", [])
    if new:
        print(f"  {len(new)} new since last pull:")
        for nr in new[:5]:
            print(f"    {nr['name']:30s}  {nr['bpb']:.4f}")
        if changelog.get("frontier_changed"):
            print(f"  \u2605 FRONTIER MOVED: {changelog['old_best']:.4f} \u2192 {changelog['new_best']:.4f}")
    else:
        print(f"  no new results since last pull")

    # 5. Frontier velocity
    velocity = db.frontier_velocity()
    v = velocity.get("velocity_bpb_per_hour", 0)
    trend = velocity.get("trend", "?")
    print(f"\n  frontier velocity: {v:.4f} bpb/hr ({trend})")
    if v < 0.01 and trend == "decelerating":
        print(f"  \u26a0 Architecture search may be converged. Consider convergence training.")

    # 6. Branch health for recent results
    if new:
        import re
        checked = set()
        for nr in new[:5]:
            prefix = re.sub(r'-\d+k$|-\d+$', '', nr["name"])
            prefix = re.sub(r'-seed\d+$', '', prefix)
            if prefix not in checked:
                health = db.branch_health(prefix)
                if health.get("status") == "dead" and health.get("count", 0) >= 5:
                    print(f"  \u26a0 branch '{prefix}': {health['count']} results, none on frontier (gap: +{health['gap']:.3f})")
                checked.add(prefix)

    db.close()
    return 0


def _launch_main(argv: Sequence[str]) -> int:
    """Launch training on a remote GPU host. No bash scripts, no docker manipulation."""
    import subprocess
    import shlex
    from chronohorn.observe.serve import FLEET_HOSTS

    parser = argparse.ArgumentParser(
        prog="chronohorn launch",
        description="Launch training on a remote GPU. Syncs code, starts docker, runs trainer.",
    )
    parser.add_argument("--host", required=True, help="Remote host (e.g. slop-01)")
    parser.add_argument("--name", default=None, help="Result name (required for single run)")
    parser.add_argument("--arch", required=True, help="Architecture version (e.g. v12)")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds for multi-run (e.g. 42,43,44)")
    parser.add_argument("--name-template", default=None, help="Name template for multi-seed (e.g. noise-{seed})")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true", help="Print command without executing")
    parser.add_argument("--data-root", default="/data/parameter-golf/datasets/fineweb10B_sp1024")
    parser.add_argument("--code-dir", default="/data/chronohorn")
    parser.add_argument("--image", default="pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime")
    parser.add_argument("--sync", action="store_true", default=True, help="Sync code before launch")
    parser.add_argument("--no-sync", dest="sync", action="store_false")

    # Pass-through args for the trainer (everything after --)
    args, extra = parser.parse_known_args(argv)

    host = args.host
    code_dir = args.code_dir
    results_dir = f"{code_dir}/out/results"

    # Step 1: Sync code
    if args.sync:
        print(f"  syncing code to {host}...", end=" ", flush=True)
        r = subprocess.run(
            ["rsync", "-az", "--exclude=.git", "--exclude=__pycache__",
             "--exclude=out", "--exclude=.idea",
             "python/", f"{host}:{code_dir}/python/"],
            capture_output=True, timeout=30,
        )
        subprocess.run(
            ["rsync", "-az", "scripts/", f"{host}:{code_dir}/scripts/"],
            capture_output=True, timeout=30,
        )
        print("done" if r.returncode == 0 else f"FAILED ({r.stderr.decode()[:50]})")

    # Validate name
    if not args.seeds and not args.name:
        parser.error("--name is required for single runs (or use --seeds for multi-seed)")

    # Step 2: Build run list
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
        template = args.name_template or f"{args.name}-seed{{seed}}"
    else:
        seeds = [args.seed]
        template = args.name
        print(f"  \u26a0 single seed ({args.seed}). For reproducible results, use --seeds 42,43,44", file=sys.stderr)

    runs = []
    for seed in seeds:
        name = template.format(seed=seed)
        trainer_args = [
            "python3", "scripts/train_polyhash.py",
            "--arch", args.arch,
            "--data-root", "/data",
            "--device", "cuda",
            "--seed", str(seed),
            "--steps", str(args.steps),
            "--batch-size", str(args.batch_size),
            "--seq-len", str(args.seq_len),
            "--lr", str(args.lr),
            "--lr-hash-mult", "2.0",
            "--cosine-decay",
            "--warmup-steps", str(args.warmup_steps),
            "--json", f"/results/{name}.json",
        ]
        if args.fp16:
            trainer_args.append("--fp16")
        # Add extra pass-through args
        trainer_args.extend(extra)
        runs.append((name, trainer_args))

    # Step 3: Build docker command
    inner_cmds = []
    for name, trainer_args in runs:
        cmd_str = " ".join(shlex.quote(a) for a in trainer_args)
        inner_cmds.append(f"echo {shlex.quote(name)} && {cmd_str}")
    inner_cmds.append("echo DONE")
    inner_script = " && ".join(inner_cmds)

    docker_cmd = [
        "sudo", "docker", "run", "--rm", "--gpus", "all",
        "-v", f"{args.data_root}:/data",
        "-v", f"{code_dir}:/code",
        "-v", f"{results_dir}:/results",
        "-w", "/code",
        "-e", "PYTHONPATH=python",
        "-e", "PYTHONUNBUFFERED=1",
        args.image,
        "bash", "-c", inner_script,
    ]

    ssh_cmd = ["ssh", "-o", "BatchMode=yes", host] + [
        " ".join(shlex.quote(a) for a in docker_cmd)
    ]

    # Step 4: Print summary
    print(f"\n  host:  {host}")
    print(f"  arch:  {args.arch}")
    print(f"  runs:  {len(runs)}")
    for name, _ in runs:
        print(f"    {name}")
    print(f"  steps: {args.steps:,}")
    print(f"  fp16:  {args.fp16}")
    est_tok_s = 400_000 if args.fp16 else 300_000
    est_sec = len(runs) * args.steps * args.batch_size * args.seq_len / est_tok_s
    print(f"  est:   ~{est_sec/60:.0f} min ({est_sec/3600:.1f} GPU-hours)")

    if args.dry_run:
        print(f"\n  (dry run — not launching)")
        print(f"  docker command:\n    {' '.join(docker_cmd[:10])}...")
        return 0

    # Step 5: Launch — write script to remote host, then nohup it
    # This avoids 5 layers of shell escaping (python→ssh→bash→nohup→docker→bash)
    print(f"\n  launching...", flush=True)

    # Build the script content
    docker_cmd_str = " ".join(shlex.quote(a) for a in docker_cmd)
    script_content = f"#!/bin/bash\n{docker_cmd_str}\n"
    script_path = "/tmp/chronohorn_launch.sh"

    # Write script to remote host
    write_cmd = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host, f"cat > {script_path} && chmod +x {script_path}"],
        input=script_content, capture_output=True, text=True, timeout=15,
    )
    if write_cmd.returncode != 0:
        print(f"  FAILED to write script: {write_cmd.stderr[:100]}", file=sys.stderr)
        return 1

    # Nohup the script
    r = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host,
         f"nohup {script_path} > /tmp/chronohorn_launch.log 2>&1 & echo launched"],
        capture_output=True, text=True, timeout=15,
    )
    if r.returncode == 0 and "launched" in r.stdout:
        print(f"  launched on {host}")
    else:
        print(f"  FAILED: {r.stderr[:100]}", file=sys.stderr)
        return 1

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


def _converge_main(argv: Sequence[str]) -> int:
    """Launch convergence training on the best config."""
    from chronohorn.db import ChronohornDB

    parser = argparse.ArgumentParser(prog="chronohorn converge")
    parser.add_argument("--name", help="Run name to converge (default: current best)")
    parser.add_argument("--steps", type=int, default=200000, help="Target steps")
    parser.add_argument("--host", default=None, help="Target host")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--db", default=None)
    args = parser.parse_args(argv)

    db_path = args.db or "out/chronohorn.db"
    db = ChronohornDB(db_path)

    if args.name:
        base = db.query("SELECT * FROM results WHERE name = ?", (args.name,))
        if not base:
            print(f"Run not found: {args.name}", file=sys.stderr)
            db.close()
            return 1
        base = base[0]
    else:
        frontier = db.frontier(1)
        if not frontier:
            print("No results in DB", file=sys.stderr)
            db.close()
            return 1
        base = frontier[0]

    print(f"Convergence training:")
    print(f"  Base: {base['name']} ({base['bpb']:.4f} bpb)")
    print(f"  Target: {args.steps:,} steps")

    # Get the config
    import json as _json
    config_row = db.query("SELECT json_blob FROM configs WHERE id = ?", (base.get("config_id"),))
    if config_row and config_row[0].get("json_blob"):
        cfg = _json.loads(config_row[0]["json_blob"])
        print(f"  Config: {_json.dumps({k: v for k, v in cfg.items() if v is not None}, indent=2)[:200]}")

    # Predict
    pred = db.predict_at_steps(base["name"], args.steps)
    if "predicted_bpb" in pred:
        print(f"  Predicted bpb at {args.steps:,}: {pred['predicted_bpb']:.4f} (r2={pred.get('r2', '?')})")

    # Estimate cost
    tok_s = base.get("tok_s") or 350000
    est_seconds = args.steps * 64 * 512 / tok_s
    print(f"  Estimated time: {est_seconds/3600:.1f} GPU-hours ({est_seconds/60:.0f} min)")

    if args.dry_run:
        print("  (dry run -- not launching)")
    else:
        print("  Launch with: chronohorn fleet dispatch or manual docker command")

    db.close()
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
        # Legacy alias → redirect to generic family matrix with --family causal-bank
        return emit_family_matrix_main(["--family", "causal-bank", *args[1:]])
    if args and args[0] == "forecast-results":
        return forecast_results_main(args[1:])
    if args and args[0] == "launch":
        return _launch_main(args[1:])
    if args and args[0] == "pull":
        return _pull_main(args[1:])
    if args and args[0] == "status":
        return _status_main(args[1:])
    if args and args[0] == "drain":
        return _drain_main(args[1:])
    if args and args[0] == "transform":
        return _transform_main(args[1:])
    if args and args[0] == "sync":
        return _sync_main(args[1:])
    if args and args[0] == "converge":
        return _converge_main(args[1:])
    if args and args[0] == "dispatch":
        return dispatch_main(args[1:])
    return dispatch_main(args)
