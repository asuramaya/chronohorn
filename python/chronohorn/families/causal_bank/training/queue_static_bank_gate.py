#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path
import subprocess
import sys

from chronohorn.families.causal_bank.training.causal_bank_training_support import CHRONOHORN_OUT_ROOT, CHRONOHORN_ROOT
from chronohorn.families.causal_bank.training.sweep_static_bank_gate import build_parser as build_sweep_parser
from chronohorn.families.causal_bank.training.sweep_static_bank_gate import run_sweep


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chronohorn local static-bank-gate queue for the causal-bank family."
    )
    parser.add_argument("--python-note", default=None)
    parser.add_argument(
        "--data-root",
        default=os.environ.get(
            "CHRONOHORN_TOKEN_SHARD_DATA_ROOT",
            str(CHRONOHORN_ROOT / "data" / "roots" / "fineweb10B_sp1024"),
        ),
    )
    parser.add_argument("--out-dir", default=str(CHRONOHORN_OUT_ROOT / "causal_bank"))
    parser.add_argument("--stamp", default=None)
    parser.add_argument("--profile", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--variant", default="window4")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--linear-half-life-max", type=float, default=16.0)
    parser.add_argument("--oscillatory-frac", type=float, default=0.875)
    parser.add_argument("--oscillatory-period-min", type=float, default=4.0)
    parser.add_argument("--oscillatory-period-max", type=float, default=64.0)
    parser.add_argument("--bank-gate-span", type=float, default=0.5)
    parser.add_argument("--probe-steps", default="1000,1400,1800,2200,2600")
    parser.add_argument("--probe-policy", default="adaptive")
    parser.add_argument("--probe-split", choices=["train", "test"], default="test")
    parser.add_argument("--probe-eval-batches", type=int, default=8)
    parser.add_argument("--probe-standard-eval-batches", type=int, default=None)
    parser.add_argument("--probe-micro-eval-batches", type=int, default=None)
    parser.add_argument("--probe-promotion-eval-batches", type=int, default=None)
    parser.add_argument("--probe-geometric-start", type=int, default=50)
    parser.add_argument("--probe-geometric-ratio", type=float, default=2.0)
    parser.add_argument("--probe-micro-cutoff-step", type=int, default=800)
    parser.add_argument("--probe-promotion-count", type=int, default=1)
    parser.add_argument("--final-eval-batches", type=int, default=50)
    parser.add_argument("--compile-train-step", action="store_true", default=True)
    parser.add_argument("--no-compile-train-step", action="store_false", dest="compile_train_step")
    parser.add_argument("--compile-eval", action="store_true", default=True)
    parser.add_argument("--no-compile-eval", action="store_false", dest="compile_eval")
    parser.add_argument("--quant-bits", type=int, action="append", default=[4, 6])
    parser.add_argument("--export-dir", default=None)
    parser.add_argument("--row", action="append", default=[])
    parser.add_argument(
        "--lock-dir",
        default="/tmp/chronohorn_causal_bank_static_bank_gate_queue.lock",
    )
    return parser


def _refuse_if_running() -> None:
    patterns = [
        "chronohorn train train-causal-bank-mlx",
        "chronohorn train sweep-static-bank-gate",
        "chronohorn.families.causal_bank.training.train_causal_bank_mlx",
        "chronohorn.families.causal_bank.training.sweep_static_bank_gate",
    ]
    if subprocess.run(
        ["pgrep", "-f", "|".join(patterns)],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0:
        raise RuntimeError(
            "refusing to start: another Chronohorn trainer or static-bank-gate sweep process is already running"
        )


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    stamp = args.stamp or subprocess.check_output(["date", "+%Y-%m-%d"], text=True).strip()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"causal_bank_static_bank_gate_queue_metal_{stamp}.log"
    summary_path = out_dir / f"causal_bank_static_bank_gate_sweep_{stamp}.json"
    lock_dir = Path(args.lock_dir)

    try:
        lock_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise RuntimeError(
            f"refusing to start: local Chronohorn static-bank-gate queue lock exists at {lock_dir}"
        ) from exc

    try:
        _refuse_if_running()
        with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
            tee = _Tee(sys.stdout, log_file)
            with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                print("starting local Chronohorn causal-bank static-bank-gate queue")
                print(f"python={args.python_note or sys.executable}")
                print(f"data_root={args.data_root}")
                print(f"summary_json={summary_path}")
                print(f"log_path={log_path}")

                bridge_parser = build_sweep_parser()
                bridge_argv = [
                    "--python-note",
                    args.python_note or sys.executable,
                    "--data-root",
                    args.data_root,
                    "--out-dir",
                    str(out_dir),
                    "--summary-json",
                    str(summary_path),
                    "--profile",
                    args.profile,
                    "--variant",
                    args.variant,
                    "--seq-len",
                    str(args.seq_len),
                    "--batch-size",
                    str(args.batch_size),
                    "--linear-half-life-max",
                    str(args.linear_half_life_max),
                    "--oscillatory-frac",
                    str(args.oscillatory_frac),
                    "--oscillatory-period-min",
                    str(args.oscillatory_period_min),
                    "--oscillatory-period-max",
                    str(args.oscillatory_period_max),
                    "--bank-gate-span",
                    str(args.bank_gate_span),
                    "--probe-steps",
                    args.probe_steps,
                    "--probe-policy",
                    args.probe_policy,
                    "--probe-split",
                    args.probe_split,
                    "--probe-eval-batches",
                    str(args.probe_eval_batches),
                    "--probe-geometric-start",
                    str(args.probe_geometric_start),
                    "--probe-geometric-ratio",
                    str(args.probe_geometric_ratio),
                    "--probe-micro-cutoff-step",
                    str(args.probe_micro_cutoff_step),
                    "--probe-promotion-count",
                    str(args.probe_promotion_count),
                    "--final-eval-batches",
                    str(args.final_eval_batches),
                ]
                if args.probe_standard_eval_batches is not None:
                    bridge_argv.extend(["--probe-standard-eval-batches", str(args.probe_standard_eval_batches)])
                if args.probe_micro_eval_batches is not None:
                    bridge_argv.extend(["--probe-micro-eval-batches", str(args.probe_micro_eval_batches)])
                if args.probe_promotion_eval_batches is not None:
                    bridge_argv.extend(["--probe-promotion-eval-batches", str(args.probe_promotion_eval_batches)])
                if args.export_dir:
                    bridge_argv.extend(["--export-dir", args.export_dir])
                if args.compile_train_step:
                    bridge_argv.append("--compile-train-step")
                else:
                    bridge_argv.append("--no-compile-train-step")
                if args.compile_eval:
                    bridge_argv.append("--compile-eval")
                else:
                    bridge_argv.append("--no-compile-eval")
                for bits in sorted(set(args.quant_bits)):
                    bridge_argv.extend(["--quant-bits", str(bits)])
                for row in args.row:
                    bridge_argv.extend(["--row", row])

                run_sweep(bridge_parser.parse_args(bridge_argv))
    finally:
        if lock_dir.exists():
            lock_dir.rmdir()


if __name__ == "__main__":
    main()
