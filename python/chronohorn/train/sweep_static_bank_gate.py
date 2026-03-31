#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

from chronohorn.engine.probes import resolve_probe_plan
from chronohorn.train.causal_bank_training_support import (
    DEFAULT_ROWS,
    build_output_path,
    load_existing_result,
    parse_row_spec,
    result_matches,
    summary_row,
)
from chronohorn.train.train_causal_bank_mlx import build_parser as build_bridge_parser
from chronohorn.train.train_causal_bank_mlx import run_bridge


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Restartable single-process Chronohorn static-bank-gate plateau sweep for the causal-bank family."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--python-note", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--stamp", default=time.strftime("%Y-%m-%d"))
    parser.add_argument("--summary-json", default=None)
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
    return parser


def run_sweep(args: argparse.Namespace) -> None:
    rows = [parse_row_spec(raw) for raw in args.row] if args.row else list(DEFAULT_ROWS)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = (
        Path(args.summary_json)
        if args.summary_json
        else out_dir / f"causal_bank_static_bank_gate_sweep_{args.stamp}.json"
    )

    bridge_parser = build_bridge_parser()
    summary_rows: list[dict[str, object]] = []
    sweep_started = time.time()

    print("\n  causal-bank static-bank-gate plateau sweep\n")
    print(
        f"  rows={len(rows)} profile={args.profile} variant={args.variant} "
        f"probe_steps={args.probe_steps} compile_train_step={args.compile_train_step} "
        f"compile_eval={args.compile_eval}"
    )
    print(f"  out_dir={out_dir}")
    if args.python_note:
        print(f"  python={args.python_note}")

    for index, (scale, steps, seed) in enumerate(rows, start=1):
        row_probe_plan = resolve_probe_plan(
            max_step=steps,
            raw_steps=args.probe_steps,
            policy=args.probe_policy,
            default_eval_batches=args.probe_eval_batches,
            standard_eval_batches=(
                args.probe_standard_eval_batches
                if args.probe_standard_eval_batches is not None
                else args.probe_eval_batches
            ),
            micro_eval_batches=args.probe_micro_eval_batches,
            promotion_eval_batches=args.probe_promotion_eval_batches,
            final_eval_batches=args.final_eval_batches,
            geometric_start_step=args.probe_geometric_start,
            geometric_ratio=args.probe_geometric_ratio,
            micro_cutoff_step=args.probe_micro_cutoff_step,
            promotion_count=args.probe_promotion_count,
        )
        row_probe_steps = [int(step) for step in row_probe_plan.get("steps", [])]
        json_path = build_output_path(
            out_dir,
            args.stamp,
            variant=args.variant,
            scale=scale,
            steps=steps,
            seed=seed,
            static_bank_gate=True,
            backend_label="mlx_train",
        )
        existing = load_existing_result(json_path)
        if result_matches(
            existing,
            scale=scale,
            steps=steps,
            seed=seed,
            probe_steps=row_probe_steps,
            compile_train_step=args.compile_train_step,
            compile_eval=args.compile_eval,
            probe_plan=row_probe_plan,
        ):
            print(f"  skip {index}/{len(rows)} scale={scale:.1f} steps={steps} seed={seed} json={json_path.name}")
            summary_rows.append(summary_row(existing, json_path, skipped=True))
            summary_path.write_text(
                json.dumps(
                    {
                        "sweep": {
                            "stamp": args.stamp,
                            "profile": args.profile,
                            "variant": args.variant,
                            "probe_steps": args.probe_steps,
                            "compile_train_step": args.compile_train_step,
                            "compile_eval": args.compile_eval,
                            "elapsed_sec": time.time() - sweep_started,
                        },
                        "rows": summary_rows,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            continue

        print(f"  run  {index}/{len(rows)} scale={scale:.1f} steps={steps} seed={seed} json={json_path.name}")
        bridge_argv = [
            "--data-root",
            args.data_root,
            "--profile",
            args.profile,
            "--variant",
            args.variant,
            "--scale",
            str(scale),
            "--steps",
            str(steps),
            "--seed",
            str(seed),
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
            ",".join(str(step) for step in row_probe_steps),
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
            "--json",
            str(json_path),
            "--static-bank-gate",
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
        if args.compile_eval:
            bridge_argv.append("--compile-eval")
        for bits in sorted(set(args.quant_bits)):
            bridge_argv.extend(["--quant-bits", str(bits)])

        bridge_args = bridge_parser.parse_args(bridge_argv)
        result = run_bridge(bridge_args)
        summary_rows.append(summary_row(result, json_path, skipped=False))
        summary_path.write_text(
            json.dumps(
                {
                    "sweep": {
                        "stamp": args.stamp,
                        "profile": args.profile,
                        "variant": args.variant,
                        "probe_steps": args.probe_steps,
                        "compile_train_step": args.compile_train_step,
                        "compile_eval": args.compile_eval,
                        "elapsed_sec": time.time() - sweep_started,
                    },
                    "rows": summary_rows,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        gc.collect()

    best_row = min(
        (row for row in summary_rows if row.get("test_bpb") is not None),
        key=lambda row: float(row["test_bpb"]),
        default=None,
    )
    if best_row is not None:
        print(
            f"\n  best scale={best_row['scale']} steps={best_row['steps']} "
            f"seed={best_row['seed']} bpb={best_row['test_bpb']:.4f}"
        )
    print(f"  wrote sweep summary to {summary_path}")


def main(argv: list[str] | None = None) -> None:
    run_sweep(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
