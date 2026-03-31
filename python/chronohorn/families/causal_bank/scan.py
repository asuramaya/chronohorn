from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


CHRONOHORN_MONOREPO = Path(__file__).resolve().parents[5]
DEFAULT_OUTPUT = CHRONOHORN_MONOREPO / "chronohorn" / "manifests" / "frontier_ablation_matrix.jsonl"
DEFAULT_LONG_SLOP_OUTPUT = CHRONOHORN_MONOREPO / "chronohorn" / "manifests" / "frontier_long_slop_matrix.jsonl"


def _common_snapshot_paths() -> list[str]:
    return [
        "chronohorn/python",
        "chronohorn/data/roots/fineweb10B_sp1024",
        "chronohorn/data/tokenizers",
        "open-predictive-coder/src",
    ]


def _base_job(name: str, goal: str, command: str, *, work_tokens: int) -> dict[str, object]:
    return {
        "name": name,
        "backend": "cuda",
        "resource_class": "cuda_gpu",
        "workload_kind": "training.frontier",
        "work_tokens": work_tokens,
        "goal": goal,
        "launcher": "managed_command",
        "hosts": ["slop-01", "slop-02"],
        "image": "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime",
        "gpu": True,
        "source_dir": str(CHRONOHORN_MONOREPO),
        "snapshot_paths": _common_snapshot_paths(),
        "remote_cwd_rel": "chronohorn",
        "env": {"PYTHONUNBUFFERED": "1"},
        "command": command,
    }


def _adaptive_probe_args(
    *,
    geometric_start: int,
    geometric_ratio: float,
    micro_cutoff_step: int,
    standard_eval_batches: int,
    micro_eval_batches: int,
    promotion_eval_batches: int,
    promotion_count: int,
) -> str:
    return (
        f"--probe-policy adaptive "
        f"--probe-geometric-start {geometric_start} "
        f"--probe-geometric-ratio {geometric_ratio} "
        f"--probe-micro-cutoff-step {micro_cutoff_step} "
        f"--probe-standard-eval-batches {standard_eval_batches} "
        f"--probe-micro-eval-batches {micro_eval_batches} "
        f"--probe-promotion-eval-batches {promotion_eval_batches} "
        f"--probe-promotion-count {promotion_count}"
    )


def _torch_train_command(
    *,
    row_name: str,
    scale: float = 18.0,
    variant: str = "window4",
    steps: int = 1000,
    seq_len: int = 256,
    batch_size: int = 16,
    linear_readout_kind: str = "routed_sqrelu_experts",
    linear_readout_num_experts: int = 8,
    linear_half_life_max: float = 16.0,
    oscillatory_frac: float = 0.875,
    oscillatory_period_min: float = 4.0,
    oscillatory_period_max: float = 64.0,
    static_bank_gate: bool = True,
    bank_gate_span: float = 0.5,
    local_window: int = 4,
    local_scale_override: float | None = 0.25,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    seed: int = 42,
    final_eval_batches: int = 20,
    probe_eval_batches: int = 8,
    probe_steps: str | None = None,
    probe_policy: str = "explicit",
    probe_geometric_start: int = 100,
    probe_geometric_ratio: float = 2.0,
    probe_micro_cutoff_step: int = 400,
    probe_standard_eval_batches: int | None = None,
    probe_micro_eval_batches: int | None = None,
    probe_promotion_eval_batches: int | None = None,
    probe_promotion_count: int = 1,
) -> str:
    probe_args = (
        f"--probe-steps {probe_steps or steps} --probe-eval-batches {probe_eval_batches} "
        if probe_policy == "explicit"
        else (
            _adaptive_probe_args(
                geometric_start=probe_geometric_start,
                geometric_ratio=probe_geometric_ratio,
                micro_cutoff_step=probe_micro_cutoff_step,
                standard_eval_batches=(
                    probe_standard_eval_batches
                    if probe_standard_eval_batches is not None
                    else probe_eval_batches
                ),
                micro_eval_batches=(
                    probe_micro_eval_batches
                    if probe_micro_eval_batches is not None
                    else max(1, min(probe_eval_batches, max(probe_eval_batches // 2, 1)))
                ),
                promotion_eval_batches=(
                    probe_promotion_eval_batches
                    if probe_promotion_eval_batches is not None
                    else max(probe_eval_batches * 2, probe_eval_batches)
                ),
                promotion_count=probe_promotion_count,
            )
            + " "
        )
    )
    train_command = (
        "PYTHONPATH=python python -m chronohorn train train-causal-bank-torch "
        f"--data-root /snapshot/chronohorn/data/roots/fineweb10B_sp1024 "
        f"--profile pilot --variant {variant} --scale {scale} --steps {steps} "
        f"--seq-len {seq_len} --batch-size {batch_size} --seed {seed} "
        f"--linear-half-life-max {linear_half_life_max} "
        f"--oscillatory-frac {oscillatory_frac} "
        f"--oscillatory-period-min {oscillatory_period_min} "
        f"--oscillatory-period-max {oscillatory_period_max} "
        f"--local-window {local_window} "
        f"--linear-readout-kind {linear_readout_kind} "
        f"--linear-readout-num-experts {linear_readout_num_experts} "
        + probe_args
        + f"--final-eval-batches {final_eval_batches} "
        + f"--learning-rate {learning_rate} --weight-decay {weight_decay} "
        + f"--device cuda "
        + ("--static-bank-gate " if static_bank_gate else "")
        + (f"--bank-gate-span {bank_gate_span} " if static_bank_gate else "")
        + (f"--local-scale-override {local_scale_override} " if local_scale_override is not None else "")
        + f"--json /run/results/{row_name}.json"
    )
    args = [
        'if ! python -c "import sentencepiece" >/dev/null 2>&1; then python -m pip install -q sentencepiece; fi',
        "mkdir -p /run/results",
        train_command,
    ]
    return "; ".join(args)


def build_current_regime_scan() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    work_tokens = 1000 * 256 * 16

    def add(name: str, goal: str, **kwargs: object) -> None:
        command = _torch_train_command(row_name=name, **kwargs)
        rows.append(_base_job(name, goal, command, work_tokens=work_tokens))

    add("cb-scan-readout-mlp-s18", "Readout ablation: dense MLP baseline against routed experts.", linear_readout_kind="mlp", linear_readout_num_experts=4)
    add("cb-scan-readout-routed-e4-s18", "Readout ablation: routed SqReLU experts with 4 experts.", linear_readout_num_experts=4)
    add("cb-scan-readout-routed-e8-s18", "Readout ablation: routed SqReLU experts with 8 experts.", linear_readout_num_experts=8)
    add("cb-scan-readout-routed-e16-s18", "Readout ablation: routed SqReLU experts with 16 experts.", linear_readout_num_experts=16)

    add("cb-scan-variant-base-s18", "Variant ablation: base local window variant.", variant="base", local_window=8)
    add("cb-scan-variant-window4-s18", "Variant ablation: window4 promoted variant.", variant="window4", local_window=4)
    add("cb-scan-variant-window16-s18", "Variant ablation: longer local window variant.", variant="window16", local_window=16)
    add("cb-scan-variant-shared-embedding-s18", "Variant ablation: shared embedding variant.", variant="shared_embedding", local_window=8)
    add("cb-scan-variant-gated-s18", "Variant ablation: gated mix variant.", variant="gated", local_window=8)

    add("cb-scan-scale16-routed-e8", "Scale ablation: lower width at scale 16.", scale=16.0)
    add("cb-scan-scale17-routed-e8", "Scale ablation: near-frontier width at scale 17.", scale=17.0)
    add("cb-scan-scale18-routed-e8", "Scale ablation: promoted current base at scale 18.", scale=18.0)
    add("cb-scan-scale19-routed-e8", "Scale ablation: slightly larger causal-bank width at scale 19.", scale=19.0)

    add("cb-scan-gate-off-routed-e8", "Static-bank-gate ablation: gate disabled.", static_bank_gate=False)
    add("cb-scan-gate-span025-routed-e8", "Static-bank-gate ablation: narrow gate span 0.25.", bank_gate_span=0.25)
    add("cb-scan-gate-span050-routed-e8", "Static-bank-gate ablation: promoted gate span 0.5.", bank_gate_span=0.5)
    add("cb-scan-gate-span100-routed-e8", "Static-bank-gate ablation: wide gate span 1.0.", bank_gate_span=1.0)

    add("cb-scan-hlmax8-routed-e8", "Half-life ablation: tighter long-memory cap at 8.", linear_half_life_max=8.0)
    add("cb-scan-hlmax16-routed-e8", "Half-life ablation: promoted long-memory cap at 16.", linear_half_life_max=16.0)
    add("cb-scan-hlmax32-routed-e8", "Half-life ablation: broader long-memory cap at 32.", linear_half_life_max=32.0)
    add("cb-scan-oscfrac000-routed-e8", "Oscillatory fraction ablation: no oscillatory modes.", oscillatory_frac=0.0)
    add("cb-scan-oscfrac050-routed-e8", "Oscillatory fraction ablation: half oscillatory modes.", oscillatory_frac=0.5)
    add("cb-scan-oscfrac0875-routed-e8", "Oscillatory fraction ablation: promoted 0.875 oscillatory fraction.", oscillatory_frac=0.875)
    add("cb-scan-oscfrac099-routed-e8", "Oscillatory fraction ablation: near-max oscillatory fraction.", oscillatory_frac=0.99)

    add("cb-scan-local-window1-routed-e8", "Local-path ablation: minimal local window.", local_window=1)
    add("cb-scan-local-window4-routed-e8", "Local-path ablation: promoted local window 4.", local_window=4)
    add("cb-scan-local-window8-routed-e8", "Local-path ablation: broader local window 8.", local_window=8)
    add("cb-scan-local-window16-routed-e8", "Local-path ablation: widest local window 16.", local_window=16)
    add("cb-scan-local-scale0125-routed-e8", "Local-path ablation: weaker local residual scale 0.125.", local_scale_override=0.125)
    add("cb-scan-local-scale0250-routed-e8", "Local-path ablation: promoted local residual scale 0.25.", local_scale_override=0.25)
    add("cb-scan-local-scale0500-routed-e8", "Local-path ablation: stronger local residual scale 0.5.", local_scale_override=0.5)

    add("cb-scan-lr0005-routed-e8", "Optimization ablation: lower learning rate 5e-4.", learning_rate=5e-4)
    add("cb-scan-lr0010-routed-e8", "Optimization ablation: promoted learning rate 1e-3.", learning_rate=1e-3)
    add("cb-scan-lr0015-routed-e8", "Optimization ablation: higher learning rate 1.5e-3.", learning_rate=1.5e-3)
    add("cb-scan-wd0-routed-e8", "Optimization ablation: zero weight decay.", weight_decay=0.0)
    add("cb-scan-wd1e5-routed-e8", "Optimization ablation: promoted weight decay 1e-5.", weight_decay=1e-5)
    add("cb-scan-wd5e5-routed-e8", "Optimization ablation: stronger weight decay 5e-5.", weight_decay=5e-5)
    return rows


def build_long_slop_scan() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def add(
        name: str,
        goal: str,
        *,
        steps: int,
        scale: float,
        variant: str = "window4",
        seed: int = 42,
        learning_rate: float = 0.0015,
        oscillatory_frac: float = 0.875,
        local_window: int | None = None,
    ) -> None:
        resolved_local_window = local_window
        if resolved_local_window is None:
            resolved_local_window = 16 if variant == "window16" else 4
        work_tokens = steps * 256 * 16
        command = _torch_train_command(
            row_name=name,
            scale=scale,
            variant=variant,
            steps=steps,
            seed=seed,
            learning_rate=learning_rate,
            oscillatory_frac=oscillatory_frac,
            local_window=resolved_local_window,
            linear_readout_kind="routed_sqrelu_experts",
            linear_readout_num_experts=8,
            final_eval_batches=50 if steps <= 5200 else 80,
            probe_policy="adaptive",
            probe_geometric_start=100,
            probe_geometric_ratio=2.0,
            probe_micro_cutoff_step=200,
            probe_standard_eval_batches=4,
            probe_micro_eval_batches=2,
            probe_promotion_eval_batches=12 if steps <= 5200 else 16,
            probe_promotion_count=2,
        )
        rows.append(_base_job(name, goal, command, work_tokens=work_tokens))

    # Phase A: longer ranking pilots on the best short-scan directions.
    add(
        "cb-long-rank-s18-lr0015-window4-s42",
        "Long ranking pilot: promoted window4 routed_e8 line with the winning 1.5e-3 learning rate.",
        steps=2600,
        scale=18.0,
        variant="window4",
        seed=42,
    )
    add(
        "cb-long-rank-s19-lr0015-window4-s42",
        "Long ranking pilot: scale19 routed_e8 with the winning 1.5e-3 learning rate.",
        steps=2600,
        scale=19.0,
        variant="window4",
        seed=42,
    )
    add(
        "cb-long-rank-s18-lr0015-window16-s42",
        "Long ranking pilot: window16 routed_e8 with the winning 1.5e-3 learning rate.",
        steps=2600,
        scale=18.0,
        variant="window16",
        seed=42,
    )
    add(
        "cb-long-rank-s19-lr0015-window16-s42",
        "Long ranking pilot: scale19 plus window16 to test whether both short-scan gains stack.",
        steps=2600,
        scale=19.0,
        variant="window16",
        seed=42,
    )
    add(
        "cb-long-rank-s18-lr0015-osc099-s42",
        "Long ranking pilot: higher oscillatory fraction 0.99 on the scale18 routed_e8 line.",
        steps=2600,
        scale=18.0,
        variant="window4",
        seed=42,
        oscillatory_frac=0.99,
    )
    add(
        "cb-long-rank-s19-lr0010-window4-s42",
        "Long ranking control: scale19 with the older 1e-3 learning rate for late-regime stability comparison.",
        steps=2600,
        scale=19.0,
        variant="window4",
        seed=42,
        learning_rate=0.001,
    )

    # Phase B: seed confirmations on the two most plausible long-horizon bases.
    for scale in (18.0, 19.0):
        for seed in (42, 43, 44):
            add(
                f"cb-long-confirm-s{int(scale):02d}-lr0015-window4-seed{seed}",
                f"Seed confirmation: window4 routed_e8 at scale {scale:.0f} with lr 1.5e-3.",
                steps=5200,
                scale=scale,
                variant="window4",
                seed=seed,
            )

    # Phase C: stretch runs that should define the next pure-causal reference if they hold up.
    add(
        "cb-long-stretch-s18-lr0015-window4-s42",
        "Stretch run: push the strongest promoted scale18 line deeper with adaptive probes and larger final eval.",
        steps=10400,
        scale=18.0,
        variant="window4",
        seed=42,
    )
    add(
        "cb-long-stretch-s19-lr0015-window4-s42",
        "Stretch run: push the strongest scale19 line deeper with adaptive probes and larger final eval.",
        steps=10400,
        scale=19.0,
        variant="window4",
        seed=42,
    )

    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="chronohorn fleet emit-causal-bank-matrix",
        description="Emit causal-bank CUDA scan manifests for the current pilot regime or the long slop regime.",
    )
    parser.add_argument("--output", default=None, help="Output JSONL manifest path.")
    parser.add_argument(
        "--regime",
        choices=["current", "long-slop"],
        default="current",
        help="Which causal-bank scan regime to emit.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    default_output = DEFAULT_OUTPUT if args.regime == "current" else DEFAULT_LONG_SLOP_OUTPUT
    output = Path(args.output or str(default_output)).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = build_current_regime_scan() if args.regime == "current" else build_long_slop_scan()
    with output.open("w", encoding="utf-8") as handle:
        if args.regime == "current":
            handle.write("# Current-regime causal-bank CUDA ablation scan.\n")
        else:
            handle.write("# Long-horizon two-slop causal-bank CUDA matrix.\n")
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "job_count": len(rows), "regime": args.regime}, indent=2, sort_keys=True))
    return 0
