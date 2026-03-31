from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


CHRONOHORN_MONOREPO = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = CHRONOHORN_MONOREPO / "chronohorn" / "manifests" / "frontier_ablation_matrix.jsonl"


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
) -> str:
    args = [
        "if ! python -c \"import sentencepiece\" >/dev/null 2>&1; then python -m pip install -q sentencepiece; fi",
        "mkdir -p /run/results",
        (
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
            f"--probe-steps {steps} --probe-eval-batches {probe_eval_batches} "
            f"--final-eval-batches {final_eval_batches} "
            f"--learning-rate {learning_rate} --weight-decay {weight_decay} "
            f"--device cuda "
            + ("--static-bank-gate " if static_bank_gate else "")
            + (f"--bank-gate-span {bank_gate_span} " if static_bank_gate else "")
            + (f"--local-scale-override {local_scale_override} " if local_scale_override is not None else "")
            + f"--json /run/results/{row_name}.json"
        ),
    ]
    return "; ".join(args)


def build_current_regime_scan() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    work_tokens = 1000 * 256 * 16

    def add(name: str, goal: str, **kwargs: object) -> None:
        command = _torch_train_command(row_name=name, **kwargs)
        rows.append(_base_job(name, goal, command, work_tokens=work_tokens))

    # Readout matrix
    add("cb-scan-readout-mlp-s18", "Readout ablation: dense MLP baseline against routed experts.", linear_readout_kind="mlp", linear_readout_num_experts=4)
    add("cb-scan-readout-routed-e4-s18", "Readout ablation: routed SqReLU experts with 4 experts.", linear_readout_num_experts=4)
    add("cb-scan-readout-routed-e8-s18", "Readout ablation: routed SqReLU experts with 8 experts.", linear_readout_num_experts=8)
    add("cb-scan-readout-routed-e16-s18", "Readout ablation: routed SqReLU experts with 16 experts.", linear_readout_num_experts=16)

    # Variant matrix
    add("cb-scan-variant-base-s18", "Variant ablation: base local window variant.", variant="base", local_window=8)
    add("cb-scan-variant-window4-s18", "Variant ablation: window4 promoted variant.", variant="window4", local_window=4)
    add("cb-scan-variant-window16-s18", "Variant ablation: longer local window variant.", variant="window16", local_window=16)
    add("cb-scan-variant-shared-embedding-s18", "Variant ablation: shared embedding variant.", variant="shared_embedding", local_window=8)
    add("cb-scan-variant-gated-s18", "Variant ablation: gated mix variant.", variant="gated", local_window=8)

    # Scale matrix
    add("cb-scan-scale16-routed-e8", "Scale ablation: lower width at scale 16.", scale=16.0)
    add("cb-scan-scale17-routed-e8", "Scale ablation: near-frontier width at scale 17.", scale=17.0)
    add("cb-scan-scale18-routed-e8", "Scale ablation: promoted current base at scale 18.", scale=18.0)
    add("cb-scan-scale19-routed-e8", "Scale ablation: slightly larger causal-bank width at scale 19.", scale=19.0)

    # Static bank gate matrix
    add("cb-scan-gate-off-routed-e8", "Static-bank-gate ablation: gate disabled.", static_bank_gate=False)
    add("cb-scan-gate-span025-routed-e8", "Static-bank-gate ablation: narrow gate span 0.25.", bank_gate_span=0.25)
    add("cb-scan-gate-span050-routed-e8", "Static-bank-gate ablation: promoted gate span 0.5.", bank_gate_span=0.5)
    add("cb-scan-gate-span100-routed-e8", "Static-bank-gate ablation: wide gate span 1.0.", bank_gate_span=1.0)

    # Spectral matrix
    add("cb-scan-hlmax8-routed-e8", "Half-life ablation: tighter long-memory cap at 8.", linear_half_life_max=8.0)
    add("cb-scan-hlmax16-routed-e8", "Half-life ablation: promoted long-memory cap at 16.", linear_half_life_max=16.0)
    add("cb-scan-hlmax32-routed-e8", "Half-life ablation: broader long-memory cap at 32.", linear_half_life_max=32.0)
    add("cb-scan-oscfrac000-routed-e8", "Oscillatory fraction ablation: no oscillatory modes.", oscillatory_frac=0.0)
    add("cb-scan-oscfrac050-routed-e8", "Oscillatory fraction ablation: half oscillatory modes.", oscillatory_frac=0.5)
    add("cb-scan-oscfrac0875-routed-e8", "Oscillatory fraction ablation: promoted 0.875 oscillatory fraction.", oscillatory_frac=0.875)
    add("cb-scan-oscfrac099-routed-e8", "Oscillatory fraction ablation: near-max oscillatory fraction.", oscillatory_frac=0.99)

    # Local path matrix
    add("cb-scan-local-window1-routed-e8", "Local-path ablation: minimal local window.", local_window=1)
    add("cb-scan-local-window4-routed-e8", "Local-path ablation: promoted local window 4.", local_window=4)
    add("cb-scan-local-window8-routed-e8", "Local-path ablation: broader local window 8.", local_window=8)
    add("cb-scan-local-window16-routed-e8", "Local-path ablation: widest local window 16.", local_window=16)
    add("cb-scan-local-scale0125-routed-e8", "Local-path ablation: weaker local residual scale 0.125.", local_scale_override=0.125)
    add("cb-scan-local-scale0250-routed-e8", "Local-path ablation: promoted local residual scale 0.25.", local_scale_override=0.25)
    add("cb-scan-local-scale0500-routed-e8", "Local-path ablation: stronger local residual scale 0.5.", local_scale_override=0.5)

    # Optimization matrix
    add("cb-scan-lr0005-routed-e8", "Optimization ablation: lower learning rate 5e-4.", learning_rate=5e-4)
    add("cb-scan-lr0010-routed-e8", "Optimization ablation: promoted learning rate 1e-3.", learning_rate=1e-3)
    add("cb-scan-lr0015-routed-e8", "Optimization ablation: higher learning rate 1.5e-3.", learning_rate=1.5e-3)
    add("cb-scan-wd0-routed-e8", "Optimization ablation: zero weight decay.", weight_decay=0.0)
    add("cb-scan-wd1e5-routed-e8", "Optimization ablation: promoted weight decay 1e-5.", weight_decay=1e-5)
    add("cb-scan-wd5e5-routed-e8", "Optimization ablation: stronger weight decay 5e-5.", weight_decay=5e-5)
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="chronohorn fleet emit-causal-bank-matrix",
        description="Emit the current frontier-safe causal-bank CUDA ablation scan manifest.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output JSONL manifest path.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = build_current_regime_scan()
    with output.open("w", encoding="utf-8") as handle:
        handle.write("# Current-regime causal-bank CUDA ablation scan.\n")
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "job_count": len(rows)}, indent=2, sort_keys=True))
    return 0
