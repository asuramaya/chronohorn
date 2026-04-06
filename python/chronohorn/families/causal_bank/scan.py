from __future__ import annotations

import argparse
import json
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from chronohorn.families.adapter import FamilyFrontierEmitter, FrontierTopology

CHRONOHORN_MONOREPO = Path(__file__).resolve().parents[5]
DEFAULT_OUTPUT = CHRONOHORN_MONOREPO / "chronohorn" / "manifests" / "frontier_ablation_matrix.jsonl"
DEFAULT_LONG_SLOP_OUTPUT = CHRONOHORN_MONOREPO / "chronohorn" / "manifests" / "frontier_long_slop_matrix.jsonl"
DEFAULT_EXOTIC_16MB_OUTPUT = CHRONOHORN_MONOREPO / "chronohorn" / "manifests" / "frontier_exotic_16mb.jsonl"
# Snapshot paths for the causal-bank (OPC) family.
# Includes decepticons/src because causal-bank training depends on OPC
# substrate code.  Other model families define their own snapshot_paths in their
# respective scan/manifest modules.
DEFAULT_SNAPSHOT_PATHS = (
    "chronohorn/python",
    "chronohorn/data/tokenizers",
    "decepticons/src",
)


def default_frontier_topology() -> FrontierTopology:
    return FrontierTopology(
        source_dir=str(CHRONOHORN_MONOREPO),
        remote_cwd_rel="chronohorn",
        hosts=("slop-01", "slop-02"),
        image="pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime",
        snapshot_paths=DEFAULT_SNAPSHOT_PATHS,
        env={"PYTHONUNBUFFERED": "1"},
        remote_data_root="/data/chronohorn/fineweb10B_sp1024",
    )


def _base_job(
    name: str,
    goal: str,
    command: str,
    *,
    work_tokens: int,
    topology: FrontierTopology,
    spec: dict[str, object] | None = None,
) -> dict[str, object]:
    row = {
        "name": name,
        "family": "causal-bank",
        "backend": "cuda",
        "resource_class": "cuda_gpu",
        "workload_kind": "training.frontier",
        "work_tokens": work_tokens,
        "goal": goal,
        "launcher": "managed_command",
        "hosts": list(topology.hosts),
        "image": topology.image,
        "gpu": True,
        "source_dir": topology.source_dir,
        "snapshot_paths": list(topology.snapshot_paths),
        "remote_cwd_rel": topology.remote_cwd_rel,
        "env": dict(topology.env),
        "command": command,
    }
    if spec:
        row.update(spec)
    return row


def _training_spec(
    *,
    variant: str = "window4",
    scale: float = 18.0,
    steps: int = 1000,
    seq_len: int = 256,
    batch_size: int = 16,
    linear_readout_kind: str = "routed_sqrelu_experts",
    linear_readout_num_experts: int = 4,
    linear_half_life_max: float = 16.0,
    oscillatory_frac: float = 0.875,
    oscillatory_schedule: str = "logspace",
    oscillatory_period_min: float = 4.0,
    oscillatory_period_max: float = 64.0,
    input_proj_scheme: str = "random",
    memory_kind: str = "none",
    substrate_mode: str = "frozen",
    num_blocks: int = 1,
    block_mixing_ratio: float = 0.25,
    state_dim: int = 0,
    num_heads: int = 1,
    patch_size: int = 1,
    patch_causal_decoder: str = "none",
    num_hemispheres: int = 1,
    fast_hemisphere_ratio: float = 0.25,
    fast_lr_mult: float = 4.0,
    linear_readout_depth: int = 1,
    linear_hidden_mult: float | None = None,
    local_hidden_mult: float | None = None,
    local_poly_order: int = 1,
    substrate_poly_order: int = 1,
    block_stride: int = 1,
    training_noise: float = 0.0,
    adaptive_reg: bool = False,
    trust_routing: bool = False,
    table_path: str = "",
    static_bank_gate: bool = True,
    bank_gate_span: float = 0.5,
    local_window: int = 4,
    local_scale_override: float | None = 0.25,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    seed: int = 42,
    profile: str = "pilot",
) -> dict[str, object]:
    return {
        "profile": profile,
        "variant": variant,
        "scale": scale,
        "steps": steps,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "linear_readout_kind": linear_readout_kind,
        "linear_readout_num_experts": linear_readout_num_experts,
        "linear_half_life_max": linear_half_life_max,
        "oscillatory_frac": oscillatory_frac,
        "oscillatory_schedule": oscillatory_schedule,
        "oscillatory_period_min": oscillatory_period_min,
        "oscillatory_period_max": oscillatory_period_max,
        "input_proj_scheme": input_proj_scheme,
        "memory_kind": memory_kind,
        "substrate_mode": substrate_mode,
        "num_blocks": num_blocks,
        "block_mixing_ratio": block_mixing_ratio,
        "state_dim": state_dim,
        "num_heads": num_heads,
        "patch_size": patch_size,
        "patch_causal_decoder": patch_causal_decoder,
        "num_hemispheres": num_hemispheres,
        "fast_hemisphere_ratio": fast_hemisphere_ratio,
        "fast_lr_mult": fast_lr_mult,
        "linear_readout_depth": linear_readout_depth,
        "linear_hidden_mult": linear_hidden_mult,
        "local_hidden_mult": local_hidden_mult,
        "local_poly_order": local_poly_order,
        "substrate_poly_order": substrate_poly_order,
        "block_stride": block_stride,
        "training_noise": training_noise,
        "adaptive_reg": adaptive_reg,
        "trust_routing": trust_routing,
        "table_path": table_path,
        "static_bank_gate": static_bank_gate,
        "bank_gate_span": bank_gate_span,
        "local_window": local_window,
        "local_scale_override": local_scale_override,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "seed": seed,
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
    topology: FrontierTopology,
    scale: float = 18.0,
    variant: str = "window4",
    steps: int = 1000,
    seq_len: int = 256,
    batch_size: int = 16,
    linear_readout_kind: str = "routed_sqrelu_experts",
    linear_readout_num_experts: int = 4,
    linear_half_life_max: float = 16.0,
    oscillatory_frac: float = 0.875,
    oscillatory_schedule: str = "logspace",
    oscillatory_period_min: float = 4.0,
    oscillatory_period_max: float = 64.0,
    input_proj_scheme: str = "random",
    static_bank_gate: bool = True,
    bank_gate_span: float = 0.5,
    local_window: int = 4,
    local_scale_override: float | None = 0.25,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    seed: int = 42,
    profile: str = "pilot",
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
        f"--data-root {topology.remote_data_root} "
        f"--profile {profile} --variant {variant} --scale {scale} --steps {steps} "
        f"--seq-len {seq_len} --batch-size {batch_size} --seed {seed} "
        f"--linear-half-life-max {linear_half_life_max} "
        f"--oscillatory-frac {oscillatory_frac} "
        f"--oscillatory-schedule {oscillatory_schedule} "
        f"--oscillatory-period-min {oscillatory_period_min} "
        f"--oscillatory-period-max {oscillatory_period_max} "
        f"--input-proj-scheme {input_proj_scheme} "
        f"--local-window {local_window} "
        f"--linear-readout-kind {linear_readout_kind} "
        f"--linear-readout-num-experts {linear_readout_num_experts} "
        + probe_args
        + f"--final-eval-batches {final_eval_batches} "
        + f"--learning-rate {learning_rate} --weight-decay {weight_decay} "
        + "--device cuda "
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


_SPEC_KEY_TO_FLAG: dict[str, str] = {
    "scale": "--scale",
    "steps": "--steps",
    "seq_len": "--seq-len",
    "batch_size": "--batch-size",
    "seed": "--seed",
    "variant": "--variant",
    "profile": "--profile",
    "linear_readout_kind": "--linear-readout-kind",
    "linear_readout_num_experts": "--linear-readout-num-experts",
    "linear_half_life_max": "--linear-half-life-max",
    "oscillatory_frac": "--oscillatory-frac",
    "oscillatory_schedule": "--oscillatory-schedule",
    "oscillatory_period_min": "--oscillatory-period-min",
    "oscillatory_period_max": "--oscillatory-period-max",
    "input_proj_scheme": "--input-proj-scheme",
    "memory_kind": "--memory-kind",
    "substrate_mode": "--substrate-mode",
    "num_blocks": "--num-blocks",
    "block_mixing_ratio": "--block-mixing-ratio",
    "state_dim": "--state-dim",
    "num_heads": "--num-heads",
    "patch_size": "--patch-size",
    "patch_causal_decoder": "--patch-causal-decoder",
    "num_hemispheres": "--num-hemispheres",
    "fast_hemisphere_ratio": "--fast-hemisphere-ratio",
    "fast_lr_mult": "--fast-lr-mult",
    "linear_readout_depth": "--linear-readout-depth",
    "local_window": "--local-window",
    "local_poly_order": "--local-poly-order",
    "substrate_poly_order": "--substrate-poly-order",
    "block_stride": "--block-stride",
    "training_noise": "--training-noise",
    "learning_rate": "--learning-rate",
    "weight_decay": "--weight-decay",
    "table_path": "--table-path",
}

_SPEC_BOOL_FLAGS: dict[str, str] = {
    "static_bank_gate": "--static-bank-gate",
    "adaptive_reg": "--adaptive-reg",
    "trust_routing": "--trust-routing",
}

# value -> (flag, prerequisite_bool_key_or_None)
_SPEC_CONDITIONAL_FLAGS: dict[str, tuple[str, str | None]] = {
    "bank_gate_span": ("--bank-gate-span", "static_bank_gate"),
    "local_scale_override": ("--local-scale-override", None),
    "linear_hidden_mult": ("--linear-hidden-mult", None),
    "local_hidden_mult": ("--local-hidden-mult", None),
}


def _command_from_spec(
    spec: dict[str, object],
    *,
    row_name: str,
    topology: FrontierTopology,
    probe_policy: str = "explicit",
    probe_eval_batches: int = 8,
    probe_steps: str | None = None,
    final_eval_batches: int = 20,
    probe_geometric_start: int = 100,
    probe_geometric_ratio: float = 2.0,
    probe_micro_cutoff_step: int = 400,
    probe_standard_eval_batches: int | None = None,
    probe_micro_eval_batches: int | None = None,
    probe_promotion_eval_batches: int | None = None,
    probe_promotion_count: int = 1,
) -> str:
    probe_args = (
        f"--probe-steps {probe_steps or spec.get('steps', 1000)} --probe-eval-batches {probe_eval_batches} "
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

    parts = [
        f"PYTHONPATH=python python -m chronohorn train train-causal-bank-torch"
        f" --data-root {topology.remote_data_root}",
    ]

    for key, flag in _SPEC_KEY_TO_FLAG.items():
        if key in spec:
            parts.append(f"{flag} {spec[key]}")

    for key, flag in _SPEC_BOOL_FLAGS.items():
        if spec.get(key) is True:
            parts.append(flag)

    for key, (flag, prereq) in _SPEC_CONDITIONAL_FLAGS.items():
        value = spec.get(key)
        if value is not None:
            prereq_ok = prereq is None or spec.get(prereq) is True
            if prereq_ok:
                parts.append(f"{flag} {value}")

    train_command = (
        " ".join(parts)
        + " "
        + probe_args
        + f"--final-eval-batches {final_eval_batches} --device cuda --json /run/results/{row_name}.json"
    )

    args = [
        'if ! python -c "import sentencepiece" >/dev/null 2>&1; then python -m pip install -q sentencepiece; fi',
        "mkdir -p /run/results",
        train_command,
    ]
    return "; ".join(args)


def build_current_regime_scan(topology: FrontierTopology | None = None) -> list[dict[str, object]]:
    active_topology = topology or default_frontier_topology()
    rows: list[dict[str, object]] = []
    work_tokens = 1000 * 256 * 16

    def add(name: str, goal: str, **kwargs: object) -> None:
        command = _torch_train_command(row_name=name, topology=active_topology, **kwargs)
        rows.append(
            _base_job(
                name,
                goal,
                command,
                work_tokens=work_tokens,
                topology=active_topology,
                spec=_training_spec(**kwargs),
            )
        )

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


def build_long_slop_scan(topology: FrontierTopology | None = None) -> list[dict[str, object]]:
    active_topology = topology or default_frontier_topology()
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
            topology=active_topology,
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
        rows.append(
            _base_job(
                name,
                goal,
                command,
                work_tokens=work_tokens,
                topology=active_topology,
                spec=_training_spec(
                    variant=variant,
                    scale=scale,
                    steps=steps,
                    local_window=resolved_local_window,
                    learning_rate=learning_rate,
                    oscillatory_frac=oscillatory_frac,
                    seed=seed,
                ),
            )
        )

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


def _estimate_artifact_mb(
    *,
    scale: float,
    linear_readout_kind: str = "mlp",
    linear_readout_num_experts: int = 4,
    local_window: int = 4,
) -> float:
    """Estimate int6 artifact size in MB for causal-bank models.

    Uses measured reference points to interpolate artifact sizes.
    Formula: params * 6 / 8 / 1024 / 1024
    """
    embed_dim = int(32 * scale)
    linear_modes = int(256 * scale)
    hidden = int(128 * scale)
    vocab = 1024
    # Embeddings (linear + local paths)
    params = 2 * vocab * embed_dim
    # Linear input projection
    params += linear_modes * embed_dim
    # Linear readout
    if linear_readout_kind == "mlp":
        params += hidden * vocab + hidden
    else:
        params += linear_readout_num_experts * (hidden * vocab + vocab)
        params += linear_readout_num_experts * hidden
    # Local path: local embedding + readout
    local_input = local_window * embed_dim
    params += vocab * local_input  # local embedding
    params += int(128 * scale) * vocab + int(128 * scale)  # local readout MLP
    # Bank gate (tiny)
    params += 6
    return round(params * 6 / 8 / 1024 / 1024, 2)


def build_exotic_16mb_scan(topology: FrontierTopology | None = None) -> list[dict[str, object]]:
    """Artifact-viable (<=16MB int6) exotic mutation matrix.

    All runs use adaptive probes, lr=1.5e-3, 1000 steps.  The forecaster
    ranks by marginal_gain_per_tflop after the pilots complete.  The
    control layer then recommends which rows to deepen to 2600/5200/10400.
    Max step cap: 10000.
    """
    active_topology = topology or default_frontier_topology()
    rows: list[dict[str, object]] = []

    COMMON = dict(
        learning_rate=0.0015,
        weight_decay=1e-5,
        seed=42,
        seq_len=256,
        batch_size=16,
        static_bank_gate=True,
        bank_gate_span=0.5,
        local_scale_override=0.25,
    )

    def add(
        name: str,
        goal: str,
        *,
        steps: int = 1000,
        **kwargs: object,
    ) -> None:
        merged = {**COMMON, **kwargs, "steps": steps}
        work_tokens = int(steps * int(merged.get("seq_len", 256)) * int(merged.get("batch_size", 16)))
        final_batches = 20 if steps <= 1000 else (50 if steps <= 5200 else 80)
        promo_batches = 8 if steps <= 1000 else (12 if steps <= 5200 else 16)
        merged_spec = _training_spec(**merged)
        command = _command_from_spec(
            merged_spec,
            row_name=name,
            topology=active_topology,
            probe_policy="adaptive",
            probe_geometric_start=50,
            probe_geometric_ratio=2.0,
            probe_micro_cutoff_step=200,
            probe_standard_eval_batches=4,
            probe_micro_eval_batches=2,
            probe_promotion_eval_batches=promo_batches,
            probe_promotion_count=2,
            final_eval_batches=final_batches,
        )
        rows.append(
            _base_job(
                name,
                goal,
                command,
                work_tokens=work_tokens,
                topology=active_topology,
                spec=merged_spec,
            )
        )
        rows[-1]["artifact_mb_est"] = _estimate_artifact_mb(
            scale=float(merged.get("scale", 14.0)),
            linear_readout_kind=str(merged.get("linear_readout_kind", "mlp")),
            linear_readout_num_experts=int(merged.get("linear_readout_num_experts", 4)),
            local_window=int(merged.get("local_window", 4)),
        )

    # ── Group A: capacity vs routing trade-off ──────────────────────
    add("ex-a-s12-mlp", "Capacity baseline: scale 12 MLP (8.3MB).",
        scale=12.0, linear_readout_kind="mlp")
    add("ex-a-s15-mlp", "Capacity test: scale 15 MLP (12.1MB).",
        scale=15.0, linear_readout_kind="mlp")
    add("ex-a-s17-mlp", "Max MLP capacity: scale 17 (15.0MB).",
        scale=17.0, linear_readout_kind="mlp")
    add("ex-a-s12-e2", "Routing test: scale 12 with 2 routed experts (13.2MB).",
        scale=12.0, linear_readout_kind="routed_sqrelu_experts", linear_readout_num_experts=2)
    add("ex-a-s8-e4", "Routing test: scale 8 with 4 routed experts (11.6MB).",
        scale=8.0, linear_readout_kind="routed_sqrelu_experts", linear_readout_num_experts=4)
    add("ex-a-s6-e8", "Routing diversity: scale 6 with 8 routed experts (13.4MB).",
        scale=6.0, linear_readout_kind="routed_sqrelu_experts", linear_readout_num_experts=8)
    add("ex-a-s4-e16", "Max routing: scale 4 with 16 routed experts (13.5MB).",
        scale=4.0, linear_readout_kind="routed_sqrelu_experts", linear_readout_num_experts=16)

    # ── Group B: oscillatory scheduling ─────────────────────────────
    # Never tested. Same params, different mode selection.
    add("ex-b-mincorr", "Scheduling: mincorr_greedy mode selection.",
        scale=14.0, linear_readout_kind="mlp", oscillatory_schedule="mincorr_greedy")
    add("ex-b-bucket", "Scheduling: period_bucket_greedy mode selection.",
        scale=14.0, linear_readout_kind="mlp", oscillatory_schedule="period_bucket_greedy")
    add("ex-b-mincorr-95", "Scheduling: mincorr + high oscillatory fraction 0.95.",
        scale=14.0, linear_readout_kind="mlp", oscillatory_schedule="mincorr_greedy",
        oscillatory_frac=0.95)

    # ── Group C: oscillatory fraction and period range ──────────────
    add("ex-c-osc000", "Oscillatory ablation: pure exponential decay (no oscillation).",
        scale=14.0, linear_readout_kind="mlp", oscillatory_frac=0.0)
    add("ex-c-osc050", "Oscillatory ablation: half oscillatory modes.",
        scale=14.0, linear_readout_kind="mlp", oscillatory_frac=0.5)
    add("ex-c-osc099", "Oscillatory ablation: near-max oscillatory fraction 0.99.",
        scale=14.0, linear_readout_kind="mlp", oscillatory_frac=0.99)
    add("ex-c-osc095-wide", "Wide periods: osc=0.95 with period range 2-256.",
        scale=14.0, linear_readout_kind="mlp", oscillatory_frac=0.95,
        oscillatory_period_min=2.0, oscillatory_period_max=256.0)
    add("ex-c-osc090-extreme", "Extreme periods: osc=0.90 with period range 1-512.",
        scale=14.0, linear_readout_kind="mlp", oscillatory_frac=0.90,
        oscillatory_period_min=1.0, oscillatory_period_max=512.0)

    # ── Group D: input projection schemes ───────────────────────────
    # Never tested. Same param count, different basis geometry.
    add("ex-d-orthogonal", "Projection: orthogonal_rows QR basis.",
        scale=14.0, linear_readout_kind="mlp", input_proj_scheme="orthogonal_rows")
    add("ex-d-split", "Projection: split_banks separate osc/decay subspaces.",
        scale=14.0, linear_readout_kind="mlp", input_proj_scheme="split_banks")
    add("ex-d-energy", "Projection: kernel_energy RMS-weighted basis.",
        scale=14.0, linear_readout_kind="mlp", input_proj_scheme="kernel_energy")

    # ── Group E: half-life range ────────────────────────────────────
    add("ex-e-hl8", "Half-life: short memory cap at 8.",
        scale=14.0, linear_readout_kind="mlp", linear_half_life_max=8.0)
    add("ex-e-hl64", "Half-life: extended memory cap at 64.",
        scale=14.0, linear_readout_kind="mlp", linear_half_life_max=64.0)
    add("ex-e-hl128", "Half-life: long memory cap at 128.",
        scale=14.0, linear_readout_kind="mlp", linear_half_life_max=128.0)
    add("ex-e-hl256", "Half-life: very long memory cap at 256.",
        scale=14.0, linear_readout_kind="mlp", linear_half_life_max=256.0)

    # ── Group F: mix mode ───────────────────────────────────────────
    add("ex-f-gated", "Mix mode: learned gated path mixing.",
        scale=14.0, linear_readout_kind="mlp", variant="gated", local_window=4)
    add("ex-f-gated-ls05", "Mix mode: gated with higher local scale 0.5.",
        scale=14.0, linear_readout_kind="mlp", variant="gated", local_window=4,
        local_scale_override=0.5)
    add("ex-f-gated-ls10", "Mix mode: gated with equal local scale 1.0.",
        scale=14.0, linear_readout_kind="mlp", variant="gated", local_window=4,
        local_scale_override=1.0)

    # ── Group G: local path ─────────────────────────────────────────
    add("ex-g-w1", "Local window: minimal window 1.",
        scale=14.0, linear_readout_kind="mlp", local_window=1)
    add("ex-g-w8", "Local window: broader window 8 (13.1MB).",
        scale=14.0, linear_readout_kind="mlp", local_window=8)
    add("ex-g-ls05", "Local scale: stronger local weight 0.5.",
        scale=14.0, linear_readout_kind="mlp", local_scale_override=0.5)
    add("ex-g-ls075", "Local scale: strong local weight 0.75.",
        scale=14.0, linear_readout_kind="mlp", local_scale_override=0.75)

    # ── Group H: sequence length ────────────────────────────────────
    add("ex-h-seq128-b32", "Sequence: short seq 128, batch 32 (same tokens/step).",
        scale=14.0, linear_readout_kind="mlp", seq_len=128, batch_size=32)
    add("ex-h-seq512-b8", "Sequence: long seq 512, batch 8 (same tokens/step).",
        scale=14.0, linear_readout_kind="mlp", seq_len=512, batch_size=8)
    add("ex-h-seq512-b16", "Sequence: long seq 512, batch 16 (2x tokens/step).",
        scale=14.0, linear_readout_kind="mlp", seq_len=512, batch_size=16)

    # ── Group I: learning rate ──────────────────────────────────────
    add("ex-i-lr2e3", "LR test: 2e-3 (33% above winner).",
        scale=14.0, linear_readout_kind="mlp", learning_rate=2e-3)
    add("ex-i-lr3e3", "LR test: 3e-3 (double winner).",
        scale=14.0, linear_readout_kind="mlp", learning_rate=3e-3)

    # ── Group J: interaction combos ─────────────────────────────────
    add("ex-j-s17-mincorr-split", "Combo: max MLP + mincorr + split_banks + osc=0.95 + wide periods + hlmax=64.",
        scale=17.0, linear_readout_kind="mlp", oscillatory_schedule="mincorr_greedy",
        input_proj_scheme="split_banks", oscillatory_frac=0.95,
        oscillatory_period_min=2.0, oscillatory_period_max=256.0, linear_half_life_max=64.0)
    add("ex-j-s17-mincorr-energy", "Combo: max MLP + mincorr + kernel_energy + osc=0.90 + hlmax=128.",
        scale=17.0, linear_readout_kind="mlp", oscillatory_schedule="mincorr_greedy",
        input_proj_scheme="kernel_energy", oscillatory_frac=0.90, linear_half_life_max=128.0)
    add("ex-j-s12-e2-mincorr", "Combo: scale 12 e2 + mincorr + split_banks + osc=0.95.",
        scale=12.0, linear_readout_kind="routed_sqrelu_experts", linear_readout_num_experts=2,
        oscillatory_schedule="mincorr_greedy", input_proj_scheme="split_banks", oscillatory_frac=0.95)
    add("ex-j-s8-e4-mincorr", "Combo: scale 8 e4 + mincorr + kernel_energy + osc=0.95.",
        scale=8.0, linear_readout_kind="routed_sqrelu_experts", linear_readout_num_experts=4,
        oscillatory_schedule="mincorr_greedy", input_proj_scheme="kernel_energy",
        oscillatory_frac=0.95, oscillatory_period_min=2.0, oscillatory_period_max=128.0)
    add("ex-j-s14-gated-w8", "Combo: scale 14 gated + window 8 + mincorr + osc=0.95 + ls=0.5.",
        scale=14.0, linear_readout_kind="mlp", variant="gated", local_window=8,
        oscillatory_schedule="mincorr_greedy", oscillatory_frac=0.95, local_scale_override=0.5)
    add("ex-j-s6-e8-mincorr", "Combo: scale 6 e8 + mincorr + osc=0.99 + wide periods + hlmax=64.",
        scale=6.0, linear_readout_kind="routed_sqrelu_experts", linear_readout_num_experts=8,
        oscillatory_schedule="mincorr_greedy", oscillatory_frac=0.99,
        oscillatory_period_min=1.0, oscillatory_period_max=512.0, linear_half_life_max=64.0)
    add("ex-j-s15-mincorr-hl128", "Combo: scale 15 MLP + mincorr + split_banks + hlmax=128 + osc=0.95.",
        scale=15.0, linear_readout_kind="mlp", oscillatory_schedule="mincorr_greedy",
        input_proj_scheme="split_banks", linear_half_life_max=128.0, oscillatory_frac=0.95)
    add("ex-j-s17-allknobs", "Kitchen sink: scale 17 + mincorr + split_banks + gated + osc=0.95 + wide periods + hlmax=64 + ls=0.5.",
        scale=17.0, linear_readout_kind="mlp", variant="gated", local_window=4,
        oscillatory_schedule="mincorr_greedy", input_proj_scheme="split_banks",
        oscillatory_frac=0.95, oscillatory_period_min=2.0, oscillatory_period_max=256.0,
        linear_half_life_max=64.0, local_scale_override=0.5)

    # Warn about over-budget rows
    for row in rows:
        est = row.get("artifact_mb_est", 0)
        if est > 16.0:
            warnings.warn(f"Row {row['name']} estimated at {est:.1f}MB — exceeds 16MB artifact budget")

    return rows


@dataclass(frozen=True)
class CausalBankFrontierEmitter(FamilyFrontierEmitter):
    family_id: str = "causal-bank"

    def supported_regimes(self) -> Sequence[str]:
        return ("current", "long-slop", "exotic-16mb")

    def build_scan_rows(self, *, regime: str, topology: FrontierTopology) -> list[dict[str, object]]:
        if regime == "current":
            return build_current_regime_scan(topology)
        if regime == "long-slop":
            return build_long_slop_scan(topology)
        if regime == "exotic-16mb":
            return build_exotic_16mb_scan(topology)
        raise ValueError(f"unsupported causal-bank frontier regime: {regime}")

    def default_output_path(self, *, regime: str) -> str:
        if regime == "exotic-16mb":
            return str(DEFAULT_EXOTIC_16MB_OUTPUT)
        return str(DEFAULT_OUTPUT if regime == "current" else DEFAULT_LONG_SLOP_OUTPUT)


CAUSAL_BANK_FRONTIER_EMITTER = CausalBankFrontierEmitter()


def _parse_env_pairs(values: Sequence[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for raw in values:
        key, sep, value = raw.partition("=")
        if not key or not sep:
            raise ValueError(f"invalid --env entry {raw!r}; expected KEY=VALUE")
        env[key] = value
    return env


def _topology_from_args(args: argparse.Namespace) -> FrontierTopology:
    base = default_frontier_topology()
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="chronohorn fleet emit-causal-bank-matrix",
        description="Emit causal-bank CUDA scan manifests for the current pilot regime or the long slop regime.",
    )
    parser.add_argument("--output", default=None, help="Output JSONL manifest path.")
    parser.add_argument(
        "--regime",
        choices=["current", "long-slop", "exotic-16mb"],
        default="current",
        help="Which causal-bank scan regime to emit.",
    )
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


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    topology = _topology_from_args(args)
    default_output = Path(CAUSAL_BANK_FRONTIER_EMITTER.default_output_path(regime=args.regime))
    output = Path(args.output or str(default_output)).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = CAUSAL_BANK_FRONTIER_EMITTER.build_scan_rows(regime=args.regime, topology=topology)
    with output.open("w", encoding="utf-8") as handle:
        if args.regime == "current":
            handle.write("# Current-regime causal-bank CUDA ablation scan.\n")
        else:
            handle.write("# Long-horizon two-slop causal-bank CUDA matrix.\n")
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "job_count": len(rows), "regime": args.regime}, indent=2, sort_keys=True))
    return 0
