from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Any

from decepticons.causal_bank import (
    CAUSAL_BANK_INPUT_PROJ_SCHEMES,
    CAUSAL_BANK_OSCILLATORY_SCHEDULES,
    CAUSAL_BANK_READOUT_KINDS,
    CAUSAL_BANK_STATE_IMPLS,
    CAUSAL_BANK_VARIANTS,
)

from chronohorn.engine.probes import PROBE_POLICY_CHOICES
from chronohorn.families.causal_bank.training.causal_bank_training_support import (
    _estimate_mlp_readout_flops,
    _estimate_routed_expert_readout_flops,
    _estimate_tied_readout_flops,
    solve_recursive_hidden_width,
    solve_routed_expert_hidden_width,
)


def add_causal_bank_core_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--profile", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--linear-modes", type=int, default=256)
    parser.add_argument("--linear-readout-kind", choices=CAUSAL_BANK_READOUT_KINDS, default="mlp")
    parser.add_argument("--linear-readout-depth", type=int, default=1)
    parser.add_argument("--linear-readout-num-experts", type=int, default=4)
    parser.add_argument("--readout-bands", type=int, default=1,
                        help="Split modes by timescale into N bands with separate readout heads (default: 1 = single readout)")
    parser.add_argument("--allow-experimental-recursive-readout", action="store_true")
    parser.add_argument(
        "--linear-hidden-match",
        choices=["none", "mlp_params", "mlp_flops"],
        default="mlp_flops",
    )
    parser.add_argument("--local-window", type=int, default=8)
    parser.add_argument("--scale", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--linear-half-life-max", type=float, default=None)
    parser.add_argument("--oscillatory-frac", type=float, default=None)
    parser.add_argument(
        "--oscillatory-schedule",
        choices=CAUSAL_BANK_OSCILLATORY_SCHEDULES,
        default="logspace",
    )
    parser.add_argument("--oscillatory-period-min", type=float, default=4.0)
    parser.add_argument("--oscillatory-period-max", type=float, default=64.0)
    parser.add_argument(
        "--input-proj-scheme",
        choices=CAUSAL_BANK_INPUT_PROJ_SCHEMES,
        default="random",
    )
    parser.add_argument(
        "--memory-kind",
        choices=("none", "ngram", "exact_context", "statistical_backoff"),
        default="none",
    )
    parser.add_argument(
        "--substrate-mode",
        choices=("frozen", "learnable_decays", "learnable_mixing", "learned_recurrence", "gated_retention"),
        default="frozen",
    )
    parser.add_argument("--static-bank-gate", action="store_true")
    parser.add_argument("--bank-gate-span", type=float, default=0.5)
    parser.add_argument("--linear-hidden-width", type=int, default=None)
    parser.add_argument("--linear-hidden-mult", type=float, default=None)
    parser.add_argument("--local-hidden-mult", type=float, default=None)
    parser.add_argument("--local-scale-override", type=float, default=None)
    parser.add_argument("--num-blocks", type=int, default=1)
    parser.add_argument("--block-mixing-ratio", type=float, default=0.25)
    parser.add_argument("--state-dim", type=int, default=0)
    parser.add_argument("--state-impl", choices=CAUSAL_BANK_STATE_IMPLS, default="scan")
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=1)
    parser.add_argument("--patch-causal-decoder", choices=("none", "autoregressive", "mlp_factored", "hybrid"), default="none")
    parser.add_argument("--num-hemispheres", type=int, default=1)
    parser.add_argument("--fast-hemisphere-ratio", type=float, default=0.25)
    parser.add_argument("--fast-lr-mult", type=float, default=4.0)
    parser.add_argument("--local-poly-order", type=int, default=1)
    parser.add_argument("--substrate-poly-order", type=int, default=1)
    parser.add_argument("--block-stride", type=int, default=1)
    parser.add_argument("--training-noise", type=float, default=0.0)
    parser.add_argument("--adaptive-reg", action="store_true")
    parser.add_argument("--trust-routing", action="store_true")
    parser.add_argument("--table-path", default="")
    parser.add_argument("--max-params", type=int, default=100_000_000)
    parser.add_argument("--max-readout-flop-ratio", type=float, default=1.10)
    parser.add_argument("--unsafe-large-model", action="store_true")
    parser.add_argument("--variant", choices=CAUSAL_BANK_VARIANTS, default="base")
    return parser


def add_bridge_evaluation_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--probe-steps", default=None)
    parser.add_argument("--probe-policy", choices=PROBE_POLICY_CHOICES, default="adaptive")
    parser.add_argument("--probe-split", choices=["train", "test"], default="test")
    parser.add_argument("--probe-eval-batches", type=int, default=8)
    parser.add_argument("--probe-standard-eval-batches", type=int, default=None)
    parser.add_argument("--probe-micro-eval-batches", type=int, default=None)
    parser.add_argument("--probe-promotion-eval-batches", type=int, default=None)
    parser.add_argument("--probe-geometric-start", type=int, default=50)
    parser.add_argument("--probe-geometric-ratio", type=float, default=2.0)
    parser.add_argument("--probe-micro-cutoff-step", type=int, default=800)
    parser.add_argument("--probe-promotion-count", type=int, default=1)
    parser.add_argument("--final-eval-batches", type=int, default=None)
    parser.add_argument("--export-dir", default=None)
    parser.add_argument("--json", required=True)
    return parser


def add_mlx_bridge_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--decay-bank",
        choices=["logspace", "narrow", "custom"],
        default="logspace",
    )
    parser.add_argument("--decays-json", default=None)
    parser.add_argument("--quant-bits", type=int, action="append", default=[])
    parser.add_argument("--compile-train-step", action="store_true")
    parser.add_argument("--compile-eval", action="store_true")
    return parser


def add_torch_bridge_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--device", default=None)
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument(
        "--probe-diagnostics",
        action="store_true",
        help="Run expensive model diagnostics on standard-tier probes (off by default for throughput).",
    )
    parser.add_argument("--profile-cuda", type=int, default=0, metavar="N",
                        help="Profile the first N training steps and write a Chrome trace to out/profile/")
    return parser


def add_causal_bank_training_arguments(
    parser: argparse.ArgumentParser,
    *,
    backend: str,
) -> argparse.ArgumentParser:
    add_causal_bank_core_arguments(parser)
    add_bridge_evaluation_arguments(parser)

    if backend == "mlx":
        add_mlx_bridge_arguments(parser)
    elif backend == "torch":
        add_torch_bridge_arguments(parser)
    else:
        raise ValueError(f"Unsupported training backend: {backend}")

    return parser


def build_causal_bank_variant_config(
    args: argparse.Namespace,
    *,
    ConfigClass: Any,
    scale_config: Any,
    seq_len: int,
    vocab_size: int,
):
    config = ConfigClass(
        max_seq_len=seq_len,
        linear_modes=args.linear_modes,
        local_window=args.local_window,
        linear_readout_kind=args.linear_readout_kind,
        linear_readout_depth=args.linear_readout_depth,
        linear_readout_num_experts=args.linear_readout_num_experts,
        readout_bands=args.readout_bands,
        init_seed=args.seed,
    )

    from decepticons.causal_bank import apply_variant as _apply_variant

    variant_cfg = _apply_variant(config, args.variant)

    if getattr(args, "decay_bank", None) == "narrow":
        variant_cfg = replace(variant_cfg, linear_half_life_max=32.0)
    if args.linear_half_life_max is not None:
        variant_cfg = replace(variant_cfg, linear_half_life_max=args.linear_half_life_max)
    variant_cfg = replace(
        variant_cfg,
        oscillatory_schedule=args.oscillatory_schedule,
        oscillatory_period_min=args.oscillatory_period_min,
        oscillatory_period_max=args.oscillatory_period_max,
        input_proj_scheme=args.input_proj_scheme,
    )
    if hasattr(args, "memory_kind") and args.memory_kind != "none":
        variant_cfg = replace(variant_cfg, memory_kind=args.memory_kind)
    if hasattr(args, "substrate_mode") and args.substrate_mode != "frozen":
        variant_cfg = replace(variant_cfg, substrate_mode=args.substrate_mode)
    if args.oscillatory_frac is not None:
        variant_cfg = replace(variant_cfg, oscillatory_frac=args.oscillatory_frac)
    if args.static_bank_gate:
        variant_cfg = replace(
            variant_cfg,
            static_bank_gate=True,
            bank_gate_span=args.bank_gate_span,
        )
    if hasattr(args, "num_blocks") and args.num_blocks > 1:
        variant_cfg = replace(variant_cfg, num_blocks=args.num_blocks)
    if hasattr(args, "block_mixing_ratio") and args.block_mixing_ratio != 0.25:
        variant_cfg = replace(variant_cfg, block_mixing_ratio=args.block_mixing_ratio)
    if hasattr(args, "state_dim") and args.state_dim > 0:
        variant_cfg = replace(variant_cfg, state_dim=args.state_dim)
    if hasattr(args, "state_impl") and args.state_impl != "scan":
        variant_cfg = replace(variant_cfg, state_impl=args.state_impl)
    if hasattr(args, "num_heads") and args.num_heads > 1:
        variant_cfg = replace(variant_cfg, num_heads=args.num_heads)
    if hasattr(args, "patch_size") and args.patch_size > 1:
        variant_cfg = replace(variant_cfg, patch_size=args.patch_size)
    if hasattr(args, "patch_causal_decoder") and args.patch_causal_decoder != "none":
        variant_cfg = replace(variant_cfg, patch_causal_decoder=args.patch_causal_decoder)
    if hasattr(args, "num_hemispheres") and args.num_hemispheres > 1:
        variant_cfg = replace(variant_cfg, num_hemispheres=args.num_hemispheres)
    if hasattr(args, "fast_hemisphere_ratio"):
        variant_cfg = replace(variant_cfg, fast_hemisphere_ratio=args.fast_hemisphere_ratio)
    if hasattr(args, "fast_lr_mult"):
        variant_cfg = replace(variant_cfg, fast_lr_mult=args.fast_lr_mult)
    if hasattr(args, "local_poly_order") and args.local_poly_order > 1:
        variant_cfg = replace(variant_cfg, local_poly_order=args.local_poly_order)
    if hasattr(args, "substrate_poly_order") and args.substrate_poly_order > 1:
        variant_cfg = replace(variant_cfg, substrate_poly_order=args.substrate_poly_order)
    if hasattr(args, "block_stride") and args.block_stride > 1:
        variant_cfg = replace(variant_cfg, block_stride=args.block_stride)
    if hasattr(args, "training_noise") and args.training_noise > 0:
        variant_cfg = replace(variant_cfg, training_noise=args.training_noise)
    if hasattr(args, "adaptive_reg") and args.adaptive_reg:
        variant_cfg = replace(variant_cfg, adaptive_reg=True)
    if hasattr(args, "trust_routing") and args.trust_routing:
        variant_cfg = replace(variant_cfg, trust_routing=True)
    if hasattr(args, "table_path") and args.table_path:
        variant_cfg = replace(variant_cfg, table_path=args.table_path)

    variant_cfg = scale_config(variant_cfg, args.scale)
    baseline_linear_hidden = variant_cfg.linear_hidden

    if args.linear_hidden_width is not None:
        variant_cfg = replace(variant_cfg, linear_hidden=(args.linear_hidden_width,))
    elif args.linear_hidden_mult is not None:
        variant_cfg = replace(
            variant_cfg,
            linear_hidden=tuple(
                max(int(round(width * args.linear_hidden_mult)), 1)
                for width in variant_cfg.linear_hidden
            ),
        )
    elif args.linear_readout_kind == "tied_recursive" and args.linear_hidden_match != "none":
        if len(baseline_linear_hidden) != 1:
            raise ValueError(
                "causal-bank tied_recursive matching expects exactly one baseline linear hidden width."
            )
        in_dim = variant_cfg.linear_modes + variant_cfg.embedding_dim
        matched_width = solve_recursive_hidden_width(
            baseline_hidden=baseline_linear_hidden[0],
            in_dim=in_dim,
            out_dim=vocab_size,
            depth=variant_cfg.linear_readout_depth,
            mode=args.linear_hidden_match,
        )
        variant_cfg = replace(variant_cfg, linear_hidden=(matched_width,))
    elif (
        args.linear_readout_kind == "routed_sqrelu_experts"
        and args.linear_hidden_match != "none"
    ):
        if len(baseline_linear_hidden) != 1:
            raise ValueError(
                "causal-bank routed expert matching expects exactly one baseline linear hidden width."
            )
        in_dim = variant_cfg.linear_modes + variant_cfg.embedding_dim
        matched_width = solve_routed_expert_hidden_width(
            baseline_hidden=baseline_linear_hidden[0],
            in_dim=in_dim,
            out_dim=vocab_size,
            num_experts=variant_cfg.linear_readout_num_experts,
            mode=args.linear_hidden_match,
        )
        variant_cfg = replace(variant_cfg, linear_hidden=(matched_width,))

    if args.local_hidden_mult is not None:
        variant_cfg = replace(
            variant_cfg,
            local_hidden=tuple(
                max(int(round(width * args.local_hidden_mult)), 1)
                for width in variant_cfg.local_hidden
            ),
        )
    if args.local_scale_override is not None:
        variant_cfg = replace(variant_cfg, local_scale=args.local_scale_override)
    return variant_cfg, baseline_linear_hidden


def build_causal_bank_training_runtime(
    args: argparse.Namespace,
    *,
    RuntimeConfig: Any,
    train_config_for_profile: Any,
):
    runtime = RuntimeConfig(profile=args.profile)
    base_train = train_config_for_profile(args.profile)
    return replace(
        runtime,
        train=replace(
            base_train,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            steps=args.steps,
            learning_rate=(
                base_train.learning_rate
                if args.learning_rate is None
                else args.learning_rate
            ),
            weight_decay=(
                base_train.weight_decay
                if args.weight_decay is None
                else args.weight_decay
            ),
            seeds=(args.seed,),
        ),
    )


def assert_safe_model_config(args: argparse.Namespace, config: Any) -> None:
    if (
        config.linear_readout_kind == "tied_recursive"
        and not args.allow_experimental_recursive_readout
    ):
        raise ValueError(
            f"{config.linear_readout_kind} is disabled on this path pending stability/performance fixes. "
            "Use --allow-experimental-recursive-readout to override."
        )
    max_linear_hidden = max(config.linear_hidden, default=0)
    max_local_hidden = max(config.local_hidden, default=0)
    violations: list[str] = []
    if config.embedding_dim > 4096:
        violations.append(f"embedding_dim={config.embedding_dim} > 4096")
    if config.linear_modes > 32768:
        violations.append(f"linear_modes={config.linear_modes} > 32768")
    if max_linear_hidden > 8192:
        violations.append(f"linear_hidden_max={max_linear_hidden} > 8192")
    if max_local_hidden > 8192:
        violations.append(f"local_hidden_max={max_local_hidden} > 8192")
    if violations and not args.unsafe_large_model:
        raise ValueError(
            f"Refusing suspiciously large config: {', '.join(violations)}. "
            "Use --unsafe-large-model to override."
        )


def assert_safe_readout_compute(
    args: argparse.Namespace,
    config: Any,
    *,
    baseline_linear_hidden: tuple[int, ...],
    out_dim: int,
) -> None:
    if config.linear_readout_kind == "mlp":
        return
    if len(baseline_linear_hidden) != 1 or len(config.linear_hidden) != 1:
        raise ValueError(
            "causal-bank non-MLP readout budget checks require exactly one linear hidden width."
        )
    in_dim = config.linear_modes + config.embedding_dim
    base_flops = _estimate_mlp_readout_flops(in_dim, baseline_linear_hidden[0], out_dim)
    if config.linear_readout_kind == "tied_recursive":
        candidate_flops = _estimate_tied_readout_flops(
            in_dim,
            config.linear_hidden[0],
            out_dim,
            config.linear_readout_depth,
        )
    else:
        candidate_flops = _estimate_routed_expert_readout_flops(
            in_dim,
            config.linear_hidden[0],
            out_dim,
            config.linear_readout_num_experts,
        )
    flop_ratio = candidate_flops / max(base_flops, 1)
    if flop_ratio > args.max_readout_flop_ratio and not args.unsafe_large_model:
        raise ValueError(
            f"Refusing {config.linear_readout_kind} with estimated flop ratio {flop_ratio:.3f} > "
            f"max_readout_flop_ratio={args.max_readout_flop_ratio:.3f}. "
            "Shrink the expert/recursive hidden width, change --linear-hidden-match, or use "
            "--unsafe-large-model to override."
        )
