#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from chronohorn.engine.backend_metadata import build_backend_environment_metadata
from chronohorn.engine.optimizer_policy import build_adamw_kwargs
from chronohorn.engine.performance import summarize_observed_training_performance
from chronohorn.service_log import configure_service_log, service_log


def _add_fallback_parser_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", default="")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--profile", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--linear-modes", type=int, default=256)
    parser.add_argument("--linear-readout-kind", default="mlp")
    parser.add_argument("--linear-readout-depth", type=int, default=1)
    parser.add_argument("--linear-readout-num-experts", type=int, default=4)
    parser.add_argument("--readout-bands", type=int, default=1)
    parser.add_argument("--allow-experimental-recursive-readout", action="store_true")
    parser.add_argument("--linear-hidden-match", default="mlp_flops")
    parser.add_argument("--local-window", type=int, default=8)
    parser.add_argument("--scale", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--linear-half-life-max", type=float, default=None)
    parser.add_argument("--oscillatory-frac", type=float, default=None)
    parser.add_argument("--oscillatory-schedule", default="logspace")
    parser.add_argument("--oscillatory-period-min", type=float, default=4.0)
    parser.add_argument("--oscillatory-period-max", type=float, default=64.0)
    parser.add_argument("--input-proj-scheme", default="random")
    parser.add_argument("--memory-kind", default="none")
    parser.add_argument("--substrate-mode", default="frozen")
    parser.add_argument("--static-bank-gate", action="store_true")
    parser.add_argument("--bank-gate-span", type=float, default=0.5)
    parser.add_argument("--linear-hidden-width", type=int, default=None)
    parser.add_argument("--linear-hidden-mult", type=float, default=None)
    parser.add_argument("--local-hidden-mult", type=float, default=None)
    parser.add_argument("--local-scale-override", type=float, default=None)
    parser.add_argument("--num-blocks", type=int, default=1)
    parser.add_argument("--block-mixing-ratio", type=float, default=0.25)
    parser.add_argument("--state-dim", type=int, default=0)
    parser.add_argument("--state-impl", default="scan")
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=1)
    parser.add_argument("--patch-causal-decoder", default="none")
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
    parser.add_argument("--variant", default="base")
    parser.add_argument("--device", default=None)
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--probe-diagnostics", action="store_true")
    parser.add_argument("--profile-cuda", type=int, default=0, metavar="N")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chronohorn Torch/CUDA causal-bank synthetic throughput benchmark."
    )
    try:
        from chronohorn.families.causal_bank.training.causal_bank_training_primitives import (
            add_causal_bank_core_arguments,
            add_torch_bridge_arguments,
        )

        add_causal_bank_core_arguments(parser, require_data_root=False)
        add_torch_bridge_arguments(parser)
    except ModuleNotFoundError:
        _add_fallback_parser_arguments(parser)
    parser.add_argument("--json", required=True)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--warmup-eval-batches", type=int, default=None)
    parser.add_argument("--measure-train-steps", type=int, default=50)
    parser.add_argument("--measure-eval-batches", type=int, default=50)
    return parser


def choose_device(raw: str | None) -> str:
    from chronohorn.families.causal_bank.training.causal_bank_training_stack import (
        load_training_backend_stack,
    )

    stack = load_training_backend_stack("torch")
    torch = stack.torch
    if raw:
        return raw
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def seed_everything(seed: int) -> None:
    import random

    from chronohorn.families.causal_bank.training.causal_bank_training_stack import (
        load_training_backend_stack,
    )

    stack = load_training_backend_stack("torch")
    torch = stack.torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _synchronize(torch: Any, device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps") and torch.mps.is_available():
        torch.mps.synchronize()


def _summarize_eval_benchmark(
    estimate: dict[str, Any],
    *,
    eval_batches: int,
    elapsed_sec: float,
) -> dict[str, Any]:
    eval_batches = int(max(eval_batches, 0))
    elapsed_sec = float(max(elapsed_sec, 1e-9))
    tokens_per_step = float(estimate["tokens_per_step"])
    tokens_completed = tokens_per_step * float(eval_batches)
    forward_flops_per_token = float(estimate["forward_total_flops_per_token"])
    total_forward_flops = tokens_completed * forward_flops_per_token
    tokens_per_second = tokens_completed / elapsed_sec
    return {
        "eval_batches": eval_batches,
        "tokens_completed": int(tokens_completed),
        "elapsed_sec": elapsed_sec,
        "tokens_per_second": tokens_per_second,
        "estimated_forward_tflops": tokens_per_second * forward_flops_per_token / 1e12,
        "estimated_total_forward_tflops_consumed": total_forward_flops / 1e12,
        "seconds_per_batch": elapsed_sec / max(eval_batches, 1),
    }


def _build_benchmark_payload(
    *,
    args: argparse.Namespace,
    runtime: Any,
    config: Any,
    device: str,
    backend_environment: dict[str, Any],
    param_count: int,
    performance_estimate: dict[str, Any],
    train_summary: dict[str, Any],
    eval_summary: dict[str, Any],
    peak_memory_mb: float | None,
) -> dict[str, Any]:
    return {
        "title": "causal-bank torch benchmark",
        "kind": "benchmark",
        "config": asdict(runtime),
        "benchmark": {
            "backend": "torch",
            "device": device,
            "torch_compile": bool(args.torch_compile),
            "backend_environment": backend_environment,
            "warmup_steps": int(args.warmup_steps),
            "warmup_eval_batches": int(
                args.warmup_eval_batches
                if args.warmup_eval_batches is not None
                else args.warmup_steps
            ),
            "measure_train_steps": int(args.measure_train_steps),
            "measure_eval_batches": int(args.measure_eval_batches),
            "peak_memory_mb": peak_memory_mb,
            "performance_estimate": performance_estimate,
            "train": train_summary,
            "eval": eval_summary,
        },
        "model": {
            "preset": "causal_bank_torch",
            "variant": args.variant,
            "scale": args.scale,
            "params": param_count,
            "seed": args.seed,
            "linear_modes": config.linear_modes,
            "local_window": config.local_window,
            "share_embedding": config.share_embedding,
            "linear_impl": config.linear_impl,
            "linear_readout_kind": config.linear_readout_kind,
            "linear_readout_depth": config.linear_readout_depth,
            "linear_readout_num_experts": config.linear_readout_num_experts,
            "readout_bands": config.readout_bands,
            "linear_half_life_min": config.linear_half_life_min,
            "linear_half_life_max": config.linear_half_life_max,
            "oscillatory_frac": config.oscillatory_frac,
            "oscillatory_schedule": config.oscillatory_schedule,
            "input_proj_scheme": config.input_proj_scheme,
            "substrate_mode": config.substrate_mode,
            "state_dim": config.state_dim,
            "state_impl": getattr(config, "state_impl", "scan"),
            "num_heads": config.num_heads,
            "num_hemispheres": config.num_hemispheres,
            "block_mixing_ratio": config.block_mixing_ratio,
            "patch_size": config.patch_size,
            "patch_causal_decoder": config.patch_causal_decoder,
            "static_bank_gate": config.static_bank_gate,
            "bank_gate_span": config.bank_gate_span,
            "embedding_dim": config.embedding_dim,
            "linear_hidden": list(config.linear_hidden),
            "local_hidden": list(config.local_hidden),
            "local_scale": config.local_scale,
            "mix_mode": config.mix_mode,
        },
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    from chronohorn.families.causal_bank import CAUSAL_BANK_TRAINING_ADAPTER
    from chronohorn.families.causal_bank.training.causal_bank_training_primitives import (
        build_causal_bank_training_runtime,
    )
    from chronohorn.families.causal_bank.training.causal_bank_training_stack import (
        load_training_backend_stack,
    )

    stack = load_training_backend_stack("torch")
    torch = stack.torch
    functional = stack.functional
    ModelClass = stack.ModelClass
    RuntimeConfig = stack.RuntimeConfig
    scale_config = stack.scale_config
    train_config_for_profile = stack.train_config_for_profile

    configure_service_log(Path(args.json).parent / "chronohorn.service.jsonl")
    log_component = "bench.causal_bank.torch"
    device = choose_device(args.device)
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    seed_everything(args.seed)
    runtime = build_causal_bank_training_runtime(
        args,
        RuntimeConfig=RuntimeConfig,
        train_config_for_profile=train_config_for_profile,
    )
    config, baseline_linear_hidden = CAUSAL_BANK_TRAINING_ADAPTER.build_variant_config(
        args,
        ConfigClass=stack.ConfigClass,
        scale_config=scale_config,
        seq_len=runtime.train.seq_len,
        vocab_size=args.vocab_size,
    )
    CAUSAL_BANK_TRAINING_ADAPTER.validate_config(
        args,
        config,
        baseline_linear_hidden=baseline_linear_hidden,
        out_dim=args.vocab_size,
    )
    model = ModelClass(vocab_size=args.vocab_size, config=config).to(device)
    optimizer_model = model
    param_count = sum(p.numel() for p in optimizer_model.parameters() if p.requires_grad)
    if param_count > args.max_params and not args.unsafe_large_model:
        raise ValueError(
            f"Refusing model with {param_count:,} trainable params > max_params={args.max_params:,}. "
            "Use --unsafe-large-model or raise --max-params to override."
        )
    if args.torch_compile:
        compile_mode = "max-autotune" if device.startswith("cuda") else "reduce-overhead"
        model = torch.compile(model, mode=compile_mode)

    use_fused = device.startswith("cuda")
    optimizer_kwargs = build_adamw_kwargs(
        backend="torch",
        learning_rate=runtime.train.learning_rate,
        weight_decay=runtime.train.weight_decay,
        device=device,
        fused=use_fused,
    )
    optimizer_params = (
        optimizer_model.param_groups(runtime.train.learning_rate)
        if hasattr(optimizer_model, "param_groups")
        else optimizer_model.parameters()
    )
    optimizer = torch.optim.AdamW(optimizer_params, **optimizer_kwargs)

    performance_estimate = CAUSAL_BANK_TRAINING_ADAPTER.estimate_training_performance(
        config=config,
        vocab_size=args.vocab_size,
        batch_size=runtime.train.batch_size,
        seq_len=runtime.train.seq_len,
        trainable_param_count=param_count,
    )
    backend_environment = build_backend_environment_metadata(
        backend="torch",
        stack=stack,
        device=device,
    )

    x = torch.randint(
        0,
        args.vocab_size,
        (runtime.train.batch_size, runtime.train.seq_len),
        device=device,
        dtype=torch.long,
    )
    y = torch.randint(
        0,
        args.vocab_size,
        (runtime.train.batch_size, runtime.train.seq_len),
        device=device,
        dtype=torch.long,
    )

    warmup_eval_batches = args.warmup_eval_batches if args.warmup_eval_batches is not None else args.warmup_steps
    service_log(
        log_component,
        "benchmark started",
        device=device,
        torch_compile=bool(args.torch_compile),
        warmup_steps=int(args.warmup_steps),
        warmup_eval_batches=int(warmup_eval_batches),
        measure_train_steps=int(args.measure_train_steps),
        measure_eval_batches=int(args.measure_eval_batches),
        scale=args.scale,
        seq_len=runtime.train.seq_len,
        batch_size=runtime.train.batch_size,
        state_impl=getattr(config, "state_impl", "scan"),
        substrate_mode=getattr(config, "substrate_mode", "frozen"),
        readout_bands=getattr(config, "readout_bands", 1),
    )

    model.train()
    for _ in range(max(int(args.warmup_steps), 0)):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        if hasattr(model, "substrate_regularization"):
            loss = loss + model.substrate_regularization(step=0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), runtime.train.grad_clip)
        optimizer.step()
    _synchronize(torch, device)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_start = time.perf_counter()
    last_train_loss = None
    for _ in range(max(int(args.measure_train_steps), 1)):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        if hasattr(model, "substrate_regularization"):
            loss = loss + model.substrate_regularization(step=0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), runtime.train.grad_clip)
        optimizer.step()
        last_train_loss = loss.detach()
    _synchronize(torch, device)
    train_elapsed = time.perf_counter() - train_start
    peak_memory_mb = None
    if device.startswith("cuda") and torch.cuda.is_available():
        peak_memory_mb = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
    train_summary = summarize_observed_training_performance(
        performance_estimate,
        steps_completed=max(int(args.measure_train_steps), 1),
        elapsed_sec=train_elapsed,
    )
    train_summary["final_loss"] = None if last_train_loss is None else float(last_train_loss.item())

    model.eval()
    with torch.inference_mode():
        for _ in range(max(int(warmup_eval_batches), 0)):
            logits = model(x)
            functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        _synchronize(torch, device)
        eval_start = time.perf_counter()
        total_eval_loss = None
        for _ in range(max(int(args.measure_eval_batches), 1)):
            logits = model(x)
            loss = functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            total_eval_loss = loss if total_eval_loss is None else total_eval_loss + loss
        _synchronize(torch, device)
    eval_elapsed = time.perf_counter() - eval_start
    eval_summary = _summarize_eval_benchmark(
        performance_estimate,
        eval_batches=max(int(args.measure_eval_batches), 1),
        elapsed_sec=eval_elapsed,
    )
    if total_eval_loss is not None:
        eval_summary["avg_loss"] = float((total_eval_loss / max(int(args.measure_eval_batches), 1)).item())

    result = _build_benchmark_payload(
        args=args,
        runtime=runtime,
        config=config,
        device=device,
        backend_environment=backend_environment,
        param_count=param_count,
        performance_estimate=performance_estimate,
        train_summary=train_summary,
        eval_summary=eval_summary,
        peak_memory_mb=peak_memory_mb,
    )
    service_log(
        log_component,
        "benchmark complete",
        train_tokens_per_second=train_summary.get("tokens_per_second"),
        train_tflops=train_summary.get("estimated_sustained_tflops"),
        eval_tokens_per_second=eval_summary.get("tokens_per_second"),
        eval_forward_tflops=eval_summary.get("estimated_forward_tflops"),
        peak_memory_mb=peak_memory_mb,
    )
    output_path = Path(args.json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    service_log(log_component, "json summary written", output_path=str(output_path))
    return result


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_benchmark(args)


if __name__ == "__main__":
    main()
