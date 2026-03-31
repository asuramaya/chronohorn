#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np

from chronohorn.train.causal_bank_training_support import (
    attach_replay_fixture_logprob_reference,
    bits_per_token_from_loss,
    build_causal_bank_adamw_kwargs,
    build_causal_bank_adamw_policy_defaults,
    build_backend_environment_metadata,
    build_replay_parity_fixture,
    estimate_causal_bank_training_performance,
    format_observed_training_performance,
    build_train_policy_metadata,
    parse_probe_steps,
    seed_python,
    summarize_observed_training_performance,
    summarize_named_arrays,
    write_causal_bank_export_bundle,
)
from chronohorn.train.causal_bank_training_primitives import (
    add_causal_bank_training_arguments,
    assert_safe_model_config,
    assert_safe_readout_compute,
    build_causal_bank_training_runtime,
    build_causal_bank_variant_config,
)
from chronohorn.train.causal_bank_training_stack import load_causal_bank_training_stack


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chronohorn Torch/CUDA causal-bank trainer on token shards."
    )
    add_causal_bank_training_arguments(parser, backend="torch")
    return parser


def seed_everything(seed: int) -> None:
    import random
    import numpy as np

    stack = load_causal_bank_training_stack("torch")
    torch = stack.torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(raw: str | None) -> str:
    stack = load_causal_bank_training_stack("torch")
    torch = stack.torch
    if raw:
        return raw
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def config_for_variant(args: argparse.Namespace, seq_len: int, stack):
    return build_causal_bank_variant_config(
        args,
        ConfigClass=stack.ConfigClass,
        scale_config=stack.scale_config,
        seq_len=seq_len,
        vocab_size=args.vocab_size,
    )


def build_runtime(args: argparse.Namespace, stack):
    return build_causal_bank_training_runtime(
        args,
        RuntimeConfig=stack.RuntimeConfig,
        train_config_for_profile=stack.train_config_for_profile,
    )


def assert_safe_config(args: argparse.Namespace, config) -> None:
    assert_safe_model_config(args, config)


def assert_safe_readout_budget(
    args: argparse.Namespace,
    config,
    *,
    baseline_linear_hidden: tuple[int, ...],
    out_dim: int,
) -> None:
    assert_safe_readout_compute(
        args,
        config,
        baseline_linear_hidden=baseline_linear_hidden,
        out_dim=out_dim,
    )


def evaluate(model, dataset, train_config, split: str, *, eval_batches: int | None = None) -> float:
    stack = load_causal_bank_training_stack("torch")
    F = stack.functional
    batches = train_config.eval_batches if eval_batches is None else eval_batches
    was_training = model.training
    model.eval()
    total = 0.0
    with stack.torch.no_grad():
        for _ in range(batches):
            x, y = dataset.batch(split, train_config.batch_size, train_config.seq_len)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            total += float(loss.item())
    if was_training:
        model.train()
    return total / batches


def run_bridge(args: argparse.Namespace) -> dict[str, object]:
    stack = load_causal_bank_training_stack("torch")
    torch = stack.torch
    F = stack.functional
    CausalBankModel = stack.ModelClass
    build_token_shard_torch_dataset = stack.build_token_shard_torch_dataset

    runtime = build_runtime(args, stack)
    device = choose_device(args.device)
    seed_everything(args.seed)
    dataset = build_token_shard_torch_dataset(
        args.data_root,
        vocab_size=args.vocab_size,
        device=device,
        pin_memory=device.startswith("cuda"),
    )
    config, baseline_linear_hidden = config_for_variant(args, runtime.train.seq_len, stack)
    assert_safe_config(args, config)
    assert_safe_readout_budget(
        args,
        config,
        baseline_linear_hidden=baseline_linear_hidden,
        out_dim=dataset.vocab_size,
    )
    model = CausalBankModel(vocab_size=dataset.vocab_size, config=config).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if param_count > args.max_params and not args.unsafe_large_model:
        raise ValueError(
            f"Refusing model with {param_count:,} trainable params > max_params={args.max_params:,}. "
            "Use --unsafe-large-model or raise --max-params to override."
        )
    if args.torch_compile:
        model = torch.compile(model)
    initial_trainable_state = {
        name: param.detach().cpu().to(dtype=torch.float32).numpy()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    init_report = summarize_named_arrays(initial_trainable_state)
    optimizer_kwargs = build_causal_bank_adamw_kwargs(
        backend="torch",
        learning_rate=runtime.train.learning_rate,
        weight_decay=runtime.train.weight_decay,
        device=device,
        fused=False,
    )
    optimizer_policy_defaults = build_causal_bank_adamw_policy_defaults(
        backend="torch",
        learning_rate=runtime.train.learning_rate,
        weight_decay=runtime.train.weight_decay,
        device=device,
        fused=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    backend_environment = build_backend_environment_metadata(
        backend="torch",
        stack=stack,
        device=device,
    )
    train_policy = build_train_policy_metadata(
        backend="torch",
        device=device,
        dtype_policy="fp32",
        optimizer_name=stack.optimizer_name,
        optimizer_impl=stack.optimizer_impl,
        optimizer_like=optimizer,
        learning_rate=runtime.train.learning_rate,
        weight_decay=runtime.train.weight_decay,
        grad_clip=runtime.train.grad_clip,
        compile_train_step=False,
        compile_eval=False,
        torch_compile=bool(args.torch_compile),
        init_policy="chronohorn_v1",
        init_seed=config.init_seed,
        explicit_defaults=optimizer_policy_defaults,
    )
    performance_estimate = estimate_causal_bank_training_performance(
        config=config,
        vocab_size=dataset.vocab_size,
        batch_size=runtime.train.batch_size,
        seq_len=runtime.train.seq_len,
        trainable_param_count=param_count,
    )

    probe_steps = parse_probe_steps(args.probe_steps, runtime.train.steps)
    probe_step_set = set(probe_steps)
    probe_history: list[dict[str, float | int | None]] = []
    performance_log: list[dict[str, float | int | None]] = []
    losses: list[float] = []
    best = float("inf")
    start = time.time()
    last_log_time = start
    last_log_step = 0

    print("\n  causal-bank torch trainer\n")
    print(
        f"  data_root={args.data_root} device={device} seed={args.seed} "
        f"steps={runtime.train.steps} seq_len={runtime.train.seq_len} "
        f"batch_size={runtime.train.batch_size} lr={runtime.train.learning_rate:g}"
    )
    print(
        f"  train_tokens={dataset.train_token_count:,} val_tokens={dataset.test_token_count:,} "
        f"variant={args.variant} scale={args.scale:.3f} linear_modes={config.linear_modes} "
        f"linear_readout={config.linear_readout_kind}:{config.linear_readout_depth} "
        f"linear_hidden={list(config.linear_hidden)} "
        f"local_window={config.local_window} osc_schedule={config.oscillatory_schedule} "
        f"static_bank_gate={config.static_bank_gate} params={param_count:,}"
    )

    model.train()
    for step in range(1, runtime.train.steps + 1):
        x, y = dataset.batch("train", runtime.train.batch_size, runtime.train.seq_len)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), runtime.train.grad_clip)
        optimizer.step()

        current = float(loss.item())
        losses.append(current)
        if current < best:
            best = current

        if step in probe_step_set:
            probe_loss = evaluate(
                model,
                dataset,
                runtime.train,
                args.probe_split,
                eval_batches=args.probe_eval_batches,
            )
            probe_bpt = bits_per_token_from_loss(probe_loss)
            tokens_per_byte = dataset.test_tokens_per_byte if args.probe_split == "test" else None
            probe_bpb = probe_bpt * tokens_per_byte if tokens_per_byte is not None else None
            recent_window = min(len(losses), runtime.train.log_every)
            recent_train_loss = float(sum(losses[-recent_window:]) / recent_window) if recent_window > 0 else float("nan")
            probe_history.append(
                {
                    "step": step,
                    "split": args.probe_split,
                    "eval_batches": args.probe_eval_batches,
                    "eval_loss": probe_loss,
                    "bits_per_token": probe_bpt,
                    "bpb": probe_bpb,
                    "recent_train_loss": recent_train_loss,
                    "elapsed_sec": time.time() - start,
                }
            )
            bpb_text = "n/a" if probe_bpb is None else f"{probe_bpb:.4f}"
            print(
                f"      probe {step:5d} | {args.probe_split} loss {probe_loss:.4f} "
                f"| bpt {probe_bpt:.4f} | bpb {bpb_text}"
            )

        if step % runtime.train.log_every == 0:
            recent = float(sum(losses[-runtime.train.log_every:]) / runtime.train.log_every)
            now = time.time()
            elapsed = now - start
            interval_steps = step - last_log_step
            interval_elapsed = now - last_log_time
            perf_summary = summarize_observed_training_performance(
                performance_estimate,
                steps_completed=step,
                elapsed_sec=elapsed,
                interval_steps=interval_steps,
                interval_elapsed_sec=interval_elapsed,
            )
            performance_log.append({"step": step, **perf_summary})
            print(
                f"      {step:5d} | loss {recent:.4f} | best {best:.4f} | "
                f"{format_observed_training_performance(perf_summary)}"
            )
            last_log_time = now
            last_log_step = step

    elapsed = time.time() - start
    performance_summary = summarize_observed_training_performance(
        performance_estimate,
        steps_completed=runtime.train.steps,
        elapsed_sec=elapsed,
    )
    train_eval = evaluate(model, dataset, runtime.train, "train", eval_batches=args.final_eval_batches)
    replay_fixture = build_replay_parity_fixture(
        dataset,
        split="test",
        sequence_length=runtime.train.seq_len,
    )
    was_training = model.training
    model.eval()
    with torch.no_grad():
        fixture_x = torch.tensor([replay_fixture["input_token_ids"]], dtype=torch.long, device=device)
        fixture_logits = model(fixture_x).detach().cpu().numpy()
    if was_training:
        model.train()
    replay_fixture = attach_replay_fixture_logprob_reference(replay_fixture, fixture_logits)
    test_eval = evaluate(model, dataset, runtime.train, "test", eval_batches=args.final_eval_batches)
    train_bpt = bits_per_token_from_loss(train_eval)
    test_bpt = bits_per_token_from_loss(test_eval)
    test_bpb = test_bpt * dataset.test_tokens_per_byte if dataset.test_tokens_per_byte is not None else None
    params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    payload_bytes_est = float(
        sum(param.numel() * param.element_size() for param in model.parameters() if param.requires_grad)
    )
    result = {
        "title": "causal-bank torch trainer",
        "config": asdict(runtime),
        "dataset": {
            "source_path": dataset.dataset.source_path,
            "tokenizer": dataset.dataset.tokenizer,
            "tokenizer_path": dataset.dataset.tokenizer_path,
            "train_token_count": int(dataset.train_token_count),
            "test_token_count": int(dataset.test_token_count),
            "test_tokens_per_byte": dataset.test_tokens_per_byte,
            "test_bytes_per_token": dataset.test_bytes_per_token,
        },
        "training": {
            "backend": "torch",
            "device": device,
            "backend_environment": backend_environment,
            "dtype_policy": "fp32",
            "compile_train_step": False,
            "compile_eval": False,
            "torch_compile": args.torch_compile,
            "compile": train_policy["compile"],
            "optimizer": train_policy["optimizer"],
            "train_policy_version": train_policy["version"],
            "train_policy": train_policy,
            "init_policy": "chronohorn_v1",
            "init_seed": config.init_seed,
            "initial_trainable_signature": init_report,
            "probe_steps": probe_steps,
            "probe_split": args.probe_split,
            "probe_eval_batches": args.probe_eval_batches,
            "final_eval_batches": runtime.train.eval_batches if args.final_eval_batches is None else args.final_eval_batches,
            "performance_estimate": performance_estimate,
            "performance": performance_summary,
            "performance_log": performance_log,
            "probes": probe_history,
            "replay_fixture": replay_fixture,
        },
        "model": {
            "preset": "causal_bank_torch",
            "variant": args.variant,
            "scale": args.scale,
            "params": params,
            "seed": args.seed,
            "linear_modes": config.linear_modes,
            "local_window": config.local_window,
            "share_embedding": config.share_embedding,
            "linear_impl": config.linear_impl,
            "linear_readout_kind": config.linear_readout_kind,
            "linear_readout_depth": config.linear_readout_depth,
            "linear_readout_num_experts": config.linear_readout_num_experts,
            "linear_hidden_match": args.linear_hidden_match,
            "linear_half_life_min": config.linear_half_life_min,
            "linear_half_life_max": config.linear_half_life_max,
            "oscillatory_frac": config.oscillatory_frac,
            "oscillatory_period_min": config.oscillatory_period_min,
            "oscillatory_period_max": config.oscillatory_period_max,
            "static_bank_gate": config.static_bank_gate,
            "bank_gate_span": config.bank_gate_span,
            "train_eval_loss": train_eval,
            "test_eval_loss": test_eval,
            "train_bits_per_token": train_bpt,
            "test_bits_per_token": test_bpt,
            "test_bpb": test_bpb,
            "overfit_pct": (test_eval / train_eval - 1.0) * 100.0,
            "train_time_sec": elapsed,
            "learning_rate": runtime.train.learning_rate,
            "payload_bytes_est": payload_bytes_est,
            "payload_mb_est": payload_bytes_est / (1024.0 * 1024.0),
            "embedding_dim": config.embedding_dim,
            "linear_hidden": list(config.linear_hidden),
            "local_hidden": list(config.local_hidden),
            "local_scale": config.local_scale,
            "mix_mode": config.mix_mode,
            "init_policy": "chronohorn_v1",
            "init_seed": config.init_seed,
            "initial_trainable_signature": init_report,
            "train_bpb": None,
        },
    }
    bpb_text = "n/a" if test_bpb is None else f"{test_bpb:.4f}"
    print(
        f"  Te:{test_eval:.4f} "
        f"bpt:{test_bpt:.4f} "
        f"bpb:{bpb_text} "
        f"tok/s:{performance_summary['tokens_per_second']:.0f} "
        f"TF/s:{performance_summary['estimated_sustained_tflops']:.3f}"
    )

    output_path = Path(args.json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.export_dir:
        learned_state = {
            name: param.detach().cpu()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        for name, buffer in model.named_buffers():
            learned_state[name] = buffer.detach().cpu()
        export_dir = write_causal_bank_export_bundle(
            export_root=args.export_dir,
            summary_path=output_path,
            variant=args.variant,
            scale=args.scale,
            readout_kind=config.linear_readout_kind,
            seed=args.seed,
            tokenizer_id=dataset.dataset.tokenizer,
            data_root=args.data_root,
            config=config,
            learned_state=learned_state,
            train_step=runtime.train.steps,
            train_wallclock_s=elapsed,
            sequence_length=runtime.train.seq_len,
            vocab_size=dataset.vocab_size,
            profile=args.profile,
            backend="torch",
            train_policy=train_policy,
            replay_fixture=replay_fixture,
        )
        result["model"]["export_dir"] = str(export_dir)
        result["model"]["export_manifest_path"] = str(export_dir / "manifest.json")
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\n  Wrote JSON summary to {output_path}")
    if args.export_dir:
        print(f"  Wrote opc-export bundle to {result['model']['export_dir']}")
    return result


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_bridge(args)


if __name__ == "__main__":
    main()
