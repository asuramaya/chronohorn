#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from chronohorn.engine.backend_metadata import build_backend_environment_metadata
from chronohorn.engine.budgets import DEFAULT_GOLF_V1_BUDGET
from chronohorn.engine.forecasting import build_result_forecast
from chronohorn.engine.optimizer_policy import (
    build_adamw_kwargs,
    build_adamw_policy_defaults,
    build_train_policy_metadata,
)
from chronohorn.engine.performance import (
    bits_per_token_from_loss,
    format_observed_training_performance,
    summarize_observed_training_performance,
)
from chronohorn.engine.probes import (
    format_probe_plan,
    probe_entry_by_step,
    resolve_probe_plan,
)
from chronohorn.engine.signatures import summarize_named_arrays
from chronohorn.families.causal_bank import CAUSAL_BANK_TRAINING_ADAPTER
from chronohorn.families.causal_bank.training.causal_bank_training_primitives import (
    build_causal_bank_training_runtime,
)
from chronohorn.families.causal_bank.training.causal_bank_training_stack import load_training_backend_stack
from chronohorn.families.causal_bank.training.causal_bank_training_support import (
    build_compute_accounting_inputs,
    build_probe_compute_accounting_inputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chronohorn Torch/CUDA causal-bank trainer on token shards."
    )
    CAUSAL_BANK_TRAINING_ADAPTER.add_training_arguments(parser, backend="torch")
    return parser


def seed_everything(seed: int) -> None:
    import random

    stack = load_training_backend_stack("torch")
    torch = stack.torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(raw: str | None) -> str:
    stack = load_training_backend_stack("torch")
    torch = stack.torch
    if raw:
        return raw
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def config_for_variant(args: argparse.Namespace, seq_len: int, stack):
    return CAUSAL_BANK_TRAINING_ADAPTER.build_variant_config(
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


def assert_safe_readout_budget(
    args: argparse.Namespace,
    config,
    *,
    baseline_linear_hidden: tuple[int, ...],
    out_dim: int,
) -> None:
    CAUSAL_BANK_TRAINING_ADAPTER.validate_config(
        args,
        config,
        baseline_linear_hidden=baseline_linear_hidden,
        out_dim=out_dim,
    )


def evaluate(model, dataset, train_config, split: str, *, eval_batches: int | None = None) -> float:  # noqa: S307
    stack = load_training_backend_stack("torch")
    F = stack.functional
    batches = train_config.eval_batches if eval_batches is None else eval_batches
    was_training = model.training
    model.eval()
    # Reset stream so every probe measures the same data slice
    inner = dataset.dataset if hasattr(dataset, "dataset") else dataset
    stream = inner.test_stream if split == "test" else inner.train_stream
    stream.reset()
    total_loss = 0.0
    total_tokens = 0
    with stack.torch.no_grad():
        for _ in range(batches):
            x, y = dataset.batch(split, train_config.batch_size, train_config.seq_len)
            logits = model(x)
            n_tokens = y.numel()
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="sum")
            total_loss += float(loss.item())
            total_tokens += n_tokens
    if was_training:
        model.train()
    return total_loss / total_tokens if total_tokens > 0 else float("inf")


def run_bridge(args: argparse.Namespace) -> dict[str, object]:
    stack = load_training_backend_stack("torch")
    torch = stack.torch
    F = stack.functional
    CausalBankModel = stack.ModelClass
    build_token_shard_torch_dataset = stack.build_token_shard_torch_dataset

    runtime = build_runtime(args, stack)
    device = choose_device(args.device)

    # CUDA performance defaults: TF32 matmul and cuDNN for ~1.3x throughput
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    seed_everything(args.seed)
    dataset = build_token_shard_torch_dataset(
        args.data_root,
        vocab_size=args.vocab_size,
        device=device,
        pin_memory=device.startswith("cuda"),
    )
    config, baseline_linear_hidden = config_for_variant(args, runtime.train.seq_len, stack)
    assert_safe_readout_budget(
        args,
        config,
        baseline_linear_hidden=baseline_linear_hidden,
        out_dim=dataset.vocab_size,
    )
    model = CausalBankModel(vocab_size=dataset.vocab_size, config=config).to(device)
    optimizer_model = model
    # Inject ngram table for trust-routing mode (decepticons doesn't import chronohorn)
    if getattr(config, "trust_routing", False) and getattr(config, "table_path", ""):
        import pathlib

        from chronohorn.families.polyhash.models.ngram_table import NgramTable
        table_path = config.table_path
        if pathlib.Path(table_path).exists():
            model.set_ngram_table(NgramTable.load(table_path))
        else:
            model.set_ngram_table(NgramTable(vocab_size=dataset.vocab_size))
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if param_count > args.max_params and not args.unsafe_large_model:
        raise ValueError(
            f"Refusing model with {param_count:,} trainable params > max_params={args.max_params:,}. "
            "Use --unsafe-large-model or raise --max-params to override."
        )
    if args.torch_compile:
        compile_mode = "max-autotune" if device.startswith("cuda") else "reduce-overhead"
        model = torch.compile(model, mode=compile_mode)
    initial_trainable_state = {
        name: param.detach().cpu().to(dtype=torch.float32).numpy()
        for name, param in optimizer_model.named_parameters()
        if param.requires_grad
    }
    init_report = summarize_named_arrays(initial_trainable_state)
    use_fused = device.startswith("cuda")
    optimizer_kwargs = build_adamw_kwargs(
        backend="torch",
        learning_rate=runtime.train.learning_rate,
        weight_decay=runtime.train.weight_decay,
        device=device,
        fused=use_fused,
    )
    optimizer_policy_defaults = build_adamw_policy_defaults(
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
    performance_estimate = CAUSAL_BANK_TRAINING_ADAPTER.estimate_training_performance(
        config=config,
        vocab_size=dataset.vocab_size,
        batch_size=runtime.train.batch_size,
        seq_len=runtime.train.seq_len,
        trainable_param_count=param_count,
    )

    effective_final_eval_batches = (
        runtime.train.eval_batches if args.final_eval_batches is None else args.final_eval_batches
    )
    probe_plan = resolve_probe_plan(
        max_step=runtime.train.steps,
        raw_steps=args.probe_steps,
        policy=args.probe_policy,
        default_eval_batches=runtime.train.eval_batches,
        standard_eval_batches=(
            args.probe_standard_eval_batches
            if args.probe_standard_eval_batches is not None
            else args.probe_eval_batches
        ),
        micro_eval_batches=args.probe_micro_eval_batches,
        promotion_eval_batches=args.probe_promotion_eval_batches,
        final_eval_batches=effective_final_eval_batches,
        geometric_start_step=args.probe_geometric_start,
        geometric_ratio=args.probe_geometric_ratio,
        micro_cutoff_step=args.probe_micro_cutoff_step,
        promotion_count=args.probe_promotion_count,
    )
    probe_steps = [int(step) for step in probe_plan.get("steps", [])]
    probe_step_set = set(probe_steps)
    probe_history: list[dict[str, float | int | None]] = []
    effective_probe_eval_batches = int(probe_plan.get("eval_batches", {}).get("standard") or runtime.train.eval_batches)
    performance_log: list[dict[str, float | int | None]] = []
    losses: list[float] = []
    best = float("inf")
    start = time.time()
    last_log_time = start
    last_log_step = 0
    cumulative_probe_tflops_est = 0.0
    cumulative_probe_elapsed_sec = 0.0
    last_log_probe_tflops_est = 0.0
    last_log_probe_elapsed_sec = 0.0

    # Substrate training hints — the code tells the operator what the mode needs
    from decepticons.causal_bank import substrate_training_hints
    hints = substrate_training_hints(config)
    for warning in hints.get("warnings", []):
        print(f"\n  \u26a0 {warning}", file=sys.stderr)

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
    if probe_steps:
        print(f"  {format_probe_plan(probe_plan)}")

    # Optional CUDA profiling: writes Chrome trace for the first N steps
    _profiler = None
    if getattr(args, "profile_cuda", 0) > 0 and device.startswith("cuda"):
        profile_dir = Path("out/profile")
        profile_dir.mkdir(parents=True, exist_ok=True)
        _profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=args.profile_cuda, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profile_dir)),
            record_shapes=True,
            with_stack=True,
        )
        _profiler.__enter__()
        print(f"  CUDA profiling enabled: first {args.profile_cuda} steps -> {profile_dir}/")

    # Enforce substrate warmup: freeze substrate params for N steps, then unfreeze
    _warmup_steps = hints.get("warmup_steps", 0)
    _substrate_params = []
    if _warmup_steps > 0:
        from decepticons.causal_bank import learnable_substrate_keys
        _substrate_keys = learnable_substrate_keys(config)
        for pname, param in model.named_parameters():
            if any(k in pname for k in _substrate_keys):
                _substrate_params.append((pname, param))
                param.requires_grad = False
        if _substrate_params:
            print(f"  substrate warmup: {len(_substrate_params)} params frozen for {_warmup_steps} steps")

    model.train()
    for step in range(1, runtime.train.steps + 1):
        # Unfreeze substrate params after warmup
        if step == _warmup_steps + 1 and _substrate_params:
            for pname, param in _substrate_params:
                param.requires_grad = True
            print(f"  substrate warmup complete: {len(_substrate_params)} params unfrozen at step {step}")
        x, y = dataset.batch("train", runtime.train.batch_size, runtime.train.seq_len)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        if hasattr(model, 'substrate_regularization'):
            loss = loss + model.substrate_regularization(step=step)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), runtime.train.grad_clip)
        optimizer.step()

        if _profiler is not None:
            _profiler.step()

        current = float(loss.item())
        losses.append(current)
        if current < best:
            best = current

        if step in probe_step_set:
            probe_entry = probe_entry_by_step(probe_plan, step) or {}
            row_probe_eval_batches = int(probe_entry.get("eval_batches") or effective_probe_eval_batches)
            probe_started = time.time()
            probe_loss = evaluate(
                model,
                dataset,
                runtime.train,
                args.probe_split,
                eval_batches=row_probe_eval_batches,
            )
            probe_elapsed_sec = time.time() - probe_started
            probe_bpt = bits_per_token_from_loss(probe_loss)
            tokens_per_byte = dataset.test_tokens_per_byte if args.probe_split == "test" else None
            probe_bpb = probe_bpt * tokens_per_byte if tokens_per_byte is not None else None
            recent_window = min(len(losses), runtime.train.log_every)
            recent_train_loss = float(sum(losses[-recent_window:]) / recent_window) if recent_window > 0 else float("nan")
            probe_row = {
                "step": step,
                "tier": probe_entry.get("tier"),
                "split": args.probe_split,
                "eval_batches": row_probe_eval_batches,
                "eval_loss": probe_loss,
                "bits_per_token": probe_bpt,
                "bpb": probe_bpb,
                "recent_train_loss": recent_train_loss,
                "elapsed_sec": probe_elapsed_sec,
            }
            probe_row["compute"] = build_probe_compute_accounting_inputs(
                performance_estimate,
                [probe_row],
                split=args.probe_split,
                eval_batches=row_probe_eval_batches,
            )["per_probe"][0]
            cumulative_probe_tflops_est += float(probe_row["compute"].get("eval_tflops_est") or 0.0)
            cumulative_probe_elapsed_sec += float(probe_elapsed_sec)
            probe_history.append(probe_row)
            # Write incremental probe for live ingestion
            _probe_path = Path(args.json).parent / f"{Path(args.json).stem}.probes.jsonl"
            with _probe_path.open("a") as _pf:
                _pf.write(json.dumps({"step": step, "bpb": probe_bpb, "loss": probe_loss, "elapsed_sec": probe_elapsed_sec, "eval_batches": row_probe_eval_batches}) + "\n")
            bpb_text = "n/a" if probe_bpb is None else f"{probe_bpb:.4f}"
            print(
                f"      probe {step:5d} | {args.probe_split} loss {probe_loss:.4f} "
                f"| bpt {probe_bpt:.4f} | bpb {bpb_text}"
            )
            # Run diagnostics on standard+ tier probes (skip micro for speed)
            if row_probe_eval_batches >= 8:
                try:
                    from decepticons.models.diagnostics import diagnose
                    diag_tokens = torch.randint(0, dataset.vocab_size, (2, 64), device=device)
                    diag = diagnose(model, diag_tokens, vocab_size=dataset.vocab_size)
                    probe_row["diagnostics"] = {
                        "modes_alive_pct": diag["summary"]["modes_alive_pct"],
                        "dominant_timescale": diag["summary"]["dominant_timescale"],
                        "findings": diag.get("findings", []),
                    }
                    if diag.get("phase"):
                        probe_row["diagnostics"]["phase_mismatch"] = diag["phase"].get("mismatch_by_band")
                    if diag.get("readout_selectivity"):
                        probe_row["diagnostics"]["readout_by_timescale"] = diag["readout_selectivity"].get("by_timescale")
                    for finding in diag.get("findings", []):
                        print(f"        {finding}")
                except Exception as diag_exc:
                    print(f"        (diagnostics skipped: {diag_exc})")

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
                probe_tflops_consumed_est=cumulative_probe_tflops_est,
                probe_elapsed_sec=cumulative_probe_elapsed_sec,
                interval_probe_tflops_est=cumulative_probe_tflops_est - last_log_probe_tflops_est,
                interval_probe_elapsed_sec=cumulative_probe_elapsed_sec - last_log_probe_elapsed_sec,
            )
            performance_log.append({"step": step, **perf_summary})
            print(
                f"      {step:5d} | loss {recent:.4f} | best {best:.4f} | "
                f"{format_observed_training_performance(perf_summary)}"
            )
            last_log_time = now
            last_log_step = step
            last_log_probe_tflops_est = cumulative_probe_tflops_est
            last_log_probe_elapsed_sec = cumulative_probe_elapsed_sec

    elapsed = time.time() - start
    performance_summary = summarize_observed_training_performance(
        performance_estimate,
        steps_completed=runtime.train.steps,
        elapsed_sec=elapsed,
        probe_tflops_consumed_est=cumulative_probe_tflops_est,
        probe_elapsed_sec=cumulative_probe_elapsed_sec,
    )
    train_eval = evaluate(model, dataset, runtime.train, "train", eval_batches=args.final_eval_batches)
    replay_fixture = CAUSAL_BANK_TRAINING_ADAPTER.build_replay_fixture(
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
    replay_fixture = CAUSAL_BANK_TRAINING_ADAPTER.attach_replay_reference(
        replay_fixture,
        fixture_logits,
    )
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
            "probe_policy": probe_plan.get("policy"),
            "probe_steps": probe_steps,
            "probe_plan": probe_plan,
            "probe_split": args.probe_split,
            "probe_eval_batches": effective_probe_eval_batches,
            "final_eval_batches": effective_final_eval_batches,
            "performance_estimate": performance_estimate,
            "performance": performance_summary,
            "performance_log": performance_log,
            "probes": probe_history,
            "replay_fixture": replay_fixture,
            "compute_accounting_inputs": build_compute_accounting_inputs(
                performance_estimate,
                train_steps_completed=runtime.train.steps,
                train_elapsed_sec=elapsed,
                probe_rows=probe_history,
                probe_split=args.probe_split,
                probe_eval_batches=effective_probe_eval_batches,
                final_eval_batches=effective_final_eval_batches,
                final_eval_splits=2,
                replay_tokens=len(replay_fixture.get("input_token_ids", [])),
                performance_summary=performance_summary,
            ),
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
            "readout_bands": config.readout_bands,
            "linear_hidden_match": args.linear_hidden_match,
            "linear_half_life_min": config.linear_half_life_min,
            "linear_half_life_max": config.linear_half_life_max,
            "oscillatory_frac": config.oscillatory_frac,
            "oscillatory_schedule": config.oscillatory_schedule,
            "oscillatory_period_min": config.oscillatory_period_min,
            "oscillatory_period_max": config.oscillatory_period_max,
            "input_proj_scheme": config.input_proj_scheme,
            "substrate_mode": config.substrate_mode,
            "state_dim": config.state_dim,
            "state_impl": getattr(config, "state_impl", "scan"),
            "num_heads": config.num_heads,
            "block_mixing_ratio": config.block_mixing_ratio,
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
        export_dir = CAUSAL_BANK_TRAINING_ADAPTER.write_export_bundle(
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
    result["forecast"] = build_result_forecast(result, budget=DEFAULT_GOLF_V1_BUDGET)
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
