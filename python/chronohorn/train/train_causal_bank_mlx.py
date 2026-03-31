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
    build_backend_environment_metadata,
    build_replay_parity_fixture,
    estimate_causal_bank_training_performance,
    format_observed_training_performance,
    build_train_policy_metadata,
    build_output_path,
    load_existing_result,
    parse_probe_steps,
    save_state_npz,
    seed_python,
    summarize_observed_training_performance,
    summarize_named_arrays,
    summary_row,
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
        description="Chronohorn MLX/Metal causal-bank trainer on token shards."
    )
    add_causal_bank_training_arguments(parser, backend="mlx")
    parser.add_argument("--save-state", default=None)
    return parser


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


def run_bridge(args: argparse.Namespace) -> dict[str, object]:
    stack = load_causal_bank_training_stack("mlx")
    mx = stack.mx
    nn = stack.nn
    CausalBankModel = stack.ModelClass
    build_token_shard_dataset = stack.build_token_shard_dataset
    bits_per_token_from_loss_impl = stack.bits_per_token_from_loss
    build_compiled_loss = stack.build_compiled_loss
    count_trainable_params = stack.count_trainable_params
    evaluate = stack.evaluate
    seed_everything = stack.seed_everything
    train_model = stack.train_model
    estimate_trainable_payload_bytes = stack.estimate_trainable_payload_bytes
    quantize_trainable_params = stack.quantize_trainable_params

    runtime = build_runtime(args, stack)
    seed_everything(args.seed)
    dataset = build_token_shard_dataset(args.data_root, vocab_size=args.vocab_size)
    config, baseline_linear_hidden = config_for_variant(args, runtime.train.seq_len, stack)
    assert_safe_config(args, config)
    assert_safe_readout_budget(
        args,
        config,
        baseline_linear_hidden=baseline_linear_hidden,
        out_dim=dataset.vocab_size,
    )
    model = CausalBankModel(vocab_size=dataset.vocab_size, config=config)
    params = count_trainable_params(model)
    if params > args.max_params and not args.unsafe_large_model:
        raise ValueError(
            f"Refusing model with {params:,} trainable params > max_params={args.max_params:,}. "
            "Use --unsafe-large-model or raise --max-params to override."
        )
    probe_steps = parse_probe_steps(args.probe_steps, runtime.train.steps)
    if args.decay_bank == "custom":
        if not args.decays_json:
            raise ValueError("--decay-bank custom requires --decays-json")
        decay_payload = json.loads(Path(args.decays_json).read_text(encoding="utf-8"))
        decays = decay_payload.get("decays")
        if decays is None:
            raise ValueError(f"Decay bank JSON is missing 'decays': {args.decays_json}")
        model.set_linear_decays(decays)
    initial_trainable_state = {
        name: np.array(value)
        for name, value in dict(nn.utils.tree_flatten(model.trainable_parameters())).items()
    }
    init_report = summarize_named_arrays(initial_trainable_state)

    print("\n  causal-bank mlx trainer\n")
    print(
        f"  data_root={args.data_root} seed={args.seed} steps={runtime.train.steps} "
        f"seq_len={runtime.train.seq_len} batch_size={runtime.train.batch_size} "
        f"lr={runtime.train.learning_rate:g}"
    )
    print(
        f"  train_shards={len(dataset.train_files)} val_shards={len(dataset.test_files)} "
        f"train_tokens={dataset.train_token_count:,} val_tokens={dataset.test_token_count:,}"
    )
    print(
        f"  variant={args.variant} scale={args.scale:.3f} linear_modes={config.linear_modes} "
        f"local_window={config.local_window} decay_bank={args.decay_bank} "
        f"linear_readout={config.linear_readout_kind}:{config.linear_readout_depth} "
        f"half_life_max={config.linear_half_life_max:.1f} "
        f"osc_frac={config.oscillatory_frac:.2f} "
        f"osc_schedule={config.oscillatory_schedule} "
        f"static_bank_gate={config.static_bank_gate} "
        f"linear_hidden={list(config.linear_hidden)} "
        f"local_hidden={list(config.local_hidden)} local_scale={config.local_scale:.3f} "
        f"params={params:,}"
    )
    if probe_steps:
        print(
            f"  probe_steps={probe_steps} probe_split={args.probe_split} "
            f"probe_eval_batches={args.probe_eval_batches}"
        )
    if args.compile_train_step or args.compile_eval:
        print(
            f"  compile_train_step={args.compile_train_step} "
            f"compile_eval={args.compile_eval}"
        )

    compiled_eval_loss = None
    if args.compile_eval:
        compiled_eval_loss = build_compiled_loss(model)
        warm_x, warm_y = dataset.batch("train", runtime.train.batch_size, runtime.train.seq_len)
        compiled_warm_loss = compiled_eval_loss(warm_x, warm_y)
        mx.eval(compiled_warm_loss)

    probe_history: list[dict[str, float | int | None]] = []
    probe_step_set = set(probe_steps)
    run_start = time.time()
    performance_estimate = estimate_causal_bank_training_performance(
        config=config,
        vocab_size=dataset.vocab_size,
        batch_size=runtime.train.batch_size,
        seq_len=runtime.train.seq_len,
        trainable_param_count=params,
    )
    performance_log: list[dict[str, float | int | None]] = []
    optimizer_kwargs = build_causal_bank_adamw_kwargs(
        backend="mlx",
        learning_rate=runtime.train.learning_rate,
        weight_decay=runtime.train.weight_decay,
    )
    backend_environment = build_backend_environment_metadata(
        backend="mlx",
        stack=stack,
        device="mlx",
    )

    def on_step(step: int, current_model, losses: list[float]) -> None:
        if step not in probe_step_set:
            return
        probe_loss = evaluate(
            current_model,
            dataset,
            runtime.train,
            args.probe_split,
            eval_batches=args.probe_eval_batches,
            compiled_loss=compiled_eval_loss,
        )
        probe_bpt = bits_per_token_from_loss_impl(probe_loss)
        tokens_per_byte = dataset.test_tokens_per_byte if args.probe_split == "test" else None
        probe_bpb = probe_bpt * tokens_per_byte if tokens_per_byte is not None else None
        recent_window = min(len(losses), runtime.train.log_every)
        recent_train_loss = float(sum(losses[-recent_window:]) / recent_window) if recent_window > 0 else float("nan")
        row = {
            "step": step,
            "split": args.probe_split,
            "eval_batches": args.probe_eval_batches,
            "eval_loss": probe_loss,
            "bits_per_token": probe_bpt,
            "bpb": probe_bpb,
            "recent_train_loss": recent_train_loss,
            "elapsed_sec": time.time() - run_start,
        }
        probe_history.append(row)
        bpb_text = "n/a" if probe_bpb is None else f"{probe_bpb:.4f}"
        print(
            f"      probe {step:5d} | {args.probe_split} loss {probe_loss:.4f} "
            f"| bpt {probe_bpt:.4f} | bpb {bpb_text}"
        )

    def on_log(
        step: int,
        recent_loss: float,
        best_loss: float,
        elapsed_sec: float,
        interval_steps: int,
        interval_elapsed_sec: float,
    ) -> str:
        del recent_loss, best_loss
        perf_summary = summarize_observed_training_performance(
            performance_estimate,
            steps_completed=step,
            elapsed_sec=elapsed_sec,
            interval_steps=interval_steps,
            interval_elapsed_sec=interval_elapsed_sec,
        )
        performance_log.append({"step": step, **perf_summary})
        return format_observed_training_performance(perf_summary)

    metrics = train_model(
        model,
        dataset,
        runtime.train,
        args.seed,
        "causal_bank_mlx_train",
        on_step=on_step if probe_steps else None,
        on_log=on_log,
        optimizer_kwargs=optimizer_kwargs,
        compile_train_step=args.compile_train_step,
        compiled_eval_loss=compiled_eval_loss,
        final_eval_batches=args.final_eval_batches,
    )
    performance_summary = summarize_observed_training_performance(
        performance_estimate,
        steps_completed=runtime.train.steps,
        elapsed_sec=metrics.train_time_sec,
    )
    train_eval = metrics.train_loss
    replay_fixture = build_replay_parity_fixture(
        dataset,
        split="test",
        sequence_length=runtime.train.seq_len,
    )
    fixture_x = mx.array(np.asarray([replay_fixture["input_token_ids"]], dtype=np.int32))
    fixture_logits = model(fixture_x)
    mx.eval(fixture_logits)
    replay_fixture = attach_replay_fixture_logprob_reference(
        replay_fixture,
        np.array(fixture_logits),
    )
    test_eval = metrics.test_loss
    train_bpt = bits_per_token_from_loss_impl(train_eval)
    test_bpt = bits_per_token_from_loss_impl(test_eval)
    test_bpb = test_bpt * dataset.test_tokens_per_byte if dataset.test_tokens_per_byte is not None else None
    train_policy = build_train_policy_metadata(
        backend="mlx",
        device="mlx",
        dtype_policy="fp32",
        optimizer_name=stack.optimizer_name,
        optimizer_impl=stack.optimizer_impl,
        optimizer_like=stack.optim_module.AdamW,
        learning_rate=runtime.train.learning_rate,
        weight_decay=runtime.train.weight_decay,
        grad_clip=runtime.train.grad_clip,
        compile_train_step=args.compile_train_step,
        compile_eval=args.compile_eval,
        torch_compile=False,
        init_policy="chronohorn_v1",
        init_seed=config.init_seed,
        explicit_defaults=optimizer_kwargs,
    )

    result = {
        "title": "causal-bank mlx trainer",
        "config": asdict(runtime),
        "dataset": {
            "source_path": dataset.source_path,
            "tokenizer": dataset.tokenizer,
            "tokenizer_path": dataset.tokenizer_path,
            "train_token_count": int(dataset.train_token_count),
            "test_token_count": int(dataset.test_token_count),
            "test_tokens_per_byte": dataset.test_tokens_per_byte,
            "test_bytes_per_token": dataset.test_bytes_per_token,
        },
        "training": {
            "backend": "mlx",
            "device": "mlx",
            "backend_environment": backend_environment,
            "dtype_policy": "fp32",
            "compile_train_step": args.compile_train_step,
            "compile_eval": args.compile_eval,
            "torch_compile": False,
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
            "preset": "causal_bank",
            "variant": args.variant,
            "scale": args.scale,
            "params": metrics.params,
            "seed": args.seed,
            "linear_modes": config.linear_modes,
            "local_window": config.local_window,
            "share_embedding": config.share_embedding,
            "linear_impl": config.linear_impl,
            "linear_readout_kind": config.linear_readout_kind,
            "linear_readout_depth": config.linear_readout_depth,
            "linear_readout_num_experts": config.linear_readout_num_experts,
            "linear_hidden_match": args.linear_hidden_match,
            "decay_bank": args.decay_bank,
            "decays_json": args.decays_json,
            "linear_half_life_min": config.linear_half_life_min,
            "linear_half_life_max": config.linear_half_life_max,
            "oscillatory_frac": config.oscillatory_frac,
            "oscillatory_period_min": config.oscillatory_period_min,
            "oscillatory_period_max": config.oscillatory_period_max,
            "static_bank_gate": config.static_bank_gate,
            "bank_gate_span": config.bank_gate_span,
            "local_hidden_mult": args.local_hidden_mult,
            "local_scale_override": args.local_scale_override,
            "train_eval_loss": train_eval,
            "test_eval_loss": test_eval,
            "train_bits_per_token": train_bpt,
            "test_bits_per_token": test_bpt,
            "train_bpb": None,
            "test_bpb": test_bpb,
            "overfit_pct": metrics.overfit_pct,
            "train_time_sec": metrics.train_time_sec,
            "learning_rate": runtime.train.learning_rate,
            "payload_bytes_est": None,
            "payload_mb_est": None,
            "embedding_dim": config.embedding_dim,
            "linear_hidden": list(config.linear_hidden),
            "local_hidden": list(config.local_hidden),
            "local_scale": config.local_scale,
            "mix_mode": config.mix_mode,
            "init_policy": "chronohorn_v1",
            "init_seed": config.init_seed,
            "initial_trainable_signature": init_report,
        },
    }

    flat_full = dict(nn.utils.tree_flatten(model.parameters()))
    trainable_names = {name for name, _ in nn.utils.tree_flatten(model.trainable_parameters())}
    result["model"]["payload_bytes_est"] = estimate_trainable_payload_bytes(flat_full, trainable_names)
    result["model"]["payload_mb_est"] = result["model"]["payload_bytes_est"] / (1024.0 * 1024.0)
    if args.save_state:
        save_path = Path(args.save_state)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_state_npz(save_path, flat_full)
        result["model"]["saved_state_path"] = str(save_path)

    quant_rows = []
    for bits in sorted(set(args.quant_bits)):
        quantized_state, stats = quantize_trainable_params(flat_full, trainable_names, bits)
        model.update(nn.utils.tree_unflatten(list(quantized_state.items())))
        q_test_eval = evaluate(
            model,
            dataset,
            runtime.train,
            "test",
            eval_batches=args.final_eval_batches,
            compiled_loss=compiled_eval_loss,
        )
        q_test_bpt = bits_per_token_from_loss_impl(q_test_eval)
        quant_rows.append(
            {
                "scheme": f"uniform_int{bits}",
                "bits": float(bits),
                "test_eval_loss": q_test_eval,
                "test_bits_per_token": q_test_bpt,
                "test_bpb": q_test_bpt * dataset.test_tokens_per_byte,
                **stats,
            }
        )
    if quant_rows:
        result["quantization"] = quant_rows
        model.update(nn.utils.tree_unflatten(list(flat_full.items())))
    print(
        f"  Te:{test_eval:.4f} "
        f"bpt:{result['model']['test_bits_per_token']:.4f} "
        f"bpb:{result['model']['test_bpb']:.4f} "
        f"Of:{metrics.overfit_pct:+.1f}% "
        f"T:{metrics.train_time_sec:.0f}s "
        f"tok/s:{performance_summary['tokens_per_second']:.0f} "
        f"TF/s:{performance_summary['estimated_sustained_tflops']:.3f}"
    )
    for row in quant_rows:
        print(
            f"  {row['scheme']}: "
            f"bpb:{row['test_bpb']:.4f} "
            f"bpt:{row['test_bits_per_token']:.4f} "
            f"payload_mb_est:{row['payload_mb_est']:.3f}"
        )

    output_path = Path(args.json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.export_dir:
        export_dir = write_causal_bank_export_bundle(
            export_root=args.export_dir,
            summary_path=output_path,
            variant=args.variant,
            scale=args.scale,
            readout_kind=config.linear_readout_kind,
            seed=args.seed,
            tokenizer_id=dataset.tokenizer,
            data_root=args.data_root,
            config=config,
            learned_state=flat_full,
            train_step=runtime.train.steps,
            train_wallclock_s=metrics.train_time_sec,
            sequence_length=runtime.train.seq_len,
            vocab_size=dataset.vocab_size,
            profile=args.profile,
            backend="mlx",
            train_policy=train_policy,
            replay_fixture=replay_fixture,
        )
        result["model"]["export_dir"] = str(export_dir)
        result["model"]["export_manifest_path"] = str(export_dir / "manifest.json")
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\n  Wrote JSON summary to {output_path}")
    if args.export_dir:
        print(f"  Wrote opc-export bundle to {result['model']['export_dir']}")
    if args.save_state:
        print(f"  Wrote NPZ state to {args.save_state}")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_bridge(args)


if __name__ == "__main__":
    main()
