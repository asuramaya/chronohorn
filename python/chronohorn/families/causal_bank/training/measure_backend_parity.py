#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from chronohorn.engine.backend_metadata import build_backend_environment_metadata
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
from chronohorn.engine.signatures import summarize_named_arrays
from chronohorn.families.causal_bank import CAUSAL_BANK_TRAINING_ADAPTER
from chronohorn.families.causal_bank.training.causal_bank_training_support import (
    build_compute_accounting_inputs,
    seed_python,
)
from chronohorn.families.causal_bank.training.causal_bank_training_stack import (
    TrainingBackendStack,
    load_training_backend_stack,
)
from chronohorn.families.causal_bank.training.causal_bank_training_primitives import (
    add_causal_bank_core_arguments,
    build_causal_bank_training_runtime,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chronohorn fixed-batch backend parity harness for MLX or Torch."
    )
    parser.add_argument("--backend", choices=["mlx", "torch"], required=True)
    add_causal_bank_core_arguments(parser)
    parser.set_defaults(steps=1)
    parser.add_argument("--device", default=None, help="Torch device override; ignored by MLX.")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--json", required=True)
    return parser


def _stable_batch_digest(x_np: np.ndarray, y_np: np.ndarray) -> str:
    digest = hashlib.sha256()
    for label, array in (("x", x_np), ("y", y_np)):
        arr = np.ascontiguousarray(array)
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(arr.shape).encode("utf-8"))
        digest.update(b"\0")
        digest.update(arr.tobytes())
    return digest.hexdigest()


def _torch_cross_entropy(stack: TrainingBackendStack, logits, y):
    F = stack.functional
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))


def _mlx_cross_entropy(stack: TrainingBackendStack, logits, y):
    mx = stack.mx
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    gathered = mx.take_along_axis(log_probs, y[..., None], axis=-1)
    return -mx.mean(gathered)


def _collect_named_arrays_torch(model, *, include_buffers: bool = False) -> dict[str, np.ndarray]:
    named = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            named[name] = param.detach().cpu().numpy().astype(np.float32, copy=False)
    if include_buffers:
        for name, buf in model.named_buffers():
            named[f"buffer::{name}"] = buf.detach().cpu().numpy().astype(np.float32, copy=False)
    return named


def _collect_named_arrays_mlx(model, *, include_frozen: bool = False) -> dict[str, np.ndarray]:
    frozen_suffixes = ("linear_in_proj", "linear_decays", "linear_kernel")
    leaf_types = ()
    try:
        import mlx.core as mx

        leaf_types = (type(mx.array(np.zeros((1,), dtype=np.float32))),)
    except Exception:
        leaf_types = ()

    named: dict[str, np.ndarray] = {}
    seen: set[int] = set()

    def recurse(value: Any, path: str) -> None:
        value_id = id(value)
        if value_id in seen:
            return
        if isinstance(value, (str, bytes, int, float, bool, np.generic, type(None))):
            return
        if leaf_types and isinstance(value, leaf_types):
            key = path.rstrip(".")
            if include_frozen or not any(key == suffix or key.endswith("." + suffix) for suffix in frozen_suffixes):
                named[key] = np.asarray(value, dtype=np.float32)
            elif include_frozen:
                named[f"buffer::{key}"] = np.asarray(value, dtype=np.float32)
            return
        if isinstance(value, dict):
            seen.add(value_id)
            for key, subvalue in value.items():
                recurse(subvalue, f"{path}{key}.")
            return
        if isinstance(value, (list, tuple)):
            seen.add(value_id)
            for index, subvalue in enumerate(value):
                recurse(subvalue, f"{path}{index}.")
            return
        if hasattr(value, "__dict__"):
            seen.add(value_id)
            for key, subvalue in vars(value).items():
                if key.startswith("_"):
                    continue
                recurse(subvalue, f"{path}{key}.")
            return

    recurse(model, "")
    return named


def _summaries_for_backend(backend: str, model) -> tuple[dict[str, Any], dict[str, Any]]:
    if backend == "torch":
        return summarize_named_arrays(_collect_named_arrays_torch(model)), summarize_named_arrays(
            _collect_named_arrays_torch(model, include_buffers=True)
        )
    return summarize_named_arrays(_collect_named_arrays_mlx(model)), summarize_named_arrays(
        _collect_named_arrays_mlx(model, include_frozen=True)
    )


def _build_model_and_runtime(args: argparse.Namespace, stack: TrainingBackendStack):
    runtime = build_causal_bank_training_runtime(
        args,
        RuntimeConfig=stack.RuntimeConfig,
        train_config_for_profile=stack.train_config_for_profile,
    )
    config, baseline_linear_hidden = CAUSAL_BANK_TRAINING_ADAPTER.build_variant_config(
        args,
        ConfigClass=stack.ConfigClass,
        scale_config=stack.scale_config,
        seq_len=runtime.train.seq_len,
        vocab_size=args.vocab_size,
    )
    CAUSAL_BANK_TRAINING_ADAPTER.validate_config(
        args,
        config,
        baseline_linear_hidden=baseline_linear_hidden,
        out_dim=args.vocab_size,
    )
    return config, runtime


def _build_dataset(args: argparse.Namespace, stack: TrainingBackendStack):
    build_token_shard_dataset = stack.build_token_shard_dataset
    return build_token_shard_dataset(args.data_root, vocab_size=args.vocab_size)


def _convert_batch_for_backend(
    batch_x_np: np.ndarray,
    batch_y_np: np.ndarray,
    stack: TrainingBackendStack,
    device: str | None,
):
    if stack.torch is not None:
        torch = stack.torch
        dtype = torch.long
        batch_x = torch.from_numpy(batch_x_np.astype(np.int64, copy=False)).to(device=device, dtype=dtype)
        batch_y = torch.from_numpy(batch_y_np.astype(np.int64, copy=False)).to(device=device, dtype=dtype)
        return batch_x, batch_y
    mx = stack.mx
    return mx.array(batch_x_np.astype(np.int32, copy=False)), mx.array(batch_y_np.astype(np.int32, copy=False))


def _torch_device(args: argparse.Namespace, stack: TrainingBackendStack) -> str:
    torch = stack.torch
    if args.device is not None:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("Requested --device cuda, but torch.cuda.is_available() is false.")
        if args.device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("Requested --device mps, but torch.backends.mps.is_available() is false.")
        return args.device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _repeat_fixed_batch_torch(
    model,
    batch_x,
    batch_y,
    runtime,
    stack: TrainingBackendStack,
    *,
    steps: int,
    device: str,
) -> dict[str, Any]:
    torch = stack.torch
    F = stack.functional
    optimizer_kwargs = build_adamw_kwargs(
        backend="torch",
        learning_rate=runtime.train.learning_rate,
        weight_decay=runtime.train.weight_decay,
        device=device,
        fused=False,
    )
    optimizer_policy_defaults = build_adamw_policy_defaults(
        backend="torch",
        learning_rate=runtime.train.learning_rate,
        weight_decay=runtime.train.weight_decay,
        device=device,
        fused=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    grad_clip = float(runtime.train.grad_clip)
    model.train()

    fixed_before = float(_torch_cross_entropy(stack, model(batch_x), batch_y).item())
    grad_norm = None
    update_norm = None
    first_step_loss = None
    step_history: list[float] = []
    start = time.time()
    for step_index in range(max(steps, 1)):
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x)
        loss = _torch_cross_entropy(stack, logits, batch_y)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item())
        before = [param.detach().clone() for param in model.parameters() if param.requires_grad]
        optimizer.step()
        after = [param.detach().clone() for param in model.parameters() if param.requires_grad]
        delta_sq = 0.0
        for left, right in zip(before, after):
            diff = (right - left).detach().float()
            delta_sq += float(torch.sum(diff * diff).item())
        update_norm = float(math.sqrt(max(delta_sq, 0.0)))
        step_logits = model(batch_x)
        step_loss = float(F.cross_entropy(step_logits.reshape(-1, step_logits.shape[-1]), batch_y.reshape(-1)).item())
        step_history.append(step_loss)
        if first_step_loss is None:
            first_step_loss = step_loss

    final_loss = step_history[-1] if step_history else fixed_before
    return {
        "fixed_batch_loss": fixed_before,
        "first_step_loss": first_step_loss,
        "final_fixed_batch_loss": final_loss,
        "grad_norm": grad_norm,
        "update_norm": update_norm,
        "step_history": step_history,
        "step_supported": True,
        "train_elapsed_sec": time.time() - start,
        "steps_ran": int(max(steps, 1)),
        "optimizer": {
            "name": "AdamW",
            "implementation": "torch.optim.AdamW",
            "learning_rate": float(runtime.train.learning_rate),
            "weight_decay": float(runtime.train.weight_decay),
            "grad_clip": float(runtime.train.grad_clip),
            "defaults": optimizer_policy_defaults,
        },
    }


def _repeat_fixed_batch_mlx(
    model,
    batch_x,
    batch_y,
    runtime,
    stack: TrainingBackendStack,
    *,
    steps: int,
) -> dict[str, Any]:
    mx = stack.mx
    nn = stack.nn
    optim = stack.optim_module
    fixed_before = float(_mlx_cross_entropy(stack, model(batch_x), batch_y))
    optimizer_kwargs = build_adamw_kwargs(
        backend="mlx",
        learning_rate=runtime.train.learning_rate,
        weight_decay=runtime.train.weight_decay,
    )

    optimizer_report = build_train_policy_metadata(
        backend="mlx",
        device="mlx",
        dtype_policy="fp32",
        optimizer_name="AdamW",
        optimizer_impl="mlx.optimizers.AdamW",
        optimizer_like=optim.AdamW,
        learning_rate=runtime.train.learning_rate,
        weight_decay=runtime.train.weight_decay,
        grad_clip=runtime.train.grad_clip,
        compile_train_step=False,
        compile_eval=False,
        torch_compile=False,
        init_policy="chronohorn_v1",
        init_seed=int(model.config.init_seed),
        explicit_defaults=optimizer_kwargs,
    )["optimizer"]

    try:
        optimizer = optim.AdamW(**optimizer_kwargs)
        grad_clip = float(runtime.train.grad_clip)

        def train_loss_fn(current_model, x, y):
            return _mlx_cross_entropy(stack, current_model(x), y)

        value_and_grad = nn.value_and_grad(model, train_loss_fn)
        grad_norm = None
        update_norm = None
        first_step_loss = None
        step_history: list[float] = []
        start = time.time()

        def flatten_trainable_numpy() -> dict[str, np.ndarray]:
            flat = dict(nn.utils.tree_flatten(model.trainable_parameters()))
            out: dict[str, np.ndarray] = {}
            for name, value in flat.items():
                mx.eval(value)
                out[str(name)] = np.asarray(value, dtype=np.float32)
            return out

        for _ in range(max(steps, 1)):
            before = flatten_trainable_numpy()
            loss, grads = value_and_grad(model, batch_x, batch_y)
            grads, grad_norm_raw = optim.clip_grad_norm(grads, max_norm=grad_clip)
            optimizer.update(model, grads)
            mx.eval(loss, model.state, optimizer.state)
            if grad_norm_raw is not None:
                mx.eval(grad_norm_raw)
                grad_norm = float(grad_norm_raw.item())
            after = flatten_trainable_numpy()
            delta_sq = 0.0
            for name, left in before.items():
                right = after.get(name)
                if right is None:
                    continue
                diff = right.astype(np.float64, copy=False) - left.astype(np.float64, copy=False)
                delta_sq += float(np.square(diff, dtype=np.float64).sum())
            update_norm = float(math.sqrt(max(delta_sq, 0.0)))
            step_loss = float(_mlx_cross_entropy(stack, model(batch_x), batch_y))
            step_history.append(step_loss)
            if first_step_loss is None:
                first_step_loss = step_loss

        final_loss = step_history[-1] if step_history else fixed_before
        return {
            "fixed_batch_loss": fixed_before,
            "first_step_loss": first_step_loss,
            "final_fixed_batch_loss": final_loss,
            "grad_norm": grad_norm,
            "update_norm": update_norm,
            "step_history": step_history,
            "step_supported": True,
            "step_reason": None,
            "train_elapsed_sec": time.time() - start,
            "steps_ran": int(max(steps, 1)),
            "optimizer": optimizer_report,
        }
    except Exception as exc:
        return {
            "fixed_batch_loss": fixed_before,
            "first_step_loss": None,
            "final_fixed_batch_loss": fixed_before,
            "grad_norm": None,
            "update_norm": None,
            "step_history": [],
            "step_supported": False,
            "step_reason": f"MLX parity step failed: {exc}",
            "train_elapsed_sec": 0.0,
            "steps_ran": 0,
            "optimizer": optimizer_report,
        }


def _build_model(args: argparse.Namespace, stack: TrainingBackendStack, config):
    CausalBankModel = stack.ModelClass
    return CausalBankModel(args.vocab_size, config)


def run_parity(args: argparse.Namespace) -> dict[str, Any]:
    seed_python(args.seed)
    stack = load_training_backend_stack(args.backend)
    if args.backend == "torch":
        torch = stack.torch
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    config, runtime = _build_model_and_runtime(args, stack)
    dataset = _build_dataset(args, stack)
    model = _build_model(args, stack, config)

    device = _torch_device(args, stack) if args.backend == "torch" else "mlx"
    backend_environment = build_backend_environment_metadata(
        backend=args.backend,
        stack=stack,
        device=device,
    )
    batch_x_np, batch_y_np = dataset.batch_numpy(args.split, runtime.train.batch_size, runtime.train.seq_len)
    batch_digest = _stable_batch_digest(batch_x_np, batch_y_np)
    batch_x, batch_y = _convert_batch_for_backend(batch_x_np, batch_y_np, stack, device if args.backend == "torch" else None)

    if args.backend == "torch":
        model = model.to(device)
    init_trainable_signature, init_state_signature = _summaries_for_backend(args.backend, model)
    performance_estimate = CAUSAL_BANK_TRAINING_ADAPTER.estimate_training_performance(
        config=config,
        vocab_size=args.vocab_size,
        batch_size=runtime.train.batch_size,
        seq_len=runtime.train.seq_len,
        trainable_param_count=int(init_trainable_signature["value_count"]),
    )

    if args.backend == "torch":
        torch = stack.torch
        with torch.no_grad():
            initial_loss = float(_torch_cross_entropy(stack, model(batch_x), batch_y).item())
        step_report = _repeat_fixed_batch_torch(
            model,
            batch_x,
            batch_y,
            runtime,
            stack,
            steps=args.steps,
            device=device,
        )
    else:
        initial_loss = float(_mlx_cross_entropy(stack, model(batch_x), batch_y))
        step_report = _repeat_fixed_batch_mlx(model, batch_x, batch_y, runtime, stack, steps=args.steps)

    tokens_per_byte = getattr(dataset, "test_tokens_per_byte", None) if args.split == "test" else None
    fixed_bpt = bits_per_token_from_loss(initial_loss)
    fixed_bpb = fixed_bpt * float(tokens_per_byte) if tokens_per_byte is not None else None
    first_step_bpt = bits_per_token_from_loss(step_report["first_step_loss"]) if step_report["first_step_loss"] is not None else None
    first_step_bpb = first_step_bpt * float(tokens_per_byte) if (first_step_bpt is not None and tokens_per_byte is not None) else None
    final_loss = float(step_report["final_fixed_batch_loss"])
    final_bpt = bits_per_token_from_loss(final_loss)
    final_bpb = final_bpt * float(tokens_per_byte) if tokens_per_byte is not None else None

    train_policy = build_train_policy_metadata(
        backend=args.backend,
        device=device,
        dtype_policy="fp32",
        optimizer_name=stack.optimizer_name,
        optimizer_impl=step_report["optimizer"]["implementation"],
        optimizer_like=stack.optim_module.AdamW if args.backend == "mlx" else stack.torch.optim.AdamW,
        learning_rate=runtime.train.learning_rate,
        weight_decay=runtime.train.weight_decay,
        grad_clip=runtime.train.grad_clip,
        compile_train_step=False,
        compile_eval=False,
        torch_compile=False,
        init_policy="chronohorn_v1",
        init_seed=int(args.seed),
        explicit_defaults=step_report["optimizer"]["defaults"],
    )
    train_policy["optimizer"] = step_report["optimizer"]
    performance_summary = summarize_observed_training_performance(
        performance_estimate,
        steps_completed=int(step_report["steps_ran"]),
        elapsed_sec=max(float(step_report["train_elapsed_sec"]), 1e-9),
    )

    report = {
        "backend": args.backend,
        "device": device,
        "backend_environment": backend_environment,
        "data_root": str(args.data_root),
        "split": args.split,
        "seed": int(args.seed),
        "profile": args.profile,
        "batch": {
            "size": int(runtime.train.batch_size),
            "seq_len": int(runtime.train.seq_len),
            "digest_sha256": batch_digest,
        },
        "config": asdict(config),
        "runtime": {
            "steps": int(args.steps),
            "learning_rate": float(runtime.train.learning_rate),
            "weight_decay": float(runtime.train.weight_decay),
            "grad_clip": float(runtime.train.grad_clip),
        },
        "model": {
            "variant": args.variant,
            "linear_readout_kind": config.linear_readout_kind,
            "linear_readout_depth": int(config.linear_readout_depth),
            "linear_readout_num_experts": int(config.linear_readout_num_experts),
            "param_count": int(sum(np.asarray(value).size for value in _collect_named_arrays_torch(model, include_buffers=True).values()))
            if args.backend == "torch"
            else int(sum(np.asarray(value).size for value in _collect_named_arrays_mlx(model, include_frozen=True).values())),
            "initial_trainable_signature": init_trainable_signature,
            "initial_state_signature": init_state_signature,
        },
        "diagnostics": {
            "initial_loss": initial_loss,
            "initial_bits_per_token": fixed_bpt,
            "initial_bpb": fixed_bpb,
            "first_step_loss": step_report["first_step_loss"],
            "first_step_bits_per_token": first_step_bpt,
            "first_step_bpb": first_step_bpb,
            "final_fixed_batch_loss": final_loss,
            "final_bits_per_token": final_bpt,
            "final_bpb": final_bpb,
            "grad_norm": step_report["grad_norm"],
            "update_norm": step_report["update_norm"],
            "step_supported": step_report["step_supported"],
            "step_reason": step_report.get("step_reason"),
            "step_history": step_report["step_history"],
            "train_elapsed_sec": step_report["train_elapsed_sec"],
        },
        "performance_estimate": performance_estimate,
        "performance": performance_summary,
        "train_policy": train_policy,
        "optimizer": step_report["optimizer"],
        "compute_accounting_inputs": build_compute_accounting_inputs(
            performance_estimate,
            train_steps_completed=int(step_report["steps_ran"]),
            train_elapsed_sec=float(step_report["train_elapsed_sec"]),
            probe_rows=[],
            probe_split=None,
            probe_eval_batches=None,
            performance_summary=performance_summary,
        ),
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    report = run_parity(args)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    fixed = report["diagnostics"]["initial_loss"]
    first = report["diagnostics"]["first_step_loss"]
    step_state = "supported" if report["diagnostics"]["step_supported"] else "forward-only"
    print(
        f"chronohorn parity backend={args.backend} split={args.split} step={step_state} "
        f"init_loss={fixed:.6f} first_step_loss={first if first is None else f'{first:.6f}'} "
        f"{format_observed_training_performance(report['performance'])} "
        f"sha256={report['model']['initial_trainable_signature']['sha256']} json={json_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
