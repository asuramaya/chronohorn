from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import json
import math
import hashlib
from dataclasses import asdict
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np


CHRONOHORN_PYTHON_ROOT = Path(__file__).resolve().parents[2]
CHRONOHORN_ROOT = CHRONOHORN_PYTHON_ROOT.parent
CHRONOHORN_OUT_ROOT = CHRONOHORN_ROOT / "out"

for path in (CHRONOHORN_PYTHON_ROOT,):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


def import_symbol(module_name: str, attr: str):
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ImportError(f"Could not resolve {attr!r} from {module_name!r}.") from exc


def bits_per_token_from_loss(token_loss_nats: float) -> float:
    return token_loss_nats / math.log(2.0)


def parse_probe_steps(raw: str | None, max_step: int) -> list[int]:
    if not raw:
        return []
    values: list[int] = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        step = int(token)
        if step <= 0:
            raise ValueError(f"Probe step must be positive, got {step}.")
        if step > max_step:
            continue
        values.append(step)
    return sorted(set(values))


def save_state_npz(path: Path, state: dict[str, object]) -> None:
    arrays = {name: np.array(value) for name, value in state.items()}
    np.savez(path, **arrays)


def build_causal_bank_deterministic_substrate(config: Any) -> dict[str, Any]:
    return {
        "substrate_family": "causal-bank",
        "layer_count": 0,
        "hidden_size": int(config.embedding_dim),
        "readout_kind": config.linear_readout_kind,
        "readout_shape": {
            "linear_hidden": [int(width) for width in config.linear_hidden],
            "linear_readout_depth": int(config.linear_readout_depth),
            "linear_readout_num_experts": int(config.linear_readout_num_experts),
            "local_hidden": [int(width) for width in config.local_hidden],
            "local_scale": float(config.local_scale),
        },
        "routing_kind": "softmax" if config.linear_readout_kind == "routed_sqrelu_experts" else "none",
        "routing_shape": {
            "experts": int(config.linear_readout_num_experts),
            "top_k": int(getattr(config, "linear_readout_top_k", 0) or 0),
        },
        "activation_kind": "gelu" if config.linear_readout_kind != "routed_sqrelu_experts" else "relu_squared",
        "memory_kind": "linear_bank+local_window",
        "feature_view_kind": config.mix_mode,
        "local_window": int(config.local_window),
        "oscillatory_schedule": config.oscillatory_schedule,
        "half_life_policy": {
            "min": float(config.linear_half_life_min),
            "max": float(config.linear_half_life_max),
            "oscillatory_frac": float(config.oscillatory_frac),
        },
        "gate_policy": {
            "static_bank_gate": bool(config.static_bank_gate),
            "bank_gate_span": float(config.bank_gate_span),
        },
        "trainable_init_policy": {
            "name": "chronohorn_v1",
            "seed": int(getattr(config, "init_seed", 42)),
        },
        "share_embedding": bool(config.share_embedding),
        "linear_modes": int(config.linear_modes),
    }


def summarize_named_arrays(named_arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    ordered = sorted((name, np.asarray(value, dtype=np.float32)) for name, value in named_arrays.items())
    digest = hashlib.sha256()
    total_count = 0
    total_sum = 0.0
    total_sq_sum = 0.0
    first_values: list[float] = []
    for name, array in ordered:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(b"\0")
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.tobytes())
        flat = contiguous.reshape(-1)
        total_count += int(flat.size)
        total_sum += float(flat.astype(np.float64, copy=False).sum())
        total_sq_sum += float(np.square(flat, dtype=np.float64).sum())
        if len(first_values) < 8:
            take = min(8 - len(first_values), int(flat.size))
            first_values.extend(float(v) for v in flat[:take].tolist())
    rms = math.sqrt(total_sq_sum / total_count) if total_count > 0 else 0.0
    return {
        "tensor_count": len(ordered),
        "value_count": total_count,
        "sha256": digest.hexdigest(),
        "sum": total_sum,
        "rms": rms,
        "first_values": first_values,
    }


def _jsonable_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable_metadata(subvalue) for key, subvalue in value.items()}
    if isinstance(value, tuple):
        return [_jsonable_metadata(subvalue) for subvalue in value]
    if isinstance(value, list):
        return [_jsonable_metadata(subvalue) for subvalue in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def describe_optimizer_defaults(optimizer_like: Any) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    raw_defaults = getattr(optimizer_like, "defaults", None)
    if isinstance(raw_defaults, dict) and raw_defaults:
        for key, value in raw_defaults.items():
            defaults[str(key)] = _jsonable_metadata(value)
        return defaults
    try:
        signature = inspect.signature(optimizer_like)
    except (TypeError, ValueError):
        return defaults
    for name, param in signature.parameters.items():
        if param.default is not inspect._empty:
            defaults[str(name)] = _jsonable_metadata(param.default)
    return defaults


def build_train_policy_metadata(
    *,
    backend: str,
    device: str,
    dtype_policy: str,
    optimizer_name: str,
    optimizer_impl: str,
    optimizer_like: Any,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float,
    compile_train_step: bool,
    compile_eval: bool,
    torch_compile: bool,
    init_policy: str,
    init_seed: int,
    explicit_defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    defaults = (
        _jsonable_metadata(explicit_defaults)
        if explicit_defaults is not None
        else describe_optimizer_defaults(optimizer_like)
    )
    return {
        "version": "chronohorn_train_policy_v1",
        "backend": backend,
        "device": device,
        "dtype_policy": dtype_policy,
        "compile": {
            "train_step": bool(compile_train_step),
            "eval": bool(compile_eval),
            "torch_compile": bool(torch_compile),
        },
        "optimizer": {
            "name": optimizer_name,
            "implementation": optimizer_impl,
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "grad_clip": float(grad_clip),
            "defaults": defaults,
        },
        "init": {
            "policy": init_policy,
            "seed": int(init_seed),
        },
    }


def build_causal_bank_adamw_kwargs(
    *,
    backend: str,
    learning_rate: float,
    weight_decay: float,
    device: str | None = None,
    fused: bool = False,
) -> dict[str, Any]:
    if backend == "torch":
        kwargs = {
            "lr": float(learning_rate),
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": float(weight_decay),
            "amsgrad": False,
            "foreach": False,
            "capturable": False,
            "differentiable": False,
            "maximize": False,
        }
        if device is not None and str(device).startswith("cuda"):
            kwargs["fused"] = bool(fused)
        return kwargs
    if backend == "mlx":
        return {
            "learning_rate": float(learning_rate),
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": float(weight_decay),
            "bias_correction": True,
        }
    raise ValueError(f"Unsupported AdamW backend: {backend}")


def build_causal_bank_adamw_policy_defaults(
    *,
    backend: str,
    learning_rate: float,
    weight_decay: float,
    device: str | None = None,
    fused: bool = False,
) -> dict[str, Any]:
    defaults = build_causal_bank_adamw_kwargs(
        backend=backend,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
        fused=fused,
    )
    if backend == "torch":
        defaults = dict(defaults)
        defaults["bias_correction"] = True
    return defaults


def _safe_call(func: Any) -> Any:
    try:
        return func()
    except Exception:
        return None


def build_backend_environment_metadata(*, backend: str, stack: Any, device: str) -> dict[str, Any]:
    if backend == "torch":
        torch = stack.torch
        metadata: dict[str, Any] = {
            "backend": "torch",
            "device": device,
            "version": getattr(torch, "__version__", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "mps_available": bool(torch.backends.mps.is_available()),
            "num_threads": int(torch.get_num_threads()),
            "num_interop_threads": int(torch.get_num_interop_threads()),
        }
        if hasattr(torch, "get_float32_matmul_precision"):
            metadata["float32_matmul_precision"] = torch.get_float32_matmul_precision()
        if hasattr(torch, "version"):
            metadata["cuda_version"] = getattr(torch.version, "cuda", None)
        cuda_backends = getattr(torch.backends, "cuda", None)
        if cuda_backends is not None and hasattr(cuda_backends, "matmul"):
            metadata["cuda_matmul_allow_tf32"] = getattr(cuda_backends.matmul, "allow_tf32", None)
        cudnn_backends = getattr(torch.backends, "cudnn", None)
        if cudnn_backends is not None:
            metadata["cudnn_enabled"] = getattr(cudnn_backends, "enabled", None)
            metadata["cudnn_allow_tf32"] = getattr(cudnn_backends, "allow_tf32", None)
            metadata["cudnn_version"] = _safe_call(cudnn_backends.version)
        if str(device).startswith("cuda") and torch.cuda.is_available():
            index = torch.cuda.current_device()
            metadata["device_index"] = int(index)
            metadata["device_name"] = torch.cuda.get_device_name(index)
            metadata["device_capability"] = list(torch.cuda.get_device_capability(index))
        return metadata
    if backend == "mlx":
        mx = stack.mx
        version = getattr(mx, "__version__", None)
        if version is None:
            try:
                version = importlib.metadata.version("mlx")
            except importlib.metadata.PackageNotFoundError:
                version = None
        return {
            "backend": "mlx",
            "device": device,
            "version": version,
        }
    raise ValueError(f"Unsupported backend environment metadata: {backend}")


def _estimate_dense_linear_flops(in_dim: int, out_dim: int) -> int:
    return int(2 * in_dim * out_dim)


def _estimate_gelu_flops(width: int) -> int:
    return int(8 * width)


def _estimate_relu_squared_flops(width: int) -> int:
    return int(2 * width)


def _estimate_softmax_flops(width: int) -> int:
    return int(5 * width)


def _estimate_sigmoid_flops(width: int) -> int:
    return int(4 * width)


def _estimate_cross_entropy_flops(vocab_size: int) -> int:
    return int(6 * vocab_size)


def _estimate_logit_feature_flops(vocab_size: int) -> int:
    return int(13 * vocab_size)


def _fft_real_transform_flops(length: int) -> int:
    if length <= 1:
        return 0
    return int(round(2.5 * length * math.log2(length)))


def _append_perf_component(
    components: list[dict[str, Any]],
    *,
    name: str,
    forward_flops_per_token: float,
    train_step_flops_per_token_est: float,
    category: str,
    notes: str,
) -> None:
    components.append(
        {
            "name": name,
            "category": category,
            "forward_flops_per_token": float(forward_flops_per_token),
            "train_step_flops_per_token_est": float(train_step_flops_per_token_est),
            "notes": notes,
        }
    )


def _append_mlp_perf_components(
    components: list[dict[str, Any]],
    *,
    prefix: str,
    in_dim: int,
    hidden_dims: tuple[int, ...],
    out_dim: int,
) -> None:
    prev = int(in_dim)
    for index, hidden_dim in enumerate(hidden_dims):
        linear = _estimate_dense_linear_flops(prev, int(hidden_dim))
        _append_perf_component(
            components,
            name=f"{prefix}.layer{index}.linear",
            forward_flops_per_token=linear,
            train_step_flops_per_token_est=3.0 * linear,
            category="trainable_linear",
            notes="Dense trainable projection; estimate uses forward plus input-gradient and weight-gradient passes.",
        )
        gelu = _estimate_gelu_flops(int(hidden_dim))
        _append_perf_component(
            components,
            name=f"{prefix}.layer{index}.gelu",
            forward_flops_per_token=gelu,
            train_step_flops_per_token_est=2.0 * gelu,
            category="activation",
            notes="Elementwise GELU estimate with backward pass.",
        )
        prev = int(hidden_dim)
    out_linear = _estimate_dense_linear_flops(prev, int(out_dim))
    _append_perf_component(
        components,
        name=f"{prefix}.out.linear",
        forward_flops_per_token=out_linear,
        train_step_flops_per_token_est=3.0 * out_linear,
        category="trainable_linear",
        notes="Final trainable projection to logits.",
    )


def estimate_causal_bank_training_performance(
    *,
    config: Any,
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    trainable_param_count: int,
) -> dict[str, Any]:
    seq_len = int(seq_len)
    batch_size = int(batch_size)
    vocab_size = int(vocab_size)
    trainable_param_count = int(trainable_param_count)
    tokens_per_step = int(batch_size * seq_len)
    linear_readout_in_dim = int(config.linear_modes + config.embedding_dim)
    components: list[dict[str, Any]] = []

    if bool(config.enable_linear):
        projection = _estimate_dense_linear_flops(int(config.embedding_dim), int(config.linear_modes))
        _append_perf_component(
            components,
            name="linear.path.input_projection",
            forward_flops_per_token=projection,
            train_step_flops_per_token_est=2.0 * projection,
            category="fixed_linear",
            notes="Token embedding projected into the frozen linear bank; backward only needs input gradients.",
        )
        if str(config.linear_impl) == "kernel":
            kernel = _estimate_dense_linear_flops(int(config.linear_modes), seq_len)
            _append_perf_component(
                components,
                name="linear.path.kernel_recurrence",
                forward_flops_per_token=kernel,
                train_step_flops_per_token_est=2.0 * kernel,
                category="fixed_linear",
                notes="Dense causal kernel application over the frozen mode bank.",
            )
        else:
            n_fft = 1 << int(math.ceil(math.log2(max(2 * seq_len - 1, 1))))
            freq_bins = n_fft // 2 + 1
            fft_batch = batch_size * int(config.linear_modes)
            fft_total_step = (
                (fft_batch + int(config.linear_modes)) * _fft_real_transform_flops(n_fft)
                + fft_batch * _fft_real_transform_flops(n_fft)
                + 6 * fft_batch * freq_bins
            )
            fft_per_token = fft_total_step / max(tokens_per_step, 1)
            _append_perf_component(
                components,
                name="linear.path.fft_recurrence",
                forward_flops_per_token=fft_per_token,
                train_step_flops_per_token_est=2.0 * fft_per_token,
                category="fixed_linear",
                notes="Approximate real-FFT convolution cost for the frozen bank.",
            )

        if str(config.linear_readout_kind) == "mlp":
            _append_mlp_perf_components(
                components,
                prefix="linear.readout",
                in_dim=linear_readout_in_dim,
                hidden_dims=tuple(int(width) for width in config.linear_hidden),
                out_dim=vocab_size,
            )
        elif str(config.linear_readout_kind) == "tied_recursive":
            hidden_dim = int(config.linear_hidden[0])
            in_proj = _estimate_dense_linear_flops(linear_readout_in_dim, hidden_dim)
            _append_perf_component(
                components,
                name="linear.readout.in_proj",
                forward_flops_per_token=in_proj,
                train_step_flops_per_token_est=3.0 * in_proj,
                category="trainable_linear",
                notes="Trainable input projection into the tied recursive readout.",
            )
            block = _estimate_dense_linear_flops(hidden_dim, hidden_dim)
            hidden_add = float(hidden_dim)
            gelu = float(_estimate_gelu_flops(hidden_dim))
            for depth_index in range(int(config.linear_readout_depth)):
                _append_perf_component(
                    components,
                    name=f"linear.readout.block{depth_index}.residual_add",
                    forward_flops_per_token=hidden_add,
                    train_step_flops_per_token_est=2.0 * hidden_add,
                    category="elementwise",
                    notes="Depth-specific hidden delta add in the tied recursive block.",
                )
                _append_perf_component(
                    components,
                    name=f"linear.readout.block{depth_index}.linear",
                    forward_flops_per_token=block,
                    train_step_flops_per_token_est=3.0 * block,
                    category="trainable_linear",
                    notes="Shared trainable block projection in the tied recursive readout.",
                )
                _append_perf_component(
                    components,
                    name=f"linear.readout.block{depth_index}.gelu",
                    forward_flops_per_token=gelu,
                    train_step_flops_per_token_est=2.0 * gelu,
                    category="activation",
                    notes="Elementwise GELU estimate with backward pass.",
                )
            out_linear = _estimate_dense_linear_flops(hidden_dim, vocab_size)
            _append_perf_component(
                components,
                name="linear.readout.out",
                forward_flops_per_token=out_linear,
                train_step_flops_per_token_est=3.0 * out_linear,
                category="trainable_linear",
                notes="Final trainable tied-recursive projection to logits.",
            )
        else:
            hidden_dim = int(config.linear_hidden[0])
            expert_count = int(config.linear_readout_num_experts)
            router = _estimate_dense_linear_flops(linear_readout_in_dim, expert_count)
            _append_perf_component(
                components,
                name="linear.readout.router",
                forward_flops_per_token=router,
                train_step_flops_per_token_est=3.0 * router,
                category="trainable_linear",
                notes="Trainable router projection over experts.",
            )
            router_softmax = _estimate_softmax_flops(expert_count)
            _append_perf_component(
                components,
                name="linear.readout.router_softmax",
                forward_flops_per_token=router_softmax,
                train_step_flops_per_token_est=2.0 * router_softmax,
                category="activation",
                notes="Router softmax normalization estimate.",
            )
            for expert_index in range(expert_count):
                expert_in = _estimate_dense_linear_flops(linear_readout_in_dim, hidden_dim)
                _append_perf_component(
                    components,
                    name=f"linear.readout.expert{expert_index}.in",
                    forward_flops_per_token=expert_in,
                    train_step_flops_per_token_est=3.0 * expert_in,
                    category="trainable_linear",
                    notes="Per-expert trainable input projection.",
                )
                relu_sq = _estimate_relu_squared_flops(hidden_dim)
                _append_perf_component(
                    components,
                    name=f"linear.readout.expert{expert_index}.relu_squared",
                    forward_flops_per_token=relu_sq,
                    train_step_flops_per_token_est=2.0 * relu_sq,
                    category="activation",
                    notes="Elementwise ReLU-squared activation estimate.",
                )
                expert_out = _estimate_dense_linear_flops(hidden_dim, vocab_size)
                _append_perf_component(
                    components,
                    name=f"linear.readout.expert{expert_index}.out",
                    forward_flops_per_token=expert_out,
                    train_step_flops_per_token_est=3.0 * expert_out,
                    category="trainable_linear",
                    notes="Per-expert trainable output projection to logits.",
                )
            expert_mix = 2.0 * expert_count * vocab_size
            _append_perf_component(
                components,
                name="linear.readout.expert_mix",
                forward_flops_per_token=expert_mix,
                train_step_flops_per_token_est=2.0 * expert_mix,
                category="elementwise",
                notes="Weighted expert reduction over logits.",
            )

        if bool(config.static_bank_gate) and float(getattr(config, "oscillatory_frac", 0.0)) > 0.0:
            mode_gate = float(config.linear_modes)
            _append_perf_component(
                components,
                name="linear.path.static_bank_gate",
                forward_flops_per_token=mode_gate,
                train_step_flops_per_token_est=2.0 * mode_gate,
                category="elementwise",
                notes="Static bank gate broadcast over modes.",
            )

    if bool(config.enable_local):
        local_in_dim = int(config.local_window * config.embedding_dim)
        _append_mlp_perf_components(
            components,
            prefix="local.readout",
            in_dim=local_in_dim,
            hidden_dims=tuple(int(width) for width in config.local_hidden),
            out_dim=vocab_size,
        )
        local_embed_grad = float(config.local_window * config.embedding_dim)
        _append_perf_component(
            components,
            name="local.embedding_gradient_accumulation",
            forward_flops_per_token=0.0,
            train_step_flops_per_token_est=local_embed_grad,
            category="gradient_accumulation",
            notes="Approximate gradient accumulation cost from the stacked local window view.",
        )

    if bool(config.enable_linear):
        linear_embed_grad = float(config.embedding_dim)
        _append_perf_component(
            components,
            name="linear.embedding_gradient_accumulation",
            forward_flops_per_token=0.0,
            train_step_flops_per_token_est=linear_embed_grad,
            category="gradient_accumulation",
            notes="Approximate gradient accumulation cost from the linear embedding path.",
        )

    if bool(config.enable_linear) and bool(config.enable_local):
        if str(config.mix_mode) == "gated":
            per_path_features = float(_estimate_logit_feature_flops(vocab_size))
            _append_perf_component(
                components,
                name="mix.gate_features",
                forward_flops_per_token=2.0 * per_path_features,
                train_step_flops_per_token_est=4.0 * per_path_features,
                category="fixed_reduction",
                notes="Entropy, max, and variance summaries over both logit streams.",
            )
            gate_proj = _estimate_dense_linear_flops(6, 1)
            _append_perf_component(
                components,
                name="mix.gate_projection",
                forward_flops_per_token=gate_proj,
                train_step_flops_per_token_est=3.0 * gate_proj,
                category="trainable_linear",
                notes="Trainable gate projection over six logit summary features.",
            )
            gate_sigmoid = _estimate_sigmoid_flops(1)
            _append_perf_component(
                components,
                name="mix.gate_sigmoid",
                forward_flops_per_token=gate_sigmoid,
                train_step_flops_per_token_est=2.0 * gate_sigmoid,
                category="activation",
                notes="Sigmoid gate activation estimate.",
            )
        mix_logits = float(2 * vocab_size)
        _append_perf_component(
            components,
            name="mix.logit_combine",
            forward_flops_per_token=mix_logits,
            train_step_flops_per_token_est=2.0 * mix_logits,
            category="elementwise",
            notes="Final linear-plus-local logit combination.",
        )

    loss_forward = float(_estimate_cross_entropy_flops(vocab_size))
    _append_perf_component(
        components,
        name="loss.cross_entropy",
        forward_flops_per_token=loss_forward,
        train_step_flops_per_token_est=2.0 * loss_forward,
        category="loss",
        notes="Softmax cross-entropy estimate over the vocab dimension.",
    )

    optimizer_flops_per_step_est = float(12 * trainable_param_count)
    optimizer_flops_per_token_est = optimizer_flops_per_step_est / max(tokens_per_step, 1)
    _append_perf_component(
        components,
        name="optimizer.adamw",
        forward_flops_per_token=0.0,
        train_step_flops_per_token_est=optimizer_flops_per_token_est,
        category="optimizer",
        notes="Approximate AdamW parameter update cost, counted as 12 flops per trainable scalar per step.",
    )

    forward_model_flops_per_token = float(
        sum(
            component["forward_flops_per_token"]
            for component in components
            if component["category"] not in {"loss", "optimizer"}
        )
    )
    forward_loss_flops_per_token = float(
        sum(component["forward_flops_per_token"] for component in components if component["category"] == "loss")
    )
    forward_total_flops_per_token = float(
        sum(component["forward_flops_per_token"] for component in components)
    )
    train_step_flops_per_token_est = float(
        sum(component["train_step_flops_per_token_est"] for component in components)
    )
    return {
        "version": "chronohorn_perf_v1",
        "counting_convention": {
            "dense_mac_flops": 2,
            "gelu_flops_per_element": 8,
            "relu_squared_flops_per_element": 2,
            "softmax_flops_per_element": 5,
            "sigmoid_flops_per_element": 4,
            "cross_entropy_flops_per_vocab": 6,
            "adamw_flops_per_parameter_step": 12,
        },
        "batch_size": batch_size,
        "seq_len": seq_len,
        "tokens_per_step": tokens_per_step,
        "trainable_param_count": trainable_param_count,
        "linear_impl": str(config.linear_impl),
        "forward_model_flops_per_token": forward_model_flops_per_token,
        "forward_loss_flops_per_token": forward_loss_flops_per_token,
        "forward_total_flops_per_token": forward_total_flops_per_token,
        "optimizer_flops_per_step_est": optimizer_flops_per_step_est,
        "optimizer_flops_per_token_est": optimizer_flops_per_token_est,
        "train_step_flops_per_token_est": train_step_flops_per_token_est,
        "forward_total_flops_per_step": forward_total_flops_per_token * tokens_per_step,
        "train_step_flops_per_step_est": train_step_flops_per_token_est * tokens_per_step,
        "components": components,
    }


def summarize_observed_training_performance(
    estimate: dict[str, Any],
    *,
    steps_completed: int,
    elapsed_sec: float,
    interval_steps: int | None = None,
    interval_elapsed_sec: float | None = None,
) -> dict[str, Any]:
    tokens_per_step = float(estimate["tokens_per_step"])
    steps_completed = int(steps_completed)
    elapsed_sec = float(max(elapsed_sec, 1e-9))
    interval_steps = steps_completed if interval_steps is None else int(interval_steps)
    interval_elapsed_sec = elapsed_sec if interval_elapsed_sec is None else float(max(interval_elapsed_sec, 1e-9))
    tokens_completed = float(tokens_per_step * steps_completed)
    interval_tokens = float(tokens_per_step * interval_steps)
    sustained_tokens_per_second = tokens_completed / elapsed_sec
    interval_tokens_per_second = interval_tokens / interval_elapsed_sec
    sustained_tflops = sustained_tokens_per_second * float(estimate["train_step_flops_per_token_est"]) / 1e12
    interval_tflops = interval_tokens_per_second * float(estimate["train_step_flops_per_token_est"]) / 1e12
    sustained_forward_tflops = sustained_tokens_per_second * float(estimate["forward_total_flops_per_token"]) / 1e12
    interval_forward_tflops = interval_tokens_per_second * float(estimate["forward_total_flops_per_token"]) / 1e12
    return {
        "version": estimate["version"],
        "steps_completed": steps_completed,
        "tokens_completed": int(tokens_completed),
        "elapsed_sec": elapsed_sec,
        "samples_per_second": (tokens_completed / max(float(estimate["seq_len"]), 1.0)) / elapsed_sec,
        "tokens_per_second": sustained_tokens_per_second,
        "estimated_sustained_tflops": sustained_tflops,
        "estimated_sustained_forward_tflops": sustained_forward_tflops,
        "interval_steps": interval_steps,
        "interval_elapsed_sec": interval_elapsed_sec,
        "interval_samples_per_second": (interval_tokens / max(float(estimate["seq_len"]), 1.0)) / interval_elapsed_sec,
        "interval_tokens_per_second": interval_tokens_per_second,
        "estimated_interval_tflops": interval_tflops,
        "estimated_interval_forward_tflops": interval_forward_tflops,
        "seconds_per_step": elapsed_sec / max(steps_completed, 1),
    }


def format_observed_training_performance(summary: dict[str, Any]) -> str:
    return (
        f"{summary['interval_tokens_per_second']:.0f} tok/s "
        f"| est {summary['estimated_interval_tflops']:.3f} TF/s"
    )


def build_replay_parity_fixture(
    dataset: Any,
    *,
    split: str,
    sequence_length: int,
    token_count: int = 16,
) -> dict[str, Any]:
    base_dataset = getattr(dataset, "dataset", dataset)
    if not hasattr(base_dataset, "batch_numpy"):
        raise TypeError("Replay parity fixture requires a dataset with batch_numpy().")
    stream = base_dataset.train_stream if split == "train" else base_dataset.test_stream
    slice_len = max(1, min(int(token_count), int(sequence_length)))
    state = (stream.file_idx, stream.tokens, stream.pos)
    try:
        x_np, y_np = base_dataset.batch_numpy(split, 1, slice_len)
    finally:
        stream.file_idx, stream.tokens, stream.pos = state
    source_file = None
    if getattr(stream, "files", None):
        source_file = stream.files[stream.file_idx].name
    return {
        "split": split,
        "token_count": slice_len,
        "source_file": source_file,
        "source_file_index": int(stream.file_idx),
        "source_token_offset": int(stream.pos),
        "input_token_ids": [int(token) for token in x_np[0].tolist()],
        "target_token_ids": [int(token) for token in y_np[0].tolist()],
    }


def attach_replay_fixture_logprob_reference(
    replay_fixture: dict[str, Any],
    logits: np.ndarray,
) -> dict[str, Any]:
    logits_np = np.asarray(logits, dtype=np.float64)
    if logits_np.ndim != 3 or logits_np.shape[0] != 1:
        raise ValueError(f"Replay fixture logits must have shape [1, T, V], got {logits_np.shape}")
    input_ids = [int(token) for token in replay_fixture.get("input_token_ids", [])]
    target_ids = [int(token) for token in replay_fixture.get("target_token_ids", [])]
    usable = min(len(target_ids), logits_np.shape[1])
    runner_target_count = max(min(len(input_ids) - 1, usable), 0)
    replay_fixture["runner_target_count"] = int(runner_target_count)
    if runner_target_count <= 0:
        return replay_fixture
    clipped = logits_np[:, :runner_target_count, :]
    max_logits = np.max(clipped, axis=-1, keepdims=True)
    log_probs = clipped - max_logits - np.log(np.sum(np.exp(clipped - max_logits), axis=-1, keepdims=True))
    gathered = [float(log_probs[0, idx, target_ids[idx]]) for idx in range(runner_target_count)]
    replay_fixture["expected_first_target_position"] = 1
    replay_fixture["expected_first_target_token_id"] = int(target_ids[0])
    replay_fixture["expected_first_gold_logprob"] = float(gathered[0])
    replay_fixture["expected_prefix_gold_logprob_sum"] = float(sum(gathered))
    return replay_fixture


def write_causal_bank_export_bundle(
    *,
    export_root: str | Path,
    summary_path: Path,
    variant: str,
    scale: float,
    readout_kind: str,
    seed: int,
    tokenizer_id: str,
    data_root: str | Path,
    config: Any,
    learned_state: dict[str, object],
    train_step: int,
    train_wallclock_s: float,
    sequence_length: int,
    vocab_size: int,
    profile: str,
    backend: str,
    train_policy: dict[str, Any] | None = None,
    replay_fixture: dict[str, Any] | None = None,
) -> Path:
    from chronohorn.export.bundle import write_opc_export_bundle

    export_dir = Path(export_root).expanduser() / summary_path.stem
    resolved_summary_path = summary_path.expanduser()
    bundled_summary_ref = "summary.json"
    variant_slug = (
        f"{variant}_scale{str(scale).replace('.', 'p')}_"
        f"{readout_kind}_seed{seed}"
    )
    bundle_root = write_opc_export_bundle(
        export_dir,
        model_family_id="causal-bank",
        model_variant_id=variant_slug,
        kernel_version="open-predictive-coder-draft",
        tokenizer_id=tokenizer_id,
        data_root_id=str(Path(data_root).expanduser()),
        deterministic_substrate=build_causal_bank_deterministic_substrate(config),
        learned_state=learned_state,
        artifact_role="replay",
        train_step=train_step,
        train_wallclock_s=train_wallclock_s,
        sequence_length=sequence_length,
        vocab_size=vocab_size,
        dtype_policy="fp32",
        quantization_policy="none",
        notes={
            "summary_json": bundled_summary_ref,
            "summary_json_source": str(resolved_summary_path),
            "seed": seed,
            "profile": profile,
            "backend": backend,
            "train_policy": train_policy,
            "replay_fixture": replay_fixture,
        },
    )
    if resolved_summary_path.is_file():
        (bundle_root / bundled_summary_ref).write_bytes(resolved_summary_path.read_bytes())
    return export_dir


def _estimate_mlp_readout_params(in_dim: int, hidden_dim: int, out_dim: int) -> int:
    return in_dim * hidden_dim + hidden_dim + hidden_dim * out_dim + out_dim


def _estimate_tied_readout_params(in_dim: int, hidden_dim: int, out_dim: int, depth: int) -> int:
    return (
        in_dim * hidden_dim
        + hidden_dim
        + hidden_dim * hidden_dim
        + hidden_dim
        + hidden_dim * out_dim
        + out_dim
        + depth * hidden_dim
    )


def _estimate_mlp_readout_flops(in_dim: int, hidden_dim: int, out_dim: int) -> int:
    return in_dim * hidden_dim + hidden_dim * out_dim


def _estimate_tied_readout_flops(in_dim: int, hidden_dim: int, out_dim: int, depth: int) -> int:
    return in_dim * hidden_dim + depth * hidden_dim * hidden_dim + hidden_dim * out_dim


def _estimate_routed_expert_readout_params(in_dim: int, hidden_dim: int, out_dim: int, num_experts: int) -> int:
    router = in_dim * num_experts + num_experts
    per_expert = in_dim * hidden_dim + hidden_dim + hidden_dim * out_dim + out_dim
    return router + num_experts * per_expert


def _estimate_routed_expert_readout_flops(in_dim: int, hidden_dim: int, out_dim: int, num_experts: int) -> int:
    router = in_dim * num_experts
    per_expert = in_dim * hidden_dim + hidden_dim * out_dim
    return router + num_experts * per_expert


def solve_recursive_hidden_width(
    *,
    baseline_hidden: int,
    in_dim: int,
    out_dim: int,
    depth: int,
    mode: str,
) -> int:
    if mode == "mlp_params":
        budget = _estimate_mlp_readout_params(in_dim, baseline_hidden, out_dim)
        a = 1.0
        b = float(in_dim + out_dim + depth + 2)
    elif mode == "mlp_flops":
        budget = _estimate_mlp_readout_flops(in_dim, baseline_hidden, out_dim)
        a = float(depth)
        b = float(in_dim + out_dim)
    else:
        raise ValueError(f"Unknown linear_hidden_match mode: {mode}")
    root = (-b + math.sqrt(max(b * b + 4.0 * a * budget, 0.0))) / (2.0 * a)
    width = max(int(math.floor(root)), 1)
    if mode == "mlp_params":
        while _estimate_tied_readout_params(in_dim, width, out_dim, depth) > budget and width > 1:
            width -= 1
    else:
        while _estimate_tied_readout_flops(in_dim, width, out_dim, depth) > budget and width > 1:
            width -= 1
    return width


def solve_routed_expert_hidden_width(
    *,
    baseline_hidden: int,
    in_dim: int,
    out_dim: int,
    num_experts: int,
    mode: str,
) -> int:
    if mode == "mlp_params":
        budget = _estimate_mlp_readout_params(in_dim, baseline_hidden, out_dim)
        constant = num_experts * (in_dim + out_dim + 1) + in_dim * num_experts + num_experts
        coeff = num_experts * (in_dim + out_dim + 1)
        width = max(int((budget - constant) // max(coeff, 1)), 1)
        while _estimate_routed_expert_readout_params(in_dim, width, out_dim, num_experts) > budget and width > 1:
            width -= 1
        return width
    if mode == "mlp_flops":
        budget = _estimate_mlp_readout_flops(in_dim, baseline_hidden, out_dim)
        constant = in_dim * num_experts
        coeff = num_experts * (in_dim + out_dim)
        width = max(int((budget - constant) // max(coeff, 1)), 1)
        while _estimate_routed_expert_readout_flops(in_dim, width, out_dim, num_experts) > budget and width > 1:
            width -= 1
        return width
    raise ValueError(f"Unknown linear_hidden_match mode: {mode}")


def _slugify_path_token(raw: str) -> str:
    chars: list[str] = []
    for ch in raw:
        if ch.isalnum():
            chars.append(ch.lower())
        else:
            chars.append("_")
    text = "".join(chars)
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_") or "default"


def build_output_path(
    out_dir: Path,
    stamp: str,
    *,
    variant: str,
    scale: float,
    steps: int,
    seed: int,
    static_bank_gate: bool,
    backend_label: str,
) -> Path:
    variant_token = _slugify_path_token(variant)
    gate_token = "static_bank_gate" if static_bank_gate else "ungated"
    backend_token = _slugify_path_token(backend_label)
    return out_dir / (
        f"causal_bank_{variant_token}_scale_{str(scale).replace('.', 'p')}_"
        f"steps{steps}_{gate_token}_{backend_token}_seed{seed}_{stamp}.json"
    )


def load_existing_result(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def result_matches(
    result: dict[str, object] | None,
    *,
    scale: float,
    steps: int,
    seed: int,
    probe_steps: list[int],
    compile_train_step: bool,
    compile_eval: bool,
) -> bool:
    if result is None:
        return False
    training = result.get("training")
    model = result.get("model")
    config = result.get("config")
    if not isinstance(training, dict) or not isinstance(model, dict) or not isinstance(config, dict):
        return False
    train_cfg = config.get("train")
    if not isinstance(train_cfg, dict):
        return False
    if train_cfg.get("steps") != steps:
        return False
    if model.get("seed") != seed:
        return False
    if abs(float(model.get("scale", 0.0)) - scale) > 1e-9:
        return False
    if training.get("compile_train_step") != compile_train_step:
        return False
    if training.get("compile_eval") != compile_eval:
        return False
    stored_probe_steps = training.get("probe_steps")
    if stored_probe_steps != probe_steps:
        return False
    return True


def summary_row(result: dict[str, Any], json_path: Path, *, skipped: bool) -> dict[str, Any]:
    model = result.get("model", {})
    training = result.get("training", {})
    config = result.get("config", {})
    train_cfg = config.get("train", {}) if isinstance(config, dict) else {}
    quant_rows = result.get("quantization", [])
    best_quant_bpb = None
    if isinstance(quant_rows, list) and quant_rows:
        best_quant_bpb = min(
            (row.get("test_bpb") for row in quant_rows if isinstance(row, dict) and row.get("test_bpb") is not None),
            default=None,
        )
    return {
        "json_path": str(json_path),
        "status": "skipped_existing" if skipped else "completed",
        "scale": model.get("scale"),
        "seed": model.get("seed"),
        "steps": train_cfg.get("steps") if isinstance(train_cfg, dict) else None,
        "test_bpb": model.get("test_bpb"),
        "int_best_bpb": best_quant_bpb,
        "train_time_sec": model.get("train_time_sec"),
        "estimated_train_tflops": (
            training.get("performance", {}).get("estimated_sustained_tflops")
            if isinstance(training.get("performance"), dict)
            else None
        ),
        "probe_steps": training.get("probe_steps"),
        "probes": training.get("probes"),
    }


def parse_row_spec(raw: str) -> tuple[float, int, int]:
    parts = [part.strip() for part in raw.split(":")]
    if len(parts) != 3:
        raise ValueError(f"Row must look like scale:steps:seed, got {raw!r}")
    scale = float(parts[0])
    steps = int(parts[1])
    seed = int(parts[2])
    if steps <= 0:
        raise ValueError(f"Row steps must be positive, got {raw!r}")
    return scale, steps, seed


DEFAULT_ROWS = [
    (16.0, 2200, 42),
    (17.0, 1800, 42),
    (17.0, 2200, 42),
    (17.0, 2600, 42),
    (18.0, 2200, 42),
    (18.0, 2600, 42),
]


def seed_python(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
