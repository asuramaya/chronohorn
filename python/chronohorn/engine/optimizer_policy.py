from __future__ import annotations

import inspect
from typing import Any

import numpy as np


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


def build_adamw_kwargs(
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


def build_adamw_policy_defaults(
    *,
    backend: str,
    learning_rate: float,
    weight_decay: float,
    device: str | None = None,
    fused: bool = False,
) -> dict[str, Any]:
    defaults = build_adamw_kwargs(
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
