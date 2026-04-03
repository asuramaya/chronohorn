from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from chronohorn.engine.importing import import_symbol
from chronohorn.train.runtime_config import RuntimeConfig, train_config_for_profile


@dataclass(frozen=True)
class TrainingBackendStack:
    backend: str
    ConfigClass: Any
    ModelClass: Any
    scale_config: Callable[..., Any]
    RuntimeConfig: Any
    train_config_for_profile: Callable[..., Any]
    optimizer_name: str
    optimizer_impl: str
    device_name: str
    build_token_shard_dataset: Callable[..., Any] | None = None
    build_token_shard_torch_dataset: Callable[..., Any] | None = None
    bits_per_token_from_loss: Callable[..., Any] | None = None
    estimate_trainable_payload_bytes: Callable[..., Any] | None = None
    quantize_trainable_params: Callable[..., Any] | None = None
    build_compiled_loss: Callable[..., Any] | None = None
    count_trainable_params: Callable[..., Any] | None = None
    evaluate: Callable[..., Any] | None = None
    seed_everything: Callable[..., Any] | None = None
    train_model: Callable[..., Any] | None = None
    optim_module: Any | None = None
    mx: Any | None = None
    nn: Any | None = None
    torch: Any | None = None
    functional: Any | None = None


def _load_causal_bank_stack(backend: str) -> TrainingBackendStack:
    """Load training stack for the causal-bank family."""
    if backend == "mlx":
        ConfigClass = import_symbol("chronohorn.models.causal_bank_mlx", "CausalBankConfig")
        ModelClass = import_symbol("chronohorn.models.causal_bank_mlx", "CausalBankModel")
        scale_config = import_symbol("chronohorn.models.causal_bank_mlx", "scale_config")

        from chronohorn.train.runtime import (
            build_compiled_loss,
            count_trainable_params,
            evaluate,
            seed_everything,
            train_model,
        )

        build_token_shard_dataset = import_symbol(
            "chronohorn.train.token_shard_dataset",
            "build_token_shard_dataset",
        )
        bits_per_token_from_loss = import_symbol(
            "chronohorn.models.quantize",
            "bits_per_token_from_loss",
        )
        estimate_trainable_payload_bytes = import_symbol(
            "chronohorn.models.quantize",
            "estimate_trainable_payload_bytes",
        )
        quantize_trainable_params = import_symbol(
            "chronohorn.models.quantize",
            "quantize_trainable_params",
        )

        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim

        return TrainingBackendStack(
            backend="mlx",
            ConfigClass=ConfigClass,
            ModelClass=ModelClass,
            scale_config=scale_config,
            RuntimeConfig=RuntimeConfig,
            train_config_for_profile=train_config_for_profile,
            optimizer_name="AdamW",
            optimizer_impl="mlx.optimizers.AdamW",
            device_name="mlx",
            build_token_shard_dataset=build_token_shard_dataset,
            bits_per_token_from_loss=bits_per_token_from_loss,
            estimate_trainable_payload_bytes=estimate_trainable_payload_bytes,
            quantize_trainable_params=quantize_trainable_params,
            build_compiled_loss=build_compiled_loss,
            count_trainable_params=count_trainable_params,
            evaluate=evaluate,
            seed_everything=seed_everything,
            train_model=train_model,
            optim_module=optim,
            mx=mx,
            nn=nn,
        )

    if backend == "torch":
        ConfigClass = import_symbol("chronohorn.models.causal_bank_torch", "CausalBankConfig")
        ModelClass = import_symbol("chronohorn.models.causal_bank_torch", "CausalBankModel")
        scale_config = import_symbol("chronohorn.models.causal_bank_torch", "scale_config")

        build_token_shard_torch_dataset = import_symbol(
            "chronohorn.train.token_shard_dataset_torch",
            "build_token_shard_torch_dataset",
        )
        build_token_shard_dataset = import_symbol(
            "chronohorn.train.token_shard_dataset",
            "build_token_shard_dataset",
        )

        import torch
        import torch.nn.functional as F

        return TrainingBackendStack(
            backend="torch",
            ConfigClass=ConfigClass,
            ModelClass=ModelClass,
            scale_config=scale_config,
            RuntimeConfig=RuntimeConfig,
            train_config_for_profile=train_config_for_profile,
            optimizer_name="AdamW",
            optimizer_impl="torch.optim.AdamW",
            device_name="torch",
            build_token_shard_dataset=build_token_shard_dataset,
            build_token_shard_torch_dataset=build_token_shard_torch_dataset,
            torch=torch,
            functional=F,
        )

    raise ValueError(f"Unknown training backend: {backend}")


def load_training_backend_stack(backend: str, *, family_id: str = "causal-bank") -> TrainingBackendStack:
    # Normalise legacy ID
    normalized = family_id.replace("_", "-")
    if normalized == "causal-bank":
        return _load_causal_bank_stack(backend)
    # Forward-compatible: other families can provide their own stack loader
    # via the registry in the future.
    from chronohorn.families.registry import available_family_ids
    known = ", ".join(available_family_ids()) or "none"
    raise ValueError(
        f"No training stack loader for family {family_id!r}; known families: {known}"
    )


CausalBankTrainingStack = TrainingBackendStack


def load_causal_bank_training_stack(backend: str) -> TrainingBackendStack:
    return load_training_backend_stack(backend, family_id="causal_bank")
