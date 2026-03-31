from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


RuntimeProfile = Literal["full", "pilot"]


@dataclass(frozen=True)
class TrainConfig:
    seq_len: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    steps: int = 5_000
    log_every: int = 500
    eval_batches: int = 300
    seeds: tuple[int, ...] = (42,)


@dataclass(frozen=True)
class RuntimeConfig:
    train: TrainConfig = field(default_factory=TrainConfig)
    profile: RuntimeProfile = "full"


TRAIN_PROFILES: dict[RuntimeProfile, TrainConfig] = {
    "full": TrainConfig(),
    "pilot": TrainConfig(
        seq_len=64,
        batch_size=24,
        steps=1_000,
        log_every=100,
        eval_batches=50,
        seeds=(42,),
    ),
}


def train_config_for_profile(profile: RuntimeProfile) -> TrainConfig:
    return TRAIN_PROFILES[profile]
