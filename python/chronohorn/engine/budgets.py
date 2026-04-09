from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompetitionBudget:
    name: str
    train_tflops_budget: float
    artifact_limit_mb: float
    primary_metric_name: str = "bpb"

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "train_tflops_budget": float(self.train_tflops_budget),
            "artifact_limit_mb": float(self.artifact_limit_mb),
            "primary_metric_name": self.primary_metric_name,
        }


# NVIDIA H100 SXM lists up to 1,979 BF16 Tensor TFLOPS per GPU.
# This budget treats the training window as 8 GPUs for 10 minutes.
# Probe/final evaluation compute is added separately by forecasting.
H100_SXM_BF16_TFLOPS = 1_979.0
GOLF_V1_REFERENCE_GPUS = 8
GOLF_V1_TRAINING_WINDOW_SEC = 600.0
GOLF_V1_EVAL_WINDOW_SEC = 600.0
GOLF_V1_TRAIN_TFLOPS_BUDGET = (
    H100_SXM_BF16_TFLOPS * GOLF_V1_REFERENCE_GPUS * GOLF_V1_TRAINING_WINDOW_SEC
)


DEFAULT_GOLF_V1_BUDGET = CompetitionBudget(
    name="golf_v1",
    train_tflops_budget=GOLF_V1_TRAIN_TFLOPS_BUDGET,
    artifact_limit_mb=16.0,
    primary_metric_name="bpb",
)

DEFAULT_COMPETITION_BUDGETS: dict[str, CompetitionBudget] = {
    DEFAULT_GOLF_V1_BUDGET.name: DEFAULT_GOLF_V1_BUDGET,
}


def resolve_competition_budget(name: str) -> CompetitionBudget:
    budget = DEFAULT_COMPETITION_BUDGETS.get(name)
    if budget is None:
        raise KeyError(f"Unknown competition budget: {name!r}")
    return budget
