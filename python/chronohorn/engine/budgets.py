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


DEFAULT_GOLF_V1_BUDGET = CompetitionBudget(
    name="golf_v1",
    train_tflops_budget=9_500_000.0,
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
