from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


LOWER_IS_BETTER_METRICS = {"bpb", "bits_per_token", "eval_loss", "test_bpb", "test_bits_per_token"}


@dataclass(frozen=True)
class RunSnapshot:
    """Agent-facing merged view of a run across manifest, launch, result, and forecast records."""

    name: str
    family: str | None
    state: str
    decision: str | None
    path: str | None
    host: str | None
    launcher: str | None
    metric_name: str | None
    metric_value: float | None
    forecast_metric_name: str | None
    forecast_metric_value: float | None
    artifact_viable: bool | None
    run_id: str | None = None
    trust_state: str | None = None
    metric_state: str | None = None
    replication_state: str | None = None
    replicate_count: int | None = None
    quarantine_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ControlAction:
    action: str
    target_name: str | None
    family: str | None
    priority: float
    rationale: str
    state: str | None = None
    host: str | None = None
    launcher: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ControlPlan:
    summary: dict[str, Any]
    actions: tuple[ControlAction, ...]
    runs: tuple[dict[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "actions": [action.as_dict() for action in self.actions],
            "runs": list(self.runs),
        }
