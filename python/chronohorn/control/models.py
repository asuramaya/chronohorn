from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


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
