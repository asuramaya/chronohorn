from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

LOWER_IS_BETTER_METRICS = {"bpb", "bits_per_token", "eval_loss", "test_bpb", "test_bits_per_token"}


@dataclass(frozen=True)
class RunRecord:
    """A single normalized runtime fact in Chronohorn's observer pipeline."""

    kind: str
    source: str
    family: str
    name: str
    status: str
    path: str | None = None
    metric_name: str | None = None
    metric_value: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _prefer_metric(snapshot: RunSnapshot) -> tuple[str | None, float | None]:
    if snapshot.forecast_metric_name and snapshot.forecast_metric_value is not None:
        return snapshot.forecast_metric_name, snapshot.forecast_metric_value
    return snapshot.metric_name, snapshot.metric_value


def _snapshot_rank_key(snapshot: RunSnapshot) -> tuple[Any, ...]:
    metric_name, metric_value = _prefer_metric(snapshot)
    if metric_name in LOWER_IS_BETTER_METRICS:
        metric_sort = float(metric_value) if metric_value is not None else float("inf")
    elif metric_value is not None:
        metric_sort = -float(metric_value)
    else:
        metric_sort = float("inf")
    return (
        snapshot.decision not in {None, "continue"},
        snapshot.artifact_viable is False,
        metric_sort,
        snapshot.name,
    )


class RunStore:
    """Small, serializable run-record store modeled after Heinrich's SignalStore."""

    def __init__(self) -> None:
        self._records: list[RunRecord] = []

    def add(self, record: RunRecord) -> None:
        self._records.append(record)

    def extend(self, records: Sequence[RunRecord]) -> None:
        self._records.extend(records)

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self):
        return iter(self._records)

    def filter(
        self,
        *,
        kind: str | None = None,
        source: str | None = None,
        family: str | None = None,
        name: str | None = None,
        status: str | None = None,
    ) -> list[RunRecord]:
        out = list(self._records)
        if kind is not None:
            out = [row for row in out if row.kind == kind]
        if source is not None:
            out = [row for row in out if row.source == source]
        if family is not None:
            out = [row for row in out if row.family == family]
        if name is not None:
            out = [row for row in out if row.name == name]
        if status is not None:
            out = [row for row in out if row.status == status]
        return out

    def runs(self) -> list[RunSnapshot]:
        grouped: dict[str, list[RunRecord]] = {}
        for record in self._records:
            grouped.setdefault(record.name, []).append(record)

        snapshots: list[RunSnapshot] = []
        for name, rows in grouped.items():
            manifest = next((row for row in reversed(rows) if row.kind == "manifest"), None)
            runtime = next((row for row in reversed(rows) if row.kind == "runtime_state"), None)
            launch = next((row for row in reversed(rows) if row.kind == "launch"), None)
            result = next((row for row in reversed(rows) if row.kind == "result"), None)
            forecast = next((row for row in reversed(rows) if row.kind == "forecast"), None)

            family = next((row.family for row in reversed(rows) if row.family), None)
            state = "unknown"
            if runtime is not None:
                state = runtime.status
            elif result is not None:
                state = result.status
            elif launch is not None:
                state = launch.status
            elif manifest is not None:
                state = manifest.status

            host = None
            launcher = None
            for record in (runtime, launch, manifest):
                if record is None:
                    continue
                host = host or str(record.metadata.get("host") or "") or None
                launcher = launcher or str(record.metadata.get("launcher") or "") or None
            path = None
            for record in (result, forecast, launch, manifest):
                if record is not None and record.path:
                    path = record.path
                    break

            artifact_viable = None
            if forecast is not None:
                artifact_viable = bool(forecast.metadata.get("artifact_viable"))

            metadata: dict[str, Any] = {}
            if manifest is not None:
                metadata["manifest"] = manifest.metadata
            if runtime is not None:
                metadata["runtime_state"] = runtime.metadata
            if launch is not None:
                metadata["launch"] = launch.metadata
            if result is not None:
                metadata["result"] = result.metadata
            if forecast is not None:
                metadata["forecast"] = forecast.metadata

            snapshots.append(
                RunSnapshot(
                    name=name,
                    family=family,
                    state=state,
                    decision=forecast.status if forecast is not None else None,
                    path=path,
                    host=host,
                    launcher=launcher,
                    metric_name=result.metric_name if result is not None else None,
                    metric_value=result.metric_value if result is not None else None,
                    forecast_metric_name=forecast.metric_name if forecast is not None else None,
                    forecast_metric_value=forecast.metric_value if forecast is not None else None,
                    artifact_viable=artifact_viable,
                    metadata=metadata,
                )
            )
        snapshots.sort(key=_snapshot_rank_key)
        return snapshots

    def best_runs(self, k: int = 10) -> list[RunSnapshot]:
        return self.runs()[: max(k, 0)]

    def summary(self) -> dict[str, Any]:
        by_kind: dict[str, int] = {}
        by_status: dict[str, int] = {}
        by_family: dict[str, int] = {}
        for row in self._records:
            by_kind[row.kind] = by_kind.get(row.kind, 0) + 1
            by_status[row.status] = by_status.get(row.status, 0) + 1
            if row.family:
                by_family[row.family] = by_family.get(row.family, 0) + 1
        runs = self.runs()
        by_state: dict[str, int] = {}
        by_decision: dict[str, int] = {}
        for run in runs:
            by_state[run.state] = by_state.get(run.state, 0) + 1
            if run.decision:
                by_decision[run.decision] = by_decision.get(run.decision, 0) + 1
        return {
            "record_count": len(self._records),
            "run_count": len(runs),
            "by_kind": by_kind,
            "by_status": by_status,
            "by_family": by_family,
            "by_state": by_state,
            "by_decision": by_decision,
        }

    def to_json(self) -> str:
        return json.dumps([record.as_dict() for record in self._records], indent=2, sort_keys=True)

    def save(self, path: str | Path) -> Path:
        output_path = Path(path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json() + "\n", encoding="utf-8")
        return output_path

    @classmethod
    def from_records(cls, records: Iterable[RunRecord]) -> "RunStore":
        store = cls()
        store.extend(list(records))
        return store

    @classmethod
    def from_json(cls, data: str) -> "RunStore":
        payload = json.loads(data)
        if not isinstance(payload, list):
            raise ValueError("run-store JSON must be a list")
        store = cls()
        for row in payload:
            if not isinstance(row, dict):
                continue
            store.add(
                RunRecord(
                    kind=str(row.get("kind", "")),
                    source=str(row.get("source", "")),
                    family=str(row.get("family", "")),
                    name=str(row.get("name", "")),
                    status=str(row.get("status", "")),
                    path=row.get("path"),
                    metric_name=row.get("metric_name"),
                    metric_value=_safe_float(row.get("metric_value")),
                    metadata=dict(row.get("metadata") or {}),
                )
            )
        return store

    @classmethod
    def load(cls, path: str | Path) -> "RunStore":
        input_path = Path(path).expanduser().resolve()
        return cls.from_json(input_path.read_text(encoding="utf-8"))
