from __future__ import annotations

from typing import Any

from chronohorn.store import LOWER_IS_BETTER_METRICS, RunSnapshot


from chronohorn.engine.results import safe_float


def metric_is_lower_better(metric_name: str | None) -> bool:
    return metric_name in LOWER_IS_BETTER_METRICS


def run_metric_name(run: RunSnapshot) -> str | None:
    return run.forecast_metric_name or run.metric_name


def current_metric(run: RunSnapshot) -> float | None:
    return safe_float(run.metric_value)


def forecast_metric(run: RunSnapshot) -> float | None:
    if run.forecast_metric_value is not None:
        return safe_float(run.forecast_metric_value)
    return safe_float(run.metric_value)


def uncertainty_bounds(run: RunSnapshot) -> tuple[float | None, float | None]:
    forecast = run.metadata.get("forecast") if isinstance(run.metadata, dict) else {}
    if not isinstance(forecast, dict):
        return None, None
    uncertainty = forecast.get("uncertainty")
    if not isinstance(uncertainty, dict):
        return None, None
    return safe_float(uncertainty.get("forecast_low_95")), safe_float(uncertainty.get("forecast_high_95"))


def pessimistic_metric(run: RunSnapshot) -> float | None:
    metric_name = run_metric_name(run)
    lower, upper = uncertainty_bounds(run)
    if metric_is_lower_better(metric_name):
        return upper if upper is not None else forecast_metric(run)
    return lower if lower is not None else forecast_metric(run)


def optimistic_metric(run: RunSnapshot) -> float | None:
    metric_name = run_metric_name(run)
    lower, upper = uncertainty_bounds(run)
    if metric_is_lower_better(metric_name):
        return lower if lower is not None else forecast_metric(run)
    return upper if upper is not None else forecast_metric(run)


def marginal_gain_per_tflop(run: RunSnapshot) -> float | None:
    forecast = run.metadata.get("forecast") if isinstance(run.metadata, dict) else {}
    if not isinstance(forecast, dict):
        return None
    return safe_float(forecast.get("marginal_gain_per_tflop"))


def marginal_gain_per_hour(run: RunSnapshot) -> float | None:
    forecast = run.metadata.get("forecast") if isinstance(run.metadata, dict) else {}
    if not isinstance(forecast, dict):
        return None
    gain = safe_float(forecast.get("marginal_gain_per_tflop"))
    sustained = safe_float(forecast.get("estimated_sustained_total_tflops"))
    if sustained is None:
        sustained = safe_float(forecast.get("estimated_sustained_tflops"))
    if gain is None or sustained is None:
        return None
    return gain * sustained * 3600.0


def remaining_wallclock_sec(run: RunSnapshot) -> float | None:
    forecast = run.metadata.get("forecast") if isinstance(run.metadata, dict) else {}
    if not isinstance(forecast, dict):
        return None
    compute_axis = forecast.get("compute_axis")
    if not isinstance(compute_axis, dict):
        return None
    return safe_float(compute_axis.get("projected_remaining_wallclock_sec"))


def control_rank_score(run: RunSnapshot) -> float:
    metric_name = run_metric_name(run)
    predicted = forecast_metric(run)
    pessimistic = pessimistic_metric(run)
    confidence = None
    forecast = run.metadata.get("forecast") if isinstance(run.metadata, dict) else {}
    if isinstance(forecast, dict):
        confidence = str(forecast.get("forecast_confidence") or "").lower() or None
    confidence_penalty = {"high": 0.0, "medium": 0.005, "low": 0.02}.get(str(confidence), 0.01)
    artifact_penalty = 0.05 if run.artifact_viable is False else 0.0
    uncertainty_penalty = 0.0
    if predicted is not None and pessimistic is not None:
        if metric_is_lower_better(metric_name):
            uncertainty_penalty = max(float(pessimistic) - float(predicted), 0.0)
        else:
            uncertainty_penalty = max(float(predicted) - float(pessimistic), 0.0)
    base = float(predicted) if predicted is not None else float("inf")
    if not metric_is_lower_better(metric_name):
        base = -base if predicted is not None else float("inf")
    return base + artifact_penalty + uncertainty_penalty + confidence_penalty


def dominates(candidate: RunSnapshot, incumbent: RunSnapshot, *, margin: float) -> bool:
    metric_name = run_metric_name(candidate)
    if metric_name != run_metric_name(incumbent):
        return False
    candidate_target = forecast_metric(candidate)
    incumbent_worst = pessimistic_metric(incumbent)
    if candidate_target is None or incumbent_worst is None:
        return False
    if metric_is_lower_better(metric_name):
        return float(candidate_target) + margin < float(incumbent_worst)
    return float(candidate_target) - margin > float(incumbent_worst)
