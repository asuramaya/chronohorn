from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Any
from typing import Sequence

from chronohorn.engine.budgets import (
    CompetitionBudget,
    DEFAULT_GOLF_V1_BUDGET,
    resolve_competition_budget,
)
from chronohorn.engine.forecasting import build_result_forecast
from chronohorn.engine.results import load_result_json


DEFAULT_FORECAST_BUDGET_NAME = DEFAULT_GOLF_V1_BUDGET.name
DEFAULT_FORECAST_TRAIN_TFLOPS_BUDGET = DEFAULT_GOLF_V1_BUDGET.train_tflops_budget
DEFAULT_FORECAST_ARTIFACT_LIMIT_MB = DEFAULT_GOLF_V1_BUDGET.artifact_limit_mb
DEFAULT_FORECAST_PRIMARY_METRIC = DEFAULT_GOLF_V1_BUDGET.primary_metric_name
LOWER_IS_BETTER_METRICS = {"bpb", "bits_per_token", "eval_loss"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="chronohorn fleet forecast-results",
        description=(
            "Project Chronohorn result JSONs onto the active competition budget, "
            "using compute-axis forecasting with probe overhead and uncertainty."
        ),
    )
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        help="Result JSON path or directory to scan recursively (repeatable).",
    )
    parser.add_argument(
        "--glob",
        action="append",
        default=[],
        help="Additional glob to scan for result JSONs (repeatable).",
    )
    parser.add_argument(
        "--budget-name",
        default=DEFAULT_FORECAST_BUDGET_NAME,
        help="Forecast budget label to embed in the output.",
    )
    parser.add_argument(
        "--train-tflops-budget",
        type=float,
        default=DEFAULT_FORECAST_TRAIN_TFLOPS_BUDGET,
        help="Training budget in TFLOPs used for budget-limited projection.",
    )
    parser.add_argument(
        "--artifact-limit-mb",
        type=float,
        default=DEFAULT_FORECAST_ARTIFACT_LIMIT_MB,
        help="Artifact-size budget in MB used for feasibility checks.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Maximum number of ranked rows to print in text mode.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of a text table.",
    )
    return parser.parse_args(argv)


def collect_result_paths(raw_paths: list[str], raw_globs: list[str]) -> list[Path]:
    result_paths: set[Path] = set()
    for raw_path in raw_paths:
        path = Path(raw_path).expanduser()
        if path.is_file() and path.suffix == ".json":
            result_paths.add(path.resolve())
            continue
        if path.is_dir():
            for child in path.rglob("*.json"):
                result_paths.add(child.resolve())
    for raw_glob in raw_globs:
        for match in glob.glob(raw_glob, recursive=True):
            path = Path(match).expanduser()
            if path.is_file() and path.suffix == ".json":
                result_paths.add(path.resolve())
    return sorted(result_paths)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _metric_direction(metric_name: str | None) -> str:
    return "lower_is_better" if metric_name in LOWER_IS_BETTER_METRICS else "unknown"


def _improvement_from_current(metric_name: str | None, current: float | None, forecast: float | None) -> float | None:
    if current is None or forecast is None:
        return None
    if _metric_direction(metric_name) == "lower_is_better":
        return float(current) - float(forecast)
    return float(forecast) - float(current)


def _confidence_from_curve(curve: dict[str, Any]) -> str:
    point_count = int(curve.get("point_count") or 0)
    weighted_r2 = _safe_float(curve.get("weighted_r2"))
    extrapolation_capped = bool(curve.get("extrapolation_capped"))
    if point_count < 2:
        return "low"
    if weighted_r2 is None:
        return "low" if extrapolation_capped else "medium"
    if weighted_r2 >= 0.98 and not extrapolation_capped and point_count >= 4:
        return "high"
    if weighted_r2 >= 0.9 and point_count >= 3:
        return "medium"
    return "low"


def _decision_signal(
    *,
    metric_name: str | None,
    artifact_viable: bool,
    remaining_total_tflops_est: float | None,
    forecast_metric_at_budget: float | None,
    forecast_low_95: float | None,
    current_metric_value: float | None,
    dbpb_dtotal_tflop: float | None,
) -> tuple[str, str]:
    if not artifact_viable:
        return ("artifact_blocked", "no viable artifact path under the current limit")
    if remaining_total_tflops_est is not None and remaining_total_tflops_est <= 0.0:
        return ("budget_exhausted", "the runtime budget is already exhausted")
    expected_improvement = _improvement_from_current(metric_name, current_metric_value, forecast_metric_at_budget)
    optimistic_improvement = _improvement_from_current(metric_name, current_metric_value, forecast_low_95)
    marginal_gain_per_tflop = None if dbpb_dtotal_tflop is None else -float(dbpb_dtotal_tflop)
    if expected_improvement is None:
        return ("unknown", "insufficient forecast signal")
    if expected_improvement <= 0.0:
        return ("flatlined", "the fitted curve does not predict further improvement")
    if optimistic_improvement is not None and optimistic_improvement <= 0.002:
        return ("near_saturation", "even the optimistic bound leaves little budgeted improvement")
    if marginal_gain_per_tflop is not None and marginal_gain_per_tflop <= 1e-5:
        return ("low_return", "marginal improvement per total TFLOP is already very small")
    return ("continue", "projected budgeted improvement is still material")


def _rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    metric_name = row.get("metric_name") or ""
    forecast_value = row.get("forecast_metric_at_budget")
    current_value = row.get("current_metric_value")
    artifact_ok = row.get("artifact_viable")
    forecast_sort = (
        float(forecast_value)
        if isinstance(forecast_value, (int, float)) and math.isfinite(float(forecast_value))
        else float("inf")
    )
    current_sort = (
        float(current_value)
        if isinstance(current_value, (int, float)) and math.isfinite(float(current_value))
        else float("inf")
    )
    return (metric_name, not bool(artifact_ok), forecast_sort, current_sort, row.get("path", ""))


def build_forecast_row(path: Path, forecast: dict[str, Any]) -> dict[str, Any]:
    observed = forecast.get("observed", {})
    projection = forecast.get("projection", {})
    artifact = forecast.get("artifact", {})
    curve = projection.get("curve_model", {}) if isinstance(projection.get("curve_model"), dict) else {}
    metric_name = observed.get("metric_name")
    current_metric_value = _safe_float(observed.get("metric_value"))
    forecast_metric_at_budget = _safe_float(projection.get("forecast_metric_at_budget"))
    forecast_low_95 = _safe_float(curve.get("forecast_lower_95"))
    forecast_high_95 = _safe_float(curve.get("forecast_upper_95"))
    current_total_tflops = _safe_float(projection.get("current_total_tflops_consumed_est"))
    budget_total_tflops = _safe_float(projection.get("budget_total_tflops_est"))
    remaining_total_tflops = _safe_float(projection.get("remaining_total_tflops_est"))
    compute_utilization = None
    if current_total_tflops is not None and budget_total_tflops is not None and budget_total_tflops > 0.0:
        compute_utilization = current_total_tflops / budget_total_tflops
    probe_compute = _safe_float(observed.get("probe_eval_tflops_consumed_est"))
    probe_overhead_fraction_compute = None
    if current_total_tflops is not None and current_total_tflops > 0.0 and probe_compute is not None:
        probe_overhead_fraction_compute = probe_compute / current_total_tflops
    confidence = _confidence_from_curve(curve)
    signal, reason = _decision_signal(
        metric_name=metric_name,
        artifact_viable=bool(artifact.get("has_viable_artifact_path")),
        remaining_total_tflops_est=remaining_total_tflops,
        forecast_metric_at_budget=forecast_metric_at_budget,
        forecast_low_95=forecast_low_95,
        current_metric_value=current_metric_value,
        dbpb_dtotal_tflop=_safe_float(projection.get("dbpb_dtotal_tflop")),
    )
    return {
        "path": str(path),
        "metric_name": metric_name,
        "current_metric_value": current_metric_value,
        "forecast_metric_at_budget": forecast_metric_at_budget,
        "forecast_delta_from_current": _safe_float(projection.get("forecast_delta_from_current")),
        "steps_completed": observed.get("steps_completed"),
        "budget_step_limit_est": projection.get("budget_step_limit_est"),
        "budget_total_step_limit_est": projection.get("budget_total_step_limit_est"),
        "estimated_sustained_tflops": _safe_float(observed.get("estimated_sustained_tflops")),
        "estimated_sustained_total_tflops": _safe_float(observed.get("estimated_sustained_total_tflops")),
        "tokens_per_second": _safe_float(observed.get("tokens_per_second")),
        "payload_mb_est": _safe_float(artifact.get("payload_mb_est")),
        "artifact_viable": bool(artifact.get("has_viable_artifact_path")),
        "current_within_artifact_budget": bool(artifact.get("current_within_budget")),
        "best_quantized_candidate": artifact.get("best_quantized_candidate"),
        "compute_axis": {
            "axis": curve.get("axis"),
            "consumed_tflops_est": current_total_tflops,
            "budget_tflops": budget_total_tflops,
            "remaining_tflops_est": remaining_total_tflops,
            "step_limit_est": projection.get("budget_total_step_limit_est") or projection.get("budget_step_limit_est"),
            "utilization": compute_utilization,
            "future_probe_count_est": projection.get("future_probe_count_est"),
            "future_probe_eval_tflops_est": _safe_float(projection.get("future_probe_eval_tflops_est")),
            "projected_total_wallclock_sec": _safe_float(projection.get("projected_total_wallclock_sec")),
            "projected_remaining_wallclock_sec": _safe_float(projection.get("projected_remaining_wallclock_sec")),
        },
        "compute_utilization": compute_utilization,
        "probe_overhead": {
            "probe_eval_tflops_consumed_est": probe_compute,
            "probe_overhead_fraction_compute": probe_overhead_fraction_compute,
            "probe_point_count": observed.get("probe_point_count"),
            "probe_elapsed_sec": _safe_float(observed.get("probe_elapsed_sec")),
        },
        "uncertainty": {
            "method": curve.get("method"),
            "point_count": curve.get("point_count"),
            "weighted_r2": _safe_float(curve.get("weighted_r2")),
            "residual_sigma": _safe_float(curve.get("residual_sigma")),
            "forecast_value": forecast_metric_at_budget,
            "forecast_low_95": forecast_low_95,
            "forecast_high_95": forecast_high_95,
            "extrapolation_capped": bool(curve.get("extrapolation_capped")),
            "confidence": confidence,
        },
        "forecast_confidence": confidence,
        "decision_signal": signal,
        "decision": {
            "signal": signal,
            "reason": reason,
            "confidence": confidence,
        },
        "forecast": forecast,
    }


def _format_number(value: Any, *, digits: int = 4) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    numeric = float(value)
    if not math.isfinite(numeric):
        return "-"
    return f"{numeric:.{digits}f}"


def _print_text(rows: list[dict[str, Any]], *, top: int) -> None:
    print("rank metric forecast current totalTF budgetTF util TF/s totalTF/s decision artifact path")
    for idx, row in enumerate(rows[: max(top, 0)], start=1):
        axis = row.get("compute_axis", {})
        print(
            " ".join(
                [
                    str(idx),
                    str(row.get("metric_name") or "-"),
                    _format_number(row.get("forecast_metric_at_budget")),
                    _format_number(row.get("current_metric_value")),
                    _format_number(axis.get("consumed_tflops_est"), digits=1),
                    _format_number(axis.get("budget_tflops"), digits=1),
                    _format_number(axis.get("utilization"), digits=4),
                    _format_number(row.get("estimated_sustained_tflops"), digits=3),
                    _format_number(row.get("estimated_sustained_total_tflops"), digits=3),
                    str(row.get("decision_signal") or "-"),
                    "yes" if row.get("artifact_viable") else "no",
                    str(row.get("path")),
                ]
            )
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    base_budget = resolve_competition_budget(args.budget_name)
    budget = CompetitionBudget(
        name=args.budget_name,
        train_tflops_budget=args.train_tflops_budget,
        artifact_limit_mb=args.artifact_limit_mb,
        primary_metric_name=base_budget.primary_metric_name,
    )
    result_paths = collect_result_paths(args.path or [], args.glob or [])
    if not result_paths:
        raise SystemExit("forecast-results requires at least one --path or --glob that resolves to JSON results")

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for path in result_paths:
        try:
            result = load_result_json(path)
            if not isinstance(result, dict):
                raise ValueError("JSON payload is not an object")
            if not isinstance(result.get("training"), dict) or not isinstance(result.get("model"), dict):
                raise ValueError("JSON payload does not look like a Chronohorn result summary")
            forecast = build_result_forecast(result, budget=budget)
        except Exception as exc:  # noqa: BLE001
            skipped.append({"path": str(path), "reason": str(exc)})
            continue
        rows.append(build_forecast_row(path, forecast))

    rows.sort(key=_rank_key)
    payload = {
        "budget_name": args.budget_name,
        "train_tflops_budget": args.train_tflops_budget,
        "artifact_limit_mb": args.artifact_limit_mb,
        "result_count": len(rows),
        "skipped": skipped,
        "rows": rows,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_text(rows, top=args.top)
        if skipped:
            print(f"\nskipped={len(skipped)}")
            for row in skipped[:10]:
                print(f"- {row['path']}: {row['reason']}")
    return 0
