from __future__ import annotations

import math
from typing import Any

from .probes import project_future_probe_entries
from .budgets import (
    DEFAULT_GOLF_V1_BUDGET,
    CompetitionBudget,
)
from .results import (
    extract_result_metric,
    extract_result_performance,
)


FORECAST_VERSION = "chronohorn_forecast_v2"
MAX_LOG_EXTRAPOLATION_MULTIPLE = 8.0
MAX_OBSERVED_IMPROVEMENT_MULTIPLE = 2.0
DEFAULT_PREDICTION_Z_SCORE = 1.96


from chronohorn.engine.results import safe_float


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _training_section(result: dict[str, Any]) -> dict[str, Any] | None:
    training = result.get("training")
    return training if isinstance(training, dict) else None


def _performance_estimate(training: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(training, dict):
        return None
    performance_estimate = training.get("performance_estimate")
    return performance_estimate if isinstance(performance_estimate, dict) else None


def _probe_rows(training: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(training, dict):
        return []
    probes = training.get("probes")
    if not isinstance(probes, list):
        return []
    return [row for row in probes if isinstance(row, dict)]


def _probe_steps(training: dict[str, Any] | None) -> list[int]:
    if not isinstance(training, dict):
        return []
    scheduled = training.get("probe_steps")
    if not isinstance(scheduled, list):
        return []
    steps: list[int] = []
    for raw in scheduled:
        step = _safe_int(raw)
        if step is not None and step > 0:
            steps.append(step)
    return sorted(set(steps))


def _training_probe_eval_batches(training: dict[str, Any] | None) -> int | None:
    if not isinstance(training, dict):
        return None
    return _safe_int(training.get("probe_eval_batches"))


def _training_probe_plan(training: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(training, dict):
        return None
    probe_plan = training.get("probe_plan")
    return probe_plan if isinstance(probe_plan, dict) else None


def _training_final_eval_batches(training: dict[str, Any] | None) -> int | None:
    if not isinstance(training, dict):
        return None
    return _safe_int(training.get("final_eval_batches"))


def _train_step_flops_per_step_est(performance_estimate: dict[str, Any] | None) -> float | None:
    if not isinstance(performance_estimate, dict):
        return None
    return safe_float(performance_estimate.get("train_step_flops_per_step_est"))


def _forward_total_flops_per_step_est(performance_estimate: dict[str, Any] | None) -> float | None:
    if not isinstance(performance_estimate, dict):
        return None
    return safe_float(performance_estimate.get("forward_total_flops_per_step"))


def _result_steps_completed(result: dict[str, Any]) -> int | None:
    performance = extract_result_performance(result)
    if performance.steps_completed is not None and performance.steps_completed > 0:
        return performance.steps_completed
    config = result.get("config")
    if isinstance(config, dict):
        train_cfg = config.get("train")
        if isinstance(train_cfg, dict):
            completed = _safe_int(train_cfg.get("steps"))
            if completed is not None and completed > 0:
                return completed
    return None


def _quantization_metric_key(metric_name: str) -> str:
    if metric_name in {"bpb", "test_bpb"}:
        return "test_bpb"
    if metric_name in {"bits_per_token", "test_bits_per_token"}:
        return "test_bits_per_token"
    return "test_eval_loss"


def _collect_metric_points(result: dict[str, Any], metric_name: str) -> list[tuple[int, float]]:
    points: dict[int, float] = {}
    training = _training_section(result)
    if training is not None:
        probes = training.get("probes")
        if isinstance(probes, list):
            for row in probes:
                if not isinstance(row, dict):
                    continue
                step = _safe_int(row.get("step"))
                value = safe_float(row.get(metric_name))
                if step is None or step <= 0 or value is None:
                    continue
                points[step] = value
    final_step = _result_steps_completed(result)
    metric = extract_result_metric(result, preferred_name=metric_name)
    if final_step is not None and final_step > 0 and metric.value is not None:
        points[final_step] = metric.value
    return sorted(points.items(), key=lambda item: item[0])


def _weighted_linear_fit(
    xs: list[float],
    ys: list[float],
    weights: list[float],
) -> dict[str, Any] | None:
    if not xs or len(xs) != len(ys) or len(xs) != len(weights):
        return None
    weight_sum = sum(weights)
    if weight_sum <= 0.0:
        return None
    mean_x = sum(weight * x for weight, x in zip(weights, xs)) / weight_sum
    mean_y = sum(weight * y for weight, y in zip(weights, ys)) / weight_sum
    denom = sum(weight * (x - mean_x) ** 2 for weight, x in zip(weights, xs))
    if denom <= 1e-12:
        return None
    slope = (
        sum(weight * (x - mean_x) * (y - mean_y) for weight, x, y in zip(weights, xs, ys))
        / denom
    )
    intercept = mean_y - slope * mean_x
    fitted = [intercept + slope * x for x in xs]
    ss_res = sum(weight * (y - fit) ** 2 for weight, y, fit in zip(weights, ys, fitted))
    ss_tot = sum(weight * (y - mean_y) ** 2 for weight, y in zip(weights, ys))
    weighted_r2 = None if ss_tot <= 1e-12 else max(min(1.0 - ss_res / ss_tot, 1.0), -1.0)
    residual_sigma = math.sqrt(max(ss_res / max(len(xs) - 2, 1), 0.0))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "weighted_r2": weighted_r2,
        "weighted_sse": float(ss_res),
        "residual_sigma": float(residual_sigma),
        "mean_x": float(mean_x),
        "mean_y": float(mean_y),
        "denom": float(denom),
        "weight_sum": float(weight_sum),
    }


def _log_linear_curve_fit(
    points: list[tuple[float, float]],
    *,
    requested_target_x: float | None,
    effective_target_x: float,
    current_x: float,
) -> dict[str, Any] | None:
    if not points:
        return None
    xs = [float(step) for step, _ in points]
    ys = [float(value) for _, value in points]
    if any(x <= 0.0 or not math.isfinite(x) for x in xs):
        return None
    weights = [max(x / max(xs), 1e-6) for x in xs]
    fit = _weighted_linear_fit([math.log(x) for x in xs], ys, weights)
    if fit is None:
        return None

    last_x = xs[-1]
    forecast_value = fit["intercept"] + fit["slope"] * math.log(max(effective_target_x, 1e-12))
    current_marginal = fit["slope"] / max(current_x, 1e-12)
    return {
        "method": "weighted_log_linear",
        "axis": None,
        "point_count": len(points),
        "requested_target_x": requested_target_x,
        "effective_target_x": effective_target_x,
        "extrapolation_capped": False,
        "forecast_value": float(forecast_value),
        "current_x": float(current_x),
        "last_observed_x": float(last_x),
        "last_observed_value": float(ys[-1]),
        "observed_best_value": float(min(ys)),
        "observed_improvement": float(max(ys[0] - min(ys), 0.0)),
        "intercept": float(fit["intercept"]),
        "slope_per_log_x": float(fit["slope"]),
        "marginal_per_tflop_at_current": float(current_marginal),
        "weighted_r2": fit["weighted_r2"],
        "weighted_sse": fit["weighted_sse"],
        "residual_sigma": fit["residual_sigma"],
    }


def _power_law_asymptotic_curve_fit(
    points: list[tuple[float, float]],
    *,
    requested_target_x: float | None,
    effective_target_x: float,
    current_x: float,
) -> dict[str, Any] | None:
    if len(points) < 4:
        return None
    xs = [float(step) for step, _ in points]
    ys = [float(value) for _, value in points]
    if any(x <= 0.0 or not math.isfinite(x) for x in xs):
        return None
    weights = [max(x / max(xs), 1e-6) for x in xs]
    min_y = min(ys)
    max_y = max(ys)
    observed_span = max(max_y - min_y, abs(ys[0] - ys[-1]), 1e-6)
    asymptote_low = min_y - max(observed_span * 4.0, 1e-3)
    asymptote_high = min_y - max(observed_span * 0.02, 1e-6)
    if asymptote_high <= asymptote_low:
        return None

    best: dict[str, Any] | None = None
    candidate_count = 32
    log_xs = [math.log(x) for x in xs]
    for index in range(candidate_count):
        t = index / max(candidate_count - 1, 1)
        asymptote = asymptote_low + (asymptote_high - asymptote_low) * (t**2)
        residuals = [y - asymptote for y in ys]
        if any(residual <= 1e-12 or not math.isfinite(residual) for residual in residuals):
            continue
        fit = _weighted_linear_fit(log_xs, [math.log(residual) for residual in residuals], weights)
        if fit is None:
            continue
        alpha = -fit["slope"]
        amplitude = math.exp(fit["intercept"])
        if not math.isfinite(alpha) or not math.isfinite(amplitude) or alpha <= 0.0 or amplitude <= 0.0:
            continue
        fitted = [asymptote + amplitude * (x ** (-alpha)) for x in xs]
        if any(not math.isfinite(value) for value in fitted):
            continue
        ss_res = sum(weight * (y - fit_y) ** 2 for weight, y, fit_y in zip(weights, ys, fitted))
        weighted_mean_y = sum(weight * y for weight, y in zip(weights, ys)) / sum(weights)
        ss_tot = sum(weight * (y - weighted_mean_y) ** 2 for weight, y in zip(weights, ys))
        weighted_r2 = None if ss_tot <= 1e-12 else max(min(1.0 - ss_res / ss_tot, 1.0), -1.0)
        residual_sigma = math.sqrt(max(ss_res / max(len(xs) - 3, 1), 0.0))
        candidate = {
            "method": "power_law_asymptotic",
            "axis": None,
            "point_count": len(points),
            "requested_target_x": requested_target_x,
            "effective_target_x": effective_target_x,
            "extrapolation_capped": False,
            "forecast_value": float(asymptote + amplitude * (max(effective_target_x, 1e-12) ** (-alpha))),
            "current_x": float(current_x),
            "last_observed_x": float(xs[-1]),
            "last_observed_value": float(ys[-1]),
            "observed_best_value": float(min_y),
            "observed_improvement": float(max(ys[0] - min_y, 0.0)),
            "asymptote": float(asymptote),
            "amplitude": float(amplitude),
            "alpha": float(alpha),
            "marginal_per_tflop_at_current": float(-alpha * amplitude * (max(current_x, 1e-12) ** (-alpha - 1.0))),
            "weighted_r2": weighted_r2,
            "weighted_sse": float(ss_res),
            "residual_sigma": float(residual_sigma),
        }
        if best is None:
            best = candidate
            continue
        best_dof = max(best["point_count"] - 3, 1)
        candidate_dof = max(candidate["point_count"] - 3, 1)
        best_score = float(best["weighted_sse"]) / float(best_dof)
        candidate_score = float(candidate["weighted_sse"]) / float(candidate_dof)
        if candidate_score < best_score:
            best = candidate

    return best


def _cap_observed_improvement(forecast_value: float, *, observed_best: float, observed_improvement: float) -> float:
    if observed_improvement <= 0.0:
        return max(forecast_value, observed_best)
    floor_value = observed_best - (observed_improvement * MAX_OBSERVED_IMPROVEMENT_MULTIPLE)
    return max(forecast_value, floor_value)


def _prediction_scale(target_x: float, last_x: float) -> float:
    if target_x <= last_x or last_x <= 0.0:
        return 1.0
    return 1.0 + max(math.log(target_x / last_x), 0.0) * 0.5


def _finalize_curve_model(
    curve: dict[str, Any],
    *,
    axis_name: str,
    requested_target_x: float | None,
    current_x: float,
) -> dict[str, Any]:
    result = dict(curve)
    result["axis"] = axis_name
    result["requested_target_x"] = requested_target_x
    result["current_x"] = float(current_x)
    target_x = result.get("effective_target_x")
    last_x = float(result.get("last_observed_x") or current_x or 1.0)
    if target_x is None:
        target_x = last_x
    target_x = float(target_x)
    result["effective_target_x"] = target_x
    forecast_value = float(result.get("forecast_value") or 0.0)
    forecast_value = _cap_observed_improvement(
        forecast_value,
        observed_best=float(result.get("observed_best_value") or forecast_value),
        observed_improvement=float(result.get("observed_improvement") or 0.0),
    )
    sigma = float(result.get("residual_sigma") or 0.0) * _prediction_scale(float(result["effective_target_x"]), last_x)
    one_sigma = sigma * 1.0
    two_sigma = sigma * DEFAULT_PREDICTION_Z_SCORE
    lower_1 = max(forecast_value - one_sigma, 0.0)
    upper_1 = max(forecast_value + one_sigma, 0.0)
    lower_95 = max(forecast_value - two_sigma, 0.0)
    upper_95 = max(forecast_value + two_sigma, 0.0)
    floor_value = float(result.get("observed_best_value") or forecast_value) - float(result.get("observed_improvement") or 0.0) * MAX_OBSERVED_IMPROVEMENT_MULTIPLE
    floor_value = max(floor_value, 0.0)
    lower_1 = max(lower_1, floor_value)
    lower_95 = max(lower_95, floor_value)
    result["forecast_value"] = float(forecast_value)
    result["forecast_lower_1sigma"] = float(lower_1)
    result["forecast_upper_1sigma"] = float(upper_1)
    result["forecast_lower_95"] = float(lower_95)
    result["forecast_upper_95"] = float(upper_95)
    result["prediction_sigma"] = float(sigma)
    result["prediction_scale"] = float(_prediction_scale(float(result["effective_target_x"]), last_x))
    result["uncertainty"] = {
        "sigma": float(sigma),
        "prediction_scale": float(_prediction_scale(float(result["effective_target_x"]), last_x)),
        "one_sigma": {
            "lower": float(lower_1),
            "upper": float(upper_1),
        },
        "two_sigma": {
            "lower": float(lower_95),
            "upper": float(upper_95),
        },
        "z_score": DEFAULT_PREDICTION_Z_SCORE,
    }
    return result


def _curve_model_for_points(
    points: list[tuple[float, float]],
    *,
    axis_name: str,
    target_x: float | None,
    current_x: float | None,
) -> dict[str, Any]:
    if not points:
        return {
            "method": "unavailable",
            "axis": axis_name,
            "point_count": 0,
            "requested_target_x": target_x,
            "effective_target_x": target_x,
            "extrapolation_capped": False,
            "forecast_value": None,
            "forecast_lower_1sigma": None,
            "forecast_upper_1sigma": None,
            "forecast_lower_95": None,
            "forecast_upper_95": None,
            "last_observed_x": None,
            "last_observed_value": None,
            "observed_best_value": None,
            "observed_improvement": None,
            "marginal_per_tflop_at_current": None,
            "weighted_r2": None,
            "weighted_sse": None,
            "residual_sigma": None,
            "prediction_sigma": None,
            "prediction_scale": None,
            "uncertainty": None,
        }

    ordered_points = sorted(points, key=lambda item: item[0])
    last_x = ordered_points[-1][0]
    current_x = float(current_x if current_x is not None else last_x)
    requested_target_x = target_x
    if target_x is None:
        target_x = last_x
    effective_target_x = last_x
    extrapolation_capped = False
    if target_x > last_x:
        effective_target_x = min(target_x, max(last_x * MAX_LOG_EXTRAPOLATION_MULTIPLE, last_x + 1e-12))
        extrapolation_capped = bool(effective_target_x < target_x)

    if len(ordered_points) == 1:
        last_value = ordered_points[-1][1]
        model = {
            "method": "hold_last",
            "axis": axis_name,
            "point_count": 1,
            "requested_target_x": requested_target_x,
            "effective_target_x": last_x,
            "extrapolation_capped": False,
            "forecast_value": float(last_value),
            "forecast_lower_1sigma": float(last_value),
            "forecast_upper_1sigma": float(last_value),
            "forecast_lower_95": float(last_value),
            "forecast_upper_95": float(last_value),
            "current_x": current_x,
            "last_observed_x": float(last_x),
            "last_observed_value": float(last_value),
            "observed_best_value": float(last_value),
            "observed_improvement": 0.0,
            "marginal_per_tflop_at_current": 0.0,
            "weighted_r2": None,
            "weighted_sse": 0.0,
            "residual_sigma": 0.0,
            "prediction_sigma": 0.0,
            "prediction_scale": 1.0,
            "uncertainty": {
                "sigma": 0.0,
                "prediction_scale": 1.0,
                "one_sigma": {"lower": float(last_value), "upper": float(last_value)},
                "two_sigma": {"lower": float(last_value), "upper": float(last_value)},
                "z_score": DEFAULT_PREDICTION_Z_SCORE,
            },
        }
        return model

    log_linear = _log_linear_curve_fit(
        ordered_points,
        requested_target_x=requested_target_x,
        effective_target_x=float(effective_target_x),
        current_x=current_x,
    )
    power_law = _power_law_asymptotic_curve_fit(
        ordered_points,
        requested_target_x=requested_target_x,
        effective_target_x=float(effective_target_x),
        current_x=current_x,
    )
    selected = log_linear
    if power_law is not None:
        if selected is None:
            selected = power_law
        else:
            selected_dof = max(int(selected["point_count"]) - (3 if selected["method"] == "power_law_asymptotic" else 2), 1)
            selected_score = float(selected["weighted_sse"]) / float(selected_dof)
            power_dof = max(int(power_law["point_count"]) - 3, 1)
            power_score = float(power_law["weighted_sse"]) / float(power_dof)
            selected_r2 = selected.get("weighted_r2")
            power_r2 = power_law.get("weighted_r2")
            if (
                power_score <= selected_score * 1.02
                or (power_r2 is not None and selected_r2 is not None and power_r2 >= selected_r2 - 0.02)
            ):
                selected = power_law
    if selected is None:
        selected = {
            "method": "unavailable",
            "axis": axis_name,
            "point_count": len(ordered_points),
            "requested_target_x": requested_target_x,
            "effective_target_x": effective_target_x,
            "extrapolation_capped": extrapolation_capped,
            "forecast_value": ordered_points[-1][1],
            "last_observed_x": last_x,
            "last_observed_value": ordered_points[-1][1],
            "observed_best_value": min(value for _, value in ordered_points),
            "observed_improvement": max(ordered_points[0][1] - min(value for _, value in ordered_points), 0.0),
            "marginal_per_tflop_at_current": None,
            "weighted_r2": None,
            "weighted_sse": None,
            "residual_sigma": None,
        }
        return _finalize_curve_model(selected, axis_name=axis_name, requested_target_x=requested_target_x, current_x=current_x)

    selected["extrapolation_capped"] = extrapolation_capped
    return _finalize_curve_model(selected, axis_name=axis_name, requested_target_x=requested_target_x, current_x=current_x)


def _build_compute_trajectory(
    result: dict[str, Any],
    *,
    metric_name: str | None,
    performance: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    training = _training_section(result)
    performance_estimate = _performance_estimate(training)
    train_step_flops_per_step_est = _train_step_flops_per_step_est(performance_estimate)
    forward_total_flops_per_step_est = _forward_total_flops_per_step_est(performance_estimate)
    if train_step_flops_per_step_est is None or forward_total_flops_per_step_est is None:
        return [], {
            "train_step_flops_per_step_est": train_step_flops_per_step_est,
            "forward_total_flops_per_step_est": forward_total_flops_per_step_est,
        }

    steps_completed = _result_steps_completed(result)
    probe_eval_batches_default = _training_probe_eval_batches(training)
    final_eval_batches = _training_final_eval_batches(training)
    probes = _probe_rows(training)
    points_by_step: dict[int, dict[str, Any]] = {}
    cumulative_probe_eval_tflops = 0.0
    observed_probe_steps: list[int] = []

    for row in sorted(probes, key=lambda item: _safe_int(item.get("step")) or 0):
        step = _safe_int(row.get("step"))
        if step is None or step <= 0:
            continue
        if steps_completed is not None and step > steps_completed:
            continue
        metric_value = safe_float(row.get(metric_name)) if metric_name is not None else None
        if metric_value is None:
            continue
        train_tflops = float(step) * float(train_step_flops_per_step_est) / 1e12
        eval_batches = _safe_int(row.get("eval_batches"))
        if eval_batches is None:
            eval_batches = probe_eval_batches_default
        probe_eval_tflops = 0.0
        compute = row.get("compute")
        if isinstance(compute, dict):
            probe_eval_tflops = float(safe_float(compute.get("eval_tflops_est")) or 0.0)
        elif eval_batches is not None and eval_batches > 0:
            probe_eval_tflops = float(eval_batches) * float(forward_total_flops_per_step_est) / 1e12
        cumulative_probe_eval_tflops += probe_eval_tflops
        points_by_step[step] = {
            "step": step,
            "metric_value": metric_value,
            "train_tflops": train_tflops,
            "probe_eval_tflops": cumulative_probe_eval_tflops,
            "final_eval_tflops": 0.0,
            "total_tflops": train_tflops + cumulative_probe_eval_tflops,
            "source": "probe",
        }
        observed_probe_steps.append(step)

    final_metric = extract_result_metric(result, preferred_name=metric_name) if metric_name is not None else None
    if steps_completed is not None and final_metric is not None and final_metric.value is not None:
        train_tflops = float(steps_completed) * float(train_step_flops_per_step_est) / 1e12
        final_eval_tflops = 0.0
        if final_eval_batches is not None and final_eval_batches > 0:
            final_eval_tflops = 2.0 * float(final_eval_batches) * float(forward_total_flops_per_step_est) / 1e12
        points_by_step[steps_completed] = {
            "step": steps_completed,
            "metric_value": float(final_metric.value),
            "train_tflops": train_tflops,
            "probe_eval_tflops": cumulative_probe_eval_tflops,
            "final_eval_tflops": final_eval_tflops,
            "total_tflops": train_tflops + cumulative_probe_eval_tflops + final_eval_tflops,
            "source": "final",
        }

    points = sorted(points_by_step.values(), key=lambda item: (item["total_tflops"], item["train_tflops"], item["step"]))
    info = {
        "train_step_flops_per_step_est": train_step_flops_per_step_est,
        "forward_total_flops_per_step_est": forward_total_flops_per_step_est,
        "steps_completed": steps_completed,
        "probe_eval_batches_default": probe_eval_batches_default,
        "final_eval_batches": final_eval_batches,
        "probe_steps": observed_probe_steps,
        "scheduled_probe_steps": _probe_steps(training),
    }
    return points, info


def _estimate_future_probe_count(
    *,
    steps_completed: int | None,
    budget_step_limit_est: int | None,
    scheduled_probe_steps: list[int],
    observed_probe_steps: list[int],
) -> int:
    if budget_step_limit_est is None:
        return 0
    if scheduled_probe_steps:
        count = 0
        for step in scheduled_probe_steps:
            if steps_completed is not None and step <= steps_completed:
                continue
            if step > budget_step_limit_est:
                continue
            count += 1
        return count
    if len(observed_probe_steps) >= 2:
        diffs = [later - earlier for earlier, later in zip(observed_probe_steps[:-1], observed_probe_steps[1:]) if later > earlier]
        if diffs:
            average_spacing = sum(diffs) / len(diffs)
            last_step = observed_probe_steps[-1]
            if last_step >= budget_step_limit_est:
                return 0
            return max(int(math.ceil((budget_step_limit_est - last_step) / average_spacing)), 0)
    if len(observed_probe_steps) == 1:
        last_step = observed_probe_steps[-1]
        if last_step < budget_step_limit_est:
            return 1
    return 0


def _estimate_future_probe_entries(
    *,
    training: dict[str, Any] | None,
    steps_completed: int | None,
    budget_step_limit_est: int | None,
) -> list[dict[str, Any]]:
    if budget_step_limit_est is None:
        return []
    probe_plan = _training_probe_plan(training)
    if probe_plan is None:
        return []
    return project_future_probe_entries(
        probe_plan,
        after_step=int(steps_completed or 0),
        max_step=int(budget_step_limit_est),
    )


def _build_compute_context(
    result: dict[str, Any],
    *,
    budget: CompetitionBudget,
    performance: Any,
) -> dict[str, Any]:
    training = _training_section(result)
    performance_estimate = _performance_estimate(training)
    train_step_flops_per_step_est = _train_step_flops_per_step_est(performance_estimate)
    forward_total_flops_per_step_est = _forward_total_flops_per_step_est(performance_estimate)
    steps_completed = _result_steps_completed(result)

    current_train_tflops_consumed_est = getattr(performance, "train_tflops_consumed_est", None)
    if current_train_tflops_consumed_est is None and steps_completed is not None and train_step_flops_per_step_est is not None:
        current_train_tflops_consumed_est = float(train_step_flops_per_step_est) * float(steps_completed) / 1e12

    current_probe_eval_tflops_consumed_est = getattr(performance, "probe_eval_tflops_consumed_est", None)
    current_final_eval_tflops_consumed_est = getattr(performance, "final_eval_tflops_consumed_est", None)
    current_total_tflops_consumed_est = getattr(performance, "total_tflops_consumed_est", None)
    if current_probe_eval_tflops_consumed_est is None or current_final_eval_tflops_consumed_est is None or current_total_tflops_consumed_est is None:
        current_compute_points, compute_info = _build_compute_trajectory(
            result,
            metric_name=None,
            performance=performance,
        )
        if current_probe_eval_tflops_consumed_est is None:
            current_probe_eval_tflops_consumed_est = current_compute_points[-1]["probe_eval_tflops"] if current_compute_points else None
        if current_final_eval_tflops_consumed_est is None:
            current_final_eval_tflops_consumed_est = current_compute_points[-1]["final_eval_tflops"] if current_compute_points else None
        if current_total_tflops_consumed_est is None and current_train_tflops_consumed_est is not None:
            current_total_tflops_consumed_est = float(current_train_tflops_consumed_est)
            if current_probe_eval_tflops_consumed_est is not None:
                current_total_tflops_consumed_est += float(current_probe_eval_tflops_consumed_est)
            if current_final_eval_tflops_consumed_est is not None:
                current_total_tflops_consumed_est += float(current_final_eval_tflops_consumed_est)
    else:
        compute_info = {}

    if current_total_tflops_consumed_est is None and current_train_tflops_consumed_est is not None:
        current_total_tflops_consumed_est = float(current_train_tflops_consumed_est)

    budget_step_limit_est = None
    if train_step_flops_per_step_est is not None and budget.train_tflops_budget > 0.0:
        budget_step_limit_est = max(
            int(math.floor(float(budget.train_tflops_budget) * 1e12 / float(train_step_flops_per_step_est))),
            1,
        )

    remaining_train_tflops_est = None
    if current_train_tflops_consumed_est is not None and budget.train_tflops_budget > 0.0:
        remaining_train_tflops_est = max(float(budget.train_tflops_budget) - float(current_train_tflops_consumed_est), 0.0)

    probes = _probe_rows(training)
    probe_eval_batches_default = _training_probe_eval_batches(training)
    scheduled_probe_steps = compute_info.get("scheduled_probe_steps") or _probe_steps(training)
    observed_probe_steps = compute_info.get("probe_steps") or []
    future_probe_entries = _estimate_future_probe_entries(
        training=training,
        steps_completed=steps_completed,
        budget_step_limit_est=budget_step_limit_est,
    )
    future_probe_count = _estimate_future_probe_count(
        steps_completed=steps_completed,
        budget_step_limit_est=budget_step_limit_est,
        scheduled_probe_steps=scheduled_probe_steps,
        observed_probe_steps=observed_probe_steps,
    )
    if future_probe_entries:
        future_probe_count = len(future_probe_entries)

    current_probe_eval_compute_known = current_probe_eval_tflops_consumed_est is not None
    if not current_probe_eval_compute_known and forward_total_flops_per_step_est is not None:
        current_probe_eval_tflops_consumed_est = 0.0
        for row in probes:
            step = _safe_int(row.get("step"))
            if step is None:
                continue
            if steps_completed is not None and step > steps_completed:
                continue
            eval_batches = _safe_int(row.get("eval_batches"))
            if eval_batches is None:
                eval_batches = probe_eval_batches_default
            if eval_batches is None or eval_batches <= 0:
                continue
            current_probe_eval_tflops_consumed_est += float(eval_batches) * float(forward_total_flops_per_step_est) / 1e12

    if current_final_eval_tflops_consumed_est is None and forward_total_flops_per_step_est is not None:
        final_eval_batches = _training_final_eval_batches(training)
        if final_eval_batches is not None and final_eval_batches > 0:
            current_final_eval_tflops_consumed_est = float(final_eval_batches) * float(forward_total_flops_per_step_est) / 1e12
        elif final_eval_batches == 0:
            current_final_eval_tflops_consumed_est = 0.0

    if current_total_tflops_consumed_est is None and current_train_tflops_consumed_est is not None:
        current_total_tflops_consumed_est = float(current_train_tflops_consumed_est)
        if current_probe_eval_tflops_consumed_est is not None:
            current_total_tflops_consumed_est += float(current_probe_eval_tflops_consumed_est)
        if current_final_eval_tflops_consumed_est is not None:
            current_total_tflops_consumed_est += float(current_final_eval_tflops_consumed_est)

    future_probe_eval_tflops_est = None
    if forward_total_flops_per_step_est is not None and future_probe_entries:
        future_probe_eval_tflops_est = (
            sum(
                float(_safe_int(entry.get("eval_batches")) or 0) * float(forward_total_flops_per_step_est)
                for entry in future_probe_entries
                if isinstance(entry, dict)
            )
            / 1e12
        )
    elif forward_total_flops_per_step_est is not None and probe_eval_batches_default is not None:
        future_probe_eval_tflops_est = float(future_probe_count) * float(probe_eval_batches_default) * float(forward_total_flops_per_step_est) / 1e12
    elif future_probe_count == 0:
        future_probe_eval_tflops_est = 0.0

    final_eval_tflops_budget_est = current_final_eval_tflops_consumed_est
    if final_eval_tflops_budget_est is None and forward_total_flops_per_step_est is not None:
        final_eval_batches = _training_final_eval_batches(training)
        if final_eval_batches is not None and final_eval_batches > 0:
            final_eval_tflops_budget_est = float(final_eval_batches) * float(forward_total_flops_per_step_est) / 1e12

    full_run_probe_eval_tflops_est = current_probe_eval_tflops_consumed_est
    if full_run_probe_eval_tflops_est is not None and future_probe_eval_tflops_est is not None:
        full_run_probe_eval_tflops_est = float(full_run_probe_eval_tflops_est) + float(future_probe_eval_tflops_est)
    elif future_probe_eval_tflops_est is not None:
        full_run_probe_eval_tflops_est = float(future_probe_eval_tflops_est)

    budget_total_tflops_est = None
    if budget.train_tflops_budget > 0.0:
        budget_total_tflops_est = float(budget.train_tflops_budget)
        if full_run_probe_eval_tflops_est is not None:
            budget_total_tflops_est += float(full_run_probe_eval_tflops_est)
        if final_eval_tflops_budget_est is not None:
            budget_total_tflops_est += float(final_eval_tflops_budget_est)

    current_total_tflops_per_train_step_est = None
    if (
        current_total_tflops_consumed_est is not None
        and steps_completed is not None
        and steps_completed > 0
    ):
        current_total_tflops_per_train_step_est = float(current_total_tflops_consumed_est) / float(steps_completed)

    remaining_total_tflops_est = None
    if current_total_tflops_consumed_est is not None and budget_total_tflops_est is not None:
        remaining_total_tflops_est = max(
            float(budget_total_tflops_est) - float(current_total_tflops_consumed_est),
            0.0,
        )

    budget_total_step_limit_est = None
    remaining_total_step_est = None
    if (
        current_total_tflops_per_train_step_est is not None
        and current_total_tflops_per_train_step_est > 0.0
        and remaining_total_tflops_est is not None
        and steps_completed is not None
    ):
        remaining_total_step_est = max(
            int(math.floor(float(remaining_total_tflops_est) / float(current_total_tflops_per_train_step_est))),
            0,
        )
        budget_total_step_limit_est = int(steps_completed + remaining_total_step_est)

    projected_total_wallclock_sec = None
    projected_remaining_wallclock_sec = None
    sustained_total_tflops = getattr(performance, "estimated_sustained_total_tflops", None)
    if sustained_total_tflops is None or sustained_total_tflops <= 0.0:
        sustained_total_tflops = getattr(performance, "estimated_sustained_tflops", None)
    if sustained_total_tflops is not None and sustained_total_tflops > 0.0:
        if budget_total_tflops_est is not None:
            projected_total_wallclock_sec = float(budget_total_tflops_est) / float(sustained_total_tflops)
        if remaining_total_tflops_est is not None:
            projected_remaining_wallclock_sec = float(remaining_total_tflops_est) / float(sustained_total_tflops)

    return {
        "train_step_flops_per_step_est": train_step_flops_per_step_est,
        "forward_total_flops_per_step_est": forward_total_flops_per_step_est,
        "current_train_tflops_consumed_est": current_train_tflops_consumed_est,
        "current_probe_eval_tflops_consumed_est": current_probe_eval_tflops_consumed_est,
        "current_final_eval_tflops_consumed_est": current_final_eval_tflops_consumed_est,
        "current_total_tflops_consumed_est": current_total_tflops_consumed_est,
        "remaining_train_tflops_est": remaining_train_tflops_est,
        "budget_step_limit_est": budget_step_limit_est,
        "future_probe_count_est": future_probe_count,
        "future_probe_entries_est": future_probe_entries,
        "future_probe_eval_tflops_est": future_probe_eval_tflops_est,
        "full_run_probe_eval_tflops_est": full_run_probe_eval_tflops_est,
        "final_eval_tflops_budget_est": final_eval_tflops_budget_est,
        "budget_total_tflops_est": budget_total_tflops_est,
        "current_total_tflops_per_train_step_est": current_total_tflops_per_train_step_est,
        "remaining_total_tflops_est": remaining_total_tflops_est,
        "remaining_total_step_est": remaining_total_step_est,
        "budget_total_step_limit_est": budget_total_step_limit_est,
        "projected_total_wallclock_sec": projected_total_wallclock_sec,
        "projected_remaining_wallclock_sec": projected_remaining_wallclock_sec,
    }


def _forecast_delta_from_current(metric_value: float | None, forecast_value: float | None) -> float | None:
    if metric_value is None or forecast_value is None:
        return None
    return float(forecast_value) - float(metric_value)


def _build_budget_curve_models(
    result: dict[str, Any],
    *,
    metric_name: str | None,
    context: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    performance = extract_result_performance(result)
    compute_points, _ = _build_compute_trajectory(result, metric_name=metric_name, performance=performance)
    if compute_points:
        train_points = [(point["train_tflops"], point["metric_value"]) for point in compute_points]
        total_points = [(point["total_tflops"], point["metric_value"]) for point in compute_points]
        current_train_x = context.get("current_train_tflops_consumed_est")
        current_total_x = context.get("current_total_tflops_consumed_est")
        train_curve = _curve_model_for_points(
            train_points,
            axis_name="train_tflops",
            target_x=float(context["budget_train_tflops_est"]) if context.get("budget_train_tflops_est") is not None else None,
            current_x=current_train_x,
        )
        if context.get("budget_total_tflops_est") is not None:
            total_curve = _curve_model_for_points(
                total_points,
                axis_name="total_tflops",
                target_x=context.get("budget_total_tflops_est"),
                current_x=current_total_x,
            )
        else:
            total_curve = train_curve
        return train_curve, total_curve, compute_points

    step_points = _collect_metric_points(result, metric_name) if metric_name is not None else []
    if not step_points:
        return (
            _curve_model_for_points([], axis_name="train_tflops", target_x=None, current_x=None),
            _curve_model_for_points([], axis_name="total_tflops", target_x=None, current_x=None),
            [],
        )
    step_series = [(float(step), float(value)) for step, value in step_points]
    step_series_points = [{"step": step, "metric_value": value} for step, value in step_points]
    current_step = _result_steps_completed(result)
    step_curve = _curve_model_for_points(
        step_series,
        axis_name="steps",
        target_x=float(context.get("budget_step_limit_est")) if context.get("budget_step_limit_est") is not None else None,
        current_x=float(current_step) if current_step is not None else None,
    )
    return step_curve, step_curve, step_series_points


def _artifact_projection(
    result: dict[str, Any],
    *,
    metric_name: str | None,
    artifact_limit_mb: float,
) -> dict[str, Any]:
    model = result.get("model")
    payload_mb_est = None
    if isinstance(model, dict):
        payload_mb_est = safe_float(model.get("payload_mb_est"))
    current_within_budget = (
        payload_mb_est is not None and payload_mb_est <= float(artifact_limit_mb)
    )
    budget_ratio = None
    if payload_mb_est is not None and artifact_limit_mb > 0.0:
        budget_ratio = payload_mb_est / float(artifact_limit_mb)

    best_viable = None
    if metric_name is not None:
        quant_key = _quantization_metric_key(metric_name)
        quant_rows = result.get("quantization")
        if isinstance(quant_rows, list):
            viable_rows = []
            for row in quant_rows:
                if not isinstance(row, dict):
                    continue
                row_payload = safe_float(row.get("payload_mb_est"))
                row_metric = safe_float(row.get(quant_key))
                if row_payload is None or row_metric is None:
                    continue
                if row_payload <= float(artifact_limit_mb):
                    viable_rows.append((row_metric, row_payload, row))
            if viable_rows:
                viable_rows.sort(key=lambda item: item[0])
                _, row_payload, row = viable_rows[0]
                best_viable = {
                    "scheme": row.get("scheme"),
                    "metric_name": metric_name,
                    "metric_value": viable_rows[0][0],
                    "payload_mb_est": row_payload,
                }

    return {
        "artifact_limit_mb": float(artifact_limit_mb),
        "payload_mb_est": payload_mb_est,
        "current_within_budget": current_within_budget,
        "budget_ratio": budget_ratio,
        "best_quantized_candidate": best_viable,
        "has_viable_artifact_path": bool(current_within_budget or best_viable is not None),
    }


def build_result_forecast(
    result: dict[str, Any],
    *,
    budget: CompetitionBudget = DEFAULT_GOLF_V1_BUDGET,
) -> dict[str, Any]:
    training = _training_section(result)
    performance_estimate = _performance_estimate(training)
    performance = extract_result_performance(result)
    metric = extract_result_metric(result, preferred_name=budget.primary_metric_name)
    context = _build_compute_context(result, budget=budget, performance=performance)

    train_curve, total_curve, compute_points = _build_budget_curve_models(
        result,
        metric_name=metric.name,
        context=context,
    )

    train_metric_forecast = train_curve.get("forecast_value")
    total_metric_forecast = total_curve.get("forecast_value")
    current_metric_value = metric.value

    return {
        "version": FORECAST_VERSION,
        "budget": budget.as_dict(),
        "observed": {
            "metric_name": metric.name,
            "metric_value": metric.value,
            "steps_completed": performance.steps_completed,
            "tokens_completed": performance.tokens_completed,
            "tokens_per_second": performance.tokens_per_second,
            "estimated_sustained_tflops": performance.estimated_sustained_tflops,
            "estimated_sustained_total_tflops": getattr(performance, "estimated_sustained_total_tflops", None),
            "seconds_per_step": performance.seconds_per_step,
            "total_elapsed_sec": getattr(performance, "total_elapsed_sec", None),
            "probe_elapsed_sec": getattr(performance, "probe_elapsed_sec", None),
            "train_tflops_consumed_est": performance.train_tflops_consumed_est,
            "probe_eval_tflops_consumed_est": performance.probe_eval_tflops_consumed_est,
            "final_eval_tflops_consumed_est": performance.final_eval_tflops_consumed_est,
            "total_tflops_consumed_est": performance.total_tflops_consumed_est,
            "train_step_flops_per_step_est": context["train_step_flops_per_step_est"],
            "forward_total_flops_per_step_est": context["forward_total_flops_per_step_est"],
            "probe_point_count": len(compute_points),
            "compute_point_count": len(compute_points),
        },
        "projection": {
            "budget_step_limit_est": context["budget_step_limit_est"],
            "budget_train_tflops_est": float(budget.train_tflops_budget) if budget.train_tflops_budget > 0.0 else None,
            "budget_total_tflops_est": context["budget_total_tflops_est"],
            "current_train_tflops_consumed_est": context["current_train_tflops_consumed_est"],
            "current_probe_eval_tflops_consumed_est": context["current_probe_eval_tflops_consumed_est"],
            "current_final_eval_tflops_consumed_est": context["current_final_eval_tflops_consumed_est"],
            "current_total_tflops_consumed_est": context["current_total_tflops_consumed_est"],
            "current_total_tflops_per_train_step_est": context["current_total_tflops_per_train_step_est"],
            "remaining_train_tflops_est": context["remaining_train_tflops_est"],
            "remaining_total_tflops_est": context["remaining_total_tflops_est"],
            "remaining_total_step_est": context["remaining_total_step_est"],
            "future_probe_count_est": context["future_probe_count_est"],
            "future_probe_eval_tflops_est": context["future_probe_eval_tflops_est"],
            "full_run_probe_eval_tflops_est": context["full_run_probe_eval_tflops_est"],
            "final_eval_tflops_budget_est": context["final_eval_tflops_budget_est"],
            "budget_total_step_limit_est": context["budget_total_step_limit_est"],
            "projected_total_wallclock_sec": context["projected_total_wallclock_sec"],
            "projected_remaining_wallclock_sec": context["projected_remaining_wallclock_sec"],
            "forecast_metric_name": metric.name,
            "forecast_metric_at_budget": total_metric_forecast,
            "forecast_metric_at_budget_lower": total_curve.get("forecast_lower_95"),
            "forecast_metric_at_budget_upper": total_curve.get("forecast_upper_95"),
            "forecast_metric_at_train_budget": train_metric_forecast,
            "forecast_metric_at_train_budget_lower": train_curve.get("forecast_lower_95"),
            "forecast_metric_at_train_budget_upper": train_curve.get("forecast_upper_95"),
            "forecast_delta_from_current": _forecast_delta_from_current(current_metric_value, total_metric_forecast),
            "dbpb_dtrain_tflop": train_curve.get("marginal_per_tflop_at_current"),
            "dbpb_dtotal_tflop": total_curve.get("marginal_per_tflop_at_current"),
            "curve_model": total_curve,
            "train_curve_model": train_curve,
        },
        "artifact": _artifact_projection(
            result,
            metric_name=metric.name,
            artifact_limit_mb=budget.artifact_limit_mb,
        ),
    }
