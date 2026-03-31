from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ResultMetric:
    name: str | None
    value: float | None


@dataclass(frozen=True)
class ResultPerformance:
    steps_completed: int | None
    tokens_completed: int | None
    tokens_per_second: float | None
    estimated_sustained_tflops: float | None
    seconds_per_step: float | None
    estimated_sustained_total_tflops: float | None = None
    total_elapsed_sec: float | None = None
    probe_elapsed_sec: float | None = None
    train_tflops_consumed_est: float | None = None
    probe_eval_tflops_consumed_est: float | None = None
    final_eval_tflops_consumed_est: float | None = None
    artifact_eval_tflops_consumed_est: float | None = None
    total_tflops_consumed_est: float | None = None


@dataclass(frozen=True)
class ResultSummary:
    path: str | None
    title: str | None
    metric: ResultMetric
    performance: ResultPerformance
    payload_mb_est: float | None
    train_time_sec: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "title": self.title,
            "metric": {
                "name": self.metric.name,
                "value": self.metric.value,
            },
            "performance": {
                "steps_completed": self.performance.steps_completed,
                "tokens_completed": self.performance.tokens_completed,
                "tokens_per_second": self.performance.tokens_per_second,
                "estimated_sustained_tflops": self.performance.estimated_sustained_tflops,
                "seconds_per_step": self.performance.seconds_per_step,
                "estimated_sustained_total_tflops": self.performance.estimated_sustained_total_tflops,
                "total_elapsed_sec": self.performance.total_elapsed_sec,
                "probe_elapsed_sec": self.performance.probe_elapsed_sec,
                "train_tflops_consumed_est": self.performance.train_tflops_consumed_est,
                "probe_eval_tflops_consumed_est": self.performance.probe_eval_tflops_consumed_est,
                "final_eval_tflops_consumed_est": self.performance.final_eval_tflops_consumed_est,
                "artifact_eval_tflops_consumed_est": self.performance.artifact_eval_tflops_consumed_est,
                "total_tflops_consumed_est": self.performance.total_tflops_consumed_est,
            },
            "payload_mb_est": self.payload_mb_est,
            "train_time_sec": self.train_time_sec,
        }


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _infer_steps_completed(
    result: dict[str, Any],
    performance: dict[str, Any] | None,
) -> int | None:
    if isinstance(performance, dict):
        steps_completed = _safe_int(performance.get("steps_completed"))
        if steps_completed is not None and steps_completed > 0:
            return steps_completed
    training = result.get("training")
    if isinstance(training, dict):
        config = result.get("config")
        if isinstance(config, dict):
            train_cfg = config.get("train")
            if isinstance(train_cfg, dict):
                steps_completed = _safe_int(train_cfg.get("steps"))
                if steps_completed is not None and steps_completed > 0:
                    return steps_completed
    return None


def _estimate_training_compute_from_summary(
    result: dict[str, Any],
    performance: dict[str, Any] | None,
    performance_estimate: dict[str, Any] | None,
) -> float | None:
    if isinstance(performance, dict):
        train_tflops_consumed_est = _safe_float(performance.get("train_tflops_consumed_est"))
        if train_tflops_consumed_est is not None:
            return train_tflops_consumed_est
    if not isinstance(performance_estimate, dict):
        return None
    steps_completed = _infer_steps_completed(result, performance)
    if steps_completed is None:
        return None
    train_step_flops_per_step_est = _safe_float(performance_estimate.get("train_step_flops_per_step_est"))
    if train_step_flops_per_step_est is None:
        return None
    return float(train_step_flops_per_step_est) * float(steps_completed) / 1e12


def _estimate_probe_eval_compute(
    result: dict[str, Any],
    *,
    performance_estimate: dict[str, Any] | None,
) -> float | None:
    training = result.get("training")
    if not isinstance(training, dict):
        return None
    performance = training.get("performance")
    if isinstance(performance, dict):
        probe_eval_tflops_consumed_est = _safe_float(performance.get("probe_eval_tflops_consumed_est"))
        if probe_eval_tflops_consumed_est is not None:
            return probe_eval_tflops_consumed_est
    if not isinstance(performance_estimate, dict):
        return None
    forward_total_flops_per_step = _safe_float(performance_estimate.get("forward_total_flops_per_step"))
    if forward_total_flops_per_step is None:
        return None
    probes = training.get("probes")
    if not isinstance(probes, list) or not probes:
        return 0.0
    top_level_eval_batches = _safe_int(training.get("probe_eval_batches"))
    total = 0.0
    for row in probes:
        if not isinstance(row, dict):
            continue
        eval_batches = _safe_int(row.get("eval_batches"))
        if eval_batches is None:
            eval_batches = top_level_eval_batches
        if eval_batches is None or eval_batches <= 0:
            continue
        total += float(eval_batches) * float(forward_total_flops_per_step) / 1e12
    return total


def _estimate_final_eval_compute(
    result: dict[str, Any],
    *,
    performance_estimate: dict[str, Any] | None,
) -> float | None:
    training = result.get("training")
    if not isinstance(training, dict):
        return None
    compute = training.get("compute_accounting_inputs")
    if isinstance(compute, dict):
        final_eval = compute.get("final_eval")
        if isinstance(final_eval, dict):
            final_eval_tflops_consumed_est = _safe_float(final_eval.get("eval_tflops_est"))
            if final_eval_tflops_consumed_est is not None:
                return final_eval_tflops_consumed_est
    performance = training.get("performance")
    if isinstance(performance, dict):
        final_eval_tflops_consumed_est = _safe_float(performance.get("final_eval_tflops_consumed_est"))
        if final_eval_tflops_consumed_est is not None:
            return final_eval_tflops_consumed_est
    if not isinstance(performance_estimate, dict):
        return None
    forward_total_flops_per_step = _safe_float(performance_estimate.get("forward_total_flops_per_step"))
    if forward_total_flops_per_step is None:
        return None
    final_eval_batches = _safe_int(training.get("final_eval_batches"))
    if final_eval_batches is None or final_eval_batches <= 0:
        return 0.0
    return 2.0 * float(final_eval_batches) * float(forward_total_flops_per_step) / 1e12


def _compute_accounting_inputs(result: dict[str, Any]) -> dict[str, Any] | None:
    training = result.get("training")
    if not isinstance(training, dict):
        return None
    compute = training.get("compute_accounting_inputs")
    return compute if isinstance(compute, dict) else None


def _load_result_payload(result: dict[str, Any], *path: str) -> Any:
    current: Any = result
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def load_result_json(path: str | Path) -> dict[str, Any]:
    result_path = Path(path).expanduser().resolve()
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Result JSON at {result_path} is not an object")
    return payload


def extract_result_metric(result: dict[str, Any], *, preferred_name: str | None = None) -> ResultMetric:
    model = result.get("model")
    if not isinstance(model, dict):
        return ResultMetric(name=None, value=None)
    metric_aliases = {
        "bpb": ("bpb", "test_bpb"),
        "bits_per_token": ("bits_per_token", "test_bits_per_token"),
        "eval_loss": ("eval_loss", "test_eval_loss"),
        "test_bpb": ("test_bpb",),
        "test_bits_per_token": ("test_bits_per_token",),
        "test_eval_loss": ("test_eval_loss",),
    }
    ordered_names = [preferred_name] if preferred_name else []
    ordered_names.extend(["bpb", "bits_per_token", "eval_loss"])
    seen: set[str] = set()
    for metric_name in ordered_names:
        if not metric_name or metric_name in seen:
            continue
        seen.add(metric_name)
        for candidate_key in metric_aliases.get(metric_name, (metric_name,)):
            value = _safe_float(model.get(candidate_key))
            if value is not None:
                canonical_name = "bpb" if candidate_key == "test_bpb" else metric_name
                if candidate_key == "test_bits_per_token":
                    canonical_name = "bits_per_token"
                if candidate_key == "test_eval_loss":
                    canonical_name = "eval_loss"
                return ResultMetric(name=canonical_name, value=value)
    return ResultMetric(name=None, value=None)


def extract_result_performance(result: dict[str, Any]) -> ResultPerformance:
    training = result.get("training")
    performance = training.get("performance") if isinstance(training, dict) else None
    performance_estimate = training.get("performance_estimate") if isinstance(training, dict) else None
    compute_inputs = _compute_accounting_inputs(result)
    compute_train = compute_inputs.get("train") if isinstance(compute_inputs, dict) else None
    compute_probe = compute_inputs.get("probe") if isinstance(compute_inputs, dict) else None
    compute_total = compute_inputs.get("total") if isinstance(compute_inputs, dict) else None
    compute_final_eval = compute_inputs.get("final_eval") if isinstance(compute_inputs, dict) else None
    compute_artifact_eval = compute_inputs.get("artifact_eval") if isinstance(compute_inputs, dict) else None
    if not isinstance(performance, dict):
        base = ResultPerformance(
            steps_completed=None,
            tokens_completed=None,
            tokens_per_second=None,
            estimated_sustained_tflops=None,
            seconds_per_step=None,
        )
        train_tflops_consumed_est = _safe_float(compute_train.get("train_tflops_est")) if isinstance(compute_train, dict) else None
        if train_tflops_consumed_est is None:
            train_tflops_consumed_est = _estimate_training_compute_from_summary(result, None, performance_estimate)
        probe_eval_tflops_consumed_est = _safe_float(compute_probe.get("eval_tflops_est")) if isinstance(compute_probe, dict) else None
        if probe_eval_tflops_consumed_est is None:
            probe_eval_tflops_consumed_est = _estimate_probe_eval_compute(result, performance_estimate=performance_estimate)
        final_eval_tflops_consumed_est = _safe_float(compute_final_eval.get("eval_tflops_est")) if isinstance(compute_final_eval, dict) else None
        if final_eval_tflops_consumed_est is None:
            final_eval_tflops_consumed_est = _estimate_final_eval_compute(result, performance_estimate=performance_estimate)
        artifact_eval_tflops_consumed_est = _safe_float(compute_artifact_eval.get("eval_tflops_est")) if isinstance(compute_artifact_eval, dict) else None
        total_elapsed_sec = _safe_float(compute_total.get("elapsed_sec")) if isinstance(compute_total, dict) else None
        probe_elapsed_sec = _safe_float(compute_probe.get("elapsed_sec_total")) if isinstance(compute_probe, dict) else None
        estimated_sustained_total_tflops = _safe_float(compute_total.get("observed_total_tflops_per_second")) if isinstance(compute_total, dict) else None
        total_tflops_consumed_est = _safe_float(compute_total.get("total_tflops_est")) if isinstance(compute_total, dict) else None
        if total_tflops_consumed_est is None and train_tflops_consumed_est is not None:
            total_tflops_consumed_est = float(train_tflops_consumed_est)
            if probe_eval_tflops_consumed_est is not None:
                total_tflops_consumed_est += float(probe_eval_tflops_consumed_est)
            if final_eval_tflops_consumed_est is not None:
                total_tflops_consumed_est += float(final_eval_tflops_consumed_est)
            if artifact_eval_tflops_consumed_est is not None:
                total_tflops_consumed_est += float(artifact_eval_tflops_consumed_est)
        return ResultPerformance(
            steps_completed=base.steps_completed,
            tokens_completed=base.tokens_completed,
            tokens_per_second=base.tokens_per_second,
            estimated_sustained_tflops=base.estimated_sustained_tflops,
            seconds_per_step=base.seconds_per_step,
            estimated_sustained_total_tflops=estimated_sustained_total_tflops,
            total_elapsed_sec=total_elapsed_sec,
            probe_elapsed_sec=probe_elapsed_sec,
            train_tflops_consumed_est=train_tflops_consumed_est,
            probe_eval_tflops_consumed_est=probe_eval_tflops_consumed_est,
            final_eval_tflops_consumed_est=final_eval_tflops_consumed_est,
            artifact_eval_tflops_consumed_est=artifact_eval_tflops_consumed_est,
            total_tflops_consumed_est=total_tflops_consumed_est,
        )
    base = ResultPerformance(
        steps_completed=_safe_int(performance.get("steps_completed")),
        tokens_completed=_safe_int(performance.get("tokens_completed")),
        tokens_per_second=_safe_float(performance.get("tokens_per_second")),
        estimated_sustained_tflops=_safe_float(performance.get("estimated_sustained_tflops")),
        seconds_per_step=_safe_float(performance.get("seconds_per_step")),
        estimated_sustained_total_tflops=_safe_float(performance.get("estimated_sustained_total_tflops")),
        total_elapsed_sec=_safe_float(performance.get("elapsed_sec")),
        probe_elapsed_sec=_safe_float(performance.get("probe_elapsed_sec")),
    )
    train_tflops_consumed_est = _safe_float(compute_train.get("train_tflops_est")) if isinstance(compute_train, dict) else None
    if train_tflops_consumed_est is None:
        train_tflops_consumed_est = _estimate_training_compute_from_summary(result, performance, performance_estimate)
    probe_eval_tflops_consumed_est = _safe_float(compute_probe.get("eval_tflops_est")) if isinstance(compute_probe, dict) else None
    if probe_eval_tflops_consumed_est is None:
        probe_eval_tflops_consumed_est = _estimate_probe_eval_compute(result, performance_estimate=performance_estimate)
    final_eval_tflops_consumed_est = _safe_float(compute_final_eval.get("eval_tflops_est")) if isinstance(compute_final_eval, dict) else None
    if final_eval_tflops_consumed_est is None:
        final_eval_tflops_consumed_est = _estimate_final_eval_compute(result, performance_estimate=performance_estimate)
    artifact_eval_tflops_consumed_est = _safe_float(compute_artifact_eval.get("eval_tflops_est")) if isinstance(compute_artifact_eval, dict) else None
    total_tflops_consumed_est = _safe_float(compute_total.get("total_tflops_est")) if isinstance(compute_total, dict) else train_tflops_consumed_est
    if total_tflops_consumed_est is not None and (not isinstance(compute_total, dict) or _safe_float(compute_total.get("total_tflops_est")) is None):
        if probe_eval_tflops_consumed_est is not None:
            total_tflops_consumed_est += float(probe_eval_tflops_consumed_est)
        if final_eval_tflops_consumed_est is not None:
            total_tflops_consumed_est += float(final_eval_tflops_consumed_est)
        if artifact_eval_tflops_consumed_est is not None:
            total_tflops_consumed_est += float(artifact_eval_tflops_consumed_est)
    return ResultPerformance(
        steps_completed=base.steps_completed,
        tokens_completed=base.tokens_completed,
        tokens_per_second=base.tokens_per_second,
        estimated_sustained_tflops=base.estimated_sustained_tflops,
        seconds_per_step=base.seconds_per_step,
        estimated_sustained_total_tflops=(
            base.estimated_sustained_total_tflops
            if base.estimated_sustained_total_tflops is not None
            else _safe_float(compute_total.get("observed_total_tflops_per_second")) if isinstance(compute_total, dict) else None
        ),
        total_elapsed_sec=(
            base.total_elapsed_sec
            if base.total_elapsed_sec is not None
            else _safe_float(compute_total.get("elapsed_sec")) if isinstance(compute_total, dict) else None
        ),
        probe_elapsed_sec=(
            base.probe_elapsed_sec
            if base.probe_elapsed_sec is not None
            else _safe_float(compute_probe.get("elapsed_sec_total")) if isinstance(compute_probe, dict) else None
        ),
        train_tflops_consumed_est=train_tflops_consumed_est,
        probe_eval_tflops_consumed_est=probe_eval_tflops_consumed_est,
        final_eval_tflops_consumed_est=final_eval_tflops_consumed_est,
        artifact_eval_tflops_consumed_est=artifact_eval_tflops_consumed_est,
        total_tflops_consumed_est=total_tflops_consumed_est,
    )


def extract_result_summary(result: dict[str, Any], *, path: str | None = None) -> ResultSummary:
    model = result.get("model")
    title = result.get("title") if isinstance(result.get("title"), str) else None
    metric = extract_result_metric(result)
    performance = extract_result_performance(result)
    payload_mb_est = _safe_float(model.get("payload_mb_est")) if isinstance(model, dict) else None
    train_time_sec = _safe_float(_load_result_payload(result, "model", "train_time_sec"))
    return ResultSummary(
        path=path,
        title=title,
        metric=metric,
        performance=performance,
        payload_mb_est=payload_mb_est,
        train_time_sec=train_time_sec,
    )
