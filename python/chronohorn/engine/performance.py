from __future__ import annotations

import math
from typing import Any


def bits_per_token_from_loss(token_loss_nats: float) -> float:
    return token_loss_nats / math.log(2.0)


def summarize_observed_training_performance(
    estimate: dict[str, Any],
    *,
    steps_completed: int,
    elapsed_sec: float,
    interval_steps: int | None = None,
    interval_elapsed_sec: float | None = None,
    probe_tflops_consumed_est: float = 0.0,
    probe_elapsed_sec: float = 0.0,
    interval_probe_tflops_est: float = 0.0,
    interval_probe_elapsed_sec: float = 0.0,
) -> dict[str, Any]:
    tokens_per_step = float(estimate["tokens_per_step"])
    steps_completed = int(steps_completed)
    elapsed_sec = float(max(elapsed_sec, 1e-9))
    interval_steps = steps_completed if interval_steps is None else int(interval_steps)
    interval_elapsed_sec = elapsed_sec if interval_elapsed_sec is None else float(max(interval_elapsed_sec, 1e-9))
    probe_tflops_consumed_est = float(max(probe_tflops_consumed_est, 0.0))
    probe_elapsed_sec = float(max(probe_elapsed_sec, 0.0))
    interval_probe_tflops_est = float(max(interval_probe_tflops_est, 0.0))
    interval_probe_elapsed_sec = float(max(interval_probe_elapsed_sec, 0.0))
    tokens_completed = float(tokens_per_step * steps_completed)
    interval_tokens = float(tokens_per_step * interval_steps)
    sustained_tokens_per_second = tokens_completed / elapsed_sec
    interval_tokens_per_second = interval_tokens / interval_elapsed_sec
    sustained_tflops = sustained_tokens_per_second * float(estimate["train_step_flops_per_token_est"]) / 1e12
    interval_tflops = interval_tokens_per_second * float(estimate["train_step_flops_per_token_est"]) / 1e12
    sustained_forward_tflops = sustained_tokens_per_second * float(estimate["forward_total_flops_per_token"]) / 1e12
    interval_forward_tflops = interval_tokens_per_second * float(estimate["forward_total_flops_per_token"]) / 1e12
    train_tflops_consumed_est = float(estimate["train_step_flops_per_step_est"]) * steps_completed / 1e12
    interval_train_tflops_consumed_est = float(estimate["train_step_flops_per_step_est"]) * interval_steps / 1e12
    total_tflops_consumed_est = train_tflops_consumed_est + probe_tflops_consumed_est
    interval_total_tflops_consumed_est = interval_train_tflops_consumed_est + interval_probe_tflops_est
    effective_train_elapsed_sec = max(elapsed_sec - probe_elapsed_sec, 1e-9)
    effective_interval_train_elapsed_sec = max(interval_elapsed_sec - interval_probe_elapsed_sec, 1e-9)
    sustained_total_tflops = total_tflops_consumed_est / elapsed_sec
    interval_total_tflops = interval_total_tflops_consumed_est / interval_elapsed_sec
    sustained_train_tflops_excluding_probe = train_tflops_consumed_est / effective_train_elapsed_sec
    interval_train_tflops_excluding_probe = (
        interval_train_tflops_consumed_est / effective_interval_train_elapsed_sec
    )
    probe_overhead_fraction_compute = (
        probe_tflops_consumed_est / total_tflops_consumed_est if total_tflops_consumed_est > 0.0 else 0.0
    )
    interval_probe_overhead_fraction_compute = (
        interval_probe_tflops_est / interval_total_tflops_consumed_est
        if interval_total_tflops_consumed_est > 0.0
        else 0.0
    )
    probe_overhead_fraction_wallclock = min(max(probe_elapsed_sec / elapsed_sec, 0.0), 1.0)
    interval_probe_overhead_fraction_wallclock = min(
        max(interval_probe_elapsed_sec / interval_elapsed_sec, 0.0),
        1.0,
    )
    return {
        "version": estimate["version"],
        "steps_completed": steps_completed,
        "tokens_completed": int(tokens_completed),
        "elapsed_sec": elapsed_sec,
        "samples_per_second": (tokens_completed / max(float(estimate["seq_len"]), 1.0)) / elapsed_sec,
        "tokens_per_second": sustained_tokens_per_second,
        "estimated_sustained_tflops": sustained_tflops,
        "estimated_sustained_total_tflops": sustained_total_tflops,
        "estimated_sustained_train_tflops_excluding_probe": sustained_train_tflops_excluding_probe,
        "estimated_sustained_forward_tflops": sustained_forward_tflops,
        "train_step_flops_per_step_est": float(estimate["train_step_flops_per_step_est"]),
        "forward_total_flops_per_step": float(estimate["forward_total_flops_per_step"]),
        "train_tflops_consumed_est": train_tflops_consumed_est,
        "probe_tflops_consumed_est": probe_tflops_consumed_est,
        "total_tflops_consumed_est": total_tflops_consumed_est,
        "probe_elapsed_sec": probe_elapsed_sec,
        "effective_train_elapsed_sec": effective_train_elapsed_sec,
        "probe_overhead_fraction_compute": probe_overhead_fraction_compute,
        "probe_overhead_fraction_wallclock": probe_overhead_fraction_wallclock,
        "interval_steps": interval_steps,
        "interval_elapsed_sec": interval_elapsed_sec,
        "interval_samples_per_second": (interval_tokens / max(float(estimate["seq_len"]), 1.0)) / interval_elapsed_sec,
        "interval_tokens_per_second": interval_tokens_per_second,
        "estimated_interval_tflops": interval_tflops,
        "estimated_interval_total_tflops": interval_total_tflops,
        "estimated_interval_train_tflops_excluding_probe": interval_train_tflops_excluding_probe,
        "estimated_interval_forward_tflops": interval_forward_tflops,
        "interval_train_tflops_consumed_est": interval_train_tflops_consumed_est,
        "interval_probe_tflops_est": interval_probe_tflops_est,
        "interval_total_tflops_consumed_est": interval_total_tflops_consumed_est,
        "interval_probe_elapsed_sec": interval_probe_elapsed_sec,
        "effective_interval_train_elapsed_sec": effective_interval_train_elapsed_sec,
        "interval_probe_overhead_fraction_compute": interval_probe_overhead_fraction_compute,
        "interval_probe_overhead_fraction_wallclock": interval_probe_overhead_fraction_wallclock,
        "seconds_per_step": elapsed_sec / max(steps_completed, 1),
    }


def format_observed_training_performance(summary: dict[str, Any]) -> str:
    return (
        f"{summary['interval_tokens_per_second']:.0f} tok/s "
        f"| train {summary['estimated_interval_tflops']:.3f} TF/s "
        f"| total {summary.get('estimated_interval_total_tflops', summary['estimated_interval_tflops']):.3f} TF/s"
    )


def estimate_eval_flops_for_batches(
    estimate: dict[str, Any],
    *,
    eval_batches: int | float,
) -> float:
    return float(max(eval_batches, 0.0)) * float(estimate["forward_total_flops_per_step"])


def summarize_compute_accounting(
    estimate: dict[str, Any],
    *,
    steps_completed: int,
    probe_tflops_consumed_est: float = 0.0,
    final_eval_batches: int = 0,
    final_eval_splits: int = 0,
    artifact_eval_batches: int = 0,
    artifact_eval_runs: int = 0,
    replay_tokens: int = 0,
) -> dict[str, Any]:
    train_step_flops_per_step_est = float(estimate["train_step_flops_per_step_est"])
    forward_total_flops_per_step = float(estimate["forward_total_flops_per_step"])
    forward_total_flops_per_token = float(estimate["forward_total_flops_per_token"])
    steps_completed = int(max(steps_completed, 0))
    probe_tflops_consumed_est = float(max(probe_tflops_consumed_est, 0.0))
    train_tflops_consumed_est = train_step_flops_per_step_est * float(steps_completed) / 1e12
    final_eval_tflops_consumed_est = (
        forward_total_flops_per_step * float(max(final_eval_batches, 0)) * float(max(final_eval_splits, 0)) / 1e12
    )
    artifact_eval_tflops_consumed_est = (
        forward_total_flops_per_step
        * float(max(artifact_eval_batches, 0))
        * float(max(artifact_eval_runs, 0))
        / 1e12
    )
    replay_tflops_consumed_est = forward_total_flops_per_token * float(max(replay_tokens, 0)) / 1e12
    eval_tflops_consumed_est = (
        probe_tflops_consumed_est
        + final_eval_tflops_consumed_est
        + artifact_eval_tflops_consumed_est
        + replay_tflops_consumed_est
    )
    total_tflops_consumed_est = train_tflops_consumed_est + eval_tflops_consumed_est
    probe_tflops_per_train_step_est = (
        probe_tflops_consumed_est / float(steps_completed) if steps_completed > 0 else 0.0
    )
    effective_total_tflops_per_train_step_est = (
        (train_tflops_consumed_est + probe_tflops_consumed_est) / float(steps_completed)
        if steps_completed > 0
        else None
    )
    return {
        "train_step_flops_per_step_est": train_step_flops_per_step_est,
        "forward_eval_flops_per_batch_est": forward_total_flops_per_step,
        "forward_eval_flops_per_token_est": forward_total_flops_per_token,
        "train_tflops_consumed_est": train_tflops_consumed_est,
        "probe_tflops_consumed_est": probe_tflops_consumed_est,
        "final_eval_tflops_consumed_est": final_eval_tflops_consumed_est,
        "artifact_eval_tflops_consumed_est": artifact_eval_tflops_consumed_est,
        "replay_tflops_consumed_est": replay_tflops_consumed_est,
        "eval_tflops_consumed_est": eval_tflops_consumed_est,
        "total_tflops_consumed_est": total_tflops_consumed_est,
        "probe_tflops_per_train_step_est": probe_tflops_per_train_step_est,
        "effective_total_tflops_per_train_step_est": effective_total_tflops_per_train_step_est,
        "probe_overhead_fraction_compute": (
            probe_tflops_consumed_est / total_tflops_consumed_est if total_tflops_consumed_est > 0.0 else 0.0
        ),
        "eval_overhead_fraction_compute": (
            eval_tflops_consumed_est / total_tflops_consumed_est if total_tflops_consumed_est > 0.0 else 0.0
        ),
    }
