#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import statistics
import sys
from typing import Iterable


PROBE_LINE_RE = re.compile(
    r"probe\s+(?P<step>\d+)\s+\|\s+(?P<split>\w+)\s+loss\s+(?P<loss>[0-9.]+)\s+\|\s+bpt\s+(?P<bpt>[0-9.]+)\s+\|\s+bpb\s+(?P<bpb>[0-9.]+|n/a)"
)
RUN_LINE_RE = re.compile(
    r"run\s+(?P<label>\S+)\s+scale=(?P<scale>[0-9.]+)\s+steps=(?P<steps>\d+)\s+seed=(?P<seed>\d+)"
)
FINAL_LINE_RE = re.compile(
    r"Te:(?P<loss>[0-9.]+)\s+bpt:(?P<bpt>[0-9.]+)\s+bpb:(?P<bpb>[0-9.]+)\s+Of:(?P<overfit>[+-]?[0-9.]+)%\s+T:(?P<time>[0-9.]+)s"
)
VARIANT_LINE_RE = re.compile(
    r"variant=(?P<variant>\S+)\s+scale=(?P<scale>[0-9.]+).*static_bank_gate=(?P<static>\S+)"
)


@dataclass(frozen=True)
class ProbePoint:
    recipe_key: str
    recipe_label: str
    step: int
    bpb: float
    elapsed_sec: float | None
    source: str
    path: str
    fidelity_rank: int


@dataclass(frozen=True)
class CurveFit:
    name: str
    projected_bpb: float
    projected_gain: float
    sse: float
    detail: str


def linear_regression(xs: list[float], ys: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n == 0:
        raise ValueError("cannot regress empty vectors")
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if var_x == 0.0:
        return 0.0, mean_y
    cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = cov_xy / var_x
    intercept = mean_y - slope * mean_x
    return slope, intercept


def fit_log_curve(steps: list[int], bpbs: list[float], next_step: int, last_bpb: float) -> CurveFit:
    xs = [math.log(step) for step in steps]
    slope, intercept = linear_regression(xs, bpbs)
    projected = intercept + slope * math.log(next_step)
    sse = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, bpbs))
    return CurveFit(
        name="log-linear",
        projected_bpb=projected,
        projected_gain=last_bpb - projected,
        sse=sse,
        detail=f"bpb = {intercept:.6f} + {slope:.6f} * log(step)",
    )


def fit_power_curve(steps: list[int], bpbs: list[float], next_step: int, last_bpb: float) -> CurveFit:
    best: CurveFit | None = None
    for alpha_i in range(10, 401):
        alpha = alpha_i / 100.0
        xs = [step ** (-alpha) for step in steps]
        slope, intercept = linear_regression(xs, bpbs)
        if slope < 0.0:
            continue
        projected = intercept + slope * (next_step ** (-alpha))
        sse = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, bpbs))
        fit = CurveFit(
            name="power-tail",
            projected_bpb=projected,
            projected_gain=last_bpb - projected,
            sse=sse,
            detail=f"bpb = {intercept:.6f} + {slope:.6f} * step^-{alpha:.2f}",
        )
        if best is None or fit.sse < best.sse:
            best = fit
    if best is None:
        return fit_log_curve(steps, bpbs, next_step, last_bpb)
    return best


def choose_best_fit(steps: list[int], bpbs: list[float], next_step: int) -> CurveFit | None:
    if len(steps) < 3:
        return None
    last_bpb = bpbs[-1]
    fits = [
        fit_log_curve(steps, bpbs, next_step, last_bpb),
        fit_power_curve(steps, bpbs, next_step, last_bpb),
    ]
    return min(fits, key=lambda fit: fit.sse)


def dedupe_points(points: Iterable[ProbePoint]) -> list[ProbePoint]:
    best_by_step: dict[int, ProbePoint] = {}
    for point in sorted(points, key=lambda row: (row.step, row.fidelity_rank, row.path), reverse=True):
        incumbent = best_by_step.get(point.step)
        if incumbent is None or point.fidelity_rank > incumbent.fidelity_rank:
            best_by_step[point.step] = point
    return [best_by_step[step] for step in sorted(best_by_step)]


def parse_json_points(path: Path) -> list[ProbePoint]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return []
    model = payload.get("model") or {}
    training = payload.get("training") or {}
    config = payload.get("config") or {}
    train_cfg = config.get("train") or {}
    step_count = train_cfg.get("steps")
    if not isinstance(step_count, int):
        return []
    scale = model.get("scale")
    seed = model.get("seed")
    variant = model.get("variant", "unknown")
    gate = "staticgate" if model.get("static_bank_gate") else "ungated"
    recipe_key = f"{variant}|scale={scale}|seed={seed}|gate={gate}"
    recipe_label = f"{variant} scale={scale} seed={seed} gate={gate}"

    points: list[ProbePoint] = []
    probes = training.get("probes") or []
    if isinstance(probes, list):
        for row in probes:
            if not isinstance(row, dict):
                continue
            step = row.get("step")
            bpb = row.get("bpb")
            if isinstance(step, int) and isinstance(bpb, (int, float)):
                points.append(
                    ProbePoint(
                        recipe_key=recipe_key,
                        recipe_label=recipe_label,
                        step=step,
                        bpb=float(bpb),
                        elapsed_sec=float(row["elapsed_sec"]) if isinstance(row.get("elapsed_sec"), (int, float)) else None,
                        source="json_probe",
                        path=str(path),
                        fidelity_rank=1,
                    )
                )
    final_bpb = model.get("test_bpb")
    final_time = model.get("train_time_sec")
    if isinstance(final_bpb, (int, float)):
        points.append(
            ProbePoint(
                recipe_key=recipe_key,
                recipe_label=recipe_label,
                step=step_count,
                bpb=float(final_bpb),
                elapsed_sec=float(final_time) if isinstance(final_time, (int, float)) else None,
                source="json_final",
                path=str(path),
                fidelity_rank=3,
            )
        )
    return points


def parse_log_points(path: Path) -> list[ProbePoint]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    points: list[ProbePoint] = []
    current_key: str | None = None
    current_label: str | None = None
    current_step: int | None = None

    for line in lines:
        run_match = RUN_LINE_RE.search(line)
        if run_match:
            scale = run_match.group("scale")
            seed = run_match.group("seed")
            label = run_match.group("label")
            current_step = int(run_match.group("steps"))
            current_key = f"{label}|scale={scale}|seed={seed}"
            current_label = f"{label} scale={scale} seed={seed}"
            continue

        variant_match = VARIANT_LINE_RE.search(line)
        if variant_match and current_key is None:
            scale = variant_match.group("scale")
            label = variant_match.group("variant")
            gate = "staticgate" if variant_match.group("static") == "True" else "ungated"
            current_key = f"{label}|scale={scale}|gate={gate}"
            current_label = f"{label} scale={scale} gate={gate}"

        probe_match = PROBE_LINE_RE.search(line)
        if probe_match and current_key is not None and current_label is not None:
            bpb_raw = probe_match.group("bpb")
            if bpb_raw != "n/a":
                points.append(
                    ProbePoint(
                        recipe_key=current_key,
                        recipe_label=current_label,
                        step=int(probe_match.group("step")),
                        bpb=float(bpb_raw),
                        elapsed_sec=None,
                        source="log_probe",
                        path=str(path),
                        fidelity_rank=0,
                    )
                )
            continue

        final_match = FINAL_LINE_RE.search(line)
        if final_match and current_key is not None and current_label is not None and current_step is not None:
            points.append(
                ProbePoint(
                    recipe_key=current_key,
                    recipe_label=current_label,
                    step=current_step,
                    bpb=float(final_match.group("bpb")),
                    elapsed_sec=float(final_match.group("time")),
                    source="log_final",
                    path=str(path),
                    fidelity_rank=2,
                )
            )
    return points


def load_points(paths: list[Path]) -> list[ProbePoint]:
    points: list[ProbePoint] = []
    for path in paths:
        if path.suffix == ".json":
            points.extend(parse_json_points(path))
        else:
            points.extend(parse_log_points(path))
    return points


def choose_next_step(steps: list[int], explicit_next_step: int | None) -> int:
    recent_gap = 400
    if len(steps) > 1:
        gaps = [b - a for a, b in zip(steps, steps[1:]) if b > a]
        if gaps:
            recent_gap = int(round(statistics.median(gaps)))
    if explicit_next_step is not None:
        if explicit_next_step > steps[-1]:
            return explicit_next_step
        return steps[-1] + recent_gap
    if len(steps) == 1:
        return steps[-1] + 400
    return steps[-1] + recent_gap


def summarize_group(
    label: str,
    points: list[ProbePoint],
    next_step: int,
    min_gain_next: float,
    min_gain_per_100: float,
    min_gain_per_hour: float,
) -> str:
    deduped = dedupe_points(points)
    steps = [point.step for point in deduped]
    bpbs = [point.bpb for point in deduped]
    lines = [f"recipe: {label}"]
    lines.append("points:")
    for point in deduped:
        elapsed = "-" if point.elapsed_sec is None else f"{point.elapsed_sec:.0f}s"
        lines.append(
            f"  step {point.step:5d} | bpb {point.bpb:.4f} | source {point.source:<10} | elapsed {elapsed} | {Path(point.path).name}"
        )

    if len(deduped) < 2:
        lines.append("recommendation: HOLD")
        lines.append("reason: only one point available; need at least two steps to estimate marginal gain.")
        return "\n".join(lines)

    deltas = []
    for prev, curr in zip(deduped, deduped[1:]):
        gain = prev.bpb - curr.bpb
        step_delta = curr.step - prev.step
        gain_per_100 = gain * 100.0 / step_delta if step_delta > 0 else float("nan")
        gain_per_hour = None
        if prev.elapsed_sec is not None and curr.elapsed_sec is not None:
            time_delta = curr.elapsed_sec - prev.elapsed_sec
            if time_delta > 0:
                gain_per_hour = gain * 3600.0 / time_delta
        deltas.append((prev, curr, gain, gain_per_100, gain_per_hour))

    lines.append("marginals:")
    for prev, curr, gain, gain_per_100, gain_per_hour in deltas:
        mph = "-" if gain_per_hour is None else f"{gain_per_hour:.4f}/hr"
        lines.append(
            f"  {prev.step:5d}->{curr.step:5d} | gain {gain:+.4f} | {gain_per_100:+.4f}/100 steps | {mph}"
        )

    recent = deltas[-1]
    _, curr, recent_gain, recent_gain_per_100, recent_gain_per_hour = recent
    fit = choose_best_fit(steps, bpbs, next_step)
    if fit is None:
        step_delta = curr.step - deduped[-2].step
        projected_gain = recent_gain_per_100 * max(next_step - curr.step, 0) / 100.0
        projected_bpb = curr.bpb - projected_gain
        fit = CurveFit(
            name="recent-slope",
            projected_bpb=projected_bpb,
            projected_gain=projected_gain,
            sse=0.0,
            detail="projection from most recent marginal gain only",
        )

    lines.append("projection:")
    lines.append(f"  next_step: {next_step}")
    lines.append(f"  fit: {fit.name} | {fit.detail}")
    lines.append(f"  projected_bpb: {fit.projected_bpb:.4f}")
    lines.append(f"  projected_gain: {fit.projected_gain:+.4f}")

    should_continue = (
        fit.projected_gain >= min_gain_next
        and recent_gain_per_100 >= min_gain_per_100
        and (recent_gain_per_hour is None or recent_gain_per_hour >= min_gain_per_hour)
    )
    should_stop = (
        fit.projected_gain < (min_gain_next * 0.5)
        or recent_gain_per_100 < (min_gain_per_100 * 0.5)
        or (recent_gain_per_hour is not None and recent_gain_per_hour < (min_gain_per_hour * 0.5))
    )

    if should_continue:
        lines.append("recommendation: CONTINUE")
        lines.append(
            "reason: projected next-step gain and recent marginal gain are both above the configured stop floor."
        )
    elif should_stop:
        lines.append("recommendation: STOP")
        lines.append(
            "reason: projected gain has fallen below the configured floor, so more steps are unlikely to be the best frontier spend."
        )
    else:
        lines.append("recommendation: REVIEW")
        lines.append(
            "reason: the curve is flattening but not decisively dead; run one more checkpoint only if this recipe is still frontier-adjacent."
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Chronohorn probe/final outputs and print a stop/continue recommendation."
    )
    parser.add_argument("paths", nargs="+", help="JSON or log files from Chronohorn frontier runs.")
    parser.add_argument("--recipe-substring", default=None, help="Only analyze recipes whose label contains this substring.")
    parser.add_argument("--next-step", type=int, default=None, help="Step to project toward. Defaults to last step plus recent median gap.")
    parser.add_argument("--min-gain-next", type=float, default=0.010, help="Minimum projected bpb gain required to continue.")
    parser.add_argument(
        "--min-gain-per-100",
        type=float,
        default=0.002,
        help="Minimum recent bpb gain per 100 steps required to continue.",
    )
    parser.add_argument(
        "--min-gain-per-hour",
        type=float,
        default=0.020,
        help="Minimum recent bpb gain per training hour required to continue when elapsed time is available.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = [Path(raw) for raw in args.paths]
    points = load_points(paths)
    if not points:
        print("No probe or final points found in the provided inputs.", file=sys.stderr)
        return 1

    grouped: dict[str, list[ProbePoint]] = defaultdict(list)
    labels: dict[str, str] = {}
    for point in points:
        if args.recipe_substring and args.recipe_substring not in point.recipe_label:
            continue
        grouped[point.recipe_key].append(point)
        labels[point.recipe_key] = point.recipe_label

    if not grouped:
        print("No recipes matched the requested filter.", file=sys.stderr)
        return 1

    for index, key in enumerate(sorted(grouped)):
        label = labels[key]
        deduped = dedupe_points(grouped[key])
        next_step = choose_next_step([point.step for point in deduped], args.next_step)
        if index:
            print()
        print(
            summarize_group(
                label=label,
                points=grouped[key],
                next_step=next_step,
                min_gain_next=args.min_gain_next,
                min_gain_per_100=args.min_gain_per_100,
                min_gain_per_hour=args.min_gain_per_hour,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
