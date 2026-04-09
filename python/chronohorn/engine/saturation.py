"""Saturation analysis for learning curves.

Computes per-probe metrics that reveal when a model has stopped learning:
  - doubling_gain: bpb improvement when training steps double
  - learning_rate: bpb improvement per 1K steps at each point
  - efficiency: bpb improvement per wall-second
  - asymptote: power-law fit from probes available so far
  - asymptote_stability: how much the asymptote estimate has shifted recently
  - saturation_step: estimated step where doubling gain drops below threshold
  - headroom: current bpb - asymptote estimate
  - status: "learning", "decelerating", "saturated", "overfitting"
"""
from __future__ import annotations

import math
from typing import Any


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def analyze_saturation(
    probes: list[dict[str, Any]],
    saturation_threshold: float = 0.01,
    asymptote_floor: float = 0.0,
) -> dict[str, Any]:
    """Analyze a learning curve for saturation.

    probes: list of {"step": int, "bpb": float, ...} sorted by step.
    saturation_threshold: doubling gain below this = saturated.
    asymptote_floor: minimum plausible bpb for the asymptote fit.

    Returns dict with all saturation metrics.
    """
    if len(probes) < 2:
        return {"status": "insufficient_data", "probes": len(probes)}

    steps = [p["step"] for p in probes]
    bpbs = [p["bpb"] for p in probes]

    # --- Per-interval metrics ---
    intervals = []
    for i in range(1, len(probes)):
        prev_probe = probes[i - 1]
        probe = probes[i]
        s0, s1 = steps[i - 1], steps[i]
        b0, b1 = bpbs[i - 1], bpbs[i]
        ds = s1 - s0
        db = b0 - b1  # positive = improving
        rate_per_1k = db / max(ds, 1) * 1000
        tf0 = prev_probe.get("tflops", prev_probe.get("tf"))
        tf1 = probe.get("tflops", probe.get("tf"))
        d_tf = None
        gain_per_tflop = None
        if isinstance(tf0, (int, float)) and isinstance(tf1, (int, float)) and tf1 > tf0:
            d_tf = float(tf1) - float(tf0)
            gain_per_tflop = db / d_tf if d_tf > 0 else None
        elapsed0 = prev_probe.get("elapsed_sec")
        elapsed1 = probe.get("elapsed_sec")
        d_elapsed = None
        gain_per_hour = None
        if isinstance(elapsed0, (int, float)) and isinstance(elapsed1, (int, float)) and elapsed1 > elapsed0:
            d_elapsed = float(elapsed1) - float(elapsed0)
            gain_per_hour = db / (d_elapsed / 3600.0) if d_elapsed > 0 else None

        # Is this approximately a doubling?
        ratio = s1 / max(s0, 1)
        is_doubling = 1.5 < ratio < 2.5

        intervals.append({
            "step_from": s0,
            "step_to": s1,
            "bpb_from": round(b0, 4),
            "bpb_to": round(b1, 4),
            "gain": round(db, 4),
            "rate_per_1k": round(rate_per_1k, 4),
            "is_doubling": is_doubling,
            "step_ratio": round(ratio, 2),
            "delta_tflops": round(d_tf, 4) if d_tf is not None else None,
            "gain_per_tflop": round(gain_per_tflop, 6) if gain_per_tflop is not None else None,
            "delta_elapsed_sec": round(d_elapsed, 4) if d_elapsed is not None else None,
            "gain_per_hour": round(gain_per_hour, 4) if gain_per_hour is not None else None,
        })

    # --- Doubling gains (only from approximate doublings) ---
    doubling_gains = [iv["gain"] for iv in intervals if iv["is_doubling"]]

    # --- Asymptote from power-law fit ---
    asymptote = None
    asymptote_alpha = None
    asymptote_r2 = None
    asymptote_reliable = False
    if len(probes) >= 4:
        asymptote, asymptote_alpha, asymptote_r2, asymptote_reliable = _fit_asymptote(steps, bpbs, asymptote_floor=asymptote_floor)

    # --- Asymptote stability (fit from last N-1 vs last N points) ---
    asymptote_stability = None
    if len(probes) >= 5:
        asym_prev, _, _, _ = _fit_asymptote(steps[:-1], bpbs[:-1], asymptote_floor=asymptote_floor)
        if asymptote is not None and asym_prev is not None:
            asymptote_stability = round(abs(asymptote - asym_prev), 4)

    # --- Headroom ---
    headroom = None
    if asymptote is not None:
        headroom = round(bpbs[-1] - asymptote, 4)

    # --- Saturation step estimate ---
    # Extrapolate where doubling gain would drop below threshold
    saturation_step = None
    if len(doubling_gains) >= 2 and doubling_gains[-1] > 0:
        # Fit exponential decay to doubling gains
        # gain(n) = g0 * r^n where n = doubling index
        recent = doubling_gains[-2:]
        if recent[0] > 0 and recent[1] > 0:
            decay_ratio = recent[1] / recent[0]
            if 0 < decay_ratio < 1:
                # How many more doublings until gain < threshold?
                doublings_remaining = math.log(saturation_threshold / recent[-1]) / math.log(decay_ratio)
                if 0 < doublings_remaining < 30:  # cap at ~1B steps
                    saturation_step = int(steps[-1] * (2 ** doublings_remaining))

    # --- Trajectory / phase ---
    last_gain = intervals[-1]["gain"] if intervals else 0
    last_rate = intervals[-1]["rate_per_1k"] if intervals else 0
    recent_doubling = doubling_gains[-1] if doubling_gains else None
    early_window = max(len(intervals) // 2, 1)
    early_gains = [float(iv["gain"]) for iv in intervals[:early_window]]
    late_gains = [float(iv["gain"]) for iv in intervals[-early_window:]]
    early_gain_median = _median(early_gains)
    late_gain_median = _median(late_gains)
    phase_acceleration = None
    if early_gain_median is not None and late_gain_median is not None:
        phase_acceleration = round(late_gain_median / max(abs(early_gain_median), 1e-6), 4)
    grok_candidate = bool(
        len(intervals) >= 4
        and early_gain_median is not None
        and late_gain_median is not None
        and early_gain_median <= max(saturation_threshold * 0.5, 0.0015)
        and late_gain_median >= max(early_gain_median * 2.5, saturation_threshold * 1.5)
        and last_gain > 0
    )

    recent_gain_per_tflop = intervals[-1]["gain_per_tflop"] if intervals else None
    total_gain_per_tflop = None
    first_tf = probes[0].get("tflops", probes[0].get("tf"))
    last_tf = probes[-1].get("tflops", probes[-1].get("tf"))
    if isinstance(first_tf, (int, float)) and isinstance(last_tf, (int, float)) and last_tf > first_tf:
        total_gain_per_tflop = round((bpbs[0] - bpbs[-1]) / (float(last_tf) - float(first_tf)), 6)

    if last_gain < -0.005:
        status = "overfitting"
    elif recent_doubling is not None and recent_doubling < saturation_threshold:
        status = "saturated"
    elif recent_doubling is not None and recent_doubling < saturation_threshold * 3:
        status = "decelerating"
    elif last_rate > 0.001:
        status = "learning"
    else:
        status = "plateau"

    if status == "overfitting":
        direction = "regressing"
        phase = "overfit"
    elif grok_candidate:
        direction = "accelerating"
        phase = "late_acceleration"
    elif len(probes) < 4:
        direction = "probing"
        phase = "bootstrap"
    elif status == "learning":
        direction = "improving"
        phase = "climbing"
    elif status == "decelerating":
        direction = "slowing"
        phase = "consolidation"
    elif status == "saturated":
        direction = "flat"
        phase = "asymptotic"
    elif status == "plateau":
        direction = "flat"
        phase = "plateau"
    else:
        direction = "unknown"
        phase = "unknown"

    return {
        "status": status,
        "direction": direction,
        "phase": phase,
        "grok_candidate": grok_candidate,
        "phase_acceleration": phase_acceleration,
        "current_bpb": round(bpbs[-1], 4),
        "best_bpb": round(min(bpbs), 4),
        "current_step": steps[-1],
        "asymptote": round(asymptote, 4) if asymptote is not None else None,
        "asymptote_alpha": round(asymptote_alpha, 4) if asymptote_alpha is not None else None,
        "asymptote_r2": round(asymptote_r2, 4) if asymptote_r2 is not None else None,
        "asymptote_reliable": asymptote_reliable,
        "asymptote_stability": asymptote_stability,
        "headroom": headroom,
        "saturation_step": saturation_step,
        "last_doubling_gain": round(recent_doubling, 4) if recent_doubling is not None else None,
        "last_rate_per_1k": round(last_rate, 4),
        "last_gain": round(last_gain, 4),
        "recent_gain_per_tflop": recent_gain_per_tflop,
        "total_gain_per_tflop": total_gain_per_tflop,
        "num_probes": len(probes),
        "intervals": intervals,
        "doubling_gains": [round(g, 4) for g in doubling_gains],
    }


def _fit_asymptote(
    steps: list[int | float],
    bpbs: list[float],
    *,
    asymptote_floor: float = 0.0,
) -> tuple[float | None, float | None, float | None, bool]:
    """Power-law fit: bpb = a * step^(-alpha) + c. Returns (c, alpha, r2, reliable).

    asymptote_floor: minimum plausible bpb (depends on tokenizer/task).
    """
    if len(steps) < 4:
        return None, None, None, False

    xs = [float(s) for s in steps]
    ys = [float(b) for b in bpbs]
    min_y = min(ys)
    max_y = max(ys)
    span = max(max_y - min_y, 1e-6)
    weights = [max(x / max(xs), 1e-6) for x in xs]

    best_sse = float("inf")
    best_asym = None
    best_alpha = None

    lo = max(min_y - span * 1.5, asymptote_floor)  # conservative: asymptote can't be more than 1.5× span below current best
    hi = min_y - span * 0.02
    if hi <= lo:
        return None, None, None, False

    for i in range(32):
        t = i / 31.0
        c = lo + (hi - lo) * (t ** 2)
        residuals = [y - c for y in ys]
        if any(r <= 1e-12 for r in residuals):
            continue
        log_x = [math.log(x) for x in xs]
        log_r = [math.log(r) for r in residuals]

        # Weighted linear regression: log_r = intercept + slope * log_x
        n = len(log_x)
        sw = sum(weights)
        sx = sum(w * x for w, x in zip(weights, log_x))
        sy = sum(w * y for w, y in zip(weights, log_r))
        sxy = sum(w * x * y for w, x, y in zip(weights, log_x, log_r))
        sxx = sum(w * x * x for w, x in zip(weights, log_x))
        denom = sw * sxx - sx * sx
        if abs(denom) < 1e-12:
            continue
        slope = (sw * sxy - sx * sy) / denom
        intercept = (sy - slope * sx) / sw
        alpha = -slope
        amplitude = math.exp(intercept)
        if alpha <= 0 or amplitude <= 0:
            continue

        fitted = [c + amplitude * (x ** (-alpha)) for x in xs]
        sse = sum(w * (y - f) ** 2 for w, y, f in zip(weights, ys, fitted))
        if sse < best_sse:
            best_sse = sse
            best_asym = c
            best_alpha = alpha

    if best_asym is None:
        return None, None, None, False

    # R2
    mean_y = sum(w * y for w, y in zip(weights, ys)) / sum(weights)
    ss_tot = sum(w * (y - mean_y) ** 2 for w, y in zip(weights, ys))
    r2 = 1 - best_sse / max(ss_tot, 1e-12) if ss_tot > 1e-12 else None

    # Reliability: the asymptote is reliable if:
    # 1. It didn't hit the search floor (best_asym > lo + 0.01)
    # 2. It's within a plausible range of the current best (> 50% of min_y)
    # 3. The curve has enough range to estimate (step range > 5x)
    step_range = max(xs) / max(min(xs), 1)
    reliable = (
        best_asym is not None
        and best_asym > lo + 0.01
        and best_asym > min_y * 0.5
        and step_range >= 5.0
    )
    return best_asym, best_alpha, r2, reliable


def format_saturation_summary(analysis: dict[str, Any]) -> str:
    """One-line human-readable summary."""
    s = analysis
    if s["status"] == "insufficient_data":
        return "insufficient data"

    parts = [s["status"].upper()]
    parts.append(f"bpb={s['current_bpb']:.3f}")

    if s["asymptote"] is not None:
        parts.append(f"asym={s['asymptote']:.3f}")
    if s["headroom"] is not None:
        parts.append(f"headroom={s['headroom']:.3f}")
    if s.get("phase"):
        parts.append(f"phase={s['phase']}")
    if s.get("direction"):
        parts.append(f"dir={s['direction']}")
    if s["last_doubling_gain"] is not None:
        parts.append(f"dbl_gain={s['last_doubling_gain']:.3f}")
    parts.append(f"rate={s['last_rate_per_1k']:.4f}/1Kstep")
    if s.get("recent_gain_per_tflop") is not None:
        parts.append(f"gain/TF={s['recent_gain_per_tflop']:.4f}")
    if s["saturation_step"] is not None:
        parts.append(f"sat_step~{s['saturation_step']:,}")

    return "  ".join(parts)
