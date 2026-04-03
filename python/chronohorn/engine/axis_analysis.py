"""Detect which experimental axes are exhausted vs still productive."""
from __future__ import annotations
from typing import Any


def analyze_axes(results: list[dict]) -> list[dict]:
    """Analyze diminishing returns across experimental axes.

    results: list of dicts with at least {name, bpb, config} where config has architecture fields.
    Returns: list of {axis, values, gains, status, headroom_estimate}
    """
    # Extract axis values from configs
    axes = {}  # axis_name -> [(value, bpb)]

    for r in results:
        cfg = r.get("config", {})
        if isinstance(cfg, str):
            import json
            try: cfg = json.loads(cfg)
            except Exception: continue

        bpb = r.get("bpb")
        if not bpb or bpb > 3:
            continue

        # Known axes
        axis_fields = {
            "hidden_dim": cfg.get("hidden_dim"),
            "num_layers": cfg.get("num_layers"),
            "scan_dim": cfg.get("scan_dim"),
            "buckets": cfg.get("buckets_per_scale") or cfg.get("buckets_per_table"),
            "embed_dim": cfg.get("embed_per_scale") or cfg.get("embed_per_table"),
            "steps": r.get("steps"),
            "seq_len": cfg.get("seq_len"),
        }

        for axis_name, val in axis_fields.items():
            if val is not None and isinstance(val, (int, float)) and val > 0:
                axes.setdefault(axis_name, []).append((val, bpb))

    # Analyze each axis
    analysis = []
    for axis_name, points in axes.items():
        if len(points) < 2:
            analysis.append({"axis": axis_name, "points": len(points), "status": "insufficient_data"})
            continue

        # Group by value, take best bpb per value
        by_val: dict[float, float] = {}
        for val, bpb in points:
            if val not in by_val or bpb < by_val[val]:
                by_val[val] = bpb

        sorted_vals = sorted(by_val.items())
        if len(sorted_vals) < 2:
            analysis.append({"axis": axis_name, "points": len(sorted_vals), "status": "single_value"})
            continue

        # Compute gains between consecutive points
        gains = []
        for i in range(1, len(sorted_vals)):
            v_prev, bpb_prev = sorted_vals[i-1]
            v_curr, bpb_curr = sorted_vals[i]
            gain = bpb_prev - bpb_curr  # positive = improvement
            gains.append({"from": v_prev, "to": v_curr, "gain": round(gain, 4)})

        # Determine status
        positive_gains = [g for g in gains if g["gain"] > 0.005]
        negative_gains = [g for g in gains if g["gain"] < -0.005]

        if not positive_gains:
            status = "exhausted"
        elif len(gains) >= 2 and gains[-1]["gain"] < gains[0]["gain"] * 0.3:
            status = "diminishing"
        elif negative_gains:
            status = "non_monotonic"
        else:
            status = "alive"

        last_gain = gains[-1]["gain"] if gains else 0

        analysis.append({
            "axis": axis_name,
            "points": len(sorted_vals),
            "values": [v for v, _ in sorted_vals],
            "best_bpb": min(by_val.values()),
            "gains": gains,
            "last_gain": last_gain,
            "status": status,
        })

    # Sort: alive first, then by last_gain
    analysis.sort(key=lambda x: (0 if x.get("status") == "alive" else 1 if x.get("status") == "diminishing" else 2, -(x.get("last_gain") or 0)))

    return analysis
