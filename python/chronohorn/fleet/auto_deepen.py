"""Auto-deepen: when a pilot's slope is alive, determine the next horizon."""
from __future__ import annotations

from typing import Any

from chronohorn.engine.saturation import analyze_saturation

_STEP_LADDER = [1000, 5000, 10000]


def should_deepen(
    probes: list[dict[str, Any]],
    *,
    current_steps: int,
    max_steps: int = 10000,
    min_improvement: float = 0.005,
) -> bool:
    """Check if the learning curve slope is alive and worth deepening."""
    if current_steps >= max_steps:
        return False
    if len(probes) < 2:
        return False
    analysis = analyze_saturation(probes)
    if analysis.get("direction") == "regressing":
        return False
    if analysis.get("phase") in {"late_acceleration", "climbing", "consolidation"}:
        headroom = analysis.get("headroom")
        if headroom is None or headroom > min_improvement:
            return True
    p1 = probes[-2]
    p2 = probes[-1]
    b1 = p1.get("bpb") or p1.get("test_bpb") or 0
    b2 = p2.get("bpb") or p2.get("test_bpb") or 0
    return (b1 - b2) > min_improvement


def next_step_target(current_steps: int) -> int:
    """Return the next step count in the ladder."""
    for target in _STEP_LADDER:
        if target > current_steps:
            return target
    return current_steps
