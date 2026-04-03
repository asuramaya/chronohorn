"""Experiment advisor: suggests next experiments based on frontier state."""
from __future__ import annotations

from typing import Any


def suggest_next(db) -> list[dict]:
    """Analyze the current state and suggest the highest-value next experiments."""
    suggestions = []

    # Get current state
    frontier = db.frontier(10, trust="admissible")
    if not frontier:
        provisional_frontier = db.frontier(10, trust="provisional")
        if provisional_frontier:
            provisional_best = provisional_frontier[0]
            return [{
                "action": "replicate provisional leader before more exploration",
                "reason": (
                    f"No admissible frontier exists yet. Current provisional leader is {provisional_best['name']} "
                    f"at {provisional_best['bpb']:.4f} bpb with metric_state={provisional_best.get('metric_state')}."
                ),
                "priority": "high",
                "type": "stabilize_evidence",
            }]
        return [{"action": "run any experiment", "reason": "no results yet", "priority": "high"}]

    best = frontier[0]
    best_bpb = best["bpb"]

    # Check frontier velocity
    velocity = db.frontier_velocity(trust="admissible")
    v = velocity.get("velocity_bpb_per_hour", 0)
    trend = velocity.get("trend", "unknown")

    # Check axis exhaustion
    from chronohorn.engine.axis_analysis import analyze_axes
    all_results = db.analysis_rows(max_bpb=3.0, controlled_only=True, trust="admissible")

    axes = analyze_axes(all_results)
    alive_axes = [a for a in axes if a.get("status") == "alive"]
    diminishing_axes = [a for a in axes if a.get("status") == "diminishing"]
    exhausted_axes = [a for a in axes if a.get("status") in ("exhausted", "non_monotonic")]

    # Suggestion 1: If velocity is low, suggest convergence training
    if v < 0.02 and trend != "accelerating":
        suggestions.append({
            "action": f"Train {best['name']} to convergence (200K-500K steps)",
            "reason": f"Frontier velocity is {v:.4f} bpb/hr and decelerating. Architecture search is converged.",
            "expected_gain": "0.04-0.08 bpb (power-law extrapolation)",
            "priority": "high",
            "type": "convergence",
        })

    # Suggestion 2: If an axis is alive, suggest pushing it
    for axis in alive_axes[:2]:
        vals = axis.get("values", [])
        last_val = vals[-1] if vals else "?"
        suggestions.append({
            "action": f"Push {axis['axis']} beyond {last_val}",
            "reason": f"Axis '{axis['axis']}' still showing gains (last: {axis.get('last_gain', 0):+.4f} bpb)",
            "expected_gain": f"~{abs(axis.get('last_gain', 0)) * 0.5:.4f} bpb (half of last gain)",
            "priority": "medium",
            "type": "explore_axis",
        })

    # Suggestion 3: Check for untested axes
    tested_axes = set(a["axis"] for a in axes if a.get("points", 0) >= 2)
    all_possible = {
        "hidden_dim", "num_layers", "scan_dim", "buckets", "embed_dim", "steps", "seq_len",
        "learning_rate", "weight_decay", "warmup_steps", "batch_size", "conv_kernel",
    }
    untested = all_possible - tested_axes
    if untested:
        for axis in sorted(untested):
            suggestions.append({
                "action": f"Vary {axis} (never tested)",
                "reason": f"Axis '{axis}' has never been varied. Could contain free gains.",
                "expected_gain": "unknown",
                "priority": "medium" if axis in ("learning_rate", "weight_decay") else "low",
                "type": "untested_axis",
            })

    # Suggestion 4: If all axes exhausted, suggest new architecture class
    if not alive_axes and len(exhausted_axes) >= 3:
        suggestions.append({
            "action": "Consider a new architecture class (e.g., add attention layer)",
            "reason": f"All {len(exhausted_axes)} tested axes are exhausted. Current architecture ceiling reached.",
            "expected_gain": "0.10-0.15 bpb (theoretical)",
            "priority": "high",
            "type": "new_architecture",
        })

    # Suggestion 5: LR tuning if never done
    unique_lrs = {
        row["config"].get("learning_rate")
        for row in all_results
        if row.get("config", {}).get("learning_rate") not in (None, "")
    }
    if len(unique_lrs) <= 1:
        suggestions.append({
            "action": "Tune learning rate (only one LR tested so far)",
            "reason": "All experiments used the same LR. A 3-point sweep (0.5x, 1x, 2x) could find free gains.",
            "expected_gain": "0.01-0.05 bpb",
            "priority": "high",
            "type": "hyperparameter",
        })

    # Sort by priority
    priority_order = {"high": 0, "medium": 1, "low": 2}
    suggestions.sort(key=lambda x: priority_order.get(x.get("priority", "low"), 3))

    return suggestions


def format_suggestions(suggestions: list[dict]) -> str:
    """Format suggestions as text."""
    if not suggestions:
        return "No suggestions -- run any experiment."

    lines = ["Suggested next experiments:"]
    for i, s in enumerate(suggestions):
        icon = {"high": "★", "medium": "→", "low": "·"}.get(s.get("priority"), "?")
        lines.append(f"  {icon} {s['action']}")
        lines.append(f"    reason: {s['reason']}")
        lines.append(f"    expected: {s.get('expected_gain', 'unknown')}")

    # NOT recommended
    lines.append("")
    lines.append("  NOT recommended:")
    lines.append("    - Repeating exhausted axes without changing something else")
    lines.append("    - Short runs (3K steps) when convergence training is needed")

    return "\n".join(lines)


def architecture_boundary(db) -> dict:
    """Detect if we've hit the architecture class ceiling."""
    # Get asymptote from best run's forecast
    best = db.frontier(1, trust="admissible")
    if not best:
        provisional = db.frontier(1, trust="provisional")
        if provisional:
            return {
                "status": "no_admissible_data",
                "message": (
                    f"No admissible run has enough trust to estimate an architecture boundary. "
                    f"Best provisional run is {provisional[0]['name']} at {provisional[0]['bpb']:.4f} bpb."
                ),
            }
        return {"status": "no_data"}

    best_name = best[0]["name"]

    # Try best run's forecast first
    forecast = db.query(
        "SELECT asymptote, asymptote_reliable, forecast_bpb FROM forecasts WHERE name = ?",
        (best_name,),
    )
    admissible_names = {
        name
        for name, row in db.result_trust_index(population="controlled", legality="legal").items()
        if row.get("trust_state") == "admissible"
    }

    asymptote = None
    if forecast and forecast[0].get("asymptote_reliable"):
        asymptote = forecast[0].get("asymptote")
    else:
        # Fall back to best reliable asymptote from any run
        reliable = db.query("""
            SELECT f.name, f.asymptote FROM forecasts f
            JOIN results r ON f.name = r.name
            WHERE f.asymptote_reliable = 1 AND NOT r.illegal
            ORDER BY f.asymptote ASC LIMIT 20
        """)
        reliable = [row for row in reliable if row.get("name") in admissible_names] if reliable else []
        asymptote = reliable[0]["asymptote"] if reliable else None

    if asymptote is None and forecast:
        # Last resort: use forecast_bpb even if unreliable
        asymptote = forecast[0].get("asymptote") or forecast[0].get("forecast_bpb")

    leader = 1.119  # competition leader

    result = {
        "best_name": best_name,
        "best_bpb": best[0]["bpb"],
        "asymptote": asymptote,
        "leader": leader,
        "gap_to_leader": round(best[0]["bpb"] - leader, 3),
    }

    if asymptote:
        result["asymptote_gap"] = round(asymptote - leader, 3)
        if asymptote - leader > 0.20:
            result["status"] = "architectural_ceiling"
            result["message"] = (
                f"Asymptote estimate ({asymptote:.3f}) is {asymptote - leader:.3f} bpb above leader ({leader}). "
                f"This gap is likely ARCHITECTURAL -- more training won't close it. "
                f"Consider: attention layer, different architecture class, or distillation."
            )
        elif asymptote - leader > 0.05:
            result["status"] = "approaching_ceiling"
            result["message"] = f"Asymptote ({asymptote:.3f}) is close to leader. Training to convergence may close most of the gap."
        else:
            result["status"] = "competitive"
            result["message"] = f"Asymptote ({asymptote:.3f}) is within striking distance of leader."
    else:
        result["status"] = "unknown"
        result["message"] = "No forecast available for best run."

    return result
