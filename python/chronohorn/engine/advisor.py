"""Experiment advisor: suggests next experiments based on frontier state."""
from __future__ import annotations


def suggest_next(db) -> list[dict]:
    """Analyze the current state and suggest the highest-value next experiments."""
    suggestions = []
    ablation_board = db.ablation_board(5, trust="all")

    def _ablation_suggestion(row: dict) -> dict | None:
        action = str(row.get("next_action") or "")
        name = str(row.get("name") or "candidate")
        phase = str(row.get("trajectory_phase") or "unknown")
        direction = str(row.get("trajectory_direction") or "unknown")
        headroom = row.get("headroom")
        artifact_budget_mb = row.get("artifact_budget_mb")
        artifact_budget_text = (
            f"{float(artifact_budget_mb):.0f}"
            if isinstance(artifact_budget_mb, (int, float))
            else "16"
        )
        compute_budget_tflops = row.get("compute_budget_tflops")
        compute_budget_text = (
            f"{float(compute_budget_tflops):.0f}"
            if isinstance(compute_budget_tflops, (int, float))
            else "unknown"
        )
        if action == "shrink_under_budget":
            int6_mb = row.get("int6_mb")
            size_text = f"{float(int6_mb):.2f}" if isinstance(int6_mb, (int, float)) else "unknown"
            return {
                "action": f"Shrink {name} under the {artifact_budget_text} MB artifact limit",
                "reason": (
                    f"It still looks {direction}/{phase}, but artifact size is {size_text} MB. "
                    f"Do not promote anything that misses the artifact budget."
                ),
                "expected_gain": "recover a valid scaling candidate without leaving the budget",
                "priority": "high",
                "type": "constraint_gate",
            }
        if action == "reduce_compute":
            total_tflops = row.get("tflops")
            tf_text = f"{float(total_tflops):.1f}" if isinstance(total_tflops, (int, float)) else "unknown"
            return {
                "action": f"Reduce {name} under the {compute_budget_text} TF training budget",
                "reason": (
                    f"It still looks {direction}/{phase}, but train compute is {tf_text} TF. "
                    "Keep the mechanism and cut the compute before calling it scalable."
                ),
                "expected_gain": "recover a stronger bpb-per-TF candidate",
                "priority": "high",
                "type": "constraint_gate",
            }
        if action == "replace_architecture":
            return {
                "action": f"Stop iterating {name} as a frontier candidate",
                "reason": (
                    "It violates the constant-state O(n) search constraint. "
                    "Spend compute on architectures that keep recurrent inference bounded."
                ),
                "expected_gain": "free queue capacity for scaling candidates",
                "priority": "high",
                "type": "constraint_gate",
            }
        if action == "test_next_scale":
            return {
                "action": f"Run {name} at the next scale before promotion",
                "reason": (
                    f"Trajectory is {direction}/{phase} with "
                    f"{len(row.get('tested_scales') or [])} tested scale(s). "
                    f"Use the cheap lanes to verify that the win survives scale."
                ),
                "expected_gain": "screen for scale survival",
                "priority": "high",
                "type": "rapid_ablation",
            }
        if action == "test_longer_context":
            return {
                "action": f"Run {name} at longer context before promotion",
                "reason": (
                    f"Trajectory is {direction}/{phase} but only "
                    f"{len(row.get('tested_seq_lens') or [])} context lane(s) are covered. "
                    f"O(n) wins need to survive context growth."
                ),
                "expected_gain": "screen for context retention",
                "priority": "high",
                "type": "rapid_ablation",
            }
        if action == "promote_full_data":
            headroom_text = f"{headroom:.3f}" if isinstance(headroom, (int, float)) else "unknown"
            return {
                "action": f"Promote {name} to a full-data run",
                "reason": (
                    f"It cleared the O(n), artifact-budget, scale, and context gates and still "
                    f"shows {direction}/{phase} with headroom={headroom_text}."
                ),
                "expected_gain": "validate that the mechanism survives real data scale",
                "priority": "high",
                "type": "promotion_gate",
            }
        if action == "replicate":
            return {
                "action": f"Replicate {name}",
                "reason": (
                    f"It passed the screening ladder but still has only "
                    f"{int(row.get('replicate_count') or 0)} trusted seed(s)."
                ),
                "expected_gain": "convert provisional evidence into admissible evidence",
                "priority": "high",
                "type": "stabilize_evidence",
            }
        if action == "deepen_same_lane":
            return {
                "action": f"Deepen {name} within the current lane",
                "reason": (
                    f"Trajectory is {direction}/{phase}, but the curve is still too immature "
                    f"for lane-crossing decisions."
                ),
                "expected_gain": "stabilize the asymptote and phase estimate",
                "priority": "medium",
                "type": "deepen",
            }
        return None

    # Get current state
    frontier = db.frontier(10, trust="admissible")
    if not frontier:
        ablation_suggestions = [
            suggestion
            for suggestion in (_ablation_suggestion(row) for row in ablation_board)
            if suggestion is not None
        ]
        if ablation_suggestions:
            return ablation_suggestions
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

    for row in ablation_board[:3]:
        suggestion = _ablation_suggestion(row)
        if suggestion is not None:
            suggestions.append(suggestion)

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
