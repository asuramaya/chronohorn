"""Terminal-friendly output for chronohorn: ASCII plots and formatted tables."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def ascii_sparkline(values: Sequence[float], width: int = 20) -> str:
    """Single-line sparkline using Unicode block characters."""
    if not values:
        return ""
    blocks = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
    mn, mx = min(values), max(values)
    rng = mx - mn if mx > mn else 1
    # Subsample if too many values
    if len(values) > width:
        step = len(values) / width
        values = [values[int(i * step)] for i in range(width)]
    return "".join(blocks[min(8, int((v - mn) / rng * 8))] for v in values)


def ascii_learning_curve(points: list[dict], width: int = 60, height: int = 12) -> str:
    """ASCII scatter plot of step vs bpb."""
    if not points or len(points) < 2:
        return "  (no curve data)"

    steps = [p["step"] for p in points if p.get("bpb")]
    bpbs = [p["bpb"] for p in points if p.get("bpb")]
    if len(steps) < 2:
        return "  (insufficient data)"

    import math
    s_min, s_max = min(steps), max(steps)
    b_min, b_max = min(bpbs), max(bpbs)
    b_pad = (b_max - b_min) * 0.05
    b_min -= b_pad
    b_max += b_pad

    # Build grid
    grid = [[" "] * width for _ in range(height)]

    for s, b in zip(steps, bpbs):
        # Log scale for steps
        if s_max > s_min:
            x = int((math.log(max(s, 1)) - math.log(max(s_min, 1))) / (math.log(max(s_max, 1)) - math.log(max(s_min, 1))) * (width - 1))
        else:
            x = 0
        if b_max > b_min:
            y = int((1 - (b - b_min) / (b_max - b_min)) * (height - 1))
        else:
            y = 0
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        grid[y][x] = "\u25cf"

    # Connect dots
    prev = None
    sorted_pts = sorted(zip(steps, bpbs))
    for s, b in sorted_pts:
        if s_max > s_min:
            x = int((math.log(max(s, 1)) - math.log(max(s_min, 1))) / (math.log(max(s_max, 1)) - math.log(max(s_min, 1))) * (width - 1))
        else:
            x = 0
        if b_max > b_min:
            y = int((1 - (b - b_min) / (b_max - b_min)) * (height - 1))
        else:
            y = 0
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        if prev and prev[0] < x:
            for xi in range(prev[0] + 1, x):
                frac = (xi - prev[0]) / (x - prev[0])
                yi = int(prev[1] + frac * (y - prev[1]))
                yi = max(0, min(height - 1, yi))
                if grid[yi][xi] == " ":
                    grid[yi][xi] = "\u00b7"
        prev = (x, y)

    # Render with Y axis labels
    lines = []
    for i, row in enumerate(grid):
        b_val = b_max - i * (b_max - b_min) / (height - 1)
        label = f"{b_val:6.3f}" if i % 3 == 0 else "      "
        lines.append(f"{label} \u2502{''.join(row)}\u2502")

    # X axis
    h_line = "\u2500" * width
    lines.append(f"       \u2514{h_line}\u2518")
    x_labels = f"        {s_min:<{width // 2}}{s_max:>{width // 2}}"
    lines.append(x_labels)

    return "\n".join(lines)


def ascii_frontier_table(board: list[dict], top_k: int = 15) -> str:
    """Formatted frontier leaderboard."""
    if not board:
        return "  (no results)"

    lines = []
    lines.append(f"{'#':>3s}  {'name':30s}  {'bpb':>7s}  {'trust':>11s}  {'MB':>5s}  {'steps':>7s}  {'slope':>6s}")
    lines.append("-" * 84)

    for i, r in enumerate(board[:top_k]):
        sl = f"{r.get('slope', 0):.3f}" if r.get("slope") else "-"
        mb = f"{r.get('int6_mb', 0):5.1f}" if r.get("int6_mb") else f"{(r.get('params', 0) or 0) * 6 / 8 / 1024 / 1024:5.1f}"
        steps = f"{r.get('steps', 0):>7,d}" if r.get("steps") else "-"
        trust = str(r.get("trust_state") or "-")[:11]

        lines.append(f"{i + 1:3d}  {r.get('name', '?'):30s}  {r.get('bpb', 0):7.4f}  {trust:>11s}  {mb}  {steps}  {sl:>6s}")

    return "\n".join(lines)


def ascii_ablation_table(board: list[dict], top_k: int = 10) -> str:
    """Formatted rapid-ablation board."""
    if not board:
        return "  (no rapid-ablation candidates)"

    lines = []
    lines.append(
        f"{'#':>3s}  {'action':18s}  {'name':28s}  {'bpb':>7s}  {'phase':16s}  {'dir':11s}  {'lanes':9s}  {'trust':11s}"
    )
    lines.append("-" * 122)

    for i, row in enumerate(board[:top_k]):
        scales = len(row.get("tested_scales") or [])
        contexts = len(row.get("tested_seq_lens") or [])
        lanes = f"s{scales}/c{contexts}"
        lines.append(
            f"{i + 1:3d}  "
            f"{str(row.get('next_action') or '-')[:18]:18s}  "
            f"{str(row.get('name') or '?')[:28]:28s}  "
            f"{float(row.get('bpb') or 0):7.4f}  "
            f"{str(row.get('trajectory_phase') or '-')[:16]:16s}  "
            f"{str(row.get('trajectory_direction') or '-')[:11]:11s}  "
            f"{lanes:9s}  "
            f"{str(row.get('trust_state') or '-')[:11]:11s}"
        )

    return "\n".join(lines)


def ascii_compare(runs: list[dict]) -> str:
    """Side-by-side comparison of multiple learning curves."""
    if not runs:
        return "  (no runs to compare)"

    markers = "\u25cf\u25cb\u25c6\u25c7\u25b2\u25b3\u25a0\u25a1"
    lines = []

    # Header
    lines.append("  Comparison:")
    for i, run in enumerate(runs):
        m = markers[i % len(markers)]
        pts = run.get("points", [])
        last_bpb = pts[-1]["bpb"] if pts else "?"
        lines.append(f"    {m} {run.get('name', '?'):30s}  final bpb={last_bpb}")

    # Align by step and show deltas
    all_steps = sorted(set(
        p["step"] for run in runs for p in run.get("points", [])
    ))

    if len(runs) == 2 and all_steps:
        lines.append(f"\n  {'step':>7s}")
        for run in runs:
            lines[-1] += f"  {run.get('name', '?')[:12]:>12s}"
        lines[-1] += "    delta"
        lines.append("  " + "-" * 50)

        for step in all_steps:
            vals = []
            for run in runs:
                pts = {p["step"]: p["bpb"] for p in run.get("points", [])}
                vals.append(pts.get(step))

            if all(v is not None for v in vals):
                delta = vals[1] - vals[0]
                lines.append(f"  {step:>7,d}  {vals[0]:>12.4f}  {vals[1]:>12.4f}  {delta:>+8.4f}")

    return "\n".join(lines)


def ascii_status(summary: dict, board: list[dict] | None = None) -> str:
    """Compact status summary for terminal."""
    lines = []
    lines.append(f"chronohorn: {summary.get('result_count', 0)} results")
    best = summary.get("best_bpb")
    if best:
        lines.append(f"  best: {best:.4f} bpb  (gap to 1.119: {best - 1.119:+.3f})")
    provisional_best = summary.get("provisional_best_bpb")
    best_any = summary.get("best_bpb_any")
    if provisional_best is not None and provisional_best != best:
        lines.append(f"  provisional best: {provisional_best:.4f} bpb")
    elif best_any is not None and best_any != best:
        lines.append(f"  raw best: {best_any:.4f} bpb")
    fams = summary.get("families", {})
    if fams:
        fam_str = ", ".join(f"{k}={v}" for k, v in fams.items())
        lines.append(f"  families: {fam_str}")
    populations = summary.get("populations", {})
    if populations:
        controlled = populations.get("controlled", {})
        imported = populations.get("imported_archive", {})
        unknown = populations.get("unknown", {})
        lines.append(
            "  controlled/imported/unknown legal: "
            f"{controlled.get('legal_count', 0)}/"
            f"{imported.get('legal_count', 0)}/"
            f"{unknown.get('legal_count', 0)}"
        )
    if summary.get("illegal_result_count") is not None:
        lines.append(f"  illegal: {summary.get('illegal_result_count', 0)}")
    trust = summary.get("trust", {})
    trust_counts = trust.get("counts", {})
    if trust_counts:
        lines.append(
            "  trust: "
            f"admissible={trust_counts.get('admissible', 0)}, "
            f"provisional={trust_counts.get('provisional', 0)}, "
            f"quarantined={trust_counts.get('quarantined', 0)}"
        )

    if board:
        lines.append(f"\n{ascii_frontier_table(board, top_k=10)}")

    return "\n".join(lines)


def ascii_evidence_matrix(
    payload: dict[str, Any],
    *,
    top_k: int = 20,
    top_manifests: int = 12,
) -> str:
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    manifests = list(payload.get("manifests") or []) if isinstance(payload, dict) else []
    rows = list(payload.get("rows") or []) if isinstance(payload, dict) else []

    lines: list[str] = []
    lines.append(
        "evidence: "
        f"rows={summary.get('row_count', 0)} "
        f"manifests={summary.get('manifest_count', 0)} "
        f"population={summary.get('population', 'all')} "
        f"legality={summary.get('legality', 'all')} "
        f"trust={summary.get('trust', 'all')}"
    )

    phase_counts = summary.get("phase_counts", {})
    if phase_counts:
        lines.append(
            "  phases: "
            f"intent={phase_counts.get('intent', 0)} "
            f"execution={phase_counts.get('execution', 0)} "
            f"observation={phase_counts.get('observation', 0)} "
            f"interpretation={phase_counts.get('interpretation', 0)}"
        )
    trust_counts = summary.get("trust_counts", {})
    if trust_counts:
        lines.append(
            "  admissibility: "
            f"admissible={trust_counts.get('admissible', 0)} "
            f"provisional={trust_counts.get('provisional', 0)} "
            f"quarantined={trust_counts.get('quarantined', 0)} "
            f"unobserved={trust_counts.get('unobserved', 0)}"
        )
    role_counts = summary.get("surface_role_counts", {})
    if role_counts:
        lines.append(
            "  manifest roles: "
            f"live={role_counts.get('live_control_input', 0)} "
            f"mixed={role_counts.get('live_mixed', 0)} "
            f"archive={role_counts.get('evidence_archive', 0)} "
            f"imports={role_counts.get('archive_import', 0)} "
            f"unknown={role_counts.get('unknown_provenance', 0)}"
        )

    if manifests:
        lines.append("")
        lines.append(f"{'role':16s}  {'manifest':32s}  {'jobs':>4s}  {'pend':>4s}  {'run':>3s}  {'obs':>4s}  {'adm':>3s}  {'hint':14s}")
        lines.append("-" * 94)
        for row in manifests[: max(0, top_manifests)]:
            lines.append(
                f"{str(row.get('surface_role') or '')[:16]:16s}  "
                f"{str(row.get('manifest') or '')[:32]:32s}  "
                f"{int(row.get('jobs') or 0):4d}  "
                f"{int(row.get('pending') or 0):4d}  "
                f"{int(row.get('running') or 0):3d}  "
                f"{int(row.get('observed') or 0):4d}  "
                f"{int(row.get('admissible') or 0):3d}  "
                f"{str(row.get('retention_hint') or '')[:14]:14s}"
            )

    if rows:
        lines.append("")
        lines.append(f"{'name':34s}  {'family':12s}  {'phase':14s}  {'trust':11s}  {'metric':20s}  {'manifest':24s}")
        lines.append("-" * 126)
        for row in rows[: max(0, top_k)]:
            lines.append(
                f"{str(row.get('name') or '')[:34]:34s}  "
                f"{str(row.get('family') or '')[:12]:12s}  "
                f"{str(row.get('phase') or '')[:14]:14s}  "
                f"{str(row.get('admissibility') or '')[:11]:11s}  "
                f"{str(row.get('metric_state') or row.get('observation_state') or '')[:20]:20s}  "
                f"{str(row.get('manifest') or '__unknown__')[:24]:24s}"
            )
    return "\n".join(lines)
