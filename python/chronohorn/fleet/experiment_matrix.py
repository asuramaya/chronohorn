"""Structured experiment matrix: define sweeps as data, expand to manifest."""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any


def expand_matrix(spec: dict) -> list[dict]:
    """Expand a sweep spec into a list of experiment configs.

    spec = {
        "name_template": "v11-{embed_dim}d-{buckets}",
        "base": {"arch": "v11", "steps": 10000, "lr": 5e-3, ...},
        "sweep": {"buckets": [65536, 500000, 2000000], "embed_dim": [1, 2, 4, 16]},
    }
    """
    base = dict(spec.get("base", {}))
    sweep = spec.get("sweep", {})
    template = spec.get("name_template", "exp-{_index}")

    # Cartesian product of sweep axes
    keys = list(sweep.keys())
    values = [sweep[k] if isinstance(sweep[k], list) else [sweep[k]] for k in keys]

    experiments = []
    for i, combo in enumerate(itertools.product(*values)):
        cfg = dict(base)
        for k, v in zip(keys, combo):
            cfg[k] = v
        cfg["_index"] = i
        # Format name
        name = template.format(**cfg)
        cfg["name"] = name
        experiments.append(cfg)

    return check_experiments(experiments)


def check_experiments(experiments: list[dict]) -> list[dict]:
    """Add warnings to experiments that look problematic."""
    for exp in experiments:
        warnings = []
        buckets = exp.get("buckets_per_scale") or exp.get("buckets_per_table") or exp.get("buckets", 65536)
        embed = exp.get("embed_per_scale") or exp.get("embed_per_table") or exp.get("embed_dim", 16)
        steps = exp.get("steps", 10000)
        batch_size = exp.get("batch_size", 64)
        seq_len = exp.get("seq_len", 512)
        num_scales = exp.get("num_scales") or exp.get("num_tables", 8)
        hidden = exp.get("hidden_dim") or exp.get("hidden", 512)

        table_params = num_scales * buckets * embed
        tokens = steps * batch_size * seq_len
        visits = tokens / max(buckets, 1)

        if embed <= 2 and buckets > 100_000:
            warnings.append(f"{embed}d embeddings too sparse (v11 showed this fails)")
        if visits < 50:
            warnings.append(f"~{visits:.0f} visits/bucket (undertrained)")
        if table_params > 0 and hidden > 0:
            neural_est = hidden * hidden * 3 * 2  # rough SwiGLU estimate
            ratio = table_params / max(neural_est, 1)
            if ratio > 10:
                warnings.append(f"{ratio:.0f}:1 table/neural ratio (MLP too small)")

        total_mb = (table_params + hidden * 1024 * 2) * 6 / 8 / 1024 / 1024
        if total_mb > 16:
            warnings.append(f"~{total_mb:.0f}MB exceeds 16MB limit")

        if warnings:
            exp["_warnings"] = warnings

    return experiments


def estimate_cost(experiments: list[dict], reference_tok_s: float = 350_000) -> dict:
    """Estimate GPU-hours for a sweep based on reference throughput."""
    total_tokens = 0
    for exp in experiments:
        steps = exp.get("steps", 10000)
        batch_size = exp.get("batch_size", 64)
        seq_len = exp.get("seq_len", 512)
        total_tokens += steps * batch_size * seq_len

    total_seconds = total_tokens / reference_tok_s
    gpu_hours = total_seconds / 3600

    return {
        "total_experiments": len(experiments),
        "total_tokens": total_tokens,
        "estimated_gpu_hours": round(gpu_hours, 2),
        "estimated_wall_minutes": round(total_seconds / 60, 1),
        "reference_tok_s": reference_tok_s,
    }


def check_matrix_roi(experiments: list[dict], db=None) -> dict:
    """Check if a matrix sweep is worth running based on frontier ROI."""
    cost = estimate_cost(experiments)
    result = {**cost, "warnings": []}

    if db is not None:
        try:
            # Get frontier velocity
            frontier = db.frontier(5, trust="admissible")
            if frontier and len(frontier) >= 2:
                best = frontier[0]["bpb"]
                result["current_best"] = best
                # Expected improvement from this sweep: rough estimate based on axis variance
                result["warnings"].append(
                    f"Current best: {best:.4f} bpb. "
                    f"This sweep costs ~{cost['estimated_gpu_hours']:.1f} GPU-hours. "
                    f"At current frontier velocity, expected gain: ~{cost['estimated_gpu_hours'] * 0.02:.3f} bpb."
                )
        except Exception:
            pass

    return result


def matrix_to_commands(
    experiments: list[dict], script: str = "scripts/train_polyhash.py"
) -> list[dict]:
    """Convert experiment configs to training commands."""
    results = []
    for exp in experiments:
        exp = dict(exp)  # don't mutate caller's dict
        name = exp.pop("name", f"exp-{len(results)}")
        # Build command line args
        args: list[str] = []
        skip_keys = {"_index", "_warnings", "name_template"}
        for k, v in exp.items():
            if k in skip_keys:
                continue
            flag = f"--{k.replace('_', '-')}"
            if isinstance(v, bool):
                if v:
                    args.append(flag)
            else:
                args.extend([flag, str(v)])

        cmd = f"python3 {script} {' '.join(args)} --json /results/{name}.json"
        results.append({"name": name, "command": cmd, **exp})

    return results


def write_manifest(experiments: list[dict], output_path: Path) -> int:
    """Write experiments as a JSONL manifest."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for exp in experiments:
            f.write(json.dumps(exp, sort_keys=True) + "\n")
    return len(experiments)
