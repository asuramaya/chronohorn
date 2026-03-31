from __future__ import annotations

import math
from typing import Any


PROBE_PLAN_VERSION = "chronohorn_probe_plan_v1"
PROBE_POLICY_CHOICES = ("off", "explicit", "adaptive")
PROBE_TIER_CHOICES = ("micro", "standard", "promotion")


def parse_probe_steps(raw: str | None, max_step: int) -> list[int]:
    if not raw:
        return []
    values: list[int] = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        step = int(token)
        if step <= 0:
            raise ValueError(f"Probe step must be positive, got {step}.")
        if step > max_step:
            continue
        values.append(step)
    return sorted(set(values))


def _positive_int(value: int | None, fallback: int) -> int:
    if value is None:
        return int(fallback)
    return max(int(value), 1)


def _positive_float(value: float | None, fallback: float) -> float:
    if value is None:
        return float(fallback)
    numeric = float(value)
    if numeric <= 1.0:
        raise ValueError(f"Probe geometric ratio must be > 1.0, got {numeric}.")
    return numeric


def _derive_eval_batches(
    *,
    default_eval_batches: int | None,
    standard_eval_batches: int | None,
    micro_eval_batches: int | None,
    promotion_eval_batches: int | None,
    final_eval_batches: int | None,
) -> dict[str, int]:
    standard = _positive_int(standard_eval_batches, _positive_int(default_eval_batches, 8))
    micro_default = max(1, min(standard, int(math.ceil(standard / 4.0))))
    promotion_soft_cap = max(standard, standard * 2)
    if final_eval_batches is not None and final_eval_batches > 0:
        promotion_soft_cap = min(max(int(final_eval_batches), standard), promotion_soft_cap)
    return {
        "micro": _positive_int(micro_eval_batches, micro_default),
        "standard": standard,
        "promotion": _positive_int(promotion_eval_batches, promotion_soft_cap),
    }


def _build_geometric_steps(
    *,
    max_step: int,
    geometric_start_step: int,
    geometric_ratio: float,
) -> list[int]:
    max_step = max(int(max_step), 0)
    if max_step <= 0:
        return []
    current = max(1, int(geometric_start_step))
    steps: list[int] = []
    while current < max_step:
        steps.append(current)
        next_step = int(math.ceil(float(current) * float(geometric_ratio)))
        if next_step <= current:
            next_step = current + 1
        current = next_step
    if max_step not in steps:
        steps.append(max_step)
    return sorted(set(step for step in steps if step > 0 and step <= max_step))


def _classify_adaptive_entries(
    steps: list[int],
    *,
    micro_cutoff_step: int,
    promotion_count: int,
    eval_batches: dict[str, int],
) -> list[dict[str, Any]]:
    if not steps:
        return []
    promotion_count = max(int(promotion_count), 1)
    promotion_steps = set(steps[-promotion_count:])
    entries: list[dict[str, Any]] = []
    for step in steps:
        if step in promotion_steps:
            tier = "promotion"
        elif step <= int(micro_cutoff_step):
            tier = "micro"
        else:
            tier = "standard"
        entries.append(
            {
                "step": int(step),
                "tier": tier,
                "eval_batches": int(eval_batches[tier]),
            }
        )
    return entries


def _explicit_entries(steps: list[int], *, eval_batches: dict[str, int]) -> list[dict[str, Any]]:
    return [
        {
            "step": int(step),
            "tier": "standard",
            "eval_batches": int(eval_batches["standard"]),
        }
        for step in steps
    ]


def resolve_probe_plan(
    *,
    max_step: int,
    raw_steps: str | None = None,
    policy: str = "adaptive",
    default_eval_batches: int | None = None,
    standard_eval_batches: int | None = None,
    micro_eval_batches: int | None = None,
    promotion_eval_batches: int | None = None,
    final_eval_batches: int | None = None,
    geometric_start_step: int = 50,
    geometric_ratio: float = 2.0,
    micro_cutoff_step: int = 800,
    promotion_count: int = 1,
) -> dict[str, Any]:
    resolved_policy = str(policy or "adaptive").strip().lower()
    if resolved_policy not in PROBE_POLICY_CHOICES:
        raise ValueError(
            f"Unsupported probe policy {policy!r}; choose from {', '.join(PROBE_POLICY_CHOICES)}."
        )
    eval_batches = _derive_eval_batches(
        default_eval_batches=default_eval_batches,
        standard_eval_batches=standard_eval_batches,
        micro_eval_batches=micro_eval_batches,
        promotion_eval_batches=promotion_eval_batches,
        final_eval_batches=final_eval_batches,
    )
    if resolved_policy == "off":
        return {
            "version": PROBE_PLAN_VERSION,
            "policy": "off",
            "max_step": int(max_step),
            "steps": [],
            "entries": [],
            "eval_batches": eval_batches,
            "geometry": None,
            "raw_steps": raw_steps,
        }

    explicit_steps = parse_probe_steps(raw_steps, max_step)
    if explicit_steps or resolved_policy == "explicit":
        return {
            "version": PROBE_PLAN_VERSION,
            "policy": "explicit",
            "max_step": int(max_step),
            "steps": explicit_steps,
            "entries": _explicit_entries(explicit_steps, eval_batches=eval_batches),
            "eval_batches": eval_batches,
            "geometry": None,
            "raw_steps": raw_steps,
        }

    geometric_ratio = _positive_float(geometric_ratio, 2.0)
    steps = _build_geometric_steps(
        max_step=int(max_step),
        geometric_start_step=_positive_int(geometric_start_step, 50),
        geometric_ratio=geometric_ratio,
    )
    entries = _classify_adaptive_entries(
        steps,
        micro_cutoff_step=_positive_int(micro_cutoff_step, 800),
        promotion_count=_positive_int(promotion_count, 1),
        eval_batches=eval_batches,
    )
    return {
        "version": PROBE_PLAN_VERSION,
        "policy": "adaptive",
        "max_step": int(max_step),
        "steps": [int(entry["step"]) for entry in entries],
        "entries": entries,
        "eval_batches": eval_batches,
        "geometry": {
            "start_step": _positive_int(geometric_start_step, 50),
            "ratio": float(geometric_ratio),
            "micro_cutoff_step": _positive_int(micro_cutoff_step, 800),
            "promotion_count": _positive_int(promotion_count, 1),
        },
        "raw_steps": raw_steps,
    }


def probe_entry_by_step(plan: dict[str, Any] | None, step: int) -> dict[str, Any] | None:
    if not isinstance(plan, dict):
        return None
    entries = plan.get("entries")
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if isinstance(entry, dict) and int(entry.get("step", -1)) == int(step):
            return entry
    return None


def project_future_probe_entries(
    plan: dict[str, Any] | None,
    *,
    after_step: int,
    max_step: int,
) -> list[dict[str, Any]]:
    if not isinstance(plan, dict):
        return []
    if int(max_step) <= int(after_step):
        return []
    policy = str(plan.get("policy") or "").strip().lower()
    if policy == "off":
        return []
    if policy == "explicit":
        entries = plan.get("entries")
        if not isinstance(entries, list):
            return []
        return [
            dict(entry)
            for entry in entries
            if isinstance(entry, dict)
            and int(entry.get("step", -1)) > int(after_step)
            and int(entry.get("step", -1)) <= int(max_step)
        ]

    eval_batches = plan.get("eval_batches")
    geometry = plan.get("geometry")
    if not isinstance(eval_batches, dict) or not isinstance(geometry, dict):
        return []
    expanded = resolve_probe_plan(
        max_step=int(max_step),
        policy="adaptive",
        default_eval_batches=int(eval_batches.get("standard") or 8),
        standard_eval_batches=int(eval_batches.get("standard") or 8),
        micro_eval_batches=int(eval_batches.get("micro") or max(1, int(eval_batches.get("standard") or 8))),
        promotion_eval_batches=int(eval_batches.get("promotion") or int(eval_batches.get("standard") or 8)),
        geometric_start_step=int(geometry.get("start_step") or 50),
        geometric_ratio=float(geometry.get("ratio") or 2.0),
        micro_cutoff_step=int(geometry.get("micro_cutoff_step") or 800),
        promotion_count=int(geometry.get("promotion_count") or 1),
    )
    return [
        dict(entry)
        for entry in expanded.get("entries", [])
        if isinstance(entry, dict) and int(entry.get("step", -1)) > int(after_step)
    ]


def format_probe_plan(plan: dict[str, Any] | None) -> str:
    if not isinstance(plan, dict):
        return "probe_policy=unavailable"
    policy = str(plan.get("policy") or "unknown")
    steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []
    eval_batches = plan.get("eval_batches") if isinstance(plan.get("eval_batches"), dict) else {}
    if not steps:
        return f"probe_policy={policy} probe_steps=[]"
    return (
        f"probe_policy={policy} probe_steps={steps} "
        f"probe_eval_batches=micro:{eval_batches.get('micro')} "
        f"standard:{eval_batches.get('standard')} "
        f"promotion:{eval_batches.get('promotion')}"
    )
