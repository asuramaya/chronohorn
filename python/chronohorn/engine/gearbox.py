"""The gearbox: auto-select the optimal execution gear for a model by
probing it, not by hardcoding a rule per architecture.

Philosophy (the census applied to speed): a model's fastest *correct*
execution config is a property of its architecture, discoverable by a
cheap warmup probe and cacheable by architecture signature — so it is
discovered ONCE per architecture, never re-derived by hand.

    gear = tune_gear(build, sample, signature, cache_path)

Each candidate gear is benchmarked on a few forward+backward steps at the
real training shape, and admitted only if its loss matches the eager
reference within tolerance (the parity gate — fast is invalid if it isn't
also correct, the same discipline the kill chamber applies to findings).
The winner is the fastest admitted gear; the verdict is cached keyed on
the architecture signature (chronohorn.fleet.telemetry) so the next model
with the same signature skips the probe.

Born from the same wound as architecture_signature itself: session 13's
megabyte pilot mispredicted throughput 60x because execution behavior was
matched across unlike architectures. Execution behavior IS a function of
the signature; the gearbox reads that function instead of guessing it.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class Gear:
    """One execution profile — a combination of the free/near-free levers."""
    name: str
    triton: bool = False
    compile_mode: str | None = None  # None | "default" | "reduce-overhead"
    amp_dtype: str | None = None     # None | "fp16" | "bf16"
    # parity_tier: "exact" gears must match eager loss tightly; "lossy" gears
    # (amp) get a looser tolerance because reduced precision legitimately
    # perturbs a single step without breaking training convergence.
    parity_tier: str = "exact"


# The candidate gears, cheapest-risk first. eager is always the reference.
# Order matters only for logging; selection is by measured throughput.
DEFAULT_GEARS: tuple[Gear, ...] = (
    Gear("eager"),
    Gear("triton", triton=True),
    Gear("compile", compile_mode="default"),
    Gear("triton+compile", triton=True, compile_mode="default"),
    Gear("amp-fp16", amp_dtype="fp16", parity_tier="lossy"),
    Gear("triton+amp-fp16", triton=True, amp_dtype="fp16", parity_tier="lossy"),
)

# Parity tolerances on the warmup loss (nats). exact gears are provably
# equivalent numerics (fused scan, kernel fusion) and must match tightly;
# lossy gears (reduced precision) get headroom because a single-step
# perturbation of this size washes out over training.
PARITY_TOL = {"exact": 5e-3, "lossy": 8e-2}


@dataclass(frozen=True)
class GearResult:
    name: str
    tokens_per_second: float
    warmup_loss: float
    admitted: bool
    reason: str


def _signature_key(signature: Any, device: str, shape: tuple[int, int]) -> str:
    """Cache key: architecture signature + device + shape class.

    The gear depends on all three — a different card or a very different
    sequence length can move the optimum — so all three are keyed.
    """
    sig = tuple(signature) if signature else ()
    b, t = shape
    # bucket the shape so minor batch/seq changes reuse the verdict
    shape_class = f"b{1 if b<=2 else 8 if b<=12 else 32}_t{512 if t<=768 else 1024 if t<=1536 else 4096}"
    return json.dumps({"sig": sig, "dev": device, "shape": shape_class}, sort_keys=True)


def load_gear_cache(cache_path: str | Path) -> dict[str, Any]:
    p = Path(cache_path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_gear_cache(cache_path: str | Path, cache: dict[str, Any]) -> None:
    p = Path(cache_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(cache, indent=2, sort_keys=True))
    import os
    os.replace(tmp, p)


def _benchmark_gear(
    gear: Gear,
    build: Callable[[Gear], Any],
    step: Callable[[Any, Gear], float],
    sample_tokens: int,
    iters: int,
    warmup: int,
) -> GearResult:
    """Build the model in this gear, time `iters` train steps, return tok/s + loss."""
    import torch

    model = build(gear)
    losses = []
    for _ in range(warmup):
        step(model, gear)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        losses.append(float(step(model, gear)))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = (time.time() - t0) / max(iters, 1)
    tokps = sample_tokens / dt if dt > 0 else 0.0
    return GearResult(gear.name, tokps, sum(losses) / len(losses), True, "measured")


def tune_gear(
    build: Callable[[Gear], Any],
    step: Callable[[Any, Gear], float],
    *,
    signature: Any,
    sample_tokens: int,
    device: str,
    shape: tuple[int, int],
    cache_path: str | Path,
    gears: tuple[Gear, ...] = DEFAULT_GEARS,
    iters: int = 8,
    warmup: int = 3,
    log: Callable[[str], None] | None = None,
    force_retune: bool = False,
) -> dict[str, Any]:
    """Return the fastest parity-safe gear for this model.

    Cache hit (same signature+device+shape) → returns instantly.
    Miss → probes every gear, parity-gates against eager, caches the winner.

    `build(gear)` constructs a model configured for that gear.
    `step(model, gear)` runs ONE train step and returns the scalar loss;
    it owns AMP autocast / triton wiring per the gear it's handed.

    Returns {"gear": <name>, "triton": bool, "compile_mode": ..., "amp_dtype": ...,
             "speedup": float, "source": "cache"|"probe", "measurements": [...]}.
    """
    _log = log or (lambda _m: None)
    key = _signature_key(signature, device, shape)
    cache = load_gear_cache(cache_path)
    if not force_retune and key in cache:
        entry = dict(cache[key])
        entry["source"] = "cache"
        _log(f"gearbox: cache hit {entry.get('gear')} (speedup {entry.get('speedup', 1.0):.2f}x)")
        return entry

    # Probe. eager first as the parity + speed reference.
    ref = _benchmark_gear(gears[0], build, step, sample_tokens, iters, warmup)
    _log(f"gearbox: eager {ref.tokens_per_second:.0f} tok/s loss {ref.warmup_loss:.4f} (reference)")
    measurements = [ref]
    best = ref
    for gear in gears[1:]:
        try:
            r = _benchmark_gear(gear, build, step, sample_tokens, iters, warmup)
        except Exception as exc:  # noqa: BLE001 — a gear that errors is simply ineligible
            measurements.append(GearResult(gear.name, 0.0, float("nan"), False, f"error: {type(exc).__name__}"))
            _log(f"gearbox: {gear.name} unavailable ({type(exc).__name__})")
            continue
        tol = PARITY_TOL[gear.parity_tier]
        drift = abs(r.warmup_loss - ref.warmup_loss)
        admitted = drift <= tol
        r = GearResult(r.name, r.tokens_per_second, r.warmup_loss, admitted,
                       "admitted" if admitted else f"parity fail (drift {drift:.4f} > {tol})")
        measurements.append(r)
        _log(f"gearbox: {gear.name} {r.tokens_per_second:.0f} tok/s "
             f"({r.tokens_per_second/ref.tokens_per_second:.2f}x) drift {drift:.4f} — {r.reason}")
        if admitted and r.tokens_per_second > best.tokens_per_second:
            best = r

    winner = next(g for g in gears if g.name == best.name)
    entry = {
        "gear": winner.name,
        "triton": winner.triton,
        "compile_mode": winner.compile_mode,
        "amp_dtype": winner.amp_dtype,
        "speedup": round(best.tokens_per_second / ref.tokens_per_second, 3) if ref.tokens_per_second else 1.0,
        "tokens_per_second": round(best.tokens_per_second, 1),
        "source": "probe",
        "measurements": [
            {"gear": m.name, "tok_s": round(m.tokens_per_second, 1),
             "admitted": m.admitted, "reason": m.reason}
            for m in measurements
        ],
    }
    cache[key] = {k: v for k, v in entry.items() if k != "source"}
    _save_gear_cache(cache_path, cache)
    _log(f"gearbox: selected {winner.name} — {entry['speedup']}x over eager, cached by signature")
    return entry
