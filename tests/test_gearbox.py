"""The gearbox: probe→measure→parity-gate→pick→cache-by-signature.

Pure-callback core, so these tests use fake build/step functions (no
torch, no GPU) to exercise the selection logic deterministically.
"""
from __future__ import annotations

import json

from chronohorn.engine.gearbox import (
    DEFAULT_GEARS,
    Gear,
    load_gear_cache,
    tune_gear,
)

# Fake per-gear (throughput, loss) profiles. The gearbox never sees these
# directly — it calls step() which we make return the loss, and times it.
# We simulate timing by having step() busy a deterministic, gear-dependent
# amount via an iteration counter the fake exposes.


def _fakes(profile: dict[str, tuple[float, float]]):
    """profile: gear_name -> (relative_speed, loss). Higher speed = fewer
    simulated 'work units' per step = faster."""
    def build(gear: Gear):
        return {"gear": gear.name}

    calls = {"n": 0}

    def step(model, gear: Gear):
        speed, loss = profile[model["gear"]]
        # Emulate wall-time by doing `1/speed` units of real work so the
        # gearbox's time.time() measurement ranks gears correctly.
        x = 0.0
        for _ in range(int(2000 / speed)):
            x += 1.0
        calls["n"] += 1
        return loss
    return build, step, calls


def test_picks_fastest_parity_safe(tmp_path):
    # triton is fastest AND parity-safe; amp is faster still but drifts out.
    profile = {
        "eager": (1.0, 5.00),
        "triton": (5.0, 5.001),           # 5x, drift 0.001 < 5e-3 → admit
        "compile": (3.0, 5.000),          # 3x, exact
        "triton+compile": (6.0, 5.002),   # 6x, drift 0.002 → admit (winner)
        "amp-fp16": (2.0, 5.05),          # lossy tier, drift 0.05 < 8e-2 → admit
        "triton+amp-fp16": (9.0, 6.00),   # 9x but drift 1.0 >> 8e-2 → REJECT
    }
    build, step, _ = _fakes(profile)
    pick = tune_gear(build, step, signature=(("substrate_mode", "gated_delta"),),
                     sample_tokens=1000, device="cuda", shape=(8, 1024),
                     cache_path=tmp_path / "gears.json", iters=3, warmup=1)
    assert pick["gear"] == "triton+compile"  # fastest ADMITTED, not fastest overall
    assert pick["source"] == "probe"
    # the cheating fast gear must be recorded as rejected
    rej = [m for m in pick["measurements"] if m["gear"] == "triton+amp-fp16"][0]
    assert not rej["admitted"] and "parity fail" in rej["reason"]


def test_cache_hit_skips_probe(tmp_path):
    profile = {g.name: (2.0, 5.0) for g in DEFAULT_GEARS}
    build, step, calls = _fakes(profile)
    sig = (("substrate_mode", "frozen"),)
    kw = dict(signature=sig, sample_tokens=1000, device="cuda", shape=(8, 1024),
              cache_path=tmp_path / "gears.json", iters=3, warmup=1)
    first = tune_gear(build, step, **kw)
    n_after_first = calls["n"]
    second = tune_gear(build, step, **kw)
    assert first["source"] == "probe"
    assert second["source"] == "cache"
    assert calls["n"] == n_after_first  # no new step() calls on cache hit
    assert second["gear"] == first["gear"]


def test_signature_separates_cache(tmp_path):
    # Two architectures must get independent verdicts.
    cache_path = tmp_path / "gears.json"
    fast_triton = {g.name: (9.0 if g.triton else 1.0, 5.0) for g in DEFAULT_GEARS}
    build_a, step_a, _ = _fakes(fast_triton)
    tune_gear(build_a, step_a, signature=(("adaptive_substrate", "True"),),
              sample_tokens=1000, device="cuda", shape=(8, 1024),
              cache_path=cache_path, iters=3, warmup=1)
    fast_eager = {g.name: (9.0 if g.name == "eager" else 1.0, 5.0) for g in DEFAULT_GEARS}
    build_b, step_b, _ = _fakes(fast_eager)
    tune_gear(build_b, step_b, signature=(("adaptive_substrate", "False"),),
              sample_tokens=1000, device="cuda", shape=(8, 1024),
              cache_path=cache_path, iters=3, warmup=1)
    cache = load_gear_cache(cache_path)
    assert len(cache) == 2  # two distinct signature keys
    gears = {json.loads(k)["sig"][0][1]: v for k, v in cache.items()}
    assert gears["True"]["triton"] is True    # adaptive → a triton gear won
    assert gears["False"]["gear"] == "eager"  # frozen → eager fastest


def test_gear_that_errors_is_ineligible(tmp_path):
    def build(gear: Gear):
        if gear.triton:
            raise RuntimeError("no triton on this box")
        return {"gear": gear.name}

    def step(model, gear: Gear):
        return 5.0

    pick = tune_gear(build, step, signature=(), sample_tokens=1000, device="cpu",
                     shape=(8, 1024), cache_path=tmp_path / "gears.json", iters=2, warmup=1)
    # triton gears errored out; a non-triton gear must win, never crash
    assert pick["triton"] is False
    errored = [m for m in pick["measurements"] if "error" in m["reason"]]
    assert len(errored) >= 1
