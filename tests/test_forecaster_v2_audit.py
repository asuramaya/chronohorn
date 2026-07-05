"""Forecaster-v2 audit clauses: late-bend bug siren + memory probation.

From the 2026-07 audit: runs that beat their own early curve beyond
noise were bug signatures (STE clicks, causality leaks); the one honest
late-bend class is memory/PKM specimens, which get probation instead of
a siren.
"""
from __future__ import annotations

import math

from chronohorn.engine.forecasting import (
    _late_bend_audit,
    _probation_clause,
    build_result_forecast,
)


def _synthetic_result(final_bpb_offset: float = 0.0, model: dict | None = None) -> dict:
    # bpb(s) = 2.0 + 5 * s^-0.5 with small deterministic ripple.
    steps = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
    probes = []
    for i, s in enumerate(steps):
        value = 2.0 + 5.0 * s**-0.5 + 0.005 * math.sin(i * 1.7)
        if s == steps[-1]:
            value += final_bpb_offset
        probes.append({"step": s, "bpb": value})
    result: dict = {"training": {"probes": probes}}
    if model is not None:
        result["model"] = model
    return result


def test_consistent_run_not_flagged() -> None:
    audit = _late_bend_audit(_synthetic_result(), "bpb", probation=None)
    assert audit is not None
    assert audit["flag"] == "consistent"
    assert abs(audit["prefix_fit_error"]) < 0.15


def test_late_bend_flagged_as_bug_siren() -> None:
    audit = _late_bend_audit(_synthetic_result(-0.6), "bpb", probation=None)
    assert audit is not None
    assert audit["flag"] == "late_bend_suspicious"


def test_late_regression_flagged() -> None:
    audit = _late_bend_audit(_synthetic_result(+0.6), "bpb", probation=None)
    assert audit is not None
    assert audit["flag"] == "late_regression"


def test_memory_class_downgrades_siren() -> None:
    result = _synthetic_result(-0.6, model={"hash_memory": True})
    probation = _probation_clause(result)
    assert probation is not None
    assert probation["class"] == "memory_augmented"
    assert probation["kill_horizon_multiplier"] == 2.0
    audit = _late_bend_audit(result, "bpb", probation=probation)
    assert audit["flag"] == "late_bend_expected_memory_class"


def test_probation_markers() -> None:
    assert _probation_clause({"model": {}}) is None
    assert _probation_clause({"model": {"memory_kind": "none"}}) is None
    clause = _probation_clause(
        {"model": {"memory_kind": "ngram", "sticky_registers": 64}}
    )
    assert set(clause["markers"]) == {"memory_kind:ngram", "sticky_registers"}
    binding = _probation_clause({"model": {"binding_kind": "delta_rule"}})
    assert binding is not None and "binding_kind" in binding["markers"]


def test_forecast_block_carries_audit_and_probation() -> None:
    result = _synthetic_result(model={"hash_memory": True})
    forecast = build_result_forecast(result)
    assert "audit" in forecast
    assert forecast["probation"]["class"] == "memory_augmented"


def test_too_few_prefix_points_returns_none() -> None:
    result = {"training": {"probes": [{"step": s, "bpb": 3.0} for s in (100, 5000, 10000)]}}
    assert _late_bend_audit(result, "bpb", probation=None) is None
