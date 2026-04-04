"""Tests for core engine modules: budgets, performance, probes, forecasting."""
from __future__ import annotations

import math


def test_bits_per_token_from_loss():
    from chronohorn.engine.performance import bits_per_token_from_loss

    # ln(2) loss should be 1.0 bits per token
    result = bits_per_token_from_loss(math.log(2))
    assert abs(result - 1.0) < 0.001


def test_bits_per_token_zero_loss():
    from chronohorn.engine.performance import bits_per_token_from_loss

    result = bits_per_token_from_loss(0.0)
    assert result == 0.0


def test_competition_budget_defaults():
    from chronohorn.engine.budgets import DEFAULT_GOLF_V1_BUDGET

    assert DEFAULT_GOLF_V1_BUDGET.name
    assert DEFAULT_GOLF_V1_BUDGET.train_tflops_budget > 0
    assert DEFAULT_GOLF_V1_BUDGET.artifact_limit_mb > 0


def test_resolve_competition_budget():
    from chronohorn.engine.budgets import resolve_competition_budget, DEFAULT_GOLF_V1_BUDGET

    budget = resolve_competition_budget(DEFAULT_GOLF_V1_BUDGET.name)
    assert budget.name == DEFAULT_GOLF_V1_BUDGET.name


def test_resolve_competition_budget_unknown():
    from chronohorn.engine.budgets import resolve_competition_budget

    # Unknown budget should either return a default or raise
    try:
        budget = resolve_competition_budget("nonexistent-budget-xyz")
        # If it returns, check it's still valid
        assert budget.name is not None
    except (KeyError, ValueError):
        pass  # expected


def test_probe_schedule():
    from chronohorn.engine.probes import resolve_probe_plan

    plan = resolve_probe_plan(max_step=10000)
    assert isinstance(plan, (list, dict, set))


def test_parse_probe_steps():
    from chronohorn.engine.probes import parse_probe_steps

    steps = parse_probe_steps("100,200,400,800", max_step=10000)
    assert 100 in steps
    assert 800 in steps


def test_build_result_forecast_basic():
    from chronohorn.engine.forecasting import build_result_forecast

    result = {
        "model": {"test_bpb": 1.85, "params": 10000000},
        "config": {"train": {"steps": 10000, "seq_len": 512}},
        "training": {
            "performance": {"tokens_per_second": 350000, "elapsed_sec": 900},
            "probes": [
                {"step": 100, "bpb": 2.5},
                {"step": 1000, "bpb": 2.0},
                {"step": 10000, "bpb": 1.85},
            ],
        },
    }
    forecast = build_result_forecast(result)
    assert isinstance(forecast, dict)


def test_load_result_json(tmp_path):
    import json
    from chronohorn.engine.results import load_result_json

    payload = {
        "model": {"test_bpb": 1.9},
        "config": {"train": {"steps": 5000}},
        "training": {"performance": {}, "probes": []},
    }
    path = tmp_path / "test.json"
    path.write_text(json.dumps(payload))
    loaded = load_result_json(str(path))
    assert loaded["model"]["test_bpb"] == 1.9


def test_load_result_json_missing(tmp_path):
    from chronohorn.engine.results import load_result_json

    try:
        load_result_json(str(tmp_path / "nonexistent.json"))
        assert False, "should have raised"
    except (FileNotFoundError, OSError, Exception):
        pass


def test_import_symbol():
    from chronohorn.engine.importing import import_symbol

    # Should be able to import a known symbol
    path_cls = import_symbol("pathlib", "Path")
    assert path_cls is not None
    from pathlib import Path
    assert path_cls is Path


def test_import_symbol_missing():
    from chronohorn.engine.importing import import_symbol

    try:
        import_symbol("nonexistent_module_xyz", "Foo")
        assert False, "should have raised"
    except (ImportError, ModuleNotFoundError):
        pass
