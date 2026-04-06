"""Tests for engine/advisor.py and engine/axis_analysis.py."""
from __future__ import annotations

from chronohorn.db import ChronohornDB


def _seed_db(db: ChronohornDB):
    """Seed DB with a variety of results for advisor testing."""
    configs = [
        ("v12-base", 1.85, 11000000, 10000, "polyhash_v12"),
        ("v12-mlp6", 1.65, 11200000, 10000, "polyhash_v12"),
        ("v12-pkm", 1.63, 11200000, 10000, "polyhash_v12"),
        ("v12-scan64", 1.80, 10500000, 10000, "polyhash_v12"),
        ("v12-scan128", 1.78, 10800000, 10000, "polyhash_v12"),
        ("v8-full", 1.90, 9000000, 10000, "polyhash_v8"),
    ]
    for name, bpb, params, steps, arch in configs:
        db.record_result(name, {
            "model": {"test_bpb": bpb, "params": params, "architecture": arch},
            "config": {"train": {"steps": steps, "seq_len": 512}},
            "training": {"performance": {"tokens_per_second": 350000}, "probes": []},
        })


def test_suggest_next(tmp_path):
    from chronohorn.engine.advisor import suggest_next

    db = ChronohornDB(tmp_path / "test.db")
    _seed_db(db)
    suggestions = suggest_next(db)
    assert isinstance(suggestions, list)
    db.close()


def test_format_suggestions(tmp_path):
    from chronohorn.engine.advisor import format_suggestions, suggest_next

    db = ChronohornDB(tmp_path / "test.db")
    _seed_db(db)
    suggestions = suggest_next(db)
    text = format_suggestions(suggestions)
    assert isinstance(text, str)
    db.close()


def test_architecture_boundary(tmp_path):
    from chronohorn.engine.advisor import architecture_boundary

    db = ChronohornDB(tmp_path / "test.db")
    _seed_db(db)
    result = architecture_boundary(db)
    assert isinstance(result, dict)
    db.close()


def test_analyze_axes(tmp_path):
    from chronohorn.engine.axis_analysis import analyze_axes

    db = ChronohornDB(tmp_path / "test.db")
    _seed_db(db)
    results = db.query("SELECT * FROM results WHERE bpb > 0")
    result = analyze_axes([dict(r) for r in results])
    assert isinstance(result, (dict, list))
    db.close()
