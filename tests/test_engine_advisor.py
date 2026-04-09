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


def test_suggest_next_prefers_lane_screening_before_promotion(tmp_path):
    from chronohorn.engine.advisor import suggest_next

    db = ChronohornDB(tmp_path / "test.db")
    db.record_job(
        "cb-screen-s8",
        manifest="screen.jsonl",
        family="causal-bank",
        config={"family": "causal-bank", "scale": 8.0, "seq_len": 256, "profile": "pilot"},
        steps=4000,
        seed=42,
        batch_size=8,
        job_spec={"work_tokens": 200_000_000},
    )
    db.record_result(
        "cb-screen-s8",
        {
            "model": {"test_bpb": 1.92, "params": 7_000_000, "architecture": "causal-bank"},
            "config": {"train": {"steps": 4000, "seq_len": 256, "scale": 8.0, "profile": "pilot"}},
            "training": {
                "performance": {"tokens_per_second": 10_000, "elapsed_sec": 100.0},
                "probes": [
                    {"step": 250, "bpb": 2.42, "tflops": 0.05, "elapsed_sec": 10.0},
                    {"step": 500, "bpb": 2.20, "tflops": 0.10, "elapsed_sec": 20.0},
                    {"step": 1000, "bpb": 2.00, "tflops": 0.25, "elapsed_sec": 40.0},
                    {"step": 4000, "bpb": 1.92, "tflops": 1.00, "elapsed_sec": 100.0},
                ],
            },
        },
    )

    suggestions = suggest_next(db)
    assert suggestions
    assert any("next scale" in str(item.get("action", "")).lower() for item in suggestions)
    db.close()


def test_suggest_next_requests_budget_shrink_for_overbudget_candidate(tmp_path):
    from chronohorn.engine.advisor import suggest_next

    db = ChronohornDB(tmp_path / "test.db")
    db.record_job(
        "cb-overbudget-s12",
        manifest="screen.jsonl",
        family="causal-bank",
        config={"family": "causal-bank", "scale": 12.0, "seq_len": 512, "profile": "pilot"},
        steps=4000,
        seed=42,
        batch_size=8,
        job_spec={"work_tokens": 200_000_000},
    )
    db.record_result(
        "cb-overbudget-s12",
        {
            "model": {"test_bpb": 1.84, "params": 30_000_000, "architecture": "causal-bank"},
            "config": {"train": {"steps": 4000, "seq_len": 512, "scale": 12.0, "profile": "pilot"}},
            "training": {
                "performance": {"tokens_per_second": 8_000, "elapsed_sec": 120.0},
                "probes": [
                    {"step": 250, "bpb": 2.32, "tflops": 0.08, "elapsed_sec": 12.0},
                    {"step": 500, "bpb": 2.12, "tflops": 0.16, "elapsed_sec": 24.0},
                    {"step": 1000, "bpb": 1.95, "tflops": 0.35, "elapsed_sec": 48.0},
                    {"step": 4000, "bpb": 1.84, "tflops": 1.20, "elapsed_sec": 120.0},
                ],
            },
        },
    )

    suggestions = suggest_next(db)
    assert suggestions
    assert any("artifact limit" in str(item.get("action", "")).lower() for item in suggestions)
    db.close()
