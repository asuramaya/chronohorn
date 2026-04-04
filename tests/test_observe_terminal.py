"""Tests for observe/terminal.py — ASCII rendering functions."""
from __future__ import annotations

from chronohorn.db import ChronohornDB


def _seed_db(db):
    for i, bpb in enumerate([1.85, 1.90, 1.95]):
        db.record_result(f"run-{i}", {
            "model": {"test_bpb": bpb, "params": 10000000 + i * 100000, "architecture": "polyhash_v12"},
            "config": {"train": {"steps": 10000}},
            "training": {
                "performance": {"tokens_per_second": 350000, "elapsed_sec": 900},
                "probes": [
                    {"step": 1000, "bpb": bpb + 0.5},
                    {"step": 5000, "bpb": bpb + 0.2},
                    {"step": 10000, "bpb": bpb},
                ],
            },
        })
    db.record_probe("run-0", 1000, 2.35)
    db.record_probe("run-0", 5000, 2.05)
    db.record_probe("run-0", 10000, 1.85)


def test_ascii_frontier_table(tmp_path):
    from chronohorn.observe.terminal import ascii_frontier_table

    db = ChronohornDB(tmp_path / "test.db")
    _seed_db(db)
    frontier = db.frontier(5, population="all")
    text = ascii_frontier_table(frontier)
    assert isinstance(text, str)
    assert len(text) > 0
    db.close()


def test_ascii_learning_curve(tmp_path):
    from chronohorn.observe.terminal import ascii_learning_curve

    db = ChronohornDB(tmp_path / "test.db")
    _seed_db(db)
    curve = db.learning_curve("run-0")
    text = ascii_learning_curve(curve)
    assert isinstance(text, str)
    db.close()


def test_ascii_status(tmp_path):
    from chronohorn.observe.terminal import ascii_status

    db = ChronohornDB(tmp_path / "test.db")
    _seed_db(db)
    summary = db.summary()
    text = ascii_status(summary)
    assert isinstance(text, str)
    db.close()


def test_ascii_sparkline():
    from chronohorn.observe.terminal import ascii_sparkline

    values = [2.5, 2.3, 2.1, 1.95, 1.85, 1.80]
    spark = ascii_sparkline(values)
    assert isinstance(spark, str)
    assert len(spark) > 0
