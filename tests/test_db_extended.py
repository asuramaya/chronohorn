"""Extended DB tests — journal, predictions, config_diff, what_varied, branch_health, frontier_velocity."""
from __future__ import annotations

from chronohorn.db import ChronohornDB


def _seed_branched(db):
    """Seed with results that have branch-like naming."""
    configs = [
        ("v12-base-seed42", 1.85, "polyhash_v12"),
        ("v12-base-seed43", 1.87, "polyhash_v12"),
        ("v12-base-seed44", 1.86, "polyhash_v12"),
        ("v12-pkm-seed42", 1.63, "polyhash_v12"),
        ("v12-pkm-seed43", 1.65, "polyhash_v12"),
        ("v8-full-seed42", 1.90, "polyhash_v8"),
    ]
    for name, bpb, arch in configs:
        db.record_result(name, {
            "model": {"test_bpb": bpb, "params": 11000000, "architecture": arch},
            "config": {
                "train": {"steps": 10000, "seq_len": 512, "learning_rate": 0.005},
                "architecture": {"hidden_dim": 320, "num_layers": 6},
            },
            "training": {
                "performance": {"tokens_per_second": 350000, "elapsed_sec": 900},
                "probes": [
                    {"step": 1000, "bpb": bpb + 0.5},
                    {"step": 5000, "bpb": bpb + 0.2},
                    {"step": 10000, "bpb": bpb},
                ],
            },
        })


def test_journal_write_and_read(tmp_path):
    db = ChronohornDB(tmp_path / "test.db")
    db.record_journal("observation", "scan half-life is 8.5 tokens", tags=["scan", "geometry"])
    db.record_journal("decision", "skip rotation scan", tags=["scan"])
    entries = db.journal_entries(limit=10)
    assert len(entries) == 2
    db.close()


def test_journal_filter_by_kind(tmp_path):
    db = ChronohornDB(tmp_path / "test.db")
    db.record_journal("observation", "note 1")
    db.record_journal("decision", "note 2")
    entries = db.journal_entries(kind="decision", limit=10)
    assert len(entries) == 1
    db.close()


def test_config_diff(tmp_path):
    db = ChronohornDB(tmp_path / "test.db")
    _seed_branched(db)
    diff = db.config_diff("v12-base-seed42", "v12-pkm-seed42")
    assert isinstance(diff, dict)
    db.close()


def test_what_varied(tmp_path):
    db = ChronohornDB(tmp_path / "test.db")
    _seed_branched(db)
    result = db.what_varied(["v12-base-seed42", "v12-base-seed43", "v12-base-seed44"])
    assert isinstance(result, dict)
    db.close()


def test_branch_health(tmp_path):
    db = ChronohornDB(tmp_path / "test.db")
    _seed_branched(db)
    health = db.branch_health("v12-base")
    assert isinstance(health, dict)
    db.close()


def test_frontier_velocity(tmp_path):
    db = ChronohornDB(tmp_path / "test.db")
    _seed_branched(db)
    velocity = db.frontier_velocity()
    assert isinstance(velocity, dict)
    db.close()


def test_predict_at_steps(tmp_path):
    db = ChronohornDB(tmp_path / "test.db")
    _seed_branched(db)
    pred = db.predict_at_steps("v12-base-seed42", 50000)
    assert isinstance(pred, dict)
    db.close()


def test_find_similar(tmp_path):
    db = ChronohornDB(tmp_path / "test.db")
    _seed_branched(db)
    similar = db.find_similar({"hidden_dim": 320, "num_layers": 6}, threshold=0.5)
    assert isinstance(similar, list)
    db.close()


def test_seed_groups(tmp_path):
    db = ChronohornDB(tmp_path / "test.db")
    _seed_branched(db)
    result = db.seed_groups()
    assert isinstance(result, (dict, list))
    db.close()


def test_detect_groups(tmp_path):
    db = ChronohornDB(tmp_path / "test.db")
    _seed_branched(db)
    groups = db.detect_groups()
    assert isinstance(groups, (dict, list))
    db.close()
