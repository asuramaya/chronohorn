from __future__ import annotations
import json
from pathlib import Path
from chronohorn.db import ChronohornDB


def test_create_and_query(tmp_path):
    db = ChronohornDB(tmp_path / "test.db")
    assert db.result_count() == 0
    assert db.best_bpb() is None
    db.close()


def test_record_result(tmp_path):
    db = ChronohornDB(tmp_path / "test.db")
    payload = {
        "model": {"test_bpb": 1.85, "params": 15000000, "scale": 10.0, "local_window": 4},
        "config": {"train": {"steps": 10000, "seq_len": 512}},
        "training": {
            "performance": {"tokens_per_second": 50000, "estimated_sustained_tflops": 5.0, "elapsed_sec": 1000},
            "probes": [
                {"step": 800, "bpb": 2.1},
                {"step": 1000, "bpb": 2.05},
            ],
        },
    }
    db.record_result("test-run", payload)
    assert db.result_count() == 1
    assert abs(db.best_bpb(population="all") - 1.85) < 0.01
    db.close()


def test_frontier(tmp_path):
    db = ChronohornDB(tmp_path / "test.db")
    for i, bpb in enumerate([2.1, 1.9, 2.0, 1.85]):
        db.record_result(f"run-{i}", {"model": {"test_bpb": bpb}, "config": {}, "training": {"performance": {}, "probes": []}})
    board = db.frontier(3, population="all")
    assert len(board) == 3
    assert board[0]["bpb"] == 1.85
    db.close()


def test_probes(tmp_path):
    db = ChronohornDB(tmp_path / "test.db")
    db.record_probe("job1", 100, 2.5)
    db.record_probe("job1", 200, 2.3)
    db.record_probe("job1", 400, 2.1)
    curve = db.learning_curve("job1")
    assert len(curve) == 3
    assert curve[0]["step"] == 100
    assert curve[2]["bpb"] == 2.1
    db.close()


def test_illegal_detection(tmp_path):
    db = ChronohornDB(tmp_path / "test.db")
    # Illegal: patch name with low bpb and causal-bank markers
    db.record_result("sub1-patch4-test", {
        "model": {"test_bpb": 0.55, "linear_readout_kind": "mlp", "local_window": 4},
        "config": {"train": {"steps": 2000}},
        "training": {"performance": {}, "probes": []},
    })
    # Legal
    db.record_result("sub1-hybrid-test", {
        "model": {"test_bpb": 1.95}, "config": {"train": {"steps": 2000}},
        "training": {"performance": {}, "probes": []},
    })
    board = db.frontier(10, population="all")
    assert len(board) == 1
    assert board[0]["name"] == "sub1-hybrid-test"
    db.close()


def test_raw_query(tmp_path):
    db = ChronohornDB(tmp_path / "test.db")
    db.record_result("a", {"model": {"test_bpb": 2.0}, "config": {}, "training": {"performance": {}, "probes": []}})
    db.record_result("b", {"model": {"test_bpb": 1.8}, "config": {}, "training": {"performance": {}, "probes": []}})
    rows = db.query("SELECT name, bpb FROM results ORDER BY bpb")
    assert rows[0]["name"] == "b"
    db.close()


def test_rebuild_from_archive(tmp_path):
    rd = tmp_path / "results"
    rd.mkdir()
    (rd / "job1.json").write_text(json.dumps({
        "model": {"test_bpb": 1.9}, "config": {"train": {"steps": 1000}},
        "training": {"performance": {}, "probes": [{"step": 1000, "bpb": 1.9}]},
    }))
    db = ChronohornDB(tmp_path / "test.db")
    count = db.rebuild_from_archive(str(rd))
    assert count == 1
    assert db.result_count() == 1
    db.close()


def test_events(tmp_path):
    db = ChronohornDB(tmp_path / "test.db")
    db.record_event("launched", name="job1", host="slop-01")
    db.record_event("completed", name="job1")
    events = db.events_recent(10)
    assert len(events) == 2
    assert events[0]["event"] == "launched"
    db.close()


def test_jobs_lifecycle(tmp_path):
    db = ChronohornDB(tmp_path / "test.db")
    db.record_job("job1", config={"scale": 10.0}, steps=2000, lr=0.0015)
    pending = db.pending_jobs()
    assert len(pending) == 1
    db.record_launch("job1", host="slop-01", launcher="managed_command")
    pending = db.pending_jobs()
    assert len(pending) == 0
    running = db.running_jobs()
    assert len(running) == 1
    db.close()
