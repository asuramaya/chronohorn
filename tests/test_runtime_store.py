from __future__ import annotations
import json
from pathlib import Path
from chronohorn.runtime_store import IncrementalStore

def test_initial_scan(tmp_path):
    rd = tmp_path / "results"; rd.mkdir()
    (rd / "job1.json").write_text(json.dumps({"model": {"test_bpb": 2.05}}))
    (rd / "job2.json").write_text(json.dumps({"model": {"test_bpb": 1.95}}))
    store = IncrementalStore(result_dir=rd)
    store.refresh()
    assert store.result_count == 2
    assert store.best_bpb == 1.95

def test_incremental_detects_new_file(tmp_path):
    rd = tmp_path / "results"; rd.mkdir()
    (rd / "job1.json").write_text(json.dumps({"model": {"test_bpb": 2.05}}))
    store = IncrementalStore(result_dir=rd)
    store.refresh()
    assert store.result_count == 1
    (rd / "job2.json").write_text(json.dumps({"model": {"test_bpb": 1.85}}))
    new = store.refresh()
    assert store.result_count == 2
    assert len(new) == 1
    assert new[0] == "job2"
    assert store.best_bpb == 1.85

def test_no_change_returns_empty(tmp_path):
    rd = tmp_path / "results"; rd.mkdir()
    (rd / "job1.json").write_text(json.dumps({"model": {"test_bpb": 2.05}}))
    store = IncrementalStore(result_dir=rd)
    store.refresh()
    new = store.refresh()
    assert len(new) == 0

def test_leaderboard_sorted(tmp_path):
    rd = tmp_path / "results"; rd.mkdir()
    for i, bpb in enumerate([2.1, 1.9, 2.0]):
        (rd / f"job{i}.json").write_text(json.dumps({"model": {"test_bpb": bpb}}))
    store = IncrementalStore(result_dir=rd)
    store.refresh()
    board = store.leaderboard(3)
    assert board[0]["bpb"] == 1.9
    assert board[2]["bpb"] == 2.1
