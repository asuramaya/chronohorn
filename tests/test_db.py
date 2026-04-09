from __future__ import annotations

import json
import sqlite3
import threading

import pytest

from chronohorn.db import ChronohornDB


def _has_decepticons() -> bool:
    try:
        import decepticons  # noqa: F401
        return True
    except ImportError:
        return False


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


def test_record_result_derives_bpb_from_bits_per_token(tmp_path):
    db = ChronohornDB(tmp_path / "test.db")
    payload = {
        "model": {"test_bits_per_token": 4.872, "params": 1000},
        "config": {},
        "training": {"performance": {}, "probes": []},
    }
    db.record_result("derived-bpb", payload)
    rows = db.query("SELECT name, bpb FROM results WHERE name = ?", ("derived-bpb",))
    assert len(rows) == 1
    assert abs(rows[0]["bpb"] - 2.0) < 0.01
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


@pytest.mark.skipif(
    not _has_decepticons(),
    reason="decepticons not installed — causal-bank illegal detection requires adapter",
)
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


def test_rebuild_from_archive_ingests_bits_per_token_only_results(tmp_path):
    rd = tmp_path / "results"
    rd.mkdir()
    (rd / "job1.json").write_text(
        json.dumps(
            {
                "model": {"test_bits_per_token": 4.872},
                "config": {"train": {"steps": 1000}},
                "training": {"performance": {}, "probes": []},
            }
        )
    )
    db = ChronohornDB(tmp_path / "test.db")
    count = db.rebuild_from_archive(str(rd))
    rows = db.query("SELECT bpb FROM results WHERE name = 'job1'")
    assert count == 1
    assert len(rows) == 1
    assert abs(rows[0]["bpb"] - 2.0) < 0.01
    db.close()


def test_events(tmp_path):
    db = ChronohornDB(tmp_path / "test.db")
    db.record_event("launched", name="job1", host="slop-01")
    db.record_event("completed", name="job1")
    events = db.events_recent(10)
    assert len(events) == 2
    assert events[0]["event"] == "launched"
    db.close()


def test_wait_write_raises_on_failed_write_and_writer_recovers(tmp_path):
    db = ChronohornDB(tmp_path / "test.db")
    with pytest.raises(sqlite3.OperationalError):
        db._write(
            "INSERT INTO definitely_missing_table(x) VALUES (?)",
            (1,),
            wait=True,
        )
    db.record_event("writer_recovered")
    events = db.events_recent(10)
    assert len(events) == 1
    assert events[0]["event"] == "writer_recovered"
    db.close()


def test_write_many_failure_rolls_back_partial_batch(tmp_path):
    db = ChronohornDB(tmp_path / "test.db")
    with pytest.raises(sqlite3.OperationalError):
        db._write_many(
            [
                (
                    "INSERT INTO events (ts, event, data) VALUES (?, ?, ?)",
                    (123.0, "partial_batch", None),
                ),
                (
                    "INSERT INTO definitely_missing_table(x) VALUES (?)",
                    (1,),
                ),
            ],
            wait=True,
        )
    assert db.events_recent(10) == []
    db.record_event("writer_recovered_after_batch")
    events = db.events_recent(10)
    assert len(events) == 1
    assert events[0]["event"] == "writer_recovered_after_batch"
    db.close()


def test_write_without_wait_commits_before_return(tmp_path):
    db = ChronohornDB(tmp_path / "test.db")
    db._write(
        "INSERT INTO events (ts, event, data) VALUES (?, ?, ?)",
        (123.0, "sync_commit", None),
    )
    events = db.events_recent(10)
    assert len(events) == 1
    assert events[0]["event"] == "sync_commit"
    db.close()


def test_concurrent_writes_are_serialized_and_preserved(tmp_path):
    db = ChronohornDB(tmp_path / "test.db")
    barrier = threading.Barrier(8)
    errors: list[BaseException] = []

    def _worker(idx: int) -> None:
        try:
            barrier.wait(timeout=5)
            db.record_event("parallel_write", slot=idx)
        except BaseException as exc:  # pragma: no cover - failures surface in assertion
            errors.append(exc)

    threads = [threading.Thread(target=_worker, args=(idx,)) for idx in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert errors == []
    events = db.events_recent(20)
    assert len(events) == 8
    assert {json.loads(event["data"])["slot"] for event in events} == set(range(8))
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


def test_open_read_only_rejects_writes(tmp_path):
    db = ChronohornDB(tmp_path / "test.db")
    db.record_event("seeded")
    db.close()

    ro = ChronohornDB.open_read_only(tmp_path / "test.db")
    with pytest.raises(sqlite3.OperationalError):
        ro.query("CREATE TABLE should_fail (id INTEGER)")
    ro.close()


def test_open_read_only_missing_db_raises(tmp_path):
    with pytest.raises(sqlite3.OperationalError):
        ChronohornDB.open_read_only(tmp_path / "missing" / "test.db")


def test_active_jobs_matches_manifest_by_path_or_basename(tmp_path):
    db = ChronohornDB(tmp_path / "test.db")
    manifest_path = tmp_path / "nested" / "scan.jsonl"
    manifest_path.parent.mkdir()
    manifest_path.write_text("")
    manifest_abs = str(manifest_path.resolve())

    db.record_job("job1", manifest=manifest_abs)

    assert [job["name"] for job in db.active_jobs(manifest=manifest_abs)] == ["job1"]
    assert [job["name"] for job in db.active_jobs(manifest="scan.jsonl")] == ["job1"]
    db.close()
