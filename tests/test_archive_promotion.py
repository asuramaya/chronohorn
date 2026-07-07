"""imported_archive → controlled trust ratchet (task #24).

Population is derived from a run's job manifest. Promotion re-points that
manifest so the population query reads the run as controlled. The ratchet is
promote-ONLY (never demotes), and Rule A promotes automatically only when a
controlled run REPRODUCES the archived one at matching config within tolerance.
"""
from __future__ import annotations

from chronohorn.db import (
    ChronohornDB,
    IMPORTED_RESULT_MANIFEST,
    PROMOTED_RESULT_MANIFEST,
)


def _mk(db: ChronohornDB, name: str, bpb: float, manifest: str, seed: int) -> None:
    db.record_result(name, {
        "model": {"test_bpb": bpb, "linear_modes": 512, "scale": 2.0},
        "config": {"train": {"steps": 1000, "lr": 0.001, "batch_size": 8, "seq_len": 512}},
        "training": {"performance": {}, "probes": []},
    })
    db._write(
        "INSERT OR REPLACE INTO jobs (name, manifest, steps, seed, lr, batch_size, state) "
        "VALUES (?, ?, ?, ?, ?, ?, 'completed')",
        (name, manifest, 1000, seed, 0.001, 8), wait=True)


def _manifest(db: ChronohornDB, name: str) -> str:
    return db._read("SELECT manifest FROM jobs WHERE name = ?", (name,))[0]["manifest"]


def test_promote_ratchet_flips_population_and_guards(tmp_path):
    db = ChronohornDB(tmp_path / "t.db")
    _mk(db, "ctrl", 1.900, "real.jsonl", 1)
    _mk(db, "imp", 1.905, IMPORTED_RESULT_MANIFEST, 2)

    out = db.promote_imported_run("imp", reason="manual: known-good provenance")
    assert out["promoted"] is True
    assert _manifest(db, "imp") == PROMOTED_RESULT_MANIFEST  # now reads as controlled

    # audit trail written
    events = db._read("SELECT event FROM events WHERE event = 'archive_promoted'")
    assert len(events) == 1

    # ratchet is promote-only: refuse already-promoted, already-controlled, no-job
    for bad in ("imp", "ctrl", "does-not-exist"):
        try:
            db.promote_imported_run(bad, reason="x")
            raise AssertionError(f"ratchet should have refused {bad}")
        except ValueError:
            pass


def test_auto_promote_only_reproduced(tmp_path):
    db = ChronohornDB(tmp_path / "t.db")
    _mk(db, "ctrl", 1.900, "real.jsonl", 1)              # controlled reference
    _mk(db, "imp_near", 1.905, IMPORTED_RESULT_MANIFEST, 2)   # Δ0.005 — reproduced
    _mk(db, "imp_far", 2.200, IMPORTED_RESULT_MANIFEST, 3)    # Δ0.300 — not reproduced

    # dry_run must not write
    candidates = db.auto_promote_reproduced(dry_run=True, legality="all")
    assert {c["name"] for c in candidates} == {"imp_near"}
    assert _manifest(db, "imp_near") == IMPORTED_RESULT_MANIFEST

    # real run promotes only the reproduced one
    db.auto_promote_reproduced(dry_run=False, legality="all")
    assert _manifest(db, "imp_near") == PROMOTED_RESULT_MANIFEST
    assert _manifest(db, "imp_far") == IMPORTED_RESULT_MANIFEST
