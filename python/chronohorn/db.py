"""ChronohornDB: SQLite as the runtime's memory.

The database is the live truth. JSON files are archives.
All writes go to the DB first. All reads come from the DB.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Sequence


DEFAULT_DB_PATH = Path("out/chronohorn.db")


class ChronohornDB:
    def __init__(self, path: Path | str = DEFAULT_DB_PATH) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS configs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hash TEXT UNIQUE NOT NULL,
                scale REAL,
                seq_len INTEGER,
                substrate_mode TEXT,
                num_blocks INTEGER DEFAULT 1,
                block_mixing_ratio REAL DEFAULT 0.25,
                block_stride INTEGER DEFAULT 1,
                state_dim INTEGER DEFAULT 0,
                num_heads INTEGER DEFAULT 1,
                patch_size INTEGER DEFAULT 1,
                patch_decoder TEXT DEFAULT 'none',
                num_hemispheres INTEGER DEFAULT 1,
                readout TEXT DEFAULT 'mlp',
                readout_depth INTEGER DEFAULT 1,
                local_window INTEGER DEFAULT 4,
                oscillatory_frac REAL,
                oscillatory_schedule TEXT DEFAULT 'logspace',
                input_proj_scheme TEXT DEFAULT 'random',
                memory_kind TEXT DEFAULT 'none',
                local_poly_order INTEGER DEFAULT 1,
                substrate_poly_order INTEGER DEFAULT 1,
                training_noise REAL DEFAULT 0,
                adaptive_reg BOOLEAN DEFAULT 0,
                params INTEGER,
                int6_mb REAL,
                json_blob TEXT
            );

            CREATE TABLE IF NOT EXISTS jobs (
                name TEXT PRIMARY KEY,
                config_id INTEGER REFERENCES configs(id),
                manifest TEXT,
                parent TEXT,
                state TEXT DEFAULT 'pending',
                steps INTEGER,
                seed INTEGER DEFAULT 42,
                lr REAL,
                batch_size INTEGER DEFAULT 16,
                host TEXT,
                launcher TEXT,
                launched_at REAL,
                completed_at REAL,
                container TEXT,
                remote_run TEXT,
                command TEXT
            );

            CREATE TABLE IF NOT EXISTS results (
                name TEXT PRIMARY KEY REFERENCES jobs(name),
                config_id INTEGER REFERENCES configs(id),
                bpb REAL NOT NULL,
                train_bpb REAL,
                overfit_pct REAL,
                tok_s REAL,
                tflops_s REAL,
                total_tflops REAL,
                wall_sec REAL,
                slope REAL,
                illegal BOOLEAN DEFAULT 0,
                json_archive TEXT,
                created_at REAL
            );

            CREATE TABLE IF NOT EXISTS probes (
                name TEXT NOT NULL,
                step INTEGER NOT NULL,
                bpb REAL,
                loss REAL,
                tflops REAL,
                elapsed_sec REAL,
                PRIMARY KEY (name, step)
            );

            CREATE TABLE IF NOT EXISTS forecasts (
                name TEXT PRIMARY KEY,
                method TEXT,
                r2 REAL,
                forecast_bpb REAL,
                marginal_per_tflop REAL,
                forecast_low REAL,
                forecast_high REAL,
                updated_at REAL
            );

            CREATE TABLE IF NOT EXISTS fleet (
                ts REAL,
                host TEXT,
                online BOOLEAN,
                gpu_busy BOOLEAN,
                containers TEXT,
                PRIMARY KEY (ts, host)
            );

            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                event TEXT NOT NULL,
                data TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_results_bpb ON results(bpb);
            CREATE INDEX IF NOT EXISTS idx_results_illegal ON results(illegal);
            CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state);
            CREATE INDEX IF NOT EXISTS idx_jobs_manifest ON jobs(manifest);
            CREATE INDEX IF NOT EXISTS idx_probes_name ON probes(name);
            CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
        """)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # === CONFIG MANAGEMENT ===

    def _config_hash(self, cfg: dict[str, Any]) -> str:
        # Hash the architecture-relevant fields only
        keys = sorted(k for k in cfg if k not in ('steps', 'seed', 'lr', 'batch_size', 'profile'))
        blob = json.dumps({k: cfg.get(k) for k in keys}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def upsert_config(self, cfg: dict[str, Any]) -> int:
        h = self._config_hash(cfg)
        row = self._conn.execute("SELECT id FROM configs WHERE hash = ?", (h,)).fetchone()
        if row:
            return row["id"]
        self._conn.execute("""
            INSERT INTO configs (hash, scale, seq_len, substrate_mode, num_blocks,
                block_mixing_ratio, block_stride, state_dim, num_heads, patch_size,
                patch_decoder, num_hemispheres, readout, readout_depth, local_window,
                oscillatory_frac, oscillatory_schedule, input_proj_scheme, memory_kind,
                local_poly_order, substrate_poly_order, training_noise, adaptive_reg,
                params, int6_mb, json_blob)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            h, cfg.get("scale"), cfg.get("seq_len"), cfg.get("substrate_mode"),
            cfg.get("num_blocks", 1), cfg.get("block_mixing_ratio", 0.25),
            cfg.get("block_stride", 1), cfg.get("state_dim", 0),
            cfg.get("num_heads", 1), cfg.get("patch_size", 1),
            cfg.get("patch_causal_decoder", "none"), cfg.get("num_hemispheres", 1),
            cfg.get("linear_readout_kind", "mlp"), cfg.get("linear_readout_depth", 1),
            cfg.get("local_window", 4), cfg.get("oscillatory_frac"),
            cfg.get("oscillatory_schedule", "logspace"), cfg.get("input_proj_scheme", "random"),
            cfg.get("memory_kind", "none"), cfg.get("local_poly_order", 1),
            cfg.get("substrate_poly_order", 1), cfg.get("training_noise", 0),
            cfg.get("adaptive_reg", False), cfg.get("params"), cfg.get("int6_mb"),
            json.dumps(cfg, sort_keys=True),
        ))
        self._conn.commit()
        return self._conn.execute("SELECT id FROM configs WHERE hash = ?", (h,)).fetchone()["id"]

    # === WRITE PATH ===

    def record_job(self, name: str, *, manifest: str = "", parent: str = "",
                   config: dict | None = None, steps: int = 0, seed: int = 42,
                   lr: float = 0, batch_size: int = 16, command: str = "") -> None:
        config_id = self.upsert_config(config) if config else None
        self._conn.execute("""
            INSERT OR REPLACE INTO jobs (name, config_id, manifest, parent, state, steps, seed, lr, batch_size, command)
            VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?)
        """, (name, config_id, manifest, parent, steps, seed, lr, batch_size, command))
        self._conn.commit()

    def record_launch(self, name: str, *, host: str, launcher: str = "",
                      container: str = "", remote_run: str = "") -> None:
        self._conn.execute("""
            UPDATE jobs SET state = 'dispatched', host = ?, launcher = ?,
                launched_at = ?, container = ?, remote_run = ?
            WHERE name = ?
        """, (host, launcher, time.time(), container, remote_run, name))
        self._conn.commit()

    def record_running(self, name: str) -> None:
        self._conn.execute("UPDATE jobs SET state = 'running' WHERE name = ?", (name,))
        self._conn.commit()

    def record_probe(self, name: str, step: int, bpb: float, *,
                     loss: float = 0, tflops: float = 0, elapsed_sec: float = 0) -> None:
        self._conn.execute("""
            INSERT OR REPLACE INTO probes (name, step, bpb, loss, tflops, elapsed_sec)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (name, step, bpb, loss, tflops, elapsed_sec))
        self._conn.commit()

    def record_result(self, name: str, payload: dict[str, Any], *,
                      json_archive: str = "") -> None:
        m = payload.get("model", {})
        perf = payload.get("training", {}).get("performance", {})
        cfg = payload.get("config", {})
        train = cfg.get("train", {}) if isinstance(cfg.get("train"), dict) else cfg

        bpb = m.get("test_bpb", 0)
        tflops_s = perf.get("estimated_sustained_tflops", 0)
        elapsed = perf.get("elapsed_sec", 0)

        # Detect illegal
        patch_size = train.get("patch_size", 1)
        decoder = train.get("patch_causal_decoder", "NOT_SET")
        illegal = patch_size > 1 and decoder in ("none", "NOT_SET")
        if not illegal and "patch" in name and "cpatch" not in name and "hybrid" not in name and bpb < 1.0:
            illegal = True

        # Compute slope from probes
        probes = payload.get("training", {}).get("probes", [])
        slope = None
        if len(probes) >= 2:
            p1, p2 = probes[-2], probes[-1]
            b1 = p1.get("bpb") or p1.get("test_bpb") or 0
            b2 = p2.get("bpb") or p2.get("test_bpb") or 0
            s1, s2 = p1.get("step", 0), p2.get("step", 0)
            if b1 > b2 and s2 > s1:
                slope = (b1 - b2) / (s2 - s1) * 1000

        # Build config dict for upsert
        config_dict = {
            "scale": m.get("scale"), "seq_len": train.get("seq_len"),
            "substrate_mode": m.get("substrate_mode", train.get("substrate_mode")),
            "num_blocks": m.get("num_blocks", train.get("num_blocks", 1)),
            "patch_size": train.get("patch_size", 1),
            "patch_causal_decoder": train.get("patch_causal_decoder", "none"),
            "linear_readout_kind": m.get("linear_readout_kind"),
            "local_window": m.get("local_window"),
            "oscillatory_frac": m.get("oscillatory_frac"),
            "params": m.get("params"),
            "int6_mb": round(m.get("params", 0) * 6 / 8 / 1024 / 1024, 2) if m.get("params") else None,
        }
        config_id = self.upsert_config(config_dict)

        self._conn.execute("""
            INSERT OR REPLACE INTO results (name, config_id, bpb, train_bpb, overfit_pct,
                tok_s, tflops_s, total_tflops, wall_sec, slope, illegal, json_archive, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            name, config_id, bpb, m.get("train_bpb"), m.get("overfit_pct"),
            perf.get("tokens_per_second", 0), tflops_s,
            round(tflops_s * elapsed, 1) if tflops_s and elapsed else 0,
            round(elapsed, 1), slope, illegal, json_archive, time.time(),
        ))

        # Ingest probes
        for p in probes:
            pbpb = p.get("bpb") or p.get("test_bpb")
            if pbpb and p.get("step"):
                self._conn.execute("""
                    INSERT OR REPLACE INTO probes (name, step, bpb, loss, elapsed_sec)
                    VALUES (?, ?, ?, ?, ?)
                """, (name, p["step"], pbpb, p.get("eval_loss", 0), p.get("elapsed_sec", 0)))

        # Update job state
        self._conn.execute("""
            UPDATE jobs SET state = 'completed', completed_at = ? WHERE name = ?
        """, (time.time(), name))
        self._conn.commit()

    def record_fleet(self, host: str, *, online: bool, gpu_busy: bool = False,
                     containers: list[str] | None = None) -> None:
        self._conn.execute("""
            INSERT INTO fleet (ts, host, online, gpu_busy, containers) VALUES (?, ?, ?, ?, ?)
        """, (time.time(), host, online, gpu_busy, json.dumps(containers or [])))
        self._conn.commit()

    def record_event(self, event: str, **data: Any) -> None:
        self._conn.execute("""
            INSERT INTO events (ts, event, data) VALUES (?, ?, ?)
        """, (time.time(), event, json.dumps(data) if data else None))
        self._conn.commit()

    # === READ PATH ===

    def frontier(self, top_k: int = 20) -> list[dict]:
        rows = self._conn.execute("""
            SELECT r.*, c.scale, c.seq_len, c.substrate_mode, c.num_blocks,
                   c.patch_size, c.patch_decoder, c.readout, c.params, c.int6_mb
            FROM results r LEFT JOIN configs c ON r.config_id = c.id
            WHERE NOT r.illegal ORDER BY r.bpb LIMIT ?
        """, (top_k,)).fetchall()
        return [dict(r) for r in rows]

    def learning_curve(self, name: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT step, bpb, tflops, elapsed_sec FROM probes WHERE name = ? ORDER BY step",
            (name,)
        ).fetchall()
        return [dict(r) for r in rows]

    def compare_curves(self, names: list[str]) -> dict[str, list[dict]]:
        return {name: self.learning_curve(name) for name in names}

    def marginal_rank(self, top_k: int = 20) -> list[dict]:
        rows = self._conn.execute("""
            SELECT r.name, r.bpb, r.slope, r.total_tflops, r.illegal,
                   c.substrate_mode, c.num_blocks, c.patch_size
            FROM results r LEFT JOIN configs c ON r.config_id = c.id
            WHERE NOT r.illegal AND r.slope IS NOT NULL AND r.slope > 0
            ORDER BY r.slope DESC LIMIT ?
        """, (top_k,)).fetchall()
        return [dict(r) for r in rows]

    def pending_jobs(self, manifest: str | None = None) -> list[dict]:
        if manifest:
            rows = self._conn.execute(
                "SELECT * FROM jobs WHERE state = 'pending' AND manifest = ?", (manifest,)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM jobs WHERE state = 'pending'").fetchall()
        return [dict(r) for r in rows]

    def running_jobs(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM jobs WHERE state IN ('dispatched', 'running')"
        ).fetchall()
        return [dict(r) for r in rows]

    def result_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]

    def best_bpb(self, legal_only: bool = True) -> float | None:
        where = "WHERE NOT illegal" if legal_only else ""
        row = self._conn.execute(f"SELECT MIN(bpb) FROM results {where}").fetchone()
        return row[0] if row else None

    def events_recent(self, limit: int = 30) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM events ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def fleet_latest(self) -> dict[str, dict]:
        rows = self._conn.execute("""
            SELECT f.* FROM fleet f
            INNER JOIN (SELECT host, MAX(ts) as max_ts FROM fleet GROUP BY host) latest
            ON f.host = latest.host AND f.ts = latest.max_ts
        """).fetchall()
        result = {}
        for r in rows:
            result[r["host"]] = {
                "online": bool(r["online"]),
                "gpu_busy": bool(r["gpu_busy"]),
                "containers": json.loads(r["containers"]) if r["containers"] else [],
            }
        return result

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        """Raw SQL query for ad hoc analysis."""
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # === BULK IMPORT ===

    def rebuild_from_archive(self, result_dir: str = "out/results") -> int:
        """One-time rebuild from JSON archive files."""
        count = 0
        for p in sorted(Path(result_dir).glob("*.json")):
            try:
                payload = json.loads(p.read_text())
                if isinstance(payload, dict) and payload.get("model", {}).get("test_bpb"):
                    self.record_result(p.stem, payload, json_archive=str(p))
                    count += 1
            except (json.JSONDecodeError, OSError):
                pass
        self.record_event("rebuild", count=count, source=result_dir)
        return count

    def ingest_manifest(self, manifest_path: str) -> int:
        """Import jobs from a JSONL manifest."""
        count = 0
        path = Path(manifest_path)
        for line in path.read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            try:
                row = json.loads(line)
                name = row.get("name", "")
                if not name:
                    continue
                # Don't overwrite completed jobs
                existing = self._conn.execute("SELECT state FROM jobs WHERE name = ?", (name,)).fetchone()
                if existing and existing["state"] == "completed":
                    continue
                config = {k: row.get(k) for k in row if k not in ("name", "command", "goal", "hosts", "image", "gpu", "source_dir", "snapshot_paths", "remote_cwd_rel", "env", "launcher", "backend", "resource_class", "workload_kind", "work_tokens", "family")}
                self.record_job(
                    name, manifest=path.name, config=config,
                    steps=row.get("steps", 0), seed=row.get("seed", 42),
                    lr=row.get("learning_rate", 0), batch_size=row.get("batch_size", 16),
                    command=row.get("command", ""),
                )
                count += 1
            except (json.JSONDecodeError, KeyError):
                pass
        self.record_event("ingest_manifest", path=str(path), count=count)
        return count
