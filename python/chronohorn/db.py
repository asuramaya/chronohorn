"""ChronohornDB: SQLite as the runtime's memory.

The database is the live truth. JSON files are archives.
All writes go to the DB first. All reads come from the DB.

Single-writer discipline: all mutations are serialized through a
dedicated writer thread via a queue. Reads use a separate connection
in autocommit mode so each SELECT sees the latest committed data.
"""
from __future__ import annotations

import hashlib
import json
import logging
import queue
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

CURRENT_SCHEMA_VERSION = 1

DEFAULT_DB_PATH = Path("out/chronohorn.db")


class ChronohornDB:
    def __init__(self, path: Path | str = DEFAULT_DB_PATH, read_only: bool = False) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._read_only = read_only

        # Main connection for reads (thread-safe with check_same_thread=False).
        # isolation_level=None puts connection in autocommit mode so each
        # SELECT sees the latest committed data from the writer connection.
        self._conn = sqlite3.connect(
            str(self._path), check_same_thread=False, isolation_level=None
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")

        if not read_only:
            # Write queue + dedicated writer thread
            self._write_queue: queue.Queue = queue.Queue()
            self._writer_conn = sqlite3.connect(
                str(self._path), check_same_thread=False
            )
            self._writer_conn.row_factory = sqlite3.Row
            self._writer_conn.execute("PRAGMA journal_mode=WAL")
            self._writer_conn.execute("PRAGMA synchronous=NORMAL")
            self._writer_thread = threading.Thread(
                target=self._writer_loop, daemon=False
            )
            self._writer_thread.start()

        self._create_tables()
        self._migrate()

    @classmethod
    def open_read_only(cls, path: Path | str = DEFAULT_DB_PATH) -> "ChronohornDB":
        """Open DB for read-only queries. Safe for concurrent access."""
        return cls(path, read_only=True)

    # === WRITER INFRASTRUCTURE ===

    def _writer_loop(self) -> None:
        """Dedicated writer thread: processes all mutations sequentially."""
        while True:
            event = None
            try:
                sql, params, event = self._write_queue.get()
                if sql is None:  # shutdown signal
                    self._write_queue.task_done()
                    break
                self._writer_conn.execute(sql, params)
                self._writer_conn.commit()
                if event:
                    event.set()  # signal completion
            except Exception:
                logger.exception("DB write failed: %s", sql[:200] if sql else "(none)")
                if event:
                    event.set()
            self._write_queue.task_done()

    def _write(self, sql: str, params: tuple = (), *, wait: bool = False) -> None:
        """Queue a write. If wait=True, blocks until the write completes."""
        if self._read_only:
            raise RuntimeError("DB is read-only")
        event = threading.Event() if wait else None
        self._write_queue.put((sql, params, event))
        if event:
            event.wait()

    def _write_many(
        self, operations: list[tuple[str, tuple]], *, wait: bool = False
    ) -> None:
        """Queue multiple writes as a batch."""
        if self._read_only:
            raise RuntimeError("DB is read-only")
        if not operations:
            return
        event = threading.Event() if wait else None
        for i, (sql, params) in enumerate(operations):
            is_last = i == len(operations) - 1
            self._write_queue.put((sql, params, event if is_last else None))
        if event:
            event.wait()

    # === SCHEMA MANAGEMENT ===

    def _create_tables(self) -> None:
        # Use writer connection directly at startup (before any queued writes)
        conn = self._writer_conn if not self._read_only else self._conn
        conn.executescript(
            """
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
        """
        )
        conn.commit()

    def _migrate(self) -> None:
        """Run schema migrations if needed."""
        conn = self._writer_conn if not self._read_only else self._conn
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER)"
        )
        row = conn.execute(
            "SELECT MAX(version) as v FROM schema_version"
        ).fetchone()
        current = row["v"] if row and row["v"] else 0

        if current < 1:
            # Version 1: initial schema (already created by _create_tables)
            conn.execute("INSERT INTO schema_version (version) VALUES (1)")
            conn.commit()

        # Future migrations go here:
        # if current < 2:
        #     conn.execute("ALTER TABLE results ADD COLUMN new_field TEXT")
        #     conn.execute("INSERT INTO schema_version (version) VALUES (2)")
        #     conn.commit()

    def close(self) -> None:
        if not self._read_only:
            self._write_queue.join()  # drain pending writes
            self._write_queue.put((None, None, None))  # shutdown signal
            self._writer_thread.join(timeout=10)
            self._writer_conn.close()
        self._conn.close()

    # === CONFIG MANAGEMENT ===

    def _config_hash(self, cfg: dict[str, Any]) -> str:
        # Hash the architecture-relevant fields only
        keys = sorted(
            k
            for k in cfg
            if k not in ("steps", "seed", "lr", "batch_size", "profile")
        )
        blob = json.dumps({k: cfg.get(k) for k in keys}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def upsert_config(self, cfg: dict[str, Any]) -> int:
        h = self._config_hash(cfg)
        row = self._conn.execute(
            "SELECT id FROM configs WHERE hash = ?", (h,)
        ).fetchone()
        if row:
            return row["id"]
        self._write(
            """
            INSERT INTO configs (hash, scale, seq_len, substrate_mode, num_blocks,
                block_mixing_ratio, block_stride, state_dim, num_heads, patch_size,
                patch_decoder, num_hemispheres, readout, readout_depth, local_window,
                oscillatory_frac, oscillatory_schedule, input_proj_scheme, memory_kind,
                local_poly_order, substrate_poly_order, training_noise, adaptive_reg,
                params, int6_mb, json_blob)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                h,
                cfg.get("scale"),
                cfg.get("seq_len"),
                cfg.get("substrate_mode"),
                cfg.get("num_blocks", 1),
                cfg.get("block_mixing_ratio", 0.25),
                cfg.get("block_stride", 1),
                cfg.get("state_dim", 0),
                cfg.get("num_heads", 1),
                cfg.get("patch_size", 1),
                cfg.get("patch_causal_decoder", "none"),
                cfg.get("num_hemispheres", 1),
                cfg.get("linear_readout_kind", "mlp"),
                cfg.get("linear_readout_depth", 1),
                cfg.get("local_window", 4),
                cfg.get("oscillatory_frac"),
                cfg.get("oscillatory_schedule", "logspace"),
                cfg.get("input_proj_scheme", "random"),
                cfg.get("memory_kind", "none"),
                cfg.get("local_poly_order", 1),
                cfg.get("substrate_poly_order", 1),
                cfg.get("training_noise", 0),
                cfg.get("adaptive_reg", False),
                cfg.get("params"),
                cfg.get("int6_mb"),
                json.dumps(cfg, sort_keys=True),
            ),
            wait=True,
        )
        # Now read the id (reader conn in autocommit sees the committed write)
        row = self._conn.execute(
            "SELECT id FROM configs WHERE hash = ?", (h,)
        ).fetchone()
        return row["id"]

    # === WRITE PATH ===

    def record_job(
        self,
        name: str,
        *,
        manifest: str = "",
        parent: str = "",
        config: dict | None = None,
        steps: int = 0,
        seed: int = 42,
        lr: float = 0,
        batch_size: int = 16,
        command: str = "",
    ) -> None:
        config_id = self.upsert_config(config) if config else None
        self._write(
            """
            INSERT OR REPLACE INTO jobs (name, config_id, manifest, parent, state,
                steps, seed, lr, batch_size, command)
            VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?)
        """,
            (
                name,
                config_id,
                manifest,
                parent,
                steps,
                seed,
                lr,
                batch_size,
                command,
            ),
            wait=True,
        )

    def record_launch(
        self,
        name: str,
        *,
        host: str,
        launcher: str = "",
        container: str = "",
        remote_run: str = "",
    ) -> None:
        # Upsert: update if exists, insert if not
        existing = self._conn.execute(
            "SELECT name FROM jobs WHERE name = ?", (name,)
        ).fetchone()
        if existing:
            self._write(
                """
                UPDATE jobs SET state = 'dispatched', host = ?, launcher = ?,
                    launched_at = ?, container = ?, remote_run = ?
                WHERE name = ?
            """,
                (host, launcher, time.time(), container, remote_run, name),
                wait=True,
            )
        else:
            self._write(
                """
                INSERT INTO jobs (name, state, host, launcher, launched_at,
                    container, remote_run)
                VALUES (?, 'dispatched', ?, ?, ?, ?, ?)
            """,
                (name, host, launcher, time.time(), container, remote_run),
                wait=True,
            )

    def record_running(self, name: str) -> None:
        self._write(
            "UPDATE jobs SET state = 'running' WHERE name = ?",
            (name,),
            wait=True,
        )

    def record_probe(
        self,
        name: str,
        step: int,
        bpb: float,
        *,
        loss: float = 0,
        tflops: float = 0,
        elapsed_sec: float = 0,
    ) -> None:
        self._write(
            """
            INSERT OR REPLACE INTO probes (name, step, bpb, loss, tflops, elapsed_sec)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (name, step, bpb, loss, tflops, elapsed_sec),
            wait=True,
        )

    def record_result(
        self,
        name: str,
        payload: dict[str, Any],
        *,
        json_archive: str = "",
    ) -> None:
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
        if (
            not illegal
            and "patch" in name
            and "cpatch" not in name
            and "hybrid" not in name
            and bpb < 1.0
        ):
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

        # Try to reuse config from the jobs table (manifest has richer config)
        job_row = self._conn.execute(
            "SELECT config_id FROM jobs WHERE name = ?", (name,)
        ).fetchone()
        if job_row and job_row["config_id"]:
            config_id = job_row["config_id"]
        else:
            # Fall back to extracting from result JSON
            config_dict = {
                "scale": m.get("scale"),
                "seq_len": train.get("seq_len"),
                "substrate_mode": m.get(
                    "substrate_mode", train.get("substrate_mode")
                ),
                "num_blocks": m.get("num_blocks", train.get("num_blocks", 1)),
                "patch_size": train.get("patch_size", 1),
                "patch_causal_decoder": train.get(
                    "patch_causal_decoder", "none"
                ),
                "linear_readout_kind": m.get("linear_readout_kind"),
                "local_window": m.get("local_window"),
                "oscillatory_frac": m.get("oscillatory_frac"),
                "params": m.get("params"),
                "int6_mb": round(
                    m.get("params", 0) * 6 / 8 / 1024 / 1024, 2
                )
                if m.get("params")
                else None,
            }
            config_id = self.upsert_config(config_dict)

        # Batch all writes for this result
        ops: list[tuple[str, tuple]] = []

        ops.append(
            (
                """
            INSERT OR REPLACE INTO results (name, config_id, bpb, train_bpb,
                overfit_pct, tok_s, tflops_s, total_tflops, wall_sec, slope,
                illegal, json_archive, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
                (
                    name,
                    config_id,
                    bpb,
                    m.get("train_bpb"),
                    m.get("overfit_pct"),
                    perf.get("tokens_per_second", 0),
                    tflops_s,
                    round(tflops_s * elapsed, 1)
                    if tflops_s and elapsed
                    else 0,
                    round(elapsed, 1),
                    slope,
                    illegal,
                    json_archive,
                    time.time(),
                ),
            )
        )

        # Ingest probes
        for p in probes:
            pbpb = p.get("bpb") or p.get("test_bpb")
            if pbpb and p.get("step"):
                ops.append(
                    (
                        """
                    INSERT OR REPLACE INTO probes (name, step, bpb, loss,
                        elapsed_sec)
                    VALUES (?, ?, ?, ?, ?)
                """,
                        (
                            name,
                            p["step"],
                            pbpb,
                            p.get("eval_loss", 0),
                            p.get("elapsed_sec", 0),
                        ),
                    )
                )

        # Update job state
        ops.append(
            (
                """
            UPDATE jobs SET state = 'completed', completed_at = ?
            WHERE name = ?
        """,
                (time.time(), name),
            )
        )

        self._write_many(ops, wait=True)

        # Auto-forecast
        try:
            self.compute_and_store_forecast(name)
        except Exception:
            logger.warning("Forecast failed for %s", name, exc_info=True)

    def record_fleet(
        self,
        host: str,
        *,
        online: bool,
        gpu_busy: bool = False,
        containers: list[str] | None = None,
    ) -> None:
        self._write(
            """
            INSERT INTO fleet (ts, host, online, gpu_busy, containers)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                time.time(),
                host,
                online,
                gpu_busy,
                json.dumps(containers or []),
            ),
            wait=True,
        )

    def record_event(self, event: str, **data: Any) -> None:
        self._write(
            """
            INSERT INTO events (ts, event, data) VALUES (?, ?, ?)
        """,
            (time.time(), event, json.dumps(data) if data else None),
            wait=True,
        )

    # === READ PATH ===

    def compute_and_store_forecast(self, name: str) -> None:
        """Compute forecast for a result and store in forecasts table."""
        # Get the result's probes
        probes = self.learning_curve(name)
        if len(probes) < 3:
            return  # not enough data to forecast

        # Simple power-law fit on the probe curve
        # bpb = a * step^(-b) + c
        # We use the last few points to estimate the slope
        import math

        steps = [p["step"] for p in probes if p["bpb"]]
        bpbs = [p["bpb"] for p in probes if p["bpb"]]

        if len(steps) < 2 or len(bpbs) < 2:
            return

        # Marginal: slope between last two points
        s1, s2 = steps[-2], steps[-1]
        b1, b2 = bpbs[-2], bpbs[-1]

        if s2 <= s1 or b1 <= b2:
            marginal = 0
        else:
            marginal = (b1 - b2) / max(s2 - s1, 1)  # bpb per step

        # Simple log-linear extrapolation to golf budget
        # log(bpb) ≈ intercept + slope * log(step)
        log_steps = [math.log(max(s, 1)) for s in steps]
        log_bpbs = [math.log(max(b, 0.01)) for b in bpbs]

        n = len(log_steps)
        sum_x = sum(log_steps)
        sum_y = sum(log_bpbs)
        sum_xy = sum(x * y for x, y in zip(log_steps, log_bpbs))
        sum_x2 = sum(x * x for x in log_steps)

        denom = n * sum_x2 - sum_x * sum_x
        if abs(denom) < 1e-10:
            return

        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n

        # R²
        mean_y = sum_y / n
        ss_tot = sum((y - mean_y) ** 2 for y in log_bpbs)
        ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(log_steps, log_bpbs))
        r2 = 1 - ss_res / max(ss_tot, 1e-10)

        # Forecast at golf budget (~2M steps)
        budget_steps = 2_000_000
        forecast_log_bpb = intercept + slope * math.log(budget_steps)
        forecast_bpb = math.exp(max(forecast_log_bpb, -5))  # clamp

        # Clamp: don't forecast below 80% of current best
        forecast_bpb = max(forecast_bpb, bpbs[-1] * 0.5)

        self._write("""
            INSERT OR REPLACE INTO forecasts (name, method, r2, forecast_bpb, marginal_per_tflop, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (name, "log_linear", round(r2, 4), round(forecast_bpb, 4), round(marginal * 1e6, 4), time.time()), wait=True)

    def frontier(self, top_k: int = 20) -> list[dict]:
        rows = self._conn.execute("""
            SELECT r.*, c.scale, c.seq_len, c.substrate_mode, c.num_blocks,
                   c.patch_size, c.patch_decoder, c.readout, c.params, c.int6_mb,
                   f.forecast_bpb, f.r2 as fc_r2, f.marginal_per_tflop as fc_marginal
            FROM results r
            LEFT JOIN configs c ON r.config_id = c.id
            LEFT JOIN forecasts f ON r.name = f.name
            WHERE NOT r.illegal ORDER BY r.bpb LIMIT ?
        """, (top_k,)).fetchall()
        return [dict(r) for r in rows]

    def learning_curve(self, name: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT step, bpb, tflops, elapsed_sec FROM probes "
            "WHERE name = ? ORDER BY step",
            (name,),
        ).fetchall()
        return [dict(r) for r in rows]

    def compare_curves(self, names: list[str]) -> dict[str, list[dict]]:
        return {name: self.learning_curve(name) for name in names}

    def marginal_rank(self, top_k: int = 20) -> list[dict]:
        rows = self._conn.execute(
            """
            SELECT r.name, r.bpb, r.slope, r.total_tflops, r.illegal,
                   c.substrate_mode, c.num_blocks, c.patch_size
            FROM results r LEFT JOIN configs c ON r.config_id = c.id
            WHERE NOT r.illegal AND r.slope IS NOT NULL AND r.slope > 0
            ORDER BY r.slope DESC LIMIT ?
        """,
            (top_k,),
        ).fetchall()
        return [dict(r) for r in rows]

    def pending_jobs(self, manifest: str | None = None) -> list[dict]:
        if manifest:
            rows = self._conn.execute(
                "SELECT * FROM jobs WHERE state = 'pending' AND manifest = ?",
                (manifest,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM jobs WHERE state = 'pending'"
            ).fetchall()
        return [dict(r) for r in rows]

    def running_jobs(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM jobs WHERE state IN ('dispatched', 'running')"
        ).fetchall()
        return [dict(r) for r in rows]

    def result_count(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM results"
        ).fetchone()[0]

    def best_bpb(self, legal_only: bool = True) -> float | None:
        where = "WHERE NOT illegal" if legal_only else ""
        row = self._conn.execute(
            f"SELECT MIN(bpb) FROM results {where}"
        ).fetchone()
        return row[0] if row else None

    def events_recent(self, limit: int = 30) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM events ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def fleet_latest(self) -> dict[str, dict]:
        rows = self._conn.execute(
            """
            SELECT f.* FROM fleet f
            INNER JOIN (
                SELECT host, MAX(ts) as max_ts FROM fleet GROUP BY host
            ) latest
            ON f.host = latest.host AND f.ts = latest.max_ts
        """
        ).fetchall()
        result = {}
        for r in rows:
            result[r["host"]] = {
                "online": bool(r["online"]),
                "gpu_busy": bool(r["gpu_busy"]),
                "containers": json.loads(r["containers"])
                if r["containers"]
                else [],
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
                if isinstance(payload, dict) and payload.get(
                    "model", {}
                ).get("test_bpb"):
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
                existing = self._conn.execute(
                    "SELECT state FROM jobs WHERE name = ?", (name,)
                ).fetchone()
                if existing and existing["state"] == "completed":
                    continue
                config = {
                    k: row.get(k)
                    for k in row
                    if k
                    not in (
                        "name",
                        "command",
                        "goal",
                        "hosts",
                        "image",
                        "gpu",
                        "source_dir",
                        "snapshot_paths",
                        "remote_cwd_rel",
                        "env",
                        "launcher",
                        "backend",
                        "resource_class",
                        "workload_kind",
                        "work_tokens",
                        "family",
                    )
                }
                self.record_job(
                    name,
                    manifest=path.name,
                    config=config,
                    steps=row.get("steps", 0),
                    seed=row.get("seed", 42),
                    lr=row.get("learning_rate", 0),
                    batch_size=row.get("batch_size", 16),
                    command=row.get("command", ""),
                )
                count += 1
            except (json.JSONDecodeError, KeyError):
                pass
        self.record_event("ingest_manifest", path=str(path), count=count)
        return count
