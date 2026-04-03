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
import queue
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Sequence

CURRENT_SCHEMA_VERSION = 5

DEFAULT_DB_PATH = Path("out/chronohorn.db")


class ChronohornDB:
    def __init__(self, path: Path | str = DEFAULT_DB_PATH, read_only: bool = False) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._read_only = read_only
        self._closed = False
        self._read_lock = threading.Lock()  # protects self._conn for concurrent reads

        # Main connection for reads.
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
                target=self._writer_loop, daemon=True
            )
            self._writer_thread.start()

            import atexit
            def _cleanup():
                try:
                    self.close()
                except Exception:
                    pass
            atexit.register(_cleanup)

        if not read_only:
            for _attempt in range(5):
                try:
                    self._create_tables()
                    self._migrate()
                    break
                except sqlite3.OperationalError as e:
                    if "locked" in str(e) and _attempt < 4:
                        time.sleep(0.2 * (_attempt + 1))
                        continue
                    raise

    @classmethod
    def open_read_only(cls, path: Path | str = DEFAULT_DB_PATH) -> "ChronohornDB":
        """Open DB for read-only queries. Safe for concurrent access."""
        return cls(path, read_only=True)

    # === WRITER INFRASTRUCTURE ===

    def _writer_loop(self) -> None:
        """Dedicated writer thread: processes all mutations sequentially.

        This loop must never die — if it does, all subsequent writes are lost.
        Every exception is caught and logged, and the loop continues.
        """
        while True:
            event = None
            try:
                sql, params, event = self._write_queue.get()
                if sql is None:  # shutdown signal
                    self._write_queue.task_done()
                    break
                for attempt in range(3):
                    try:
                        if isinstance(sql, list):
                            for s, p in sql:
                                self._writer_conn.execute(s, p)
                        else:
                            self._writer_conn.execute(sql, params)
                        self._writer_conn.commit()
                        break
                    except sqlite3.OperationalError as e:
                        if "locked" in str(e) and attempt < 2:
                            time.sleep(0.1 * (attempt + 1))
                            continue
                        raise
            except Exception as exc:
                import sys
                print(f"chronohorn db write error: {exc}", file=sys.stderr)
            finally:
                if event:
                    event.set()
                try:
                    self._write_queue.task_done()
                except ValueError:
                    pass  # task_done called too many times

    def _write(self, sql: str, params: tuple = (), *, wait: bool = False) -> None:
        """Queue a write. If wait=True, blocks until the write completes."""
        if self._read_only:
            raise RuntimeError("DB is read-only")
        event = threading.Event() if wait else None
        self._write_queue.put((sql, params, event))
        if event:
            if not event.wait(timeout=10):
                import sys
                print("chronohorn db: write timed out, retrying once", file=sys.stderr)
                retry_event = threading.Event()
                self._write_queue.put((sql, params, retry_event))
                if not retry_event.wait(timeout=10):
                    print("chronohorn db: write retry also timed out, write lost", file=sys.stderr)

    def _write_many(
        self, operations: list[tuple[str, tuple]], *, wait: bool = False
    ) -> None:
        """Queue multiple writes as a batch."""
        if self._read_only:
            raise RuntimeError("DB is read-only")
        if not operations:
            return
        event = threading.Event() if wait else None
        self._write_queue.put((operations, None, event))
        if event:
            if not event.wait(timeout=10):
                import sys
                print("chronohorn db: write_many timed out, retrying once", file=sys.stderr)
                retry_event = threading.Event()
                self._write_queue.put((operations, None, retry_event))
                if not retry_event.wait(timeout=10):
                    print("chronohorn db: write_many retry also timed out, write lost", file=sys.stderr)

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
                json_blob TEXT,
                family TEXT,
                family_config TEXT
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
                created_at REAL,
                family TEXT,
                steps INTEGER,
                seq_len INTEGER,
                params INTEGER,
                int6_mb REAL,
                tflops REAL
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
                updated_at REAL,
                -- saturation metrics (v2)
                asymptote REAL,
                asymptote_alpha REAL,
                asymptote_r2 REAL,
                asymptote_reliable BOOLEAN,
                asymptote_stability REAL,
                headroom REAL,
                saturation_step INTEGER,
                last_doubling_gain REAL,
                last_rate_per_1k REAL,
                saturation_status TEXT
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

            CREATE TABLE IF NOT EXISTS journal (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                kind TEXT NOT NULL,
                run_name TEXT,
                content TEXT NOT NULL,
                tags TEXT
            );

            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                run_name TEXT NOT NULL,
                target_steps INTEGER NOT NULL,
                predicted_bpb REAL NOT NULL,
                method TEXT,
                actual_bpb REAL,
                error_pct REAL,
                UNIQUE(run_name, target_steps)
            );

            CREATE INDEX IF NOT EXISTS idx_journal_ts ON journal(ts);
            CREATE INDEX IF NOT EXISTS idx_journal_kind ON journal(kind);
            CREATE INDEX IF NOT EXISTS idx_predictions_run ON predictions(run_name);
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

        if current < 2:
            # Version 2: model-family agnosticism
            # Add family columns to configs (may already exist from fresh _create_tables)
            existing_config_cols = {
                row[1]
                for row in conn.execute("PRAGMA table_info(configs)").fetchall()
            }
            if "family" not in existing_config_cols:
                conn.execute(
                    "ALTER TABLE configs ADD COLUMN family TEXT DEFAULT 'causal_bank'"
                )
            if "family_config" not in existing_config_cols:
                conn.execute(
                    "ALTER TABLE configs ADD COLUMN family_config TEXT"
                )
            # Add family column to results
            existing_result_cols = {
                row[1]
                for row in conn.execute("PRAGMA table_info(results)").fetchall()
            }
            if "family" not in existing_result_cols:
                conn.execute(
                    "ALTER TABLE results ADD COLUMN family TEXT"
                )
            conn.execute("INSERT INTO schema_version (version) VALUES (2)")
            conn.commit()

        if current < 3:
            # Version 3: denormalize key metrics into results for fast queries
            existing_result_cols = {
                row[1]
                for row in conn.execute("PRAGMA table_info(results)").fetchall()
            }
            for col, typ in [("steps", "INTEGER"), ("seq_len", "INTEGER"),
                             ("params", "INTEGER"), ("int6_mb", "REAL"),
                             ("tflops", "REAL")]:
                if col not in existing_result_cols:
                    conn.execute(f"ALTER TABLE results ADD COLUMN {col} {typ}")
            conn.execute("INSERT INTO schema_version (version) VALUES (3)")
            conn.commit()

        if current < 4:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS journal (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    kind TEXT NOT NULL,
                    run_name TEXT,
                    content TEXT NOT NULL,
                    tags TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    run_name TEXT NOT NULL,
                    target_steps INTEGER NOT NULL,
                    predicted_bpb REAL NOT NULL,
                    method TEXT,
                    actual_bpb REAL,
                    error_pct REAL,
                    UNIQUE(run_name, target_steps)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_journal_ts ON journal(ts)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_journal_kind ON journal(kind)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_run ON predictions(run_name)")
            conn.execute("INSERT INTO schema_version (version) VALUES (4)")
            conn.commit()

        if current < 5:
            # Version 5: asymptote reliability flag
            existing_forecast_cols = {
                row[1]
                for row in conn.execute("PRAGMA table_info(forecasts)").fetchall()
            }
            if "asymptote_reliable" not in existing_forecast_cols:
                conn.execute(
                    "ALTER TABLE forecasts ADD COLUMN asymptote_reliable BOOLEAN"
                )
            conn.execute("INSERT INTO schema_version (version) VALUES (5)")
            conn.commit()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not self._read_only:
            self._write_queue.put((None, None, None))  # shutdown signal
            self._writer_thread.join(timeout=5)
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
        return hashlib.sha256(blob.encode()).hexdigest()[:32]

    @staticmethod
    def _detect_family(cfg: dict[str, Any]) -> str:
        """Detect model family from config dict via the family registry.

        Looks at 'architecture' or 'family' fields.  Returns the canonical
        family ID, or the raw string if no registry match is found.
        """
        from chronohorn.families.registry import detect_family as _reg_detect
        result = _reg_detect(cfg)
        if result is not None:
            return result
        # Unknown architecture — return it as-is for forward compat
        explicit = cfg.get("family") or cfg.get("architecture")
        return str(explicit).lower() if explicit else "unknown"

    @staticmethod
    def _extract_family_config(cfg: dict[str, Any], family: str) -> str | None:
        """Extract family-specific config fields as a JSON blob.

        Fields that already have dedicated DB columns are excluded.
        Everything else is stored in the family_config JSON blob so
        new families don't need schema changes.
        """
        family_fields = {
            k: v
            for k, v in cfg.items()
            if k
            not in (
                "scale",
                "seq_len",
                "steps",
                "seed",
                "lr",
                "learning_rate",
                "batch_size",
                "profile",
                "family",
                "architecture",
                "params",
                "int6_mb",
            )
        }
        return json.dumps(family_fields, sort_keys=True) if family_fields else None

    def upsert_config(self, cfg: dict[str, Any]) -> int:
        h = self._config_hash(cfg)
        row = self._read_one(
            "SELECT id FROM configs WHERE hash = ?", (h,)
        )
        if row:
            return row["id"]

        family = self._detect_family(cfg)
        family_config = self._extract_family_config(cfg, family)

        self._write(
            """
            INSERT OR IGNORE INTO configs (hash, scale, seq_len, substrate_mode, num_blocks,
                block_mixing_ratio, block_stride, state_dim, num_heads, patch_size,
                patch_decoder, num_hemispheres, readout, readout_depth, local_window,
                oscillatory_frac, oscillatory_schedule, input_proj_scheme, memory_kind,
                local_poly_order, substrate_poly_order, training_noise, adaptive_reg,
                params, int6_mb, json_blob, family, family_config)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                family,
                family_config,
            ),
            wait=True,
        )
        # Read back the id — retry to handle WAL visibility lag
        for _ in range(5):
            row = self._read_one(
                "SELECT id FROM configs WHERE hash = ?", (h,)
            )
            if row:
                return row["id"]
            time.sleep(0.01)
        raise RuntimeError(f"upsert_config: config with hash {h} not found after insert")

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
        existing = self._read_one(
            "SELECT name FROM jobs WHERE name = ?", (name,)
        )
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
        family: str | None = None,
        compute_forecast: bool = True,
    ) -> None:
        m = payload.get("model", {})
        perf = payload.get("training", {}).get("performance", {})
        cfg = payload.get("config", {})
        train = cfg.get("train", {}) if isinstance(cfg.get("train"), dict) else cfg

        try:
            bpb = float(m.get("test_bpb", 0))
        except (TypeError, ValueError):
            import sys
            print(f"chronohorn: skipping {name} (invalid bpb: {m.get('test_bpb')!r})", file=sys.stderr)
            return
        if not bpb or bpb <= 0:
            import sys
            print(f"chronohorn: skipping {name} (bpb={bpb!r}, must be > 0)", file=sys.stderr)
            return
        tflops_s = perf.get("estimated_sustained_tflops", 0)
        elapsed = perf.get("elapsed_sec", 0)

        # Detect family from payload if not explicitly provided
        if family is None:
            arch = m.get("architecture") or train.get("architecture") or cfg.get("architecture")
            if arch:
                family = self._detect_family({"architecture": arch})
            else:
                # Pass full payload so registry can check model section for OPC markers
                family = self._detect_family(payload)

        # Delegate illegal detection to the family adapter
        # Inject name into payload so adapter can use name-based heuristics
        from chronohorn.families.registry import detect_illegal as _reg_detect_illegal
        detect_payload = {**payload, "name": name, "_name": name}
        illegal = _reg_detect_illegal(detect_payload, family_id=family)

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
        job_row = self._read_one(
            "SELECT config_id FROM jobs WHERE name = ?", (name,)
        )
        if job_row and job_row["config_id"]:
            config_id = job_row["config_id"]
        else:
            # Fall back to extracting from result JSON — store everything
            config_dict = {}
            # Merge model-level config fields
            for k, v in m.items():
                if k not in ("test_bpb", "train_bpb", "test_eval_loss", "train_eval_loss",
                              "test_bits_per_token", "train_bits_per_token", "overfit_pct",
                              "payload_bytes_est", "payload_mb_est", "initial_trainable_signature"):
                    config_dict[k] = v
            # Merge train-level config fields (lower priority)
            for k, v in train.items():
                if k not in ("steps", "seed", "batch_size", "grad_clip", "log_every",
                              "eval_batches", "seeds") and k not in config_dict:
                    config_dict[k] = v
            config_id = self.upsert_config(config_dict)

        # Compute denormalized fields for results
        steps_val = perf.get("steps_completed") or train.get("steps")
        seq_len_val = train.get("seq_len")
        params_val = m.get("params")
        int6_mb_val = round(params_val * 6 / 8 / 1024 / 1024, 2) if params_val else None
        total_tflops_val = round(tflops_s * elapsed, 1) if tflops_s and elapsed else 0

        # Estimate tflops from flops_per_step if not directly available
        flops_per_step = perf.get("train_step_flops_per_step_est", 0)
        if total_tflops_val == 0 and flops_per_step and steps_val:
            total_tflops_val = round(steps_val * flops_per_step / 1e12, 1)

        # Last resort: estimate per-probe tflops from tok/s and params
        # Rough: 6 FLOPs per param per token (forward ~ 2*params, backward ~ 4*params)
        if not flops_per_step and params_val and perf.get("tokens_per_second") and steps_val:
            est_flops_per_tok = params_val * 6
            est_flops_per_step = est_flops_per_tok * (train.get("batch_size", 64) * train.get("seq_len", 512))
            flops_per_step = est_flops_per_step
            if total_tflops_val == 0:
                total_tflops_val = round(steps_val * flops_per_step / 1e12, 1)

        # Batch all writes for this result
        ops: list[tuple[str, tuple]] = []

        ops.append(
            (
                """
            INSERT OR REPLACE INTO results (name, config_id, bpb, train_bpb,
                overfit_pct, tok_s, tflops_s, total_tflops, wall_sec, slope,
                illegal, json_archive, created_at, family,
                steps, seq_len, params, int6_mb, tflops)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
                (
                    name,
                    config_id,
                    bpb,
                    m.get("train_bpb"),
                    m.get("overfit_pct"),
                    perf.get("tokens_per_second", 0),
                    tflops_s,
                    total_tflops_val,
                    round(elapsed, 1),
                    slope,
                    illegal,
                    json_archive,
                    time.time(),
                    family,
                    steps_val,
                    seq_len_val,
                    params_val,
                    int6_mb_val,
                    total_tflops_val,
                ),
            )
        )

        # Ingest probes (flops_per_step already computed above)
        for p in probes:
            pbpb = p.get("bpb") or p.get("test_bpb")
            if pbpb and p.get("step"):
                probe_tflops = round(p["step"] * flops_per_step / 1e12, 4) if flops_per_step else 0
                ops.append(
                    (
                        """
                    INSERT OR REPLACE INTO probes (name, step, bpb, loss,
                        tflops, elapsed_sec)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                        (
                            name,
                            p["step"],
                            pbpb,
                            p.get("eval_loss", 0),
                            probe_tflops,
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

        # Auto-forecast (skipped during bulk rebuild for performance)
        if compute_forecast:
            try:
                self.compute_and_store_forecast(name)
            except Exception:
                pass  # forecast failure shouldn't block result ingestion

        # G5: Auto-check branch health after ingesting result
        try:
            # Extract prefix: just the version part (e.g., v6, v8, v8h, v11)
            parts = name.split("-")
            prefix = parts[0]
            health = self.branch_health(prefix)
            if health.get("status") == "dead" and health.get("count", 0) >= 5:
                import sys
                print(f"chronohorn: \u26a0 branch '{prefix}' has {health['count']} results, none on frontier (gap: +{health['gap']:.3f} bpb)", file=sys.stderr)
        except Exception:
            pass

        # Auto-reconcile predictions
        try:
            predictions = self._read(
                "SELECT id, predicted_bpb, target_steps FROM predictions WHERE run_name = ?",
                (name,)
            )
            for pred in predictions:
                p = dict(pred)
                if steps_val and p["target_steps"] and steps_val >= p["target_steps"]:
                    error_pct = round((bpb - p["predicted_bpb"]) / max(p["predicted_bpb"], 0.01) * 100, 1)
                    self._write(
                        "UPDATE predictions SET actual_bpb = ?, error_pct = ? WHERE id = ?",
                        (bpb, error_pct, p["id"]),
                    )
                    import sys
                    if abs(error_pct) > 10:
                        print(f"chronohorn: prediction for {name} was {p['predicted_bpb']:.4f}, actual {bpb:.4f} (error: {error_pct:+.1f}%)", file=sys.stderr)
        except Exception:
            pass

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
        """Compute forecast and saturation analysis for a result."""
        from chronohorn.engine.saturation import analyze_saturation

        probes = self.learning_curve(name)

        # Clean probes: filter NaN/inf/invalid, sort, deduplicate
        probes = [p for p in probes if p.get("bpb") and p["bpb"] > 0 and p["bpb"] < 100]
        probes.sort(key=lambda p: p["step"])
        seen_steps: set[int] = set()
        unique_probes: list[dict] = []
        for p in probes:
            if p["step"] not in seen_steps:
                seen_steps.add(p["step"])
                unique_probes.append(p)
        probes = unique_probes

        if len(probes) < 4:  # need at least 4 for meaningful fit
            return

        import math

        steps = [p["step"] for p in probes if p.get("bpb")]
        bpbs = [p["bpb"] for p in probes if p.get("bpb")]

        if len(steps) < 4 or len(bpbs) < 4:
            return

        # Quality gate: don't forecast if max step < 500
        if max(steps) < 500:
            return

        # Quality gate: don't trust saturation analysis with < 4 probes spanning < 2x step range
        step_range = max(steps) / max(min(steps), 1)
        if step_range < 2.0:
            return

        # Marginal: slope between last two points
        s1, s2 = steps[-2], steps[-1]
        b1, b2 = bpbs[-2], bpbs[-1]
        marginal = (b1 - b2) / max(s2 - s1, 1) if s2 > s1 and b1 > b2 else 0

        # Log-linear forecast
        log_steps = [math.log(max(s, 1)) for s in steps]
        log_bpbs = [math.log(max(b, 0.01)) for b in bpbs]
        n = len(log_steps)
        sum_x = sum(log_steps); sum_y = sum(log_bpbs)
        sum_xy = sum(x * y for x, y in zip(log_steps, log_bpbs))
        sum_x2 = sum(x * x for x in log_steps)
        denom = n * sum_x2 - sum_x * sum_x
        if abs(denom) < 1e-10:
            return
        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n
        mean_y = sum_y / n
        ss_tot = sum((y - mean_y) ** 2 for y in log_bpbs)
        ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(log_steps, log_bpbs))
        r2 = 1 - ss_res / max(ss_tot, 1e-10)
        budget_steps = 2_000_000
        forecast_bpb = math.exp(max(intercept + slope * math.log(budget_steps), -5))
        forecast_bpb = max(forecast_bpb, bpbs[-1] * 0.5)

        # Saturation analysis
        sat = analyze_saturation(probes)
        asymptote = sat.get("asymptote")

        self._write("""
            INSERT OR REPLACE INTO forecasts
            (name, method, r2, forecast_bpb, marginal_per_tflop, updated_at,
             asymptote, asymptote_alpha, asymptote_r2, asymptote_reliable,
             asymptote_stability,
             headroom, saturation_step, last_doubling_gain, last_rate_per_1k,
             saturation_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            name, "log_linear", round(r2, 4), round(forecast_bpb, 4),
            round(marginal * 1e6, 4), time.time(),
            asymptote, sat.get("asymptote_alpha"),
            sat.get("asymptote_r2"), sat.get("asymptote_reliable", False),
            sat.get("asymptote_stability"),
            sat.get("headroom"), sat.get("saturation_step"),
            sat.get("last_doubling_gain"), sat.get("last_rate_per_1k"),
            sat.get("status"),
        ), wait=True)

        # Flag discrepancy between log-linear and power-law forecasts
        if asymptote is not None and forecast_bpb is not None:
            if abs(asymptote - forecast_bpb) / max(asymptote, forecast_bpb, 0.01) > 0.1:
                self.record_event("forecast_discrepancy", name=name,
                    forecast_bpb=round(forecast_bpb, 4),
                    asymptote=round(asymptote, 4),
                    discrepancy_pct=round(abs(asymptote - forecast_bpb) / max(asymptote, 0.01) * 100, 1))

    def frontier(self, top_k: int = 20) -> list[dict]:
        if top_k <= 0:
            return []
        rows = self._read("""
            SELECT r.name, r.bpb, r.family, r.train_bpb, r.overfit_pct,
                   r.tok_s, r.wall_sec, r.slope, r.illegal,
                   COALESCE(r.params, c.params) as params,
                   COALESCE(r.int6_mb, c.int6_mb) as int6_mb,
                   COALESCE(r.steps, j.steps) as steps,
                   COALESCE(r.seq_len, c.seq_len) as seq_len,
                   COALESCE(r.tflops, r.total_tflops) as tflops,
                   COALESCE(f.asymptote, f.forecast_bpb) as fc_bpb,
                   COALESCE(f.asymptote_r2, f.r2) as fc_r2,
                   f.headroom, f.saturation_status,
                   c.json_blob as config_json
            FROM results r
            LEFT JOIN configs c ON r.config_id = c.id
            LEFT JOIN forecasts f ON r.name = f.name
            LEFT JOIN jobs j ON r.name = j.name
            WHERE NOT r.illegal ORDER BY r.bpb LIMIT ?
        """, (top_k,))
        canonical_keys = [
            "name", "bpb", "family", "params", "int6_mb", "steps", "seq_len",
            "tok_s", "tflops", "wall_sec", "illegal", "slope",
            "fc_bpb", "fc_r2", "headroom", "overfit_pct", "train_bpb",
            "saturation_status", "config_json",
        ]
        result = []
        for r in rows:
            d = dict(r)
            for k in canonical_keys:
                d.setdefault(k, None)
            result.append(d)
        return result

    # === G3: SIMILAR-CONFIG DETECTION ===

    def find_similar(self, config: dict, threshold: float = 0.2) -> list[dict]:
        """Find results with similar configs (within threshold on all numeric fields)."""
        rows = self._read(
            "SELECT r.name, r.bpb, r.tok_s, c.json_blob FROM results r "
            "JOIN configs c ON r.config_id = c.id "
            "WHERE NOT r.illegal AND r.bpb < 3 ORDER BY r.bpb LIMIT 100"
        )

        similar = []
        for row in rows:
            r = dict(row)
            try:
                other_cfg = json.loads(r["json_blob"]) if r["json_blob"] else {}
            except Exception:
                continue

            # Compare numeric fields
            matches = 0
            total = 0
            for k, v in config.items():
                if isinstance(v, (int, float)) and v != 0:
                    other_v = other_cfg.get(k)
                    if isinstance(other_v, (int, float)) and other_v != 0:
                        total += 1
                        if abs(v - other_v) / max(abs(v), abs(other_v)) <= threshold:
                            matches += 1

            if total > 0 and matches / total >= 0.8:  # 80% of numeric fields within threshold
                similar.append({"name": r["name"], "bpb": r["bpb"], "match_pct": round(matches / total * 100)})

        return similar[:5]

    # === G5: DEAD-BRANCH DETECTOR ===

    def branch_health(self, prefix: str) -> dict:
        """Check if an architecture branch is contributing to the frontier."""
        branch_results = self._read(
            "SELECT name, bpb FROM results WHERE (name LIKE ? OR name = ?) AND NOT illegal ORDER BY bpb LIMIT 20",
            (f"{prefix}-%", prefix)
        )
        if not branch_results:
            return {"prefix": prefix, "count": 0, "status": "no_results"}

        branch_best = min(r["bpb"] for r in branch_results)
        frontier = self.frontier(10)
        frontier_best = frontier[0]["bpb"] if frontier else None
        on_frontier = sum(1 for f in frontier if f["name"] == prefix or f["name"].startswith(prefix + "-"))

        gap = branch_best - frontier_best if frontier_best else 0

        status = "alive" if on_frontier > 0 else ("close" if gap < 0.05 else "dead")

        return {
            "prefix": prefix,
            "count": len(branch_results),
            "best_bpb": round(branch_best, 4),
            "frontier_best": round(frontier_best, 4) if frontier_best else None,
            "gap": round(gap, 4),
            "on_frontier": on_frontier,
            "status": status,
        }

    def frontier_velocity(self, window: int = 10) -> dict:
        """Track frontier improvement rate. Uses bpb ordering, not timestamps."""
        # Get all legal results sorted by creation order (or bpb as proxy)
        results = self._read("""
            SELECT name, bpb, wall_sec, created_at FROM results
            WHERE NOT illegal AND bpb < 3 ORDER BY created_at ASC
        """)

        if len(results) < 2:
            return {"velocity_bpb_per_hour": 0, "trend": "insufficient_data", "improvements": []}

        # Compute running best and track improvements
        results = [dict(r) for r in results]
        running_best = float("inf")
        improvements = []
        total_wall = 0

        for r in results:
            total_wall += r.get("wall_sec") or 0
            if r["bpb"] < running_best:
                delta = running_best - r["bpb"] if running_best < float("inf") else 0
                running_best = r["bpb"]
                if delta > 0.001:  # skip noise
                    improvements.append({"name": r["name"], "bpb": r["bpb"], "delta": round(delta, 4)})

        gpu_hours = total_wall / 3600
        total_delta = sum(i["delta"] for i in improvements)
        velocity = total_delta / max(gpu_hours, 0.01)

        # Trend: compare first half vs second half of improvements
        if len(improvements) >= 4:
            mid = len(improvements) // 2
            first_half = sum(i["delta"] for i in improvements[:mid])
            second_half = sum(i["delta"] for i in improvements[mid:])
            if second_half < first_half * 0.3:
                trend = "decelerating"
            elif second_half > first_half * 1.5:
                trend = "accelerating"
            else:
                trend = "steady"
        elif len(improvements) >= 2:
            # With few improvements, check if last is smaller than first
            if improvements[-1]["delta"] < improvements[0]["delta"] * 0.5:
                trend = "decelerating"
            else:
                trend = "steady"
        else:
            trend = "insufficient_data"

        return {
            "velocity_bpb_per_hour": round(velocity, 4),
            "total_delta": round(total_delta, 4),
            "gpu_hours": round(gpu_hours, 2),
            "trend": trend,
            "improvements": improvements[-window:],
            "current_best": running_best,
        }

    def detect_groups(self) -> list[dict]:
        """Detect experiment groups by name pattern and summarize."""
        import re as _re
        results = self._read("SELECT name, bpb, steps, family FROM results WHERE NOT illegal ORDER BY name")

        # Group by prefix (strip trailing -10k, -20k, -50k, numbers)
        groups = {}
        for r in [dict(x) for x in results]:
            # Strip step suffix and numeric suffix
            prefix = _re.sub(r'-\d+k$|-\d+$', '', r["name"])
            # Also strip -seed* suffix
            prefix = _re.sub(r'-seed\d+$', '', prefix)
            groups.setdefault(prefix, []).append(r)

        # Only return groups with 3+ members
        summaries = []
        for prefix, members in groups.items():
            if len(members) < 3:
                continue
            bpbs = [m["bpb"] for m in members]
            best = min(members, key=lambda m: m["bpb"])
            worst = max(members, key=lambda m: m["bpb"])
            summaries.append({
                "prefix": prefix,
                "count": len(members),
                "best": {"name": best["name"], "bpb": best["bpb"]},
                "worst": {"name": worst["name"], "bpb": worst["bpb"]},
                "spread": round(worst["bpb"] - best["bpb"], 4),
                "members": [m["name"] for m in sorted(members, key=lambda m: m["bpb"])],
            })

        summaries.sort(key=lambda x: x["best"]["bpb"])
        return summaries

    def learning_curve(self, name: str) -> list[dict]:
        rows = self._read(
            "SELECT step, bpb, tflops, elapsed_sec FROM probes "
            "WHERE name = ? ORDER BY step",
            (name,),
        )
        return [{"step": r["step"], "bpb": r["bpb"], "tf": r["tflops"] or 0,
                 "tflops": r["tflops"], "elapsed_sec": r["elapsed_sec"]}
                for r in rows]

    def compare_curves(self, names: list[str]) -> dict[str, list[dict]]:
        return {name: self.learning_curve(name) for name in names}

    def saturation(self, name: str) -> dict:
        """Full saturation analysis for a single experiment."""
        from chronohorn.engine.saturation import analyze_saturation
        probes = self.learning_curve(name)
        return analyze_saturation(probes)

    def saturation_frontier(self, top_k: int = 20) -> list[dict]:
        """Rank experiments by saturation status + headroom."""
        if top_k <= 0:
            return []
        rows = self._read("""
            SELECT f.name, f.forecast_bpb, f.asymptote, f.headroom,
                   f.last_doubling_gain, f.last_rate_per_1k, f.saturation_status,
                   f.saturation_step, f.asymptote_stability, f.asymptote_reliable,
                   r.bpb, r.total_tflops
            FROM forecasts f
            JOIN results r ON f.name = r.name
            WHERE f.asymptote IS NOT NULL AND NOT r.illegal
                  AND f.asymptote_reliable = 1
            ORDER BY f.asymptote ASC LIMIT ?
        """, (top_k,))
        return [dict(r) for r in rows]

    def marginal_rank(self, top_k: int = 20) -> list[dict]:
        if top_k <= 0:
            return []
        rows = self._read(
            """
            SELECT r.name, r.bpb, r.family, r.slope,
                   COALESCE(r.tflops, r.total_tflops) as total_tf,
                   r.steps, r.illegal, r.tok_s, r.wall_sec,
                   COALESCE(f.asymptote, f.forecast_bpb) as fc_bpb
            FROM results r
            LEFT JOIN forecasts f ON r.name = f.name
            WHERE NOT r.illegal AND r.slope IS NOT NULL AND r.slope > 0
            ORDER BY r.slope DESC LIMIT ?
        """,
            (top_k * 3,),  # fetch extra, will re-sort
        )
        result = []
        for r in [dict(row) for row in rows]:
            total_tf = r.get("total_tf") or 0
            # Compute real marginal: need at least some tflops
            if total_tf > 0:
                # marginal = total bpb improvement / total TFLOPs
                # Approximate: use slope * steps / tflops as proxy
                steps = r.get("steps") or 1
                marginal = r["slope"] * steps / 1000 / total_tf if total_tf > 0 else 0
            else:
                # Fallback: estimate tflops from tok/s and wall_sec
                tok_s = r.get("tok_s") or 0
                wall = r.get("wall_sec") or 0
                if tok_s > 0 and wall > 0:
                    # Rough: assume 2 FLOPs per token (very rough)
                    est_tflops = tok_s * wall * 2 / 1e12
                    marginal = r["slope"] * (r.get("steps") or 1) / 1000 / est_tflops if est_tflops > 0 else r["slope"]
                else:
                    marginal = r["slope"]  # last resort

            r["marginal"] = round(marginal, 6)
            r["slope_alive"] = (r.get("slope") or 0) > 0.005 and (r.get("steps") or 0) >= 500
            result.append(r)

        result.sort(key=lambda x: -x["marginal"])
        return result[:top_k]

    def pending_jobs(self, manifest: str | None = None) -> list[dict]:
        if manifest:
            rows = self._read(
                "SELECT * FROM jobs WHERE state = 'pending' AND manifest = ?",
                (manifest,),
            )
        else:
            rows = self._read(
                "SELECT * FROM jobs WHERE state = 'pending'"
            )
        return [dict(r) for r in rows]

    def running_jobs(self) -> list[dict]:
        rows = self._read(
            "SELECT * FROM jobs WHERE state IN ('dispatched', 'running')"
        )
        return [dict(r) for r in rows]

    def result_count(self) -> int:
        return self._read_one(
            "SELECT COUNT(*) FROM results"
        )[0]

    def best_bpb(self, legal_only: bool = True) -> float | None:
        where = "WHERE NOT illegal" if legal_only else ""
        row = self._read_one(
            f"SELECT MIN(bpb) FROM results {where}"
        )
        return row[0] if row else None

    def events_recent(self, limit: int = 30) -> list[dict]:
        if limit <= 0:
            return []
        rows = self._read(
            "SELECT * FROM events ORDER BY ts DESC LIMIT ?", (limit,)
        )
        return [dict(r) for r in reversed(rows)]

    def fleet_latest(self) -> dict[str, dict]:
        rows = self._read(
            """
            SELECT f.* FROM fleet f
            INNER JOIN (
                SELECT host, MAX(ts) as max_ts FROM fleet GROUP BY host
            ) latest
            ON f.host = latest.host AND f.ts = latest.max_ts
        """
        )
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

    def drain_status(self) -> dict:
        """Canonical drain status from jobs table."""
        counts = self.query("""
            SELECT state, COUNT(*) as c FROM jobs GROUP BY state
        """)
        by_state = {r["state"]: r["c"] for r in counts}
        pending = by_state.get("pending", 0)
        running = by_state.get("dispatched", 0) + by_state.get("running", 0)
        completed = by_state.get("completed", 0)
        failed = by_state.get("failed", 0)
        total = sum(by_state.values())

        # Per-manifest breakdown
        manifests_raw = self.query("""
            SELECT manifest,
                   COUNT(*) as total,
                   SUM(CASE WHEN state = 'completed' THEN 1 ELSE 0 END) as done
            FROM jobs WHERE manifest IS NOT NULL AND manifest != ''
            GROUP BY manifest ORDER BY manifest
        """)
        manifests = [{"name": r["manifest"], "total": r["total"], "done": r["done"]}
                     for r in manifests_raw]

        return {
            "pending": pending, "running": running, "completed": completed,
            "failed": failed, "total": total,
            "done": pending == 0 and running == 0 and total > 0,
            "manifests": manifests,
        }

    def summary(self) -> dict:
        """High-level summary of the database state."""
        n = self.result_count()
        best = self.best_bpb()
        families = self.query(
            "SELECT DISTINCT family, COUNT(*) as c FROM results GROUP BY family"
        )
        manifests = self.query(
            "SELECT COUNT(DISTINCT manifest) as c FROM jobs "
            "WHERE manifest IS NOT NULL"
        )
        return {
            "result_count": n,
            "best_bpb": best,
            "families": {r["family"]: r["c"] for r in families},
            "manifest_count": manifests[0]["c"] if manifests else 0,
        }

    def config_summary(self, name: str) -> dict:
        """Return family-specific config fields via the adapter."""
        row = self._read_one("""
            SELECT c.json_blob, c.family FROM configs c
            JOIN results r ON r.config_id = c.id
            WHERE r.name = ?
        """, (name,))
        if not row or not row["json_blob"]:
            return {}
        try:
            from chronohorn.families.registry import get_adapter
            cfg = json.loads(row["json_blob"])
            adapter = get_adapter(row["family"])
            return adapter.config_summary({"config": {"train": cfg}})
        except (KeyError, ImportError, json.JSONDecodeError):
            return json.loads(row["json_blob"])

    def config_diff(self, name1: str, name2: str) -> dict:
        """Compare configs of two runs. Returns changed, only_in_1, only_in_2."""
        row1 = self._read_one("""
            SELECT c.json_blob, c.family FROM configs c
            JOIN results r ON r.config_id = c.id WHERE r.name = ?
        """, (name1,))
        row2 = self._read_one("""
            SELECT c.json_blob, c.family FROM configs c
            JOIN results r ON r.config_id = c.id WHERE r.name = ?
        """, (name2,))
        if not row1 or not row2:
            return {"error": "one or both runs not found"}

        cfg1 = json.loads(row1["json_blob"]) if row1["json_blob"] else {}
        cfg2 = json.loads(row2["json_blob"]) if row2["json_blob"] else {}

        all_keys = set(cfg1) | set(cfg2)
        changed = {}
        only_in_1 = {}
        only_in_2 = {}
        same = {}

        for k in sorted(all_keys):
            v1 = cfg1.get(k)
            v2 = cfg2.get(k)
            if k in cfg1 and k not in cfg2:
                only_in_1[k] = v1
            elif k in cfg2 and k not in cfg1:
                only_in_2[k] = v2
            elif v1 != v2:
                changed[k] = [v1, v2]
            else:
                same[k] = v1

        # Also include metric comparison
        r1 = self._read_one("SELECT bpb, tok_s, params, steps, wall_sec FROM results WHERE name = ?", (name1,))
        r2 = self._read_one("SELECT bpb, tok_s, params, steps, wall_sec FROM results WHERE name = ?", (name2,))

        metrics = {}
        if r1 and r2:
            for k in ["bpb", "tok_s", "params", "steps", "wall_sec"]:
                v1, v2 = dict(r1).get(k), dict(r2).get(k)
                if v1 != v2:
                    metrics[k] = [v1, v2]

        return {"changed": changed, "only_in_1": only_in_1, "only_in_2": only_in_2, "same_count": len(same), "metrics": metrics}

    def what_varied(self, names: list[str] | None = None, family: str | None = None, limit: int = 50) -> dict:
        """Find config keys that vary across a set of runs."""
        if names:
            placeholders = ",".join("?" * len(names))
            rows = self._read(f"""
                SELECT r.name, c.json_blob FROM results r
                JOIN configs c ON r.config_id = c.id
                WHERE r.name IN ({placeholders})
            """, tuple(names))
        elif family:
            rows = self._read("""
                SELECT r.name, c.json_blob FROM results r
                JOIN configs c ON r.config_id = c.id
                WHERE r.family = ? ORDER BY r.bpb LIMIT ?
            """, (family, limit))
        else:
            rows = self._read("""
                SELECT r.name, c.json_blob FROM results r
                JOIN configs c ON r.config_id = c.id
                WHERE NOT r.illegal ORDER BY r.bpb LIMIT ?
            """, (limit,))

        if not rows:
            return {"varied": {}, "constant": {}, "runs": 0}

        configs = {}
        for row in rows:
            try:
                configs[row["name"]] = json.loads(row["json_blob"]) if row["json_blob"] else {}
            except (json.JSONDecodeError, TypeError):
                configs[row["name"]] = {}

        # Find all keys and their values
        all_keys = set()
        for cfg in configs.values():
            all_keys.update(cfg.keys())

        varied = {}
        constant = {}
        for k in sorted(all_keys):
            values = {}
            for name, cfg in configs.items():
                v = cfg.get(k)
                v_str = str(v)
                values.setdefault(v_str, []).append(name)
            if len(values) > 1:
                varied[k] = {v: names_list for v, names_list in values.items()}
            else:
                constant[k] = next(iter(values.keys()))

        return {"varied": varied, "constant": constant, "runs": len(configs)}

    def cost_summary(self) -> dict:
        """Aggregate GPU-hours and compute ROI."""
        rows = self._read("""
            SELECT family, COUNT(*) as runs,
                   SUM(wall_sec) as total_sec,
                   MIN(bpb) as best_bpb, MAX(bpb) as worst_bpb
            FROM results WHERE wall_sec > 0
            GROUP BY family
        """)

        total_sec = sum(r["total_sec"] or 0 for r in rows)
        total_runs = sum(r["runs"] for r in rows)

        families = {}
        for r in rows:
            families[r["family"]] = {
                "runs": r["runs"],
                "gpu_hours": round((r["total_sec"] or 0) / 3600, 2),
                "best_bpb": r["best_bpb"],
            }

        # Overall best
        best = self.best_bpb() or 0

        return {
            "total_gpu_hours": round(total_sec / 3600, 2),
            "total_runs": total_runs,
            "best_bpb": best,
            "families": families,
        }

    def cost_per_run(self, name: str) -> dict:
        """Cost analysis for a single run."""
        row = self._read_one("""
            SELECT wall_sec, bpb, slope, tok_s, params, steps FROM results WHERE name = ?
        """, (name,))
        if not row:
            return {"error": f"run not found: {name}"}
        r = dict(row)
        gpu_hours = (r["wall_sec"] or 0) / 3600
        return {
            "name": name,
            "gpu_hours": round(gpu_hours, 3),
            "bpb": r["bpb"],
            "tok_s": r["tok_s"],
            "slope": r["slope"],
            "bpb_per_gpu_hour": round(r["slope"] * 1000 / max(gpu_hours, 0.001), 4) if r["slope"] else None,
        }

    # === W7: CHANGELOG ===

    def changelog(self, since_ts: float | None = None, since_hours: float | None = None) -> dict:
        """What changed since a timestamp or N hours ago.

        Default: since the last pull_completed event (falls back to 1 hour).
        """
        if since_ts is None and since_hours is None:
            # Default: since last pull
            last_pull = self._read_one(
                "SELECT ts FROM events WHERE event = 'pull_completed' ORDER BY ts DESC LIMIT 1"
            )
            if last_pull:
                since_ts = dict(last_pull)["ts"]
            else:
                since_ts = time.time() - 3600  # fallback: 1 hour
        elif since_ts is None:
            since_ts = time.time() - since_hours * 3600

        # Check if DB was just rebuilt — everything looks "new" but isn't
        last_rebuild = self._read_one(
            "SELECT ts FROM events WHERE event = 'rebuild' ORDER BY ts DESC LIMIT 1"
        )
        if last_rebuild and since_ts and dict(last_rebuild)["ts"] > since_ts:
            since_ts = dict(last_rebuild)["ts"]

        new_results = self._read(
            "SELECT name, bpb, family, created_at FROM results WHERE created_at > ? ORDER BY bpb",
            (since_ts,)
        )
        new_results = [dict(r) for r in new_results]

        # Check if frontier changed
        all_best = self.best_bpb()
        old_best = None
        if new_results:
            # Best bpb among results created BEFORE the timestamp
            row = self._read_one("SELECT MIN(bpb) as best FROM results WHERE created_at <= ? AND NOT illegal", (since_ts,))
            old_best = dict(row).get("best") if row else None

        frontier_changed = old_best is not None and all_best is not None and all_best < old_best

        return {
            "new_results": [{"name": r["name"], "bpb": r["bpb"], "family": r["family"]} for r in new_results],
            "count": len(new_results),
            "old_best": old_best,
            "new_best": all_best,
            "frontier_changed": frontier_changed,
            "since": since_ts,
        }

    # === W12: EXPERIMENT JOURNAL ===

    def record_journal(self, kind: str, content: str, *, run_name: str | None = None, tags: list[str] | None = None) -> None:
        self._write(
            "INSERT INTO journal (ts, kind, run_name, content, tags) VALUES (?, ?, ?, ?, ?)",
            (time.time(), kind, run_name, content, json.dumps(tags) if tags else None),
            wait=True,
        )

    def journal_entries(self, kind: str | None = None, run_name: str | None = None, limit: int = 50) -> list[dict]:
        if kind and run_name:
            rows = self._read("SELECT * FROM journal WHERE kind = ? AND run_name = ? ORDER BY ts DESC LIMIT ?", (kind, run_name, limit))
        elif kind:
            rows = self._read("SELECT * FROM journal WHERE kind = ? ORDER BY ts DESC LIMIT ?", (kind, limit))
        elif run_name:
            rows = self._read("SELECT * FROM journal WHERE run_name = ? ORDER BY ts DESC LIMIT ?", (run_name, limit))
        else:
            rows = self._read("SELECT * FROM journal ORDER BY ts DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]

    # === W8: PREDICTION + AUDIT ===

    def record_prediction(self, run_name: str, target_steps: int, predicted_bpb: float, method: str = "power_law") -> None:
        self._write(
            "INSERT OR REPLACE INTO predictions (ts, run_name, target_steps, predicted_bpb, method) VALUES (?, ?, ?, ?, ?)",
            (time.time(), run_name, target_steps, predicted_bpb, method),
            wait=True,
        )

    def audit_predictions(self) -> list[dict]:
        """Check predictions against actual results."""
        rows = self._read("""
            SELECT p.run_name, p.target_steps, p.predicted_bpb, p.method,
                   r.bpb as actual_bpb, r.steps as actual_steps
            FROM predictions p
            LEFT JOIN results r ON p.run_name = r.name AND r.steps >= p.target_steps
        """)
        results = []
        for row in rows:
            r = dict(row)
            if r.get("actual_bpb") is not None and r.get("predicted_bpb"):
                r["error_pct"] = round((r["actual_bpb"] - r["predicted_bpb"]) / r["predicted_bpb"] * 100, 2)
                # Update the prediction record
                self._write(
                    "UPDATE predictions SET actual_bpb = ?, error_pct = ? WHERE run_name = ? AND target_steps = ?",
                    (r["actual_bpb"], r["error_pct"], r["run_name"], r["target_steps"]),
                )
            results.append(r)
        return results

    def predict_at_steps(self, run_name: str, target_steps: int) -> dict:
        """Extrapolate from existing probes using power-law fit."""
        import math
        probes = self.learning_curve(run_name)
        probes = [p for p in probes if p.get("bpb") and p["bpb"] > 0]
        if len(probes) < 3:
            return {"error": "need at least 3 probes", "run_name": run_name}

        steps = [p["step"] for p in probes]
        bpbs = [p["bpb"] for p in probes]

        # Log-log linear regression: log(bpb) = a + b * log(step)
        log_s = [math.log(s) for s in steps]
        log_b = [math.log(b) for b in bpbs]
        n = len(log_s)
        sx = sum(log_s); sy = sum(log_b)
        sxy = sum(x*y for x, y in zip(log_s, log_b))
        sxx = sum(x*x for x in log_s)
        denom = n * sxx - sx * sx
        if abs(denom) < 1e-10:
            return {"error": "degenerate fit", "run_name": run_name}
        b_slope = (n * sxy - sx * sy) / denom
        a_intercept = (sy - b_slope * sx) / n

        predicted = math.exp(a_intercept + b_slope * math.log(target_steps))
        # Clamp: prediction can't be better than 50% of current best
        predicted = max(predicted, min(bpbs) * 0.5)

        # R-squared
        mean_y = sy / n
        ss_tot = sum((y - mean_y)**2 for y in log_b)
        ss_res = sum((y - (a_intercept + b_slope * x))**2 for x, y in zip(log_s, log_b))
        r2 = 1 - ss_res / max(ss_tot, 1e-10)

        # Store the prediction
        self.record_prediction(run_name, target_steps, round(predicted, 4), method="power_law")

        return {
            "run_name": run_name,
            "current_steps": max(steps),
            "current_bpb": min(bpbs),
            "target_steps": target_steps,
            "predicted_bpb": round(predicted, 4),
            "r2": round(r2, 4),
            "method": "power_law",
        }

    def _read(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        """Thread-safe read: serializes access to the reader connection."""
        with self._read_lock:
            return self._conn.execute(sql, params).fetchall()

    def _read_one(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        """Thread-safe single-row read."""
        with self._read_lock:
            return self._conn.execute(sql, params).fetchone()

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        """Raw SQL query for ad hoc analysis."""
        return [dict(r) for r in self._read(sql, params)]

    # === BULK IMPORT ===

    def rebuild_from_archive(self, result_dir: str = "out/results") -> int:
        """One-time rebuild from JSON archive files.

        Skips per-result forecasts for performance. Call rebuild_forecasts()
        separately if forecasts are needed.
        """
        count = 0
        skipped = 0
        errors = 0
        for p in sorted(Path(result_dir).glob("*.json")):
            try:
                # Skip if already in DB
                existing = self._read_one("SELECT name FROM results WHERE name = ?", (p.stem,))
                if existing:
                    skipped += 1
                    continue
                payload = json.loads(p.read_text())
                if isinstance(payload, dict) and payload.get(
                    "model", {}
                ).get("test_bpb"):
                    self.record_result(p.stem, payload, json_archive=str(p), compute_forecast=False)
                    count += 1
                else:
                    skipped += 1
            except (json.JSONDecodeError, OSError):
                errors += 1
        if skipped or errors:
            import sys
            print(
                f"chronohorn: rebuild from {result_dir}: "
                f"{count} ingested, {skipped} skipped, {errors} errors",
                file=sys.stderr,
            )
        if count:
            self.record_event("rebuild", count=count, skipped=skipped, errors=errors, source=result_dir)
        return count

    def rebuild_forecasts(self) -> int:
        """Batch-compute forecasts for all results that have enough probes."""
        names = self._read("SELECT DISTINCT name FROM probes")
        count = 0
        for row in names:
            try:
                self.compute_and_store_forecast(row["name"])
                count += 1
            except Exception:
                pass
        return count

    def rebuild_from_manifests(self, manifests_dir: str = "manifests") -> int:
        """Ingest all manifests from a directory."""
        count = 0
        mdir = Path(manifests_dir)
        if mdir.exists():
            for p in sorted(mdir.glob("frontier_*.jsonl")):
                try:
                    count += self.ingest_manifest(str(p))
                except Exception:
                    pass
        return count

    def seed_groups(self) -> list[dict]:
        """Group results by config (ignoring seed) and compute statistics."""
        rows = self._read(
            "SELECT r.name, r.bpb, c.json_blob FROM results r "
            "JOIN configs c ON r.config_id = c.id WHERE NOT r.illegal"
        )

        groups: dict[str, list[dict]] = {}
        for row in rows:
            r = dict(row)
            try:
                cfg = json.loads(r["json_blob"]) if r["json_blob"] else {}
                cfg.pop("seed", None)
                cfg.pop("init_seed", None)
                key = hashlib.sha256(
                    json.dumps(cfg, sort_keys=True).encode()
                ).hexdigest()[:16]
            except (json.JSONDecodeError, TypeError):
                continue
            groups.setdefault(key, []).append({"name": r["name"], "bpb": r["bpb"]})

        # Only return groups with multiple results
        import statistics

        multi: list[dict] = []
        for key, members in groups.items():
            if len(members) >= 2:
                bpbs = [m["bpb"] for m in members]
                mean = statistics.mean(bpbs)
                std = statistics.stdev(bpbs) if len(bpbs) > 1 else 0
                multi.append({
                    "config_hash": key,
                    "runs": members,
                    "count": len(members),
                    "mean_bpb": round(mean, 4),
                    "std_bpb": round(std, 4),
                    "ci_95": round(1.96 * std, 4),
                })

        multi.sort(key=lambda x: x["mean_bpb"])
        return multi

    def record_checkpoint(self, run_name: str, step: int, path: str) -> None:
        """Record a checkpoint location for a run."""
        self.record_event("checkpoint", run_name=run_name, step=step, path=path)

    def latest_checkpoint(self, run_name: str) -> dict | None:
        """Find the latest checkpoint for a run."""
        rows = self._read(
            "SELECT data FROM events WHERE event = 'checkpoint' AND data LIKE ? "
            "ORDER BY ts DESC LIMIT 1",
            (f'%"run_name": "{run_name}"%',),
        )
        if rows:
            try:
                return json.loads(rows[0]["data"])
            except (json.JSONDecodeError, TypeError):
                pass
        return None

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
                existing = self._read_one(
                    "SELECT state FROM jobs WHERE name = ?", (name,)
                )
                if existing and existing["state"] != "pending":
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
