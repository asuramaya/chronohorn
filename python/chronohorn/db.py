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
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from chronohorn.fleet.k8s import (
    DEFAULT_K8S_EXECUTOR_NAME,
    DEFAULT_K8S_GATEWAY_HOST,
    DEFAULT_K8S_NAMESPACE,
    default_executor_name,
    default_remote_source_dir,
    default_runtime_job_name,
    infer_executor_kind,
)
from chronohorn.manifest_normalization import normalize_manifest_payload

CURRENT_SCHEMA_VERSION = 12
IMPORTED_RESULT_MANIFEST = "__imported_result__"
IMPORTED_RESULT_LAUNCHER = "result_import"
VALID_RESULT_POPULATIONS = {"controlled", "imported_archive", "unknown", "all"}
VALID_RESULT_LEGALITY = {"legal", "illegal", "all"}
VALID_RESULT_TRUST_FILTERS = {"admissible", "provisional", "quarantined", "all"}

DEFAULT_DB_PATH = Path("out/chronohorn.db")
CHRONOHORN_ROOT = Path(__file__).resolve().parents[2]
JOB_CONFIG_METADATA_KEYS = {
    "artifact_bin",
    "argv",
    "cluster_gateway_host",
    "checkpoint_path",
    "command",
    "completed_at",
    "container",
    "cwd",
    "env",
    "executor_kind",
    "executor_name",
    "generated_by",
    "goal",
    "gpu",
    "host",
    "hosts",
    "image",
    "launched_at",
    "launcher",
    "log_path",
    "manifest",
    "manifest_path",
    "parent",
    "placement_cores",
    "remote_assets",
    "remote_cwd_rel",
    "remote_run",
    "remote_source_dir",
    "remote_snapshot",
    "report_every",
    "requested_launcher",
    "resource_class",
    "rsync_excludes",
    "run_id",
    "snapshot_paths",
    "source_dir",
    "state",
    "summary_path",
    "sync_paths",
    "threads",
    "train_tokens",
    "val_tokens",
    "work_tokens",
    "workload_kind",
    "runtime_namespace",
    "runtime_job_name",
    "runtime_pod_name",
    "runtime_node_name",
}


class ChronohornDB:
    def __init__(self, path: Path | str = DEFAULT_DB_PATH, read_only: bool = False) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._read_only = read_only
        self._closed = False
        self._read_lock = threading.Lock()  # protects self._conn for concurrent reads
        self._branch_health_warned: set[str] = set()  # suppress duplicate warnings

        # Main connection for reads.
        # isolation_level=None puts connection in autocommit mode so each
        # SELECT sees the latest committed data from the writer connection.
        self._conn = sqlite3.connect(
            str(self._path), check_same_thread=False, isolation_level=None
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._archive_payload_cache: dict[str, dict[str, Any] | None] = {}

        if not read_only:
            # Write queue + dedicated writer thread
            self._write_queue: queue.Queue = queue.Queue()
            self._writer_conn = sqlite3.connect(
                str(self._path), check_same_thread=False
            )
            self._writer_conn.row_factory = sqlite3.Row
            self._writer_conn.execute("PRAGMA journal_mode=WAL")
            self._writer_conn.execute("PRAGMA synchronous=NORMAL")
            self._writer_conn.execute("PRAGMA foreign_keys=ON")
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
    def open_read_only(cls, path: Path | str = DEFAULT_DB_PATH) -> ChronohornDB:
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
                    raise RuntimeError("chronohorn db: write lost after two timeouts")

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
                family TEXT,
                parent TEXT,
                state TEXT DEFAULT 'pending',
                steps INTEGER,
                seed INTEGER,
                lr REAL,
                batch_size INTEGER,
                host TEXT,
                executor_kind TEXT,
                executor_name TEXT,
                launcher TEXT,
                requested_launcher TEXT,
                backend TEXT,
                resource_class TEXT,
                workload_kind TEXT,
                work_tokens INTEGER,
                launched_at REAL,
                completed_at REAL,
                container TEXT,
                remote_run TEXT,
                runtime_namespace TEXT,
                runtime_job_name TEXT,
                runtime_pod_name TEXT,
                runtime_node_name TEXT,
                command TEXT,
                job_json TEXT
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
                    "ALTER TABLE configs ADD COLUMN family TEXT DEFAULT NULL"
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

        if current < 6:
            self._backfill_imported_jobs(conn)
            conn.execute("INSERT INTO schema_version (version) VALUES (6)")
            conn.commit()

        if current < 7:
            self._backfill_legacy_command_jobs(conn)
            conn.execute("INSERT INTO schema_version (version) VALUES (7)")
            conn.commit()

        if current < 8:
            self._backfill_legacy_command_jobs(conn)
            conn.execute("INSERT INTO schema_version (version) VALUES (8)")
            conn.commit()

        if current < 9:
            self._backfill_legacy_command_jobs(conn)
            conn.execute("INSERT INTO schema_version (version) VALUES (9)")
            conn.commit()

        if current < 10:
            existing_job_cols = {
                row[1]
                for row in conn.execute("PRAGMA table_info(jobs)").fetchall()
            }
            for col, typ in [
                ("family", "TEXT"),
                ("requested_launcher", "TEXT"),
                ("backend", "TEXT"),
                ("resource_class", "TEXT"),
                ("workload_kind", "TEXT"),
                ("work_tokens", "INTEGER"),
                ("job_json", "TEXT"),
            ]:
                if col not in existing_job_cols:
                    conn.execute(f"ALTER TABLE jobs ADD COLUMN {col} {typ}")
            self._backfill_job_specs(conn)
            conn.execute("INSERT INTO schema_version (version) VALUES (10)")
            conn.commit()

        if current < 11:
            self._backfill_job_specs(conn)
            conn.execute("INSERT INTO schema_version (version) VALUES (11)")
            conn.commit()

        if current < 12:
            existing_job_cols = {
                row[1]
                for row in conn.execute("PRAGMA table_info(jobs)").fetchall()
            }
            for col, typ in [
                ("executor_kind", "TEXT"),
                ("executor_name", "TEXT"),
                ("runtime_namespace", "TEXT"),
                ("runtime_job_name", "TEXT"),
                ("runtime_pod_name", "TEXT"),
                ("runtime_node_name", "TEXT"),
            ]:
                if col not in existing_job_cols:
                    conn.execute(f"ALTER TABLE jobs ADD COLUMN {col} {typ}")
            self._backfill_job_specs(conn)
            conn.execute("INSERT INTO schema_version (version) VALUES (12)")
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

    @staticmethod
    def _load_json_blob(blob: Any) -> dict[str, Any]:
        if isinstance(blob, dict):
            return dict(blob)
        if not blob:
            return {}
        try:
            parsed = json.loads(blob)
        except (TypeError, json.JSONDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _coerce_int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _dump_json_blob(payload: Mapping[str, Any] | None) -> str:
        return json.dumps(dict(payload or {}), sort_keys=True)

    def _normalized_job_payload(
        self,
        name: str,
        *,
        manifest: str = "",
        parent: str = "",
        family: str | None = None,
        config: Mapping[str, Any] | None = None,
        steps: Any = None,
        seed: Any = None,
        lr: Any = None,
        batch_size: Any = None,
        command: str = "",
        job_spec: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = dict(job_spec or {})
        merged_config: dict[str, Any] = {}
        if isinstance(config, Mapping):
            merged_config.update(dict(config))
        payload_config = payload.get("config")
        if isinstance(payload_config, Mapping):
            merged_config.update(dict(payload_config))
        if merged_config:
            payload["config"] = merged_config

        payload["name"] = name
        if manifest and not payload.get("manifest"):
            payload["manifest"] = manifest
        if parent and not payload.get("parent"):
            payload["parent"] = parent
        if family and not payload.get("family"):
            payload["family"] = family
        if steps not in (None, "") and "steps" not in payload:
            payload["steps"] = steps
        if seed not in (None, "") and "seed" not in payload:
            payload["seed"] = seed
        if lr not in (None, ""):
            payload.setdefault("learning_rate", lr)
        if batch_size not in (None, "") and "batch_size" not in payload:
            payload["batch_size"] = batch_size
        if command and not payload.get("command"):
            payload["command"] = command
        return payload

    def _canonical_job_spec(
        self,
        name: str,
        *,
        manifest: str = "",
        parent: str = "",
        family: str | None = None,
        config: Mapping[str, Any] | None = None,
        steps: Any = None,
        seed: Any = None,
        lr: Any = None,
        batch_size: Any = None,
        command: str = "",
        job_spec: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = self._normalized_job_payload(
            name,
            manifest=manifest,
            parent=parent,
            family=family,
            config=config,
            steps=steps,
            seed=seed,
            lr=lr,
            batch_size=batch_size,
            command=command,
            job_spec=job_spec,
        )
        normalized = normalize_manifest_payload(payload)
        spec = dict(normalized)
        cfg = spec.get("config")
        if not isinstance(cfg, dict):
            cfg = {}
        spec["config"] = cfg
        for key in JOB_CONFIG_METADATA_KEYS:
            cfg.pop(key, None)

        family_value = spec.get("family") or payload.get("family")
        if not family_value and cfg:
            family_value = self._detect_family(cfg)
        if family_value:
            family_text = str(family_value)
            spec["family"] = family_text
            cfg.setdefault("family", family_text)

        requested_launcher = (
            payload.get("requested_launcher")
            or spec.get("requested_launcher")
            or payload.get("launcher")
            or spec.get("launcher")
        )
        if requested_launcher not in (None, ""):
            spec["requested_launcher"] = str(requested_launcher)

        executor_kind = infer_executor_kind(spec) or infer_executor_kind(payload) or infer_executor_kind(
            {"launcher": requested_launcher, "host": spec.get("host")}
        )
        if executor_kind:
            spec["executor_kind"] = executor_kind
        executor_name = default_executor_name(spec, executor_kind=executor_kind)
        if executor_name:
            spec["executor_name"] = executor_name

        for key in (
            "manifest",
            "parent",
            "goal",
            "image",
            "source_dir",
            "remote_cwd_rel",
            "run_id",
            "manifest_path",
            "host",
            "cluster_gateway_host",
            "container",
            "remote_run",
            "remote_source_dir",
            "runtime_namespace",
            "runtime_job_name",
            "runtime_pod_name",
            "runtime_node_name",
        ):
            value = payload.get(key) if payload.get(key) not in (None, "") else spec.get(key)
            if value not in (None, ""):
                spec[key] = value

        for key in ("launcher", "backend", "resource_class", "workload_kind", "command"):
            value = payload.get(key) if payload.get(key) not in (None, "") else spec.get(key)
            if value not in (None, ""):
                spec[key] = value

        for key in ("hosts", "snapshot_paths", "sync_paths"):
            value = payload.get(key)
            if isinstance(value, list) and value:
                spec[key] = list(value)

        env_map = payload.get("env")
        if isinstance(env_map, Mapping) and env_map:
            spec["env"] = dict(env_map)

        if payload.get("gpu") is not None:
            spec["gpu"] = bool(payload.get("gpu"))

        work_tokens = self._coerce_int(spec.get("work_tokens", payload.get("work_tokens")))
        if work_tokens is not None:
            spec["work_tokens"] = work_tokens

        if spec.get("executor_kind") == "k8s_cluster":
            spec.setdefault("executor_name", DEFAULT_K8S_EXECUTOR_NAME)
            spec.setdefault("cluster_gateway_host", DEFAULT_K8S_GATEWAY_HOST)
            explicit_namespace = payload.get("runtime_namespace") or payload.get("k8s_namespace")
            if explicit_namespace not in (None, ""):
                spec["runtime_namespace"] = str(explicit_namespace)
            else:
                spec["runtime_namespace"] = DEFAULT_K8S_NAMESPACE
            explicit_runtime_job_name = payload.get("runtime_job_name")
            if explicit_runtime_job_name not in (None, ""):
                spec["runtime_job_name"] = str(explicit_runtime_job_name)
            else:
                spec["runtime_job_name"] = default_runtime_job_name(spec)
            spec.setdefault("remote_source_dir", default_remote_source_dir(spec))

        learning_rate = spec.get("learning_rate")
        if learning_rate is None:
            learning_rate = spec.get("lr")
        learning_rate_value = self._coerce_float(learning_rate)
        if learning_rate_value is not None and learning_rate_value > 0:
            spec["learning_rate"] = learning_rate_value
            spec.setdefault("lr", learning_rate_value)
            cfg.setdefault("learning_rate", learning_rate_value)
        else:
            spec.pop("learning_rate", None)
            spec.pop("lr", None)
            cfg.pop("learning_rate", None)
            cfg.pop("lr", None)

        steps_value = self._coerce_int(spec.get("steps"))
        if steps_value is not None and steps_value > 0:
            spec["steps"] = steps_value
            cfg.setdefault("steps", steps_value)
        else:
            spec.pop("steps", None)
            cfg.pop("steps", None)

        batch_size_value = self._coerce_int(spec.get("batch_size"))
        if batch_size_value is not None and batch_size_value > 0:
            spec["batch_size"] = batch_size_value
            cfg.setdefault("batch_size", batch_size_value)
        else:
            spec.pop("batch_size", None)
            cfg.pop("batch_size", None)

        seed_value = self._coerce_int(spec.get("seed"))
        if seed_value is not None:
            spec["seed"] = seed_value
            cfg.setdefault("seed", seed_value)
        else:
            spec.pop("seed", None)
            cfg.pop("seed", None)

        return spec

    def _job_column_values(self, spec: Mapping[str, Any]) -> dict[str, Any]:
        learning_rate = spec.get("learning_rate")
        if learning_rate is None:
            learning_rate = spec.get("lr")
        requested_launcher = spec.get("requested_launcher") or spec.get("launcher")
        return {
            "family": str(spec.get("family") or "") or None,
            "steps": self._coerce_int(spec.get("steps")),
            "seed": self._coerce_int(spec.get("seed")),
            "lr": self._coerce_float(learning_rate),
            "batch_size": self._coerce_int(spec.get("batch_size")),
            "executor_kind": str(spec.get("executor_kind") or "") or None,
            "executor_name": str(spec.get("executor_name") or "") or None,
            "launcher": str(spec.get("launcher") or "") or None,
            "requested_launcher": str(requested_launcher or "") or None,
            "backend": str(spec.get("backend") or "") or None,
            "resource_class": str(spec.get("resource_class") or "") or None,
            "workload_kind": str(spec.get("workload_kind") or "") or None,
            "work_tokens": self._coerce_int(spec.get("work_tokens")),
            "runtime_namespace": str(spec.get("runtime_namespace") or "") or None,
            "runtime_job_name": str(spec.get("runtime_job_name") or "") or None,
            "runtime_pod_name": str(spec.get("runtime_pod_name") or "") or None,
            "runtime_node_name": str(spec.get("runtime_node_name") or "") or None,
            "command": str(spec.get("command") or "") or None,
            "job_json": self._dump_json_blob(spec),
        }

    def _job_row_to_dict(self, row: Mapping[str, Any]) -> dict[str, Any]:
        row_dict = dict(row)
        item = self._load_json_blob(row_dict.get("job_json"))
        config = item.get("config")
        if not isinstance(config, Mapping):
            config = self._load_json_blob(row_dict.get("config_json"))
        else:
            config = dict(config)
        item["config"] = dict(config)

        for key in (
            "name",
            "manifest",
            "family",
            "parent",
            "state",
            "host",
            "executor_kind",
            "executor_name",
            "launcher",
            "requested_launcher",
            "backend",
            "resource_class",
            "workload_kind",
            "container",
            "remote_run",
            "runtime_namespace",
            "runtime_job_name",
            "runtime_pod_name",
            "runtime_node_name",
        ):
            value = row_dict.get(key)
            if value not in (None, ""):
                item[key] = value

        for key in ("steps", "seed", "batch_size", "work_tokens", "launched_at", "completed_at"):
            value = row_dict.get(key)
            if value is not None:
                item[key] = value

        learning_rate = row_dict.get("lr")
        if learning_rate is not None:
            item["learning_rate"] = learning_rate
            item["lr"] = learning_rate

        command = row_dict.get("command")
        if command not in (None, ""):
            item["command"] = command

        family_value = item.get("family")
        if family_value and not item["config"].get("family"):
            item["config"]["family"] = family_value
        for key in ("steps", "seed", "batch_size"):
            value = item.get(key)
            if value is not None and item["config"].get(key) is None:
                item["config"][key] = value
        if item.get("learning_rate") is not None and item["config"].get("learning_rate") is None:
            item["config"]["learning_rate"] = item["learning_rate"]

        return item

    @staticmethod
    def _payload_experiment_config(
        model: Mapping[str, Any],
        train: Mapping[str, Any],
        cfg: Mapping[str, Any],
    ) -> dict[str, Any]:
        config_dict: dict[str, Any] = {}
        for key, value in cfg.items():
            if key in {"train", "model", "data"} and isinstance(value, Mapping):
                continue
            config_dict[key] = value
        for section_name in ("data", "model"):
            section = cfg.get(section_name)
            if not isinstance(section, Mapping):
                continue
            for key, value in section.items():
                config_dict.setdefault(key, value)
        for key, value in train.items():
            if key not in ("grad_clip", "log_every", "eval_batches", "seeds"):
                config_dict[key] = value
        for key, value in model.items():
            if key not in (
                "test_bpb",
                "train_bpb",
                "test_eval_loss",
                "train_eval_loss",
                "test_bits_per_token",
                "train_bits_per_token",
                "overfit_pct",
                "payload_bytes_est",
                "payload_mb_est",
                "initial_trainable_signature",
            ):
                config_dict.setdefault(key, value)
        return config_dict

    @staticmethod
    def _canonical_experiment_config(
        base_cfg: Mapping[str, Any] | None,
        *,
        family: str | None = None,
        result_steps: Any = None,
        result_seq_len: Any = None,
        job_steps: Any = None,
        job_seed: Any = None,
        job_lr: Any = None,
        job_batch_size: Any = None,
    ) -> dict[str, Any]:
        cfg = dict(base_cfg or {})
        if "learning_rate" not in cfg and cfg.get("lr") is not None:
            cfg["learning_rate"] = cfg["lr"]
        cfg.pop("lr", None)

        steps = result_steps if result_steps not in (None, 0, "") else job_steps
        if steps not in (None, 0, ""):
            cfg["steps"] = steps

        if result_seq_len not in (None, 0, ""):
            cfg["seq_len"] = result_seq_len

        if job_seed not in (None, ""):
            cfg["seed"] = job_seed

        if job_batch_size not in (None, 0, ""):
            cfg["batch_size"] = job_batch_size

        if job_lr not in (None, "") and (job_lr != 0 or "learning_rate" not in cfg):
            cfg["learning_rate"] = job_lr

        if family and not cfg.get("family"):
            cfg["family"] = family

        return cfg

    @staticmethod
    def _coerce_population(population: str | None) -> str:
        value = str(population or "controlled")
        if value not in VALID_RESULT_POPULATIONS:
            raise ValueError(
                f"invalid population {value!r}; expected one of {sorted(VALID_RESULT_POPULATIONS)}"
            )
        return value

    @staticmethod
    def _coerce_legality(legality: str | None) -> str:
        value = str(legality or "legal")
        if value not in VALID_RESULT_LEGALITY:
            raise ValueError(
                f"invalid legality {value!r}; expected one of {sorted(VALID_RESULT_LEGALITY)}"
            )
        return value

    @staticmethod
    def _coerce_trust_filter(trust: str | None) -> str:
        value = str(trust or "all")
        if value not in VALID_RESULT_TRUST_FILTERS:
            raise ValueError(
                f"invalid trust filter {value!r}; expected one of {sorted(VALID_RESULT_TRUST_FILTERS)}"
            )
        return value

    def _load_result_archive_payload(self, archive_path: str | None) -> dict[str, Any] | None:
        path_str = str(archive_path or "").strip()
        if not path_str:
            return None
        cached = self._archive_payload_cache.get(path_str, ...)
        if cached is not ...:
            return cached
        try:
            payload = json.loads(Path(path_str).read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                payload = None
        except (OSError, json.JSONDecodeError):
            payload = None
        self._archive_payload_cache[path_str] = payload
        return payload

    def _seed_group_sizes(
        self,
        *,
        population: str,
        legality: str,
    ) -> dict[str, int]:
        sizes: dict[str, int] = {}
        for group in self.seed_groups(population=population, legality=legality):
            count = int(group.get("count") or 0)
            for run in group.get("runs", []):
                name = run.get("name")
                if name:
                    sizes[str(name)] = count
        return sizes

    def _metric_schema_state(
        self,
        row: Mapping[str, Any],
        payload: Mapping[str, Any] | None,
    ) -> str:
        if payload is None:
            return "archive_missing"
        dataset = payload.get("dataset")
        provenance = payload.get("provenance")
        metrics = payload.get("metrics")
        if isinstance(provenance, dict):
            if provenance.get("train_bpb_basis") or provenance.get("test_bpb_basis") or provenance.get("metric_basis"):
                return "explicit_metric_basis"
        if isinstance(dataset, dict) and (
            dataset.get("test_tokens_per_byte") is not None
            or dataset.get("train_tokens_per_byte_est") is not None
        ):
            return "dataset_anchored"
        if isinstance(metrics, dict):
            return "metrics_block_only"
        return "legacy_result_schema"

    def _trust_annotation(
        self,
        row: Mapping[str, Any],
        *,
        population: str,
        seed_group_sizes: Mapping[str, int],
    ) -> dict[str, Any]:
        name = str(row.get("name") or "")
        selected_population = self._coerce_population(population)
        payload = self._load_result_archive_payload(row.get("json_archive"))
        metric_state = self._metric_schema_state(row, payload)
        replicate_count = int(seed_group_sizes.get(name, 1))
        replication_state = "replicated" if replicate_count >= 2 else "single_seed"

        if row.get("illegal"):
            trust_state = "quarantined"
            quarantine_reason = "illegal"
        elif selected_population in {"imported_archive", "unknown"}:
            trust_state = "quarantined"
            quarantine_reason = f"{selected_population}_provenance"
        elif metric_state in {"archive_missing", "legacy_result_schema"}:
            trust_state = "provisional"
            quarantine_reason = metric_state
        elif replication_state == "single_seed":
            trust_state = "provisional"
            quarantine_reason = "single_seed"
        else:
            trust_state = "admissible"
            quarantine_reason = None

        return {
            "trust_state": trust_state,
            "replication_state": replication_state,
            "replicate_count": replicate_count,
            "metric_state": metric_state,
            "quarantine_reason": quarantine_reason,
        }

    def _annotate_result_rows(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        population: str,
        legality: str,
    ) -> list[dict[str, Any]]:
        selected_population = self._coerce_population(population)
        selected_legality = self._coerce_legality(legality)
        seed_group_sizes = self._seed_group_sizes(
            population=selected_population,
            legality=selected_legality,
        )
        annotated: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item.update(
                self._trust_annotation(
                    item,
                    population=selected_population,
                    seed_group_sizes=seed_group_sizes,
                )
            )
            annotated.append(item)
        return annotated

    def _population_clauses(
        self,
        *,
        population: str | None,
        result_alias: str = "r",
        job_alias: str = "j",
    ) -> tuple[list[str], list[Any]]:
        selected = self._coerce_population(population)
        if selected == "controlled":
            return [
                f"{job_alias}.name IS NOT NULL",
                f"COALESCE({job_alias}.manifest, '') != ?",
            ], [IMPORTED_RESULT_MANIFEST]
        if selected == "imported_archive":
            return [
                f"{job_alias}.name IS NOT NULL",
                f"COALESCE({job_alias}.manifest, '') = ?",
            ], [IMPORTED_RESULT_MANIFEST]
        if selected == "unknown":
            return [f"{job_alias}.name IS NULL"], []
        return [], []

    def _legality_clauses(
        self,
        *,
        legality: str | None,
        result_alias: str = "r",
    ) -> tuple[list[str], list[Any]]:
        selected = self._coerce_legality(legality)
        if selected == "legal":
            return [f"NOT {result_alias}.illegal"], []
        if selected == "illegal":
            return [f"{result_alias}.illegal"], []
        return [], []

    def _config_from_joined_row(self, row: Mapping[str, Any]) -> dict[str, Any]:
        return self._canonical_experiment_config(
            self._load_json_blob(row.get("config_json") or row.get("json_blob")),
            family=row.get("family"),
            result_steps=row.get("result_steps", row.get("steps")),
            result_seq_len=row.get("result_seq_len", row.get("seq_len")),
            job_steps=row.get("job_steps"),
            job_seed=row.get("job_seed"),
            job_lr=row.get("job_lr"),
            job_batch_size=row.get("job_batch_size"),
        )

    def _joined_run_row(self, name: str) -> dict[str, Any] | None:
        row = self._read_one(
            """
            SELECT r.name, r.family,
                   r.steps AS result_steps,
                   r.seq_len AS result_seq_len,
                   c.json_blob AS config_json,
                   j.steps AS job_steps,
                   j.seed AS job_seed,
                   j.lr AS job_lr,
                   j.batch_size AS job_batch_size,
                   j.manifest
            FROM results r
            LEFT JOIN configs c ON r.config_id = c.id
            LEFT JOIN jobs j ON j.name = r.name
            WHERE r.name = ?
        """,
            (name,),
        )
        return dict(row) if row else None

    def population_summary(self, *, family: str | None = None) -> dict[str, Any]:
        clauses: list[str] = []
        params: list[Any] = [IMPORTED_RESULT_MANIFEST] * 6
        if family:
            clauses.append("r.family = ?")
            params.append(family)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        row = self._read_one(
            f"""
            SELECT
                COUNT(*) AS total_results,
                SUM(CASE WHEN NOT r.illegal THEN 1 ELSE 0 END) AS legal_results,
                SUM(CASE WHEN r.illegal THEN 1 ELSE 0 END) AS illegal_results,
                SUM(CASE WHEN j.name IS NOT NULL AND COALESCE(j.manifest, '') != ? THEN 1 ELSE 0 END) AS controlled_count,
                SUM(CASE WHEN j.name IS NOT NULL AND COALESCE(j.manifest, '') != ? AND NOT r.illegal THEN 1 ELSE 0 END) AS controlled_legal_count,
                MIN(CASE WHEN j.name IS NOT NULL AND COALESCE(j.manifest, '') != ? AND NOT r.illegal THEN r.bpb END) AS controlled_best_bpb,
                SUM(CASE WHEN j.name IS NOT NULL AND COALESCE(j.manifest, '') = ? THEN 1 ELSE 0 END) AS imported_count,
                SUM(CASE WHEN j.name IS NOT NULL AND COALESCE(j.manifest, '') = ? AND NOT r.illegal THEN 1 ELSE 0 END) AS imported_legal_count,
                MIN(CASE WHEN j.name IS NOT NULL AND COALESCE(j.manifest, '') = ? AND NOT r.illegal THEN r.bpb END) AS imported_best_bpb,
                SUM(CASE WHEN j.name IS NULL THEN 1 ELSE 0 END) AS unknown_count,
                SUM(CASE WHEN j.name IS NULL AND NOT r.illegal THEN 1 ELSE 0 END) AS unknown_legal_count,
                MIN(CASE WHEN j.name IS NULL AND NOT r.illegal THEN r.bpb END) AS unknown_best_bpb
            FROM results r
            LEFT JOIN jobs j ON j.name = r.name
            {where}
        """,
            tuple(params),
        )
        if not row:
            return {
                "result_count": 0,
                "legal_count": 0,
                "illegal_count": 0,
                "populations": {},
            }
        data = dict(row)
        return {
            "result_count": int(data.get("total_results") or 0),
            "legal_count": int(data.get("legal_results") or 0),
            "illegal_count": int(data.get("illegal_results") or 0),
            "populations": {
                "controlled": {
                    "count": int(data.get("controlled_count") or 0),
                    "legal_count": int(data.get("controlled_legal_count") or 0),
                    "best_bpb": data.get("controlled_best_bpb"),
                },
                "imported_archive": {
                    "count": int(data.get("imported_count") or 0),
                    "legal_count": int(data.get("imported_legal_count") or 0),
                    "best_bpb": data.get("imported_best_bpb"),
                },
                "unknown": {
                    "count": int(data.get("unknown_count") or 0),
                    "legal_count": int(data.get("unknown_legal_count") or 0),
                    "best_bpb": data.get("unknown_best_bpb"),
                },
            },
        }

    def trust_summary(
        self,
        *,
        population: str = "controlled",
        legality: str = "legal",
        family: str | None = None,
    ) -> dict[str, Any]:
        clauses: list[str] = []
        params: list[Any] = []
        population_clauses, population_params = self._population_clauses(population=population)
        legality_clauses, legality_params = self._legality_clauses(legality=legality)
        clauses.extend(population_clauses)
        clauses.extend(legality_clauses)
        params.extend(population_params)
        params.extend(legality_params)
        if family:
            clauses.append("r.family = ?")
            params.append(family)
        where_sql = " AND ".join(clauses) if clauses else "1=1"
        rows = self._read(
            f"""
            SELECT r.name, r.family, r.illegal, r.json_archive
            FROM results r
            LEFT JOIN jobs j ON j.name = r.name
            WHERE {where_sql}
            ORDER BY r.bpb
        """,
            tuple(params),
        )
        annotated = self._annotate_result_rows(
            rows,
            population=self._coerce_population(population),
            legality=self._coerce_legality(legality),
        )
        counts = {"admissible": 0, "provisional": 0, "quarantined": 0}
        metric_states: dict[str, int] = {}
        for row in annotated:
            trust_state = row["trust_state"]
            counts[trust_state] = counts.get(trust_state, 0) + 1
            metric_state = str(row.get("metric_state") or "unknown")
            metric_states[metric_state] = metric_states.get(metric_state, 0) + 1
        return {
            "population": self._coerce_population(population),
            "legality": self._coerce_legality(legality),
            "counts": counts,
            "metric_states": metric_states,
        }

    def result_trust_index(
        self,
        *,
        population: str = "controlled",
        legality: str = "legal",
    ) -> dict[str, dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        population_clauses, population_params = self._population_clauses(population=population)
        legality_clauses, legality_params = self._legality_clauses(legality=legality)
        clauses.extend(population_clauses)
        clauses.extend(legality_clauses)
        params.extend(population_params)
        params.extend(legality_params)
        where_sql = " AND ".join(clauses) if clauses else "1=1"
        rows = self._read(
            f"""
            SELECT r.name, r.family, r.illegal, r.json_archive
            FROM results r
            LEFT JOIN jobs j ON j.name = r.name
            WHERE {where_sql}
        """,
            tuple(params),
        )
        annotated = self._annotate_result_rows(
            rows,
            population=self._coerce_population(population),
            legality=self._coerce_legality(legality),
        )
        return {str(row["name"]): row for row in annotated if row.get("name")}

    @staticmethod
    def _matrix_population_label(row: Mapping[str, Any]) -> str:
        manifest = str(row.get("manifest") or "")
        if manifest == IMPORTED_RESULT_MANIFEST:
            return "imported_archive"
        if row.get("job_present"):
            return "controlled"
        return "unknown"

    @staticmethod
    def _matrix_legality_label(row: Mapping[str, Any]) -> str:
        if not row.get("result_present"):
            return "unobserved"
        return "illegal" if bool(row.get("illegal")) else "legal"

    @staticmethod
    def _intent_state(row: Mapping[str, Any]) -> str:
        population = str(row.get("population") or "")
        if population == "imported_archive":
            return "archive_import"
        if not row.get("job_present"):
            return "unknown"
        manifest = str(row.get("manifest") or "")
        return "manifested" if manifest else "db_only"

    @staticmethod
    def _execution_state(row: Mapping[str, Any]) -> str:
        population = str(row.get("population") or "")
        if population == "imported_archive":
            return "archive_import"
        if not row.get("job_present"):
            return "unknown"
        state = str(row.get("state") or "")
        if state in {"pending", "dispatched", "running", "completed"}:
            return state
        if row.get("launched_at") is not None:
            return "dispatched"
        return "unknown"

    @staticmethod
    def _observation_state(row: Mapping[str, Any]) -> str:
        if not row.get("result_present"):
            return "none"
        probe_count = int(row.get("probe_count") or 0)
        return "curve" if probe_count > 0 else "result"

    @staticmethod
    def _interpretation_state(row: Mapping[str, Any]) -> str:
        return "forecasted" if row.get("forecast_present") else "none"

    @staticmethod
    def _evidence_phase(row: Mapping[str, Any]) -> str:
        if row.get("forecast_present"):
            return "interpretation"
        if row.get("result_present"):
            return "observation"
        execution_state = str(row.get("execution_state") or "")
        if execution_state in {"dispatched", "running", "completed"}:
            return "execution"
        if row.get("job_present"):
            return "intent"
        return "unknown"

    @staticmethod
    def _retention_hint(surface_role: str) -> str:
        if surface_role in {"live_control_input", "live_mixed"}:
            return "keep_live"
        if surface_role in {"evidence_archive", "archive_import"}:
            return "archive"
        if surface_role == "unknown_provenance":
            return "investigate"
        return "candidate_prune"

    @classmethod
    def _manifest_surface_role(cls, summary: Mapping[str, Any]) -> str:
        manifest = str(summary.get("manifest") or "")
        if manifest == IMPORTED_RESULT_MANIFEST:
            return "archive_import"
        pending = int(summary.get("pending") or 0)
        running = int(summary.get("running") or 0)
        observed = int(summary.get("observed") or 0)
        if pending > 0 or running > 0:
            return "live_mixed" if observed > 0 else "live_control_input"
        if observed > 0:
            return "evidence_archive"
        if manifest == "__unknown__":
            return "unknown_provenance"
        return "candidate_prune"

    @classmethod
    def _row_surface_role(cls, row: Mapping[str, Any]) -> str:
        manifest = str(row.get("manifest") or "")
        if manifest == IMPORTED_RESULT_MANIFEST or str(row.get("population") or "") == "imported_archive":
            return "archive_import"
        if str(row.get("population") or "") == "unknown":
            return "unknown_provenance"
        state = str(row.get("state") or "")
        if state in {"pending", "dispatched", "running"}:
            return "live_mixed" if row.get("result_present") else "live_control_input"
        if row.get("result_present"):
            return "evidence_archive"
        return "candidate_prune"

    def evidence_matrix(
        self,
        *,
        top_k: int = 50,
        family: str | None = None,
        manifest: str | None = None,
        population: str = "all",
        legality: str = "all",
        trust: str = "all",
        state: str | None = None,
    ) -> dict[str, Any]:
        selected_population = self._coerce_population(population or "all")
        selected_legality = self._coerce_legality(legality or "all")
        selected_trust = self._coerce_trust_filter(trust or "all")
        family_filter = str(family or "").strip() or None
        manifest_filter = str(manifest or "").strip() or None
        state_filter = str(state or "").strip() or None

        probe_rows = self._read(
            """
            SELECT name, COUNT(*) AS probe_count, MAX(step) AS last_probe_step
            FROM probes
            GROUP BY name
        """
        )
        probe_map = {
            str(row["name"]): {
                "probe_count": int(row["probe_count"] or 0),
                "last_probe_step": int(row["last_probe_step"] or 0),
            }
            for row in probe_rows
        }

        joined_rows = [
            dict(row)
            for row in self._read(
                """
                SELECT
                    j.name AS name,
                    1 AS job_present,
                    CASE WHEN r.name IS NOT NULL THEN 1 ELSE 0 END AS result_present,
                    COALESCE(r.family, j.family, c.family, 'unknown') AS family,
                    j.manifest,
                    j.state,
                    j.parent,
                    j.launcher,
                    j.requested_launcher,
                    j.backend,
                    j.resource_class,
                    j.workload_kind,
                    j.work_tokens,
                    j.host,
                    j.launched_at,
                    j.completed_at,
                    j.command,
                    r.bpb,
                    r.illegal,
                    r.json_archive,
                    COALESCE(r.steps, j.steps) AS steps,
                    COALESCE(r.seq_len, c.seq_len) AS seq_len,
                    CASE WHEN f.name IS NOT NULL THEN 1 ELSE 0 END AS forecast_present
                FROM jobs j
                LEFT JOIN results r ON r.name = j.name
                LEFT JOIN forecasts f ON f.name = j.name
                LEFT JOIN configs c ON c.id = COALESCE(r.config_id, j.config_id)
            """
            )
        ]
        orphan_rows = [
            dict(row)
            for row in self._read(
                """
                SELECT
                    r.name AS name,
                    0 AS job_present,
                    1 AS result_present,
                    COALESCE(r.family, c.family, 'unknown') AS family,
                    '' AS manifest,
                    'unknown' AS state,
                    '' AS parent,
                    '' AS launcher,
                    '' AS requested_launcher,
                    '' AS backend,
                    '' AS resource_class,
                    '' AS workload_kind,
                    NULL AS work_tokens,
                    '' AS host,
                    NULL AS launched_at,
                    NULL AS completed_at,
                    '' AS command,
                    r.bpb AS bpb,
                    r.illegal AS illegal,
                    r.json_archive AS json_archive,
                    r.steps AS steps,
                    COALESCE(r.seq_len, c.seq_len) AS seq_len,
                    CASE WHEN f.name IS NOT NULL THEN 1 ELSE 0 END AS forecast_present
                FROM results r
                LEFT JOIN jobs j ON j.name = r.name
                LEFT JOIN forecasts f ON f.name = r.name
                LEFT JOIN configs c ON c.id = r.config_id
                WHERE j.name IS NULL
            """
            )
        ]

        seed_group_cache: dict[tuple[str, str], dict[str, int]] = {}

        def _seed_sizes(pop_label: str, leg_label: str) -> dict[str, int]:
            key = (pop_label, leg_label)
            cached = seed_group_cache.get(key)
            if cached is not None:
                return cached
            if leg_label not in {"legal", "illegal"}:
                sizes: dict[str, int] = {}
            else:
                sizes = self._seed_group_sizes(population=pop_label, legality=leg_label)
            seed_group_cache[key] = sizes
            return sizes

        rows: list[dict[str, Any]] = []
        for raw in joined_rows + orphan_rows:
            item = dict(raw)
            name = str(item.get("name") or "")
            probes = probe_map.get(name, {})
            item["probe_count"] = int(probes.get("probe_count") or 0)
            item["last_probe_step"] = int(probes.get("last_probe_step") or 0)
            item["job_present"] = bool(item.get("job_present"))
            item["result_present"] = bool(item.get("result_present"))
            item["forecast_present"] = bool(item.get("forecast_present"))
            item["population"] = self._matrix_population_label(item)
            item["legality"] = self._matrix_legality_label(item)
            item["intent_state"] = self._intent_state(item)
            item["execution_state"] = self._execution_state(item)
            item["observation_state"] = self._observation_state(item)
            item["interpretation_state"] = self._interpretation_state(item)
            item["phase"] = self._evidence_phase(item)
            item["surface_role"] = self._row_surface_role(item)
            item["retention_hint"] = self._retention_hint(item["surface_role"])

            if item["result_present"]:
                trust_annotation = self._trust_annotation(
                    item,
                    population=item["population"],
                    seed_group_sizes=_seed_sizes(item["population"], item["legality"]),
                )
                item.update(trust_annotation)
                item["admissibility"] = item["trust_state"]
            else:
                item["trust_state"] = None
                item["replication_state"] = None
                item["replicate_count"] = 0
                item["metric_state"] = None
                item["quarantine_reason"] = None
                item["admissibility"] = "unobserved"

            if family_filter and str(item.get("family") or "") != family_filter:
                continue
            if manifest_filter and str(item.get("manifest") or "__unknown__") != manifest_filter:
                continue
            if selected_population != "all" and item["population"] != selected_population:
                continue
            if selected_legality == "legal":
                if item["legality"] == "illegal":
                    continue
            elif selected_legality == "illegal":
                if item["legality"] != "illegal":
                    continue
            if selected_trust != "all":
                if item.get("trust_state") != selected_trust:
                    continue
            if state_filter and str(item.get("state") or "") != state_filter:
                continue
            rows.append(item)

        manifest_summaries: dict[str, dict[str, Any]] = {}
        for row in rows:
            manifest_name = str(row.get("manifest") or "__unknown__")
            summary = manifest_summaries.setdefault(
                manifest_name,
                {
                    "manifest": manifest_name,
                    "families": set(),
                    "jobs": 0,
                    "pending": 0,
                    "running": 0,
                    "completed": 0,
                    "observed": 0,
                    "curves": 0,
                    "forecasted": 0,
                    "admissible": 0,
                    "provisional": 0,
                    "quarantined": 0,
                    "unobserved": 0,
                },
            )
            family_value = str(row.get("family") or "")
            if family_value:
                summary["families"].add(family_value)
            summary["jobs"] += 1
            state_value = str(row.get("state") or "")
            if state_value == "pending":
                summary["pending"] += 1
            elif state_value in {"dispatched", "running"}:
                summary["running"] += 1
            elif state_value == "completed":
                summary["completed"] += 1
            if row.get("result_present"):
                summary["observed"] += 1
            if row.get("observation_state") == "curve":
                summary["curves"] += 1
            if row.get("forecast_present"):
                summary["forecasted"] += 1
            trust_state = row.get("trust_state")
            if trust_state in {"admissible", "provisional", "quarantined"}:
                summary[trust_state] += 1
            else:
                summary["unobserved"] += 1

        manifest_rows: list[dict[str, Any]] = []
        for manifest_name, summary in manifest_summaries.items():
            families = sorted(summary.pop("families"))
            summary["family_count"] = len(families)
            summary["families"] = families
            summary["surface_role"] = self._manifest_surface_role(summary)
            summary["retention_hint"] = self._retention_hint(summary["surface_role"])
            manifest_rows.append(summary)

        role_priority = {
            "live_control_input": 0,
            "live_mixed": 1,
            "evidence_archive": 2,
            "archive_import": 3,
            "unknown_provenance": 4,
            "candidate_prune": 5,
        }
        manifest_rows.sort(
            key=lambda row: (
                role_priority.get(str(row.get("surface_role") or ""), 99),
                -int(row.get("jobs") or 0),
                str(row.get("manifest") or ""),
            )
        )

        summary = {
            "population": selected_population,
            "legality": selected_legality,
            "trust": selected_trust,
            "row_count": len(rows),
            "manifest_count": len(manifest_rows),
            "population_counts": defaultdict(int),
            "legality_counts": defaultdict(int),
            "phase_counts": defaultdict(int),
            "surface_role_counts": defaultdict(int),
            "trust_counts": defaultdict(int),
            "observation_counts": defaultdict(int),
        }
        for row in rows:
            summary["population_counts"][str(row.get("population") or "unknown")] += 1
            summary["legality_counts"][str(row.get("legality") or "unknown")] += 1
            summary["phase_counts"][str(row.get("phase") or "unknown")] += 1
            summary["observation_counts"][str(row.get("observation_state") or "unknown")] += 1
            trust_label = str(row.get("admissibility") or "unknown")
            summary["trust_counts"][trust_label] += 1
        for manifest_row in manifest_rows:
            summary["surface_role_counts"][str(manifest_row.get("surface_role") or "unknown")] += 1
        summary["population_counts"] = dict(summary["population_counts"])
        summary["legality_counts"] = dict(summary["legality_counts"])
        summary["phase_counts"] = dict(summary["phase_counts"])
        summary["surface_role_counts"] = dict(summary["surface_role_counts"])
        summary["trust_counts"] = dict(summary["trust_counts"])
        summary["observation_counts"] = dict(summary["observation_counts"])

        phase_priority = {
            "execution": 0,
            "intent": 1,
            "observation": 2,
            "interpretation": 3,
            "unknown": 4,
        }
        state_priority = {"running": 0, "dispatched": 1, "pending": 2, "completed": 3, "unknown": 4}
        rows.sort(
            key=lambda row: (
                role_priority.get(str(row.get("surface_role") or ""), 99),
                phase_priority.get(str(row.get("phase") or ""), 99),
                state_priority.get(str(row.get("state") or "unknown"), 99),
                str(row.get("manifest") or "__unknown__"),
                str(row.get("name") or ""),
            )
        )

        if top_k > 0:
            rows = rows[:top_k]
        return {
            "summary": summary,
            "manifests": manifest_rows,
            "rows": rows,
        }

    def analysis_rows(
        self,
        *,
        family: str | None = None,
        limit: int | None = None,
        max_bpb: float = 3.0,
        controlled_only: bool = True,
        trust: str = "all",
    ) -> list[dict[str, Any]]:
        clauses = ["NOT r.illegal", "r.bpb < ?"]
        params: list[Any] = [max_bpb]
        if family:
            clauses.append("r.family = ?")
            params.append(family)
        if controlled_only:
            clauses.append("j.name IS NOT NULL")
            clauses.append("COALESCE(j.manifest, '') != ?")
            params.append(IMPORTED_RESULT_MANIFEST)

        sql = f"""
            SELECT r.name, r.bpb, r.family,
                   r.illegal,
                   r.json_archive,
                   r.steps AS result_steps,
                   r.seq_len AS result_seq_len,
                   c.json_blob AS config_json,
                   j.steps AS job_steps,
                   j.seed AS job_seed,
                   j.lr AS job_lr,
                   j.batch_size AS job_batch_size,
                   j.manifest
            FROM results r
            LEFT JOIN configs c ON r.config_id = c.id
            LEFT JOIN jobs j ON j.name = r.name
            WHERE {" AND ".join(clauses)}
            ORDER BY r.bpb
        """
        if limit is not None and limit > 0:
            sql += " LIMIT ?"
            params.append(limit)

        rows = self._read(sql, tuple(params))
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["config"] = self._config_from_joined_row(item)
            item["steps"] = item["config"].get("steps")
            item["seq_len"] = item["config"].get("seq_len")
            result.append(item)
        population = "controlled" if controlled_only else "all"
        annotated = self._annotate_result_rows(
            result,
            population=population,
            legality="legal",
        )
        trust_filter = self._coerce_trust_filter(trust)
        if trust_filter != "all":
            annotated = [row for row in annotated if row.get("trust_state") == trust_filter]
        return annotated

    def _backfill_imported_jobs(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            """
            SELECT r.name, r.config_id, r.steps, r.created_at, c.json_blob
            FROM results r
            LEFT JOIN jobs j ON j.name = r.name
            LEFT JOIN configs c ON c.id = r.config_id
            WHERE j.name IS NULL
        """
        ).fetchall()
        for row in rows:
            cfg = self._load_json_blob(row["json_blob"])
            seed = self._coerce_int(cfg.get("seed") or cfg.get("init_seed"))
            lr = self._coerce_float(cfg.get("learning_rate") or cfg.get("lr"))
            batch_size = self._coerce_int(cfg.get("batch_size"))
            steps = self._coerce_int(row["steps"]) or self._coerce_int(cfg.get("steps"))
            completed_at = row["created_at"] or time.time()
            conn.execute(
                """
                INSERT INTO jobs (
                    name, config_id, manifest, parent, state, steps, seed, lr,
                    batch_size, launcher, completed_at, command
                )
                VALUES (?, ?, ?, '', 'completed', ?, ?, ?, ?, ?, ?, '')
            """,
                (
                    row["name"],
                    row["config_id"],
                    IMPORTED_RESULT_MANIFEST,
                    steps,
                    seed,
                    lr,
                    batch_size,
                    IMPORTED_RESULT_LAUNCHER,
                    completed_at,
                ),
            )

    def _upsert_config_in_conn(self, conn: sqlite3.Connection, cfg: dict[str, Any]) -> int:
        h = self._config_hash(cfg)
        row = conn.execute("SELECT id FROM configs WHERE hash = ?", (h,)).fetchone()
        if row:
            return int(row["id"])

        family = self._detect_family(cfg)
        family_config = self._extract_family_config(cfg, family)
        conn.execute(
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
                cfg.get("readout"),
                cfg.get("readout_depth", 0),
                cfg.get("local_window", 0),
                cfg.get("oscillatory_frac", 0.0),
                cfg.get("oscillatory_schedule", "linear"),
                cfg.get("input_proj_scheme", "linear"),
                cfg.get("memory_kind", "none"),
                cfg.get("local_poly_order", 0),
                cfg.get("substrate_poly_order", 0),
                cfg.get("training_noise", 0.0),
                cfg.get("adaptive_reg", 0.0),
                cfg.get("params"),
                cfg.get("int6_mb"),
                json.dumps(cfg, sort_keys=True),
                family,
                family_config,
            ),
        )
        row = conn.execute("SELECT id FROM configs WHERE hash = ?", (h,)).fetchone()
        if not row:
            raise RuntimeError(f"failed to upsert config for hash {h}")
        return int(row["id"])

    def _backfill_legacy_command_jobs(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            """
            SELECT j.name, j.config_id, j.steps, j.seed, j.lr, j.batch_size, j.command, c.json_blob
            FROM jobs j
            LEFT JOIN configs c ON c.id = j.config_id
            WHERE COALESCE(j.command, '') != ''
        """
        ).fetchall()
        for row in rows:
            base_cfg = self._load_json_blob(row["json_blob"])
            normalized = normalize_manifest_payload(
                {
                    "name": row["name"],
                    "command": row["command"],
                    "config": base_cfg,
                    "steps": row["steps"],
                    "seed": row["seed"],
                    "learning_rate": row["lr"],
                    "batch_size": row["batch_size"],
                }
            )
            cfg = normalized.get("config")
            if not isinstance(cfg, dict) or not cfg:
                continue
            config_id = self._upsert_config_in_conn(conn, cfg)
            conn.execute(
                """
                UPDATE jobs
                SET config_id = ?, steps = ?, seed = ?, lr = ?, batch_size = ?
                WHERE name = ?
            """,
                (
                    config_id,
                    int(normalized.get("steps") or 0),
                    int(normalized.get("seed") or 42),
                    float(normalized.get("learning_rate") or 0.0),
                    int(normalized.get("batch_size") or 16),
                    row["name"],
                ),
            )
            conn.execute(
                "UPDATE results SET config_id = ? WHERE name = ?",
                (config_id, row["name"]),
            )

    def _load_manifest_job_specs(
        self,
        manifest_names: Sequence[str],
    ) -> dict[str, dict[str, dict[str, Any]]]:
        from chronohorn.fleet.dispatch import load_manifest

        loaded: dict[str, dict[str, dict[str, Any]]] = {}
        for manifest_name in sorted(
            {
                str(name).strip()
                for name in manifest_names
                if str(name or "").strip()
                and str(name).strip() != IMPORTED_RESULT_MANIFEST
            }
        ):
            rows: list[dict[str, Any]] | None = None
            manifest_key = Path(manifest_name).name
            candidates = [
                Path(manifest_name).expanduser(),
                CHRONOHORN_ROOT / "manifests" / manifest_key,
            ]
            for candidate in candidates:
                try:
                    rows = load_manifest(candidate)
                    break
                except FileNotFoundError:
                    continue
                except Exception as exc:
                    import sys
                    print(f"chronohorn: manifest load failed for {candidate}: {exc}", file=sys.stderr)
                    rows = None
                    break
            if not rows:
                continue
            loaded[manifest_key] = {
                str(row.get("name")): dict(row)
                for row in rows
                if row.get("name")
            }
        return loaded

    def _backfill_job_specs(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            """
            SELECT j.*, c.json_blob AS config_json
            FROM jobs j
            LEFT JOIN configs c ON c.id = j.config_id
        """
        ).fetchall()
        manifest_specs = self._load_manifest_job_specs(
            [str(row["manifest"] or "") for row in rows]
        )
        for row in rows:
            row_dict = dict(row)
            manifest_name = str(row_dict.get("manifest") or "")
            manifest_rows = manifest_specs.get(Path(manifest_name).name, {})
            manifest_spec = manifest_rows.get(str(row_dict.get("name") or ""))
            existing_spec = self._load_json_blob(row_dict.get("job_json"))
            base_cfg = self._load_json_blob(row_dict.get("config_json"))
            manifest_has_runtime_namespace = bool(
                manifest_spec
                and (
                    (manifest_spec or {}).get("runtime_namespace") not in (None, "")
                    or (manifest_spec or {}).get("k8s_namespace") not in (None, "")
                )
            )
            if existing_spec and str(existing_spec.get("executor_kind") or "") == "k8s_cluster":
                if (
                    not manifest_has_runtime_namespace
                    and str(existing_spec.get("runtime_namespace") or "") == "chronohorn"
                ):
                    existing_spec = dict(existing_spec)
                    existing_spec.pop("runtime_namespace", None)

            payload: dict[str, Any] = {}
            if manifest_spec:
                payload.update(dict(manifest_spec))
            if existing_spec:
                payload.update(existing_spec)

            requested_launcher = (
                row_dict.get("requested_launcher")
                or payload.get("requested_launcher")
                or (manifest_spec or {}).get("launcher")
                or payload.get("launcher")
            )
            if requested_launcher not in (None, ""):
                payload["requested_launcher"] = requested_launcher

            for key in (
                "host",
                "executor_kind",
                "executor_name",
                "launcher",
                "backend",
                "resource_class",
                "workload_kind",
                "container",
                "remote_run",
                "runtime_namespace",
                "runtime_job_name",
                "runtime_pod_name",
                "runtime_node_name",
            ):
                value = row_dict.get(key)
                if (
                    key == "runtime_namespace"
                    and str(row_dict.get("executor_kind") or "") == "k8s_cluster"
                    and not manifest_has_runtime_namespace
                    and str(value or "") == "chronohorn"
                ):
                    continue
                if value not in (None, ""):
                    payload[key] = value
            if row_dict.get("work_tokens") is not None:
                payload["work_tokens"] = row_dict["work_tokens"]

            spec = self._canonical_job_spec(
                str(row_dict["name"]),
                manifest=manifest_name,
                parent=str(row_dict.get("parent") or ""),
                family=row_dict.get("family"),
                config=base_cfg,
                steps=row_dict.get("steps"),
                seed=row_dict.get("seed"),
                lr=row_dict.get("lr"),
                batch_size=row_dict.get("batch_size"),
                command=str(row_dict.get("command") or ""),
                job_spec=payload,
            )
            if row_dict.get("state") not in (None, ""):
                spec["state"] = row_dict["state"]

            config_id = row_dict.get("config_id")
            cfg = spec.get("config")
            if isinstance(cfg, dict) and cfg:
                config_id = self._upsert_config_in_conn(conn, cfg)

            values = self._job_column_values(spec)
            conn.execute(
                """
                UPDATE jobs
                SET config_id = ?, family = ?, steps = ?, seed = ?, lr = ?, batch_size = ?,
                    executor_kind = ?, executor_name = ?, launcher = ?, requested_launcher = ?,
                    backend = ?, resource_class = ?, workload_kind = ?, work_tokens = ?,
                    runtime_namespace = ?, runtime_job_name = ?, runtime_pod_name = ?,
                    runtime_node_name = ?, command = ?, job_json = ?
                WHERE name = ?
            """,
                (
                    config_id,
                    values["family"],
                    values["steps"],
                    values["seed"],
                    values["lr"],
                    values["batch_size"],
                    values["executor_kind"],
                    values["executor_name"],
                    values["launcher"],
                    values["requested_launcher"],
                    values["backend"],
                    values["resource_class"],
                    values["workload_kind"],
                    values["work_tokens"],
                    values["runtime_namespace"],
                    values["runtime_job_name"],
                    values["runtime_pod_name"],
                    values["runtime_node_name"],
                    values["command"],
                    values["job_json"],
                    row_dict["name"],
                ),
            )
            if config_id is not None:
                conn.execute(
                    "UPDATE results SET config_id = ? WHERE name = ?",
                    (config_id, row_dict["name"]),
                )

    def _config_hash(self, cfg: dict[str, Any]) -> str:
        normalized = dict(cfg)
        if "learning_rate" not in normalized and normalized.get("lr") is not None:
            normalized["learning_rate"] = normalized["lr"]
        for key in ("seed", "init_seed", "profile", "lr"):
            normalized.pop(key, None)
        blob = json.dumps(normalized, sort_keys=True)
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
        family: str | None = None,
        config: dict | None = None,
        steps: int = 0,
        seed: int | None = None,
        lr: float | None = None,
        batch_size: int | None = None,
        command: str = "",
        job_spec: Mapping[str, Any] | None = None,
    ) -> None:
        spec = self._canonical_job_spec(
            name,
            manifest=manifest,
            parent=parent,
            family=family,
            config=config or {},
            steps=steps,
            seed=seed,
            lr=lr,
            batch_size=batch_size,
            command=command,
            job_spec=job_spec,
        )
        merged_config = spec.get("config")
        config_id = self.upsert_config(merged_config) if isinstance(merged_config, dict) and merged_config else None
        values = self._job_column_values(spec)
        self._write(
            """
            INSERT INTO jobs (
                name, config_id, manifest, family, parent, state, steps, seed, lr, batch_size,
                executor_kind, executor_name, launcher, requested_launcher, backend,
                resource_class, workload_kind, work_tokens, runtime_namespace,
                runtime_job_name, runtime_pod_name, runtime_node_name, command, job_json
            )
            VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                config_id = excluded.config_id,
                manifest = excluded.manifest,
                family = excluded.family,
                parent = excluded.parent,
                steps = excluded.steps,
                seed = excluded.seed,
                lr = excluded.lr,
                batch_size = excluded.batch_size,
                executor_kind = CASE
                    WHEN jobs.state IN ('dispatched', 'running', 'completed') THEN jobs.executor_kind
                    ELSE excluded.executor_kind
                END,
                executor_name = CASE
                    WHEN jobs.state IN ('dispatched', 'running', 'completed') THEN jobs.executor_name
                    ELSE excluded.executor_name
                END,
                launcher = CASE
                    WHEN jobs.state IN ('dispatched', 'running', 'completed') THEN jobs.launcher
                    ELSE excluded.launcher
                END,
                requested_launcher = excluded.requested_launcher,
                backend = excluded.backend,
                resource_class = excluded.resource_class,
                workload_kind = excluded.workload_kind,
                work_tokens = excluded.work_tokens,
                runtime_namespace = CASE
                    WHEN jobs.state IN ('dispatched', 'running', 'completed') THEN jobs.runtime_namespace
                    ELSE excluded.runtime_namespace
                END,
                runtime_job_name = CASE
                    WHEN jobs.state IN ('dispatched', 'running', 'completed') THEN jobs.runtime_job_name
                    ELSE excluded.runtime_job_name
                END,
                runtime_pod_name = CASE
                    WHEN jobs.state IN ('dispatched', 'running', 'completed') THEN jobs.runtime_pod_name
                    ELSE excluded.runtime_pod_name
                END,
                runtime_node_name = CASE
                    WHEN jobs.state IN ('dispatched', 'running', 'completed') THEN jobs.runtime_node_name
                    ELSE excluded.runtime_node_name
                END,
                command = excluded.command,
                job_json = excluded.job_json,
                state = CASE
                    WHEN jobs.state IN ('dispatched', 'running', 'completed') THEN jobs.state
                    ELSE 'pending'
                END
        """,
            (
                name,
                config_id,
                manifest,
                values["family"],
                parent,
                values["steps"],
                values["seed"],
                values["lr"],
                values["batch_size"],
                values["executor_kind"],
                values["executor_name"],
                values["launcher"],
                values["requested_launcher"],
                values["backend"],
                values["resource_class"],
                values["workload_kind"],
                values["work_tokens"],
                values["runtime_namespace"],
                values["runtime_job_name"],
                values["runtime_pod_name"],
                values["runtime_node_name"],
                values["command"],
                values["job_json"],
            ),
            wait=True,
        )

    def record_launch(
        self,
        name: str,
        *,
        host: str,
        executor_kind: str = "",
        executor_name: str = "",
        launcher: str = "",
        container: str = "",
        remote_run: str = "",
        runtime_namespace: str = "",
        runtime_job_name: str = "",
        runtime_pod_name: str = "",
        runtime_node_name: str = "",
    ) -> None:
        # Upsert: update if exists, insert if not
        existing = self._read_one(
            "SELECT j.*, c.json_blob AS config_json FROM jobs j "
            "LEFT JOIN configs c ON c.id = j.config_id WHERE j.name = ?",
            (name,),
        )
        if existing:
            existing_row = dict(existing)
            existing_job = self._job_row_to_dict(existing)
            requested_launcher = (
                existing_job.get("requested_launcher")
                or existing_row.get("requested_launcher")
                or existing_job.get("launcher")
                or existing_row.get("launcher")
            )
            existing_job["host"] = host
            if executor_kind:
                existing_job["executor_kind"] = executor_kind
            if executor_name:
                existing_job["executor_name"] = executor_name
            if launcher:
                existing_job["launcher"] = launcher
            if requested_launcher not in (None, ""):
                existing_job["requested_launcher"] = requested_launcher
            if container:
                existing_job["container"] = container
            if remote_run:
                existing_job["remote_run"] = remote_run
            if runtime_namespace:
                existing_job["runtime_namespace"] = runtime_namespace
            if runtime_job_name:
                existing_job["runtime_job_name"] = runtime_job_name
            if runtime_pod_name:
                existing_job["runtime_pod_name"] = runtime_pod_name
            if runtime_node_name:
                existing_job["runtime_node_name"] = runtime_node_name
            values = self._job_column_values(existing_job)
            merged_container = str(existing_job.get("container") or "") or None
            merged_remote_run = str(existing_job.get("remote_run") or "") or None
            self._write(
                """
                UPDATE jobs SET state = 'dispatched', host = ?, executor_kind = ?, executor_name = ?,
                    launcher = ?, requested_launcher = ?, launched_at = ?, container = ?,
                    remote_run = ?, runtime_namespace = ?, runtime_job_name = ?, runtime_pod_name = ?,
                    runtime_node_name = ?, job_json = ?
                WHERE name = ?
            """,
                (
                    host,
                    values["executor_kind"],
                    values["executor_name"],
                    values["launcher"],
                    values["requested_launcher"],
                    time.time(),
                    merged_container,
                    merged_remote_run,
                    values["runtime_namespace"],
                    values["runtime_job_name"],
                    values["runtime_pod_name"],
                    values["runtime_node_name"],
                    self._dump_json_blob(existing_job),
                    name,
                ),
                wait=True,
            )
        else:
            spec = self._canonical_job_spec(
                name,
                job_spec={
                    "name": name,
                    "host": host,
                    "executor_kind": executor_kind,
                    "executor_name": executor_name,
                    "launcher": launcher,
                    "requested_launcher": launcher,
                    "container": container,
                    "remote_run": remote_run,
                    "runtime_namespace": runtime_namespace,
                    "runtime_job_name": runtime_job_name,
                    "runtime_pod_name": runtime_pod_name,
                    "runtime_node_name": runtime_node_name,
                },
            )
            values = self._job_column_values(spec)
            self._write(
                """
                INSERT INTO jobs (
                    name, state, host, executor_kind, executor_name, launcher,
                    requested_launcher, launched_at, container, remote_run,
                    runtime_namespace, runtime_job_name, runtime_pod_name, runtime_node_name,
                    family, backend, resource_class, workload_kind, work_tokens, command, job_json
                )
                VALUES (?, 'dispatched', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    name,
                    host,
                    values["executor_kind"],
                    values["executor_name"],
                    launcher,
                    values["requested_launcher"],
                    time.time(),
                    container,
                    remote_run,
                    values["runtime_namespace"],
                    values["runtime_job_name"],
                    values["runtime_pod_name"],
                    values["runtime_node_name"],
                    values["family"],
                    values["backend"],
                    values["resource_class"],
                    values["workload_kind"],
                    values["work_tokens"],
                    values["command"],
                    values["job_json"],
                ),
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
        loss: float | None = None,
        tflops: float | None = None,
        elapsed_sec: float | None = None,
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
        payload_config = self._payload_experiment_config(m, train, cfg if isinstance(cfg, dict) else {})

        raw_bpb = m.get("test_bpb")
        if raw_bpb is None:
            import sys
            print(f"chronohorn: skipping {name} (missing test_bpb)", file=sys.stderr)
            return
        try:
            bpb = float(raw_bpb)
        except (TypeError, ValueError):
            import sys
            print(f"chronohorn: skipping {name} (invalid bpb: {raw_bpb!r})", file=sys.stderr)
            return
        if bpb <= 0:
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
            b1 = p1.get("bpb") or p1.get("test_bpb")
            b2 = p2.get("bpb") or p2.get("test_bpb")
            s1, s2 = p1.get("step"), p2.get("step")
            if b1 and b2 and s1 and s2 and b1 > b2 and s2 > s1:
                slope = (b1 - b2) / (s2 - s1) * 1000

        # Try to reuse config from the jobs table (manifest has richer config)
        job_row = self._read_one(
            "SELECT config_id FROM jobs WHERE name = ?", (name,)
        )
        if not job_row:
            self.record_job(
                name,
                manifest=IMPORTED_RESULT_MANIFEST,
                config=payload_config,
                steps=perf.get("steps_completed") or train.get("steps"),
                seed=train.get("seed") or m.get("seed"),
                lr=train.get("learning_rate") or m.get("learning_rate"),
                batch_size=train.get("batch_size"),
                command="",
            )
            job_row = self._read_one(
                "SELECT config_id FROM jobs WHERE name = ?", (name,)
            )
        if job_row and job_row["config_id"]:
            config_id = job_row["config_id"]
        else:
            config_id = self.upsert_config(payload_config)
            self._write(
                "UPDATE jobs SET config_id = ? WHERE name = ?",
                (config_id, name),
                wait=True,
            )

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
            except Exception as exc:
                import sys
                print(f"chronohorn: forecast failed for {name}: {exc}", file=sys.stderr)
                self.record_event("forecast_error", name=name, error=str(exc))

        # G5: Auto-check branch health after ingesting result (warn once per branch)
        try:
            parts = name.split("-")
            prefix = parts[0]
            if prefix not in self._branch_health_warned:
                health = self.branch_health(prefix)
                if health.get("status") == "dead" and health.get("count", 0) >= 5:
                    import sys
                    print(f"chronohorn: \u26a0 branch '{prefix}' has {health['count']} results, none on frontier (gap: +{health['gap']:.3f} bpb)", file=sys.stderr)
                    self._branch_health_warned.add(prefix)
        except Exception as exc:
            import sys
            print(f"chronohorn: branch health check failed for {name}: {exc}", file=sys.stderr)
            self.record_event("branch_health_error", name=name, error=str(exc))

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
        except Exception as exc:
            import sys
            print(f"chronohorn: prediction reconcile failed for {name}: {exc}", file=sys.stderr)
            self.record_event("prediction_reconcile_error", name=name, error=str(exc))

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

    def build_run_snapshots(
        self,
        *,
        population: str = "controlled",
        trust: str = "all",
    ) -> list:
        """Build RunSnapshot objects directly from DB — replaces RunStore pipeline.

        Returns a list of RunSnapshot (from control.models) with all fields
        populated from results + jobs + forecasts tables.
        """
        from chronohorn.control.models import RunSnapshot

        rows = self._read("""
            SELECT
                r.name,
                COALESCE(r.family, j.family) AS family,
                COALESCE(j.state, 'completed') AS state,
                r.bpb,
                r.train_bpb,
                r.json_archive,
                r.illegal,
                r.int6_mb,
                r.params,
                r.tok_s,
                r.tflops_s,
                r.wall_sec,
                j.host,
                j.launcher,
                j.manifest,
                f.forecast_bpb,
                f.marginal_per_tflop,
                f.forecast_low,
                f.forecast_high,
                f.method AS forecast_method,
                f.r2 AS forecast_r2,
                f.asymptote,
                f.headroom,
                f.saturation_status,
                COALESCE(f.asymptote_reliable, 0) AS asymptote_reliable
            FROM results r
            LEFT JOIN jobs j ON j.name = r.name
            LEFT JOIN forecasts f ON f.name = r.name
            WHERE NOT r.illegal
            ORDER BY r.bpb ASC
        """)

        trust_index = {}
        try:
            trust_index = self.result_trust_index(population=population, legality="legal")
        except Exception:
            pass

        artifact_limit_mb = 16.0
        snapshots = []
        for row in rows:
            name = row["name"]
            trust_entry = trust_index.get(name, {})
            trust_state = trust_entry.get("trust_state")
            if trust != "all" and trust_state != trust:
                continue

            forecast_bpb = row["forecast_bpb"]
            forecast_meta = {}
            if forecast_bpb is not None:
                forecast_meta = {
                    "forecast_bpb": forecast_bpb,
                    "marginal_gain_per_tflop": row["marginal_per_tflop"],
                    "forecast_confidence": "high" if row["forecast_r2"] and row["forecast_r2"] > 0.95 else "medium" if row["forecast_r2"] and row["forecast_r2"] > 0.8 else "low",
                    "uncertainty": {
                        "forecast_low_95": row["forecast_low"],
                        "forecast_high_95": row["forecast_high"],
                    },
                    "estimated_sustained_tflops": row["tflops_s"],
                }

            int6_mb = row["int6_mb"]
            artifact_viable = int6_mb is not None and int6_mb <= artifact_limit_mb

            snapshots.append(RunSnapshot(
                name=name,
                family=row["family"],
                state=row["state"],
                decision=None,
                path=row["json_archive"],
                host=row["host"],
                launcher=row["launcher"],
                metric_name="bpb",
                metric_value=row["bpb"],
                forecast_metric_name="bpb" if forecast_bpb is not None else None,
                forecast_metric_value=forecast_bpb,
                artifact_viable=artifact_viable,
                trust_state=trust_state,
                metric_state=trust_entry.get("metric_state"),
                replication_state=trust_entry.get("replication_state"),
                replicate_count=trust_entry.get("replicate_count"),
                quarantine_reason=trust_entry.get("quarantine_reason"),
                metadata={"forecast": forecast_meta} if forecast_meta else {},
            ))
        return snapshots

    def frontier(
        self,
        top_k: int = 20,
        *,
        family: str | None = None,
        population: str = "controlled",
        legality: str = "legal",
        trust: str = "all",
    ) -> list[dict]:
        if top_k <= 0:
            return []
        clauses: list[str] = []
        params: list[Any] = []
        population_clauses, population_params = self._population_clauses(population=population)
        legality_clauses, legality_params = self._legality_clauses(legality=legality)
        clauses.extend(population_clauses)
        clauses.extend(legality_clauses)
        params.extend(population_params)
        params.extend(legality_params)
        if family:
            clauses.append("r.family = ?")
            params.append(family)
        where_sql = " AND ".join(clauses) if clauses else "1=1"
        rows = self._read(f"""
            SELECT r.name, r.bpb, r.family, r.train_bpb, r.overfit_pct,
                   r.tok_s, r.wall_sec, r.slope, r.illegal,
                   r.json_archive,
                   COALESCE(r.params, c.params) as params,
                   COALESCE(r.int6_mb, c.int6_mb) as int6_mb,
                   r.steps AS result_steps,
                   r.seq_len AS result_seq_len,
                   j.steps AS job_steps,
                   j.seed AS job_seed,
                   j.lr AS job_lr,
                   j.batch_size AS job_batch_size,
                   COALESCE(r.steps, j.steps) as steps,
                   COALESCE(r.seq_len, c.seq_len) as seq_len,
                   COALESCE(r.tflops, r.total_tflops) as tflops,
                   COALESCE(f.asymptote, f.forecast_bpb) as fc_bpb,
                   COALESCE(f.asymptote_r2, f.r2) as fc_r2,
                   f.headroom, f.saturation_status,
                   c.json_blob as config_json,
                   j.manifest
            FROM results r
            LEFT JOIN configs c ON r.config_id = c.id
            LEFT JOIN forecasts f ON r.name = f.name
            LEFT JOIN jobs j ON r.name = j.name
            WHERE {where_sql}
            ORDER BY r.bpb
        """, tuple(params))
        canonical_keys = [
            "name", "bpb", "family", "params", "int6_mb", "steps", "seq_len",
            "tok_s", "tflops", "wall_sec", "illegal", "slope",
            "fc_bpb", "fc_r2", "headroom", "overfit_pct", "train_bpb",
            "saturation_status", "config_json",
        ]
        result = []
        for r in rows:
            d = dict(r)
            d["config_json"] = json.dumps(
                self._config_from_joined_row(d),
                sort_keys=True,
            )
            for k in canonical_keys:
                d.setdefault(k, None)
            result.append(d)
        annotated = self._annotate_result_rows(
            result,
            population=self._coerce_population(population),
            legality=self._coerce_legality(legality),
        )
        trust_filter = self._coerce_trust_filter(trust)
        if trust_filter != "all":
            annotated = [row for row in annotated if row.get("trust_state") == trust_filter]
        return annotated[:top_k]

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
            except (json.JSONDecodeError, TypeError):
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

    def branch_health(
        self,
        prefix: str,
        *,
        population: str = "controlled",
        legality: str = "legal",
    ) -> dict:
        """Check if an architecture branch is contributing to the frontier."""
        clauses = [
            "(r.name LIKE ? OR r.name = ?)",
        ]
        params: list[Any] = [f"{prefix}-%", prefix]
        population_clauses, population_params = self._population_clauses(population=population)
        legality_clauses, legality_params = self._legality_clauses(legality=legality)
        clauses.extend(population_clauses)
        clauses.extend(legality_clauses)
        params.extend(population_params)
        params.extend(legality_params)
        branch_results = self._read(
            f"""
            SELECT r.name, r.bpb
            FROM results r
            LEFT JOIN jobs j ON j.name = r.name
            WHERE {" AND ".join(clauses)}
            ORDER BY r.bpb LIMIT 20
        """,
            tuple(params),
        )
        if not branch_results:
            return {"prefix": prefix, "count": 0, "status": "no_results"}

        branch_best = min(r["bpb"] for r in branch_results)
        frontier = self.frontier(10, population=population, legality=legality)
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

    def frontier_velocity(
        self,
        window: int = 10,
        *,
        population: str = "controlled",
        legality: str = "legal",
        trust: str = "all",
    ) -> dict:
        """Track frontier improvement rate. Uses bpb ordering, not timestamps."""
        clauses = ["r.bpb < 3"]
        params: list[Any] = []
        population_clauses, population_params = self._population_clauses(population=population)
        legality_clauses, legality_params = self._legality_clauses(legality=legality)
        clauses.extend(population_clauses)
        clauses.extend(legality_clauses)
        params.extend(population_params)
        params.extend(legality_params)
        results = self._read(
            f"""
            SELECT r.name, r.bpb, r.wall_sec, r.created_at
                   , r.family, r.illegal, r.json_archive
            FROM results r
            LEFT JOIN jobs j ON j.name = r.name
            WHERE {" AND ".join(clauses)}
            ORDER BY r.created_at ASC
        """,
            tuple(params),
        )
        annotated = self._annotate_result_rows(
            results,
            population=self._coerce_population(population),
            legality=self._coerce_legality(legality),
        )
        trust_filter = self._coerce_trust_filter(trust)
        if trust_filter != "all":
            annotated = [row for row in annotated if row.get("trust_state") == trust_filter]
        results = annotated

        if len(results) < 2:
            return {
                "velocity_bpb_per_hour": 0,
                "trend": "insufficient_data",
                "improvements": [],
                "population": self._coerce_population(population),
                "legality": self._coerce_legality(legality),
                "trust": trust_filter,
            }

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
            "population": self._coerce_population(population),
            "legality": self._coerce_legality(legality),
            "trust": trust_filter,
        }

    def detect_groups(
        self,
        *,
        population: str = "controlled",
        legality: str = "legal",
    ) -> list[dict]:
        """Detect experiment groups by name pattern and summarize."""
        import re as _re
        clauses: list[str] = []
        params: list[Any] = []
        population_clauses, population_params = self._population_clauses(population=population)
        legality_clauses, legality_params = self._legality_clauses(legality=legality)
        clauses.extend(population_clauses)
        clauses.extend(legality_clauses)
        params.extend(population_params)
        params.extend(legality_params)
        where_sql = " AND ".join(clauses) if clauses else "1=1"
        results = self._read(
            f"""
            SELECT r.name, r.bpb, r.steps, r.family
            FROM results r
            LEFT JOIN jobs j ON j.name = r.name
            WHERE {where_sql}
            ORDER BY r.name
        """,
            tuple(params),
        )

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

    @staticmethod
    def _interpolate_curve_value(
        points: Sequence[Mapping[str, Any]],
        *,
        x_key: str,
        y_key: str,
        target: float,
    ) -> float | None:
        samples: list[tuple[float, float]] = []
        for point in points:
            x = point.get(x_key)
            y = point.get(y_key)
            if x is None or y is None:
                continue
            try:
                x_value = float(x)
                y_value = float(y)
            except (TypeError, ValueError):
                continue
            samples.append((x_value, y_value))
        if not samples:
            return None
        samples.sort(key=lambda item: item[0])
        if target == samples[0][0]:
            return samples[0][1]
        if len(samples) == 1 or target < samples[0][0] or target > samples[-1][0]:
            return None
        for (x1, y1), (x2, y2) in zip(samples, samples[1:]):
            if target == x1:
                return y1
            if x1 <= target <= x2:
                if x2 == x1:
                    return y1
                mix = (target - x1) / (x2 - x1)
                return y1 + mix * (y2 - y1)
        return samples[-1][1] if target == samples[-1][0] else None

    def architecture_audit(
        self,
        *,
        population: str = "controlled",
        legality: str = "legal",
        trust: str = "all",
        families: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        selected_population = self._coerce_population(population)
        selected_legality = self._coerce_legality(legality)
        selected_trust = self._coerce_trust_filter(trust)
        family_filter = [str(f) for f in (families or []) if f]

        clauses: list[str] = []
        params: list[Any] = []
        population_clauses, population_params = self._population_clauses(population=selected_population)
        legality_clauses, legality_params = self._legality_clauses(legality=selected_legality)
        clauses.extend(population_clauses)
        clauses.extend(legality_clauses)
        params.extend(population_params)
        params.extend(legality_params)
        if family_filter:
            placeholders = ",".join("?" * len(family_filter))
            clauses.append(f"r.family IN ({placeholders})")
            params.extend(family_filter)
        where_sql = " AND ".join(clauses) if clauses else "1=1"

        rows = [
            dict(row)
            for row in self._read(
                f"""
                SELECT r.name, r.family, r.bpb,
                       COALESCE(r.tflops, r.total_tflops) AS total_tflops,
                       COALESCE(r.params, c.params) AS params,
                       COALESCE(r.int6_mb, c.int6_mb) AS int6_mb,
                       r.json_archive,
                       r.steps AS result_steps,
                       j.steps AS job_steps,
                       j.seed AS job_seed,
                       j.lr AS job_lr,
                       j.batch_size AS job_batch_size,
                       j.manifest
                FROM results r
                LEFT JOIN jobs j ON j.name = r.name
                LEFT JOIN configs c ON c.id = r.config_id
                WHERE {where_sql}
                ORDER BY r.bpb ASC
            """,
                tuple(params),
            )
        ]
        rows = self._annotate_result_rows(
            rows,
            population=selected_population,
            legality=selected_legality,
        )
        if selected_trust != "all":
            rows = [row for row in rows if row.get("trust_state") == selected_trust]
        if not rows:
            return {
                "population": selected_population,
                "legality": selected_legality,
                "trust": selected_trust,
                "families": {},
                "endpoint_ranking": [],
                "step_envelope": [],
                "compute_envelope": [],
                "warnings": ["no matching results"],
                "verdict": {
                    "status": "no_data",
                    "message": "No matching results in the selected cohort.",
                },
            }

        family_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        name_to_family: dict[str, str] = {}
        for row in rows:
            row["steps"] = row.get("job_steps") or row.get("result_steps") or 0
            family = str(row.get("family") or "unknown")
            family_rows[family].append(row)
            name_to_family[row["name"]] = family
        selected_names = set(name_to_family)

        probe_rows = self._read(
            f"""
            SELECT p.name, r.family, p.step, p.bpb, p.tflops
            FROM probes p
            JOIN results r ON r.name = p.name
            LEFT JOIN jobs j ON j.name = r.name
            WHERE {where_sql}
            ORDER BY r.family, p.name, p.step
        """,
            tuple(params),
        )
        curves_by_family: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        for row in probe_rows:
            if row["name"] not in selected_names:
                continue
            family = str(row["family"] or "unknown")
            curves_by_family[family][row["name"]].append(
                {
                    "step": row["step"],
                    "bpb": row["bpb"],
                    "tf": row["tflops"],
                }
            )

        seed_support_by_family: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"group_count": 0, "replicated_runs": 0, "max_group_size": 1, "best_ci_95": None}
        )
        seed_support_by_run: dict[str, dict[str, Any]] = {}
        for group in self.seed_groups(population=selected_population, legality=selected_legality):
            family_names = {
                name_to_family.get(run.get("name"))
                for run in group.get("runs", [])
                if name_to_family.get(run.get("name"))
            }
            if len(family_names) != 1:
                continue
            family = next(iter(family_names))
            support = seed_support_by_family[family]
            count = int(group.get("count") or 0)
            support["group_count"] += 1
            support["replicated_runs"] += count
            support["max_group_size"] = max(support["max_group_size"], count)
            ci_95 = group.get("ci_95")
            if ci_95 is not None and (
                support["best_ci_95"] is None or ci_95 < support["best_ci_95"]
            ):
                support["best_ci_95"] = ci_95
            for run in group.get("runs", []):
                run_name = run.get("name")
                if not run_name:
                    continue
                seed_support_by_run[str(run_name)] = {
                    "group_size": count,
                    "ci_95": ci_95,
                }

        family_summary: dict[str, dict[str, Any]] = {}
        endpoint_ranking: list[dict[str, Any]] = []
        for family, family_result_rows in sorted(family_rows.items(), key=lambda item: item[1][0]["bpb"]):
            best = min(family_result_rows, key=lambda row: row["bpb"])
            step_counts: dict[int, int] = {}
            for row in family_result_rows:
                steps = int(row.get("steps") or 0)
                step_counts[steps] = step_counts.get(steps, 0) + 1
            seed_support = dict(seed_support_by_family.get(family, {}))
            if not seed_support:
                seed_support = {"group_count": 0, "replicated_runs": 0, "max_group_size": 1, "best_ci_95": None}
            best_run_seed_support = dict(seed_support_by_run.get(best["name"], {}))
            if not best_run_seed_support:
                best_run_seed_support = {"group_size": 1, "ci_95": None}
            family_summary[family] = {
                "result_count": len(family_result_rows),
                "probe_runs": len(curves_by_family.get(family, {})),
                "distinct_endpoint_steps": [
                    {"steps": step, "count": count}
                    for step, count in sorted(step_counts.items())
                ],
                "seed_support": seed_support,
                "best_run_seed_support": best_run_seed_support,
                "best": {
                    "name": best["name"],
                    "bpb": best["bpb"],
                    "trust_state": best.get("trust_state"),
                    "metric_state": best.get("metric_state"),
                    "steps": best.get("steps"),
                    "seed": best.get("job_seed"),
                    "learning_rate": best.get("job_lr"),
                    "batch_size": best.get("job_batch_size"),
                    "total_tflops": best.get("total_tflops"),
                    "int6_mb": best.get("int6_mb"),
                    "params": best.get("params"),
                    "manifest": best.get("manifest"),
                },
            }
            endpoint_ranking.append(
                {
                    "family": family,
                    "name": best["name"],
                    "bpb": best["bpb"],
                    "trust_state": best.get("trust_state"),
                    "steps": best.get("steps"),
                    "total_tflops": best.get("total_tflops"),
                    "int6_mb": best.get("int6_mb"),
                    "seed_support": best_run_seed_support["group_size"],
                }
            )

        def _build_envelope(*, x_key: str, targets: Sequence[float], label: str) -> list[dict[str, Any]]:
            envelope: list[dict[str, Any]] = []
            for target in targets:
                scores: dict[str, float] = {}
                leaders: dict[str, str] = {}
                for family, run_curves in curves_by_family.items():
                    best_value: float | None = None
                    best_run: str | None = None
                    for run_name, points in run_curves.items():
                        value = self._interpolate_curve_value(points, x_key=x_key, y_key="bpb", target=target)
                        if value is None:
                            continue
                        if best_value is None or value < best_value:
                            best_value = value
                            best_run = run_name
                    if best_value is not None and best_run is not None:
                        scores[family] = round(best_value, 4)
                        leaders[family] = best_run
                if len(scores) < 2:
                    continue
                ranked = sorted(scores.items(), key=lambda item: item[1])
                leader_family, leader_bpb = ranked[0]
                second_bpb = ranked[1][1]
                entry = {
                    label: int(target) if float(target).is_integer() else round(target, 4),
                    "scores": scores,
                    "leader": leader_family,
                    "leader_bpb": leader_bpb,
                    "leader_run": leaders[leader_family],
                    "gap_to_second": round(second_bpb - leader_bpb, 4),
                }
                envelope.append(entry)
            return envelope

        step_envelope = _build_envelope(
            x_key="step",
            targets=[200, 400, 800, 1600, 3200, 5000, 6400, 10000],
            label="steps",
        )
        compute_envelope = _build_envelope(
            x_key="tf",
            targets=[250, 500, 1000, 2000, 3000, 4000, 5000, 6000, 8000],
            label="tflops",
        )

        endpoint_leader = endpoint_ranking[0]["family"] if endpoint_ranking else None
        highest_common_step = step_envelope[-1] if step_envelope else None
        highest_common_compute = compute_envelope[-1] if compute_envelope else None

        warnings: list[str] = []
        if endpoint_leader and family_summary[endpoint_leader]["best_run_seed_support"]["group_size"] < 2:
            warnings.append(f"endpoint leader {endpoint_leader} best run is still single-seed")
        if endpoint_leader and family_summary[endpoint_leader]["best"]["metric_state"] != "dataset_anchored":
            warnings.append(
                f"endpoint leader {endpoint_leader} best run metric state is {family_summary[endpoint_leader]['best']['metric_state']}"
            )
        if highest_common_step and highest_common_step["leader"] != endpoint_leader:
            warnings.append(
                f"highest common step budget ({highest_common_step['steps']}) favors {highest_common_step['leader']}, not endpoint leader {endpoint_leader}"
            )
        if highest_common_compute and highest_common_compute["leader"] != endpoint_leader:
            warnings.append(
                f"highest common compute budget ({highest_common_compute['tflops']} TF) favors {highest_common_compute['leader']}, not endpoint leader {endpoint_leader}"
            )
        for family, summary in family_summary.items():
            if summary["probe_runs"] < 3:
                warnings.append(f"{family} has weak probe coverage ({summary['probe_runs']} probe-backed runs)")

        if not endpoint_ranking or len(endpoint_ranking) < 2:
            verdict = {
                "status": "insufficient_data",
                "message": "Need at least two families in the selected cohort to compare architectures.",
            }
        else:
            endpoint_gap = round(endpoint_ranking[1]["bpb"] - endpoint_ranking[0]["bpb"], 4)
            status = "provisionally_favor_endpoint_leader"
            message = (
                f"Endpoint best is {endpoint_leader} by {endpoint_gap} bpb "
                f"({endpoint_ranking[0]['name']} vs {endpoint_ranking[1]['name']})."
            )
            if highest_common_compute and highest_common_compute["leader"] != endpoint_leader:
                status = "falsify_before_commit"
                message = (
                    f"Endpoint best is {endpoint_leader}, but matched-compute at "
                    f"{highest_common_compute['tflops']} TF favors {highest_common_compute['leader']} "
                    f"({highest_common_compute['scores'][highest_common_compute['leader']]} vs "
                    f"{highest_common_compute['scores'][endpoint_leader]} bpb)."
                )
            elif highest_common_step and highest_common_step["leader"] != endpoint_leader:
                status = "mixed_regime_signal"
                message = (
                    f"Endpoint best is {endpoint_leader}, but the highest common step budget "
                    f"({highest_common_step['steps']}) favors {highest_common_step['leader']}."
                )
            if endpoint_leader and family_summary[endpoint_leader]["best_run_seed_support"]["group_size"] < 2:
                status = "falsify_before_commit"
                message += " Endpoint leader is still single-seed."
            verdict = {
                "status": status,
                "endpoint_leader": endpoint_leader,
                "highest_common_step_leader": highest_common_step["leader"] if highest_common_step else None,
                "highest_common_compute_leader": highest_common_compute["leader"] if highest_common_compute else None,
                "message": message,
            }

        return {
            "population": selected_population,
            "legality": selected_legality,
            "trust": selected_trust,
            "families": family_summary,
            "endpoint_ranking": endpoint_ranking,
            "step_envelope": step_envelope,
            "compute_envelope": compute_envelope,
            "warnings": warnings,
            "verdict": verdict,
        }

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

    def marginal_rank(
        self,
        top_k: int = 20,
        *,
        family: str | None = None,
        population: str = "controlled",
        legality: str = "legal",
    ) -> list[dict]:
        if top_k <= 0:
            return []
        clauses = ["r.slope IS NOT NULL", "r.slope > 0"]
        params: list[Any] = []
        population_clauses, population_params = self._population_clauses(population=population)
        legality_clauses, legality_params = self._legality_clauses(legality=legality)
        clauses.extend(population_clauses)
        clauses.extend(legality_clauses)
        params.extend(population_params)
        params.extend(legality_params)
        if family:
            clauses.append("r.family = ?")
            params.append(family)
        params.append(top_k * 3)
        rows = self._read(
            f"""
            SELECT r.name, r.bpb, r.family, r.slope,
                   COALESCE(r.tflops, r.total_tflops) as total_tf,
                   r.steps, r.illegal, r.tok_s, r.wall_sec,
                   COALESCE(f.asymptote, f.forecast_bpb) as fc_bpb
            FROM results r
            LEFT JOIN forecasts f ON r.name = f.name
            LEFT JOIN jobs j ON j.name = r.name
            WHERE {" AND ".join(clauses)}
            ORDER BY r.slope DESC LIMIT ?
        """,
            tuple(params),  # fetch extra, will re-sort
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
        return self._job_rows(states=("pending",), manifest=manifest)

    def running_jobs(self) -> list[dict]:
        return self._job_rows(states=("dispatched", "running"))

    def active_jobs(self, manifest: str | None = None) -> list[dict[str, Any]]:
        return self._job_rows(states=("pending", "dispatched", "running"), manifest=manifest)

    def job_spec(self, name: str) -> dict[str, Any] | None:
        row = self._read_one(
            """
            SELECT j.*, c.json_blob AS config_json
            FROM jobs j
            LEFT JOIN configs c ON c.id = j.config_id
            WHERE j.name = ?
        """,
            (name,),
        )
        return self._job_row_to_dict(row) if row else None

    def _job_rows(
        self,
        *,
        states: Sequence[str],
        manifest: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if states:
            clauses.append(
                "j.state IN (" + ", ".join("?" for _ in states) + ")"
            )
            params.extend(states)
        if manifest:
            clauses.append("j.manifest = ?")
            params.append(manifest)
        where_sql = " AND ".join(clauses) if clauses else "1=1"
        rows = self._read(
            f"""
            SELECT j.*, c.json_blob AS config_json
            FROM jobs j
            LEFT JOIN configs c ON c.id = j.config_id
            WHERE {where_sql}
            ORDER BY j.name
        """,
            tuple(params),
        )
        return [self._job_row_to_dict(row) for row in rows]

    def result_count(self) -> int:
        return self._read_one(
            "SELECT COUNT(*) FROM results"
        )[0]

    def best_bpb(self, legal_only: bool = True, *, population: str = "controlled") -> float | None:
        where: list[str] = []
        params: list[Any] = []
        population_clauses, population_params = self._population_clauses(population=population)
        where.extend(population_clauses)
        params.extend(population_params)
        legality = "legal" if legal_only else "all"
        legality_clauses, legality_params = self._legality_clauses(legality=legality)
        where.extend(legality_clauses)
        params.extend(legality_params)
        where_sql = " AND ".join(where) if where else "1=1"
        row = self._read_one(
            f"""
            SELECT MIN(r.bpb)
            FROM results r
            LEFT JOIN jobs j ON j.name = r.name
            WHERE {where_sql}
        """,
            tuple(params),
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
        population_summary = self.population_summary()
        trust_summary = self.trust_summary(population="all", legality="all")
        controlled_legal_trust = self.trust_summary()
        n = population_summary["result_count"]
        best_any = population_summary["populations"].get("controlled", {}).get("best_bpb")
        admissible_frontier = self.frontier(1, trust="admissible")
        provisional_frontier = self.frontier(1, trust="provisional")
        best = admissible_frontier[0]["bpb"] if admissible_frontier else best_any
        admissible_best = admissible_frontier[0]["bpb"] if admissible_frontier else None
        provisional_best = provisional_frontier[0]["bpb"] if provisional_frontier else None
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
            "best_bpb_any": best_any,
            "admissible_best_bpb": admissible_best,
            "provisional_best_bpb": provisional_best,
            "legal_result_count": population_summary["legal_count"],
            "illegal_result_count": population_summary["illegal_count"],
            "populations": population_summary["populations"],
            "trust": trust_summary,
            "controlled_legal_trust": controlled_legal_trust,
            "families": {r["family"]: r["c"] for r in families},
            "manifest_count": manifests[0]["c"] if manifests else 0,
        }

    def config_summary(self, name: str) -> dict:
        """Return family-specific config fields via the adapter."""
        row = self._joined_run_row(name)
        if not row:
            return {}
        try:
            from chronohorn.families.registry import get_adapter
            cfg = self._config_from_joined_row(row)
            adapter = get_adapter(row["family"])
            return adapter.config_summary({"config": {"train": cfg}})
        except (KeyError, ImportError, json.JSONDecodeError):
            return self._config_from_joined_row(row)

    def config_diff(self, name1: str, name2: str) -> dict:
        """Compare configs of two runs. Returns changed, only_in_1, only_in_2."""
        row1 = self._joined_run_row(name1)
        row2 = self._joined_run_row(name2)
        if not row1 or not row2:
            return {"error": "one or both runs not found"}

        cfg1 = self._config_from_joined_row(row1)
        cfg2 = self._config_from_joined_row(row2)

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
                SELECT r.name, r.family,
                       r.steps AS result_steps,
                       r.seq_len AS result_seq_len,
                       c.json_blob,
                       j.steps AS job_steps,
                       j.seed AS job_seed,
                       j.lr AS job_lr,
                       j.batch_size AS job_batch_size
                FROM results r
                LEFT JOIN configs c ON r.config_id = c.id
                LEFT JOIN jobs j ON j.name = r.name
                WHERE r.name IN ({placeholders})
            """, tuple(names))
        elif family:
            rows = self._read("""
                SELECT r.name, r.family,
                       r.steps AS result_steps,
                       r.seq_len AS result_seq_len,
                       c.json_blob,
                       j.steps AS job_steps,
                       j.seed AS job_seed,
                       j.lr AS job_lr,
                       j.batch_size AS job_batch_size
                FROM results r
                LEFT JOIN configs c ON r.config_id = c.id
                LEFT JOIN jobs j ON j.name = r.name
                WHERE r.family = ?
                  AND NOT r.illegal
                  AND j.name IS NOT NULL
                  AND COALESCE(j.manifest, '') != ?
                ORDER BY r.bpb LIMIT ?
            """, (family, IMPORTED_RESULT_MANIFEST, limit))
        else:
            rows = self._read("""
                SELECT r.name, r.family,
                       r.steps AS result_steps,
                       r.seq_len AS result_seq_len,
                       c.json_blob,
                       j.steps AS job_steps,
                       j.seed AS job_seed,
                       j.lr AS job_lr,
                       j.batch_size AS job_batch_size,
                       j.manifest
                FROM results r
                LEFT JOIN configs c ON r.config_id = c.id
                LEFT JOIN jobs j ON j.name = r.name
                WHERE NOT r.illegal
                  AND j.name IS NOT NULL
                  AND COALESCE(j.manifest, '') != ?
                ORDER BY r.bpb LIMIT ?
            """, (IMPORTED_RESULT_MANIFEST, limit))

        if not rows:
            return {"varied": {}, "constant": {}, "runs": 0}

        configs = {}
        for row in rows:
            configs[row["name"]] = self._config_from_joined_row(dict(row))

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
        errors: list[dict[str, str]] = []
        for row in names:
            try:
                self.compute_and_store_forecast(row["name"])
                count += 1
            except Exception as exc:
                errors.append({"name": row["name"], "error": str(exc)})
        if errors:
            import sys
            print(f"chronohorn: rebuild_forecasts completed with {len(errors)} errors", file=sys.stderr)
            self.record_event("rebuild_forecasts_error", count=len(errors), sample=errors[:10])
        return count

    def rebuild_from_manifests(self, manifests_dir: str = "manifests") -> int:
        """Ingest all manifests from a directory."""
        count = 0
        errors: list[dict[str, str]] = []
        mdir = Path(manifests_dir)
        if mdir.exists():
            for p in sorted(mdir.glob("frontier_*.jsonl")):
                try:
                    count += self.ingest_manifest(str(p))
                except Exception as exc:
                    errors.append({"path": str(p), "error": str(exc)})
        if errors:
            import sys
            print(f"chronohorn: rebuild_from_manifests completed with {len(errors)} errors", file=sys.stderr)
            self.record_event("rebuild_manifests_error", count=len(errors), sample=errors[:10])
        return count

    def seed_groups(
        self,
        *,
        population: str = "controlled",
        legality: str = "legal",
    ) -> list[dict]:
        """Group results by config (ignoring seed) and compute statistics."""
        clauses: list[str] = []
        params: list[Any] = []
        population_clauses, population_params = self._population_clauses(population=population)
        legality_clauses, legality_params = self._legality_clauses(legality=legality)
        clauses.extend(population_clauses)
        clauses.extend(legality_clauses)
        params.extend(population_params)
        params.extend(legality_params)
        where_sql = " AND ".join(clauses) if clauses else "1=1"
        rows = self._read(
            f"""
            SELECT r.name, r.bpb, r.family,
                   r.steps AS result_steps,
                   r.seq_len AS result_seq_len,
                   c.json_blob,
                   j.steps AS job_steps,
                   j.seed AS job_seed,
                   j.lr AS job_lr,
                   j.batch_size AS job_batch_size,
                   j.manifest
            FROM results r
            LEFT JOIN configs c ON r.config_id = c.id
            LEFT JOIN jobs j ON j.name = r.name
            WHERE {where_sql}
        """,
            tuple(params),
        )

        groups: dict[str, list[dict]] = {}
        for row in rows:
            r = dict(row)
            try:
                cfg = self._config_from_joined_row(r)
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
                row = normalize_manifest_payload(json.loads(line))
                name = row.get("name", "")
                if not name:
                    continue
                # Don't overwrite completed jobs
                existing = self._read_one(
                    "SELECT state FROM jobs WHERE name = ?", (name,)
                )
                if existing and existing["state"] != "pending":
                    continue
                self.record_job(
                    name,
                    manifest=path.name,
                    family=row.get("family"),
                    config=row.get("config"),
                    steps=row.get("steps"),
                    seed=row.get("seed"),
                    lr=row.get("learning_rate"),
                    batch_size=row.get("batch_size"),
                    command=row.get("command", ""),
                    job_spec=row,
                )
                count += 1
            except (json.JSONDecodeError, KeyError) as exc:
                import sys
                print(f"chronohorn: manifest row skipped in {path}: {exc}", file=sys.stderr)
        self.record_event("ingest_manifest", path=str(path), count=count)
        return count
