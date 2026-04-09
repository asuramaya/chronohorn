"""Chronohorn unified runtime daemon.

One process, three interfaces:
  1. HTTP visualization + control (/api/status, /api/action, /api/tools)
  2. Fleet drain loop (dispatch + result pull-back + auto-deepen)
  3. Fleet probe cache (async SSH, non-blocking)

Usage:
    chronohorn runtime --manifest <path> [--port 7070] [--poll 60]
    chronohorn runtime --manifest <path> --auto-deepen --max-steps 10000
    chronohorn runtime --port 7070  # viz only, no drain
"""
from __future__ import annotations

import json
import re
import shlex
import threading
import time
from collections.abc import Sequence
from http.server import HTTPServer
from pathlib import Path
from typing import Any

from chronohorn.db import ChronohornDB
from chronohorn.fleet.validation import validate_job_name
from chronohorn.service_log import configure_service_log, service_log


class RuntimeState:
    """Thread-safe shared state for all runtime services."""

    def __init__(self, db_path: str = "out/chronohorn.db") -> None:
        self._lock = threading.Lock()
        self.stop_event = threading.Event()
        self.db = ChronohornDB(db_path)
        self.manifests: list[str] = []
        self.result_dir: str = "out/results"
        self.poll_interval: int = 60
        self.auto_deepen: bool = False
        self.max_steps: int = 10000
        self.dispatch: bool = True
        self._component_health: dict[str, dict[str, Any]] = {}
        self._threads: dict[str, threading.Thread] = {}

    def register_thread(self, component: str, thread: threading.Thread) -> None:
        with self._lock:
            self._threads[component] = thread

    def _ensure_component(self, component: str) -> dict[str, Any]:
        entry = self._component_health.get(component)
        if entry is None:
            entry = {
                "status": "starting",
                "started_at": None,
                "last_ok_at": None,
                "last_error_at": None,
                "last_error": None,
                "ok_count": 0,
                "error_count": 0,
                "details": {},
            }
            self._component_health[component] = entry
        return entry

    def mark_component_started(self, component: str, **details: Any) -> None:
        now = time.time()
        with self._lock:
            entry = self._ensure_component(component)
            if entry["started_at"] is None:
                entry["started_at"] = now
            entry["status"] = "starting"
            if details:
                entry["details"] = dict(details)

    def mark_component_ok(self, component: str, **details: Any) -> None:
        now = time.time()
        with self._lock:
            entry = self._ensure_component(component)
            if entry["started_at"] is None:
                entry["started_at"] = now
            entry["status"] = "ok"
            entry["last_ok_at"] = now
            entry["ok_count"] = int(entry["ok_count"]) + 1
            if details:
                entry["details"] = dict(details)

    def mark_component_error(self, component: str, error: Any, **details: Any) -> None:
        now = time.time()
        with self._lock:
            entry = self._ensure_component(component)
            if entry["started_at"] is None:
                entry["started_at"] = now
            entry["status"] = "error"
            entry["last_error_at"] = now
            entry["last_error"] = str(error)
            entry["error_count"] = int(entry["error_count"]) + 1
            if details:
                entry["details"] = dict(details)

    def health_snapshot(self) -> dict[str, Any]:
        with self._lock:
            components = {}
            for name, entry in self._component_health.items():
                snapshot = dict(entry)
                snapshot["details"] = dict(entry.get("details") or {})
                thread = self._threads.get(name)
                snapshot["thread_alive"] = thread.is_alive() if thread is not None else None
                components[name] = snapshot
        return {
            "stop_requested": self.stop_event.is_set(),
            "components": components,
        }


def _resolved_manifest_paths(manifests: Sequence[str]) -> list[str]:
    resolved: list[str] = []
    for manifest in manifests:
        raw = str(manifest or "").strip()
        if not raw:
            continue
        resolved.append(str(Path(raw).expanduser().resolve()))
    return resolved


def _wait_or_stop(state: RuntimeState, timeout: float) -> bool:
    """Wait for timeout seconds or until shutdown is requested."""
    return state.stop_event.wait(timeout=max(timeout, 0.0))


def _runtime_health_payload(state: RuntimeState) -> dict[str, Any]:
    payload = state.health_snapshot()
    payload["manifests"] = _resolved_manifest_paths(state.manifests)
    payload["result_dir"] = str(Path(state.result_dir).expanduser())
    payload["poll_interval"] = state.poll_interval
    payload["dispatch"] = state.dispatch
    payload["auto_deepen"] = state.auto_deepen
    try:
        payload["drain"] = state.db.drain_status()
    except Exception as exc:
        payload["drain"] = {"error": str(exc)}
    return payload


def _fleet_probe_loop(state: RuntimeState) -> None:
    """Background thread: probe fleet via SSH every 30s."""
    from chronohorn.observe.serve import _probe_fleet

    while not state.stop_event.is_set():
        try:
            fleet = _probe_fleet()
            online_hosts = 0
            gpu_busy_hosts = 0
            for host, info in fleet.items():
                containers = [c["name"] for c in info.get("containers", [])]
                if info.get("online", False):
                    online_hosts += 1
                if containers:
                    gpu_busy_hosts += 1
                state.db.record_fleet(
                    host, online=info.get("online", False),
                    gpu_busy=len(containers) > 0,
                    containers=containers,
                )
            state.mark_component_ok(
                "fleet_probe",
                hosts=len(fleet),
                online_hosts=online_hosts,
                gpu_busy_hosts=gpu_busy_hosts,
            )
        except Exception as exc:
            state.mark_component_error("fleet_probe", exc)
            state.db.record_event("fleet_probe_error", error=str(exc)[:500])
            service_log("runtime.fleet_probe", "fleet probe failed", level="error", error=str(exc))
        if _wait_or_stop(state, 30):
            break


def _drain_loop(state: RuntimeState) -> None:
    """Background thread: drain DB-backed jobs + auto-deepen."""
    from chronohorn.fleet.auto_deepen import next_step_target
    from chronohorn.fleet.drain import drain_db_tick

    while not state.stop_event.is_set():
        manifest_paths = _resolved_manifest_paths(state.manifests)
        try:
            tick = drain_db_tick(
                db=state.db,
                manifests=manifest_paths,
                result_out_dir=Path(state.result_dir),
                dispatch=state.dispatch,
            )
            state.db.record_event(
                "drain_tick",
                manifests=manifest_paths,
                pending=tick.pending,
                running=tick.running,
                completed=tick.completed,
                launched=tick.launched,
                pulled=tick.pulled,
            )
            state.mark_component_ok(
                "drain",
                manifests=manifest_paths,
                pending=tick.pending,
                running=tick.running,
                completed=tick.completed,
                blocked=tick.blocked,
                launched=tick.launched,
                pulled=tick.pulled,
            )
        except Exception as exc:
            state.mark_component_error("drain", exc, manifests=manifest_paths)
            state.db.record_event("drain_error", error=str(exc)[:500])
            service_log("runtime.drain", "drain tick failed", level="error", error=str(exc), manifests=manifest_paths)

        # --- Auto-deepen: write new jobs directly into the DB ---
        if state.auto_deepen:
            candidates = state.db.query("""
                SELECT r.name, r.slope, j.steps, j.command, j.config_id FROM results r
                JOIN jobs j ON r.name = j.name
                WHERE r.slope > 0.05 AND j.steps < ? AND NOT r.illegal
                AND r.name NOT IN (SELECT parent FROM jobs WHERE parent IS NOT NULL AND parent != '')
            """, (state.max_steps,))

            for row in candidates:
                target = next_step_target(row["steps"])
                if target <= row["steps"]:
                    continue
                base_name = row["name"]
                raw_child_name = f"{base_name}-s{target}"
                try:
                    child_name = validate_job_name(raw_child_name)
                except ValueError as exc:
                    state.db.record_event(
                        "auto_deepen_error",
                        source=base_name,
                        target=raw_child_name,
                        error=str(exc)[:200],
                    )
                    continue

                # Skip if this child job already exists in the DB
                existing = state.db.query("SELECT name FROM jobs WHERE name = ?", (child_name,))
                if existing:
                    continue

                # Build the updated command: replace --steps N and --json path
                parent_cmd = row.get("command") or ""
                new_cmd = re.sub(r"(?<!\w)--steps\s+\d+", f"--steps {target}", parent_cmd)
                new_cmd = re.sub(
                    r"--json\s+(?:\"[^\"]+\"|\S+)",
                    f"--json {shlex.quote(f'out/results/{child_name}.json')}",
                    new_cmd,
                )

                # Fetch parent config from DB for reuse
                parent_cfg_rows = state.db.query(
                    "SELECT json_blob FROM configs WHERE id = ?", (row.get("config_id"),)
                ) if row.get("config_id") else []
                try:
                    parent_config = (
                        json.loads(parent_cfg_rows[0]["json_blob"])
                        if parent_cfg_rows and parent_cfg_rows[0].get("json_blob")
                        else {}
                    )
                except (json.JSONDecodeError, TypeError):
                    parent_config = {}
                parent_job = state.db.job_spec(base_name) or {}
                child_job = dict(parent_job)
                child_job["name"] = child_name
                child_job["parent"] = base_name
                child_job["command"] = new_cmd
                child_job["steps"] = target
                child_job["state"] = "pending"
                child_job["run_id"] = f"{child_job.get('manifest_path', '')}::{child_name}" if child_job.get("manifest_path") else child_name
                child_job["generated_by"] = "auto_deepen"

                try:
                    state.db.record_job(
                        child_name,
                        manifest=str(parent_job.get("manifest") or ""),
                        parent=base_name,
                        config=parent_config,
                        steps=target,
                        command=new_cmd,
                        job_spec=child_job,
                    )
                    state.db.record_event("auto_deepen", source=base_name,
                                          target=child_name, target_steps=target)
                except Exception as exc:
                    state.db.record_event(
                        "auto_deepen_error",
                        source=base_name,
                        target=child_name,
                        error=str(exc)[:200],
                    )

        if _wait_or_stop(state, state.poll_interval):
            break


def _make_handler(state: RuntimeState, tool_server: Any):
    """Create HTTP handler with shared state and tool server."""
    from chronohorn.observe.serve import _HTML, Handler, _build_api_data

    class RuntimeHandler(Handler):
        def do_GET(self):
            if self.path == "/" or self.path == "/index.html":
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(_HTML.encode())
            elif self.path.startswith("/api/status"):
                try:
                    data = _build_api_data(state.db)
                    data["health"] = _runtime_health_payload(state)
                except Exception as exc:
                    data = {
                        "error": str(exc),
                        "n": 0,
                        "curves": {},
                        "board": [],
                        "eff": [],
                        "fleet": {},
                        "best": None,
                        "manifests": [],
                        "configs": [],
                        "drain": {},
                        "events": [],
                        "health": _runtime_health_payload(state),
                    }
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())
            elif self.path == "/api/health":
                payload = _runtime_health_payload(state)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(json.dumps(payload).encode())
            elif self.path == "/api/tools":
                tools = tool_server.list_tools() if tool_server else []
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"tools": [t["name"] for t in tools]}).encode())
            elif self.path == "/api/events":
                events = state.db.events_recent(30)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(events).encode())
            elif self.path.startswith("/api/query"):
                from urllib.parse import parse_qs, urlparse
                params = parse_qs(urlparse(self.path).query)
                sql = params.get("sql", [""])[0]
                if not sql or not sql.strip().upper().startswith("SELECT"):
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Only SELECT queries allowed"}).encode())
                    return
                try:
                    rows = state.db.query(sql)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"rows": rows, "count": len(rows)}).encode())
                except Exception as exc:
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(exc)}).encode())
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path == "/api/action":
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                try:
                    req = json.loads(body)
                    tool_name = req.get("tool", "")
                    tool_args = req.get("args", {})
                    result = tool_server.call_tool(tool_name, tool_args) if tool_server else {"error": "no tool server"}
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps(result).encode())
                except Exception as exc:
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(exc)}).encode())
            else:
                self.send_error(404)

        def do_OPTIONS(self):
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

    RuntimeHandler.db = state.db
    RuntimeHandler.tool_server = tool_server
    return RuntimeHandler


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    from chronohorn.mcp import ToolServer
    from chronohorn.observe.serve import _launch_chrome_app

    parser = argparse.ArgumentParser(
        prog="chronohorn runtime",
        description="Unified runtime: drain + fleet probe + visualization + MCP tools.",
    )
    parser.add_argument("--manifest", action="append", default=[], help="Manifest to drain (repeatable)")
    parser.add_argument("--port", type=int, default=7070, help="HTTP port")
    parser.add_argument("--poll", type=int, default=60, help="Drain poll interval seconds")
    parser.add_argument("--result-dir", default="out/results", help="Local result directory")
    parser.add_argument("--auto-deepen", action="store_true", help="Auto-generate deepening runs when slope is alive")
    parser.add_argument("--max-steps", type=int, default=10000, help="Maximum steps for auto-deepen")
    parser.add_argument("--no-dispatch", action="store_true", help="Monitor and pull only — don't launch new jobs")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--width", type=int, default=420)
    parser.add_argument("--height", type=int, default=520)
    args = parser.parse_args(argv)

    # Initialize DB
    result_path = Path(args.result_dir)
    db_path = str(result_path.parent / "chronohorn.db")
    configure_service_log(result_path.parent / "chronohorn.service.jsonl")
    state = RuntimeState(db_path=db_path)
    state.manifests = list(args.manifest)
    state.result_dir = args.result_dir
    state.poll_interval = args.poll
    state.auto_deepen = args.auto_deepen
    state.max_steps = args.max_steps
    state.dispatch = not args.no_dispatch
    state.mark_component_started("runtime", manifests=_resolved_manifest_paths(args.manifest))

    # One-time rebuild from existing JSON archive
    count = state.db.rebuild_from_archive(args.result_dir)
    state.db.record_event("init", results=count)

    # Ingest manifests into jobs table
    for manifest in args.manifest:
        state.db.ingest_manifest(manifest)

    # Create shared MCP tool server
    tool_server = ToolServer(db=state.db)

    # Start background threads
    fleet_thread = threading.Thread(
        target=_fleet_probe_loop,
        args=(state,),
        name="ChronohornRuntimeFleetProbe",
    )
    state.register_thread("fleet_probe", fleet_thread)
    state.mark_component_started("fleet_probe", interval_sec=30)
    fleet_thread.start()
    state.db.record_event("started", component="fleet_probe")
    service_log("runtime.fleet_probe", "started", interval_sec=30)
    service_log("runtime", "initial results loaded", results=count, db_path=db_path)

    # Drain runs for all DB-tracked jobs, not just manifested ones
    drain_thread = threading.Thread(
        target=_drain_loop,
        args=(state,),
        name="ChronohornRuntimeDrain",
    )
    state.register_thread("drain", drain_thread)
    state.mark_component_started(
        "drain",
        manifests=_resolved_manifest_paths(state.manifests),
        auto_deepen=state.auto_deepen,
        dispatch=state.dispatch,
    )
    drain_thread.start()
    state.db.record_event("started", component="drain",
        manifests=_resolved_manifest_paths(state.manifests),
        auto_deepen=state.auto_deepen)
    manifest_desc = f"{len(state.manifests)} manifests" if state.manifests else "all DB jobs"
    mode = "auto-deepen" if state.auto_deepen else "manual"
    service_log(
        "runtime.drain",
        "started",
        manifests=_resolved_manifest_paths(state.manifests),
        manifest_scope=manifest_desc,
        mode=mode,
        dispatch=state.dispatch,
    )

    # Start HTTP server with action endpoint
    handler = _make_handler(state, tool_server)
    handler.result_dir = args.result_dir
    server = HTTPServer(("127.0.0.1", args.port), handler)
    state.mark_component_ok("http", port=args.port)
    state.db.record_event("started", component="http", port=args.port)

    chrome_proc = None
    if not args.no_browser:
        chrome_proc = _launch_chrome_app(args.port, args.width, args.height)
        if chrome_proc:
            service_log("runtime.http", "chrome app opened", pid=chrome_proc.pid, port=args.port)

    best = state.db.best_bpb()
    service_log("runtime.http", "server ready", url=f"http://127.0.0.1:{args.port}", best_bpb=best)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        service_log("runtime", "shutdown requested", level="warning")
    finally:
        state.stop_event.set()
        server.server_close()
        for thread in (fleet_thread, drain_thread):
            thread.join(timeout=max(state.poll_interval, 30) + 5)
            if thread.is_alive():
                service_log("runtime", "thread did not stop cleanly", level="warning", thread=thread.name)
        try:
            state.db.record_event("shutdown", component="runtime")
        except Exception as exc:
            service_log("runtime", "shutdown event write failed", level="error", error=str(exc))
        try:
            state.db.close()
        except Exception as exc:
            service_log("runtime", "db close failed", level="error", error=str(exc))
        if chrome_proc:
            chrome_proc.terminate()
    return 0
