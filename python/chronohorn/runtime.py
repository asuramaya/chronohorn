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
import sys
import threading
import time
from collections.abc import Sequence
from http.server import HTTPServer
from pathlib import Path
from typing import Any

from chronohorn.db import ChronohornDB


class RuntimeState:
    """Thread-safe shared state for all runtime services."""

    def __init__(self, db_path: str = "out/chronohorn.db") -> None:
        self._lock = threading.Lock()
        self.db = ChronohornDB(db_path)
        self.manifests: list[str] = []
        self.result_dir: str = "out/results"
        self.poll_interval: int = 60
        self.auto_deepen: bool = False
        self.max_steps: int = 10000
        self.dispatch: bool = True


def _fleet_probe_loop(state: RuntimeState) -> None:
    """Background thread: probe fleet via SSH every 30s."""
    from chronohorn.observe.serve import _probe_fleet

    while True:
        try:
            fleet = _probe_fleet()
            for host, info in fleet.items():
                containers = [c["name"] for c in info.get("containers", [])]
                state.db.record_fleet(
                    host, online=info.get("online", False),
                    gpu_busy=len(containers) > 0,
                    containers=containers,
                )
        except Exception as exc:
            import sys
            print(f"chronohorn: fleet probe failed: {exc}", file=sys.stderr)
        time.sleep(30)


def _drain_loop(state: RuntimeState) -> None:
    """Background thread: drain DB-backed jobs + auto-deepen."""
    from chronohorn.fleet.auto_deepen import next_step_target
    from chronohorn.fleet.drain import drain_db_tick

    while True:
        try:
            tick = drain_db_tick(
                db=state.db,
                manifests=[Path(manifest).name for manifest in state.manifests],
                result_out_dir=Path(state.result_dir),
                dispatch=state.dispatch,
            )
            state.db.record_event(
                "drain_tick",
                manifests=[Path(manifest).name for manifest in state.manifests],
                pending=tick.pending,
                running=tick.running,
                completed=tick.completed,
                launched=tick.launched,
                pulled=tick.pulled,
            )
        except Exception as exc:
            import sys
            print(f"chronohorn runtime: drain tick failed: {exc}", file=sys.stderr)
            state.db.record_event("drain_error", error=str(exc)[:500])

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
                child_name = f"{base_name}-s{target}"

                # Skip if this child job already exists in the DB
                existing = state.db.query("SELECT name FROM jobs WHERE name = ?", (child_name,))
                if existing:
                    continue

                # Build the updated command: replace --steps N and --json path
                parent_cmd = row.get("command") or ""
                new_cmd = re.sub(r"(?<!\w)--steps\s+\d+", f"--steps {target}", parent_cmd)
                new_cmd = re.sub(
                    r"--json\s+(?:\"[^\"]+\"|\S+)",
                    f"--json out/results/{child_name}.json",
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

        time.sleep(state.poll_interval)


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
                except Exception as exc:
                    data = {"error": str(exc), "n": 0, "curves": {}, "board": [], "eff": [], "fleet": {}, "best": None, "manifests": [], "configs": [], "drain": {}, "events": []}
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())
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
                    if tool_server:
                        result = tool_server.call_tool(tool_name, tool_args)
                    else:
                        result = {"error": "no tool server"}
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
    state = RuntimeState(db_path=db_path)
    state.manifests = list(args.manifest)
    state.result_dir = args.result_dir
    state.poll_interval = args.poll
    state.auto_deepen = args.auto_deepen
    state.max_steps = args.max_steps
    state.dispatch = not args.no_dispatch

    # One-time rebuild from existing JSON archive
    count = state.db.rebuild_from_archive(args.result_dir)
    state.db.record_event("init", results=count)

    # Ingest manifests into jobs table
    for manifest in args.manifest:
        state.db.ingest_manifest(manifest)

    # Create shared MCP tool server
    tool_server = ToolServer(db=state.db)

    # Start background threads
    threading.Thread(target=_fleet_probe_loop, args=(state,), daemon=True).start()
    state.db.record_event("started", component="fleet_probe")
    print("fleet probe: started", file=sys.stderr)
    print(f"runtime: {count} initial results in DB", file=sys.stderr)

    # Drain runs for all DB-tracked jobs, not just manifested ones
    threading.Thread(target=_drain_loop, args=(state,), daemon=True).start()
    state.db.record_event("started", component="drain",
        manifests=[Path(m).name for m in state.manifests],
        auto_deepen=state.auto_deepen)
    manifest_desc = f"{len(state.manifests)} manifests" if state.manifests else "all DB jobs"
    mode = "auto-deepen" if state.auto_deepen else "manual"
    print(f"drain: started ({manifest_desc}, {mode})", file=sys.stderr)

    # Start HTTP server with action endpoint
    handler = _make_handler(state, tool_server)
    handler.result_dir = args.result_dir
    server = HTTPServer(("127.0.0.1", args.port), handler)
    state.db.record_event("started", component="http", port=args.port)

    chrome_proc = None
    if not args.no_browser:
        chrome_proc = _launch_chrome_app(args.port, args.width, args.height)
        if chrome_proc:
            print(f"chrome: pid {chrome_proc.pid}", file=sys.stderr)

    best = state.db.best_bpb()
    best_str = f" | best: {best:.4f}" if best else ""
    print(f"chronohorn runtime: http://127.0.0.1:{args.port}{best_str}", file=sys.stderr)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutdown", file=sys.stderr)
    finally:
        if chrome_proc:
            chrome_proc.terminate()
    return 0
