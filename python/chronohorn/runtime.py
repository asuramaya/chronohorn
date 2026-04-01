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
import sys
import threading
import time
from http.server import HTTPServer
from pathlib import Path
from typing import Any, Sequence

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
        except Exception:
            pass
        time.sleep(30)



def _drain_loop(state: RuntimeState) -> None:
    """Background thread: drain manifests + auto-deepen."""
    from chronohorn.fleet.drain import drain_tick
    from chronohorn.fleet.auto_deepen import should_deepen, next_step_target
    from chronohorn.fleet.manifest_transform import load_and_transform

    while True:
        for manifest in list(state.manifests):
            try:
                tick = drain_tick(
                    manifest,
                    result_out_dir=Path(state.result_dir),
                    db=state.db,
                )
                state.db.record_event("drain_tick", manifest=Path(manifest).name,
                    pending=tick.pending, running=tick.running, completed=tick.completed,
                    launched=tick.launched, pulled=tick.pulled)
            except Exception as exc:
                state.db.record_event("drain_error", error=str(exc)[:200])

        # Auto-deepen from DB
        if state.auto_deepen:
            candidates = state.db.query("""
                SELECT r.name, r.slope, j.steps FROM results r
                JOIN jobs j ON r.name = j.name
                WHERE r.slope > 0.05 AND j.steps < ? AND NOT r.illegal
                AND r.name NOT IN (SELECT parent FROM jobs WHERE parent IS NOT NULL AND parent != '')
            """, (state.max_steps,))

            for row in candidates:
                target = next_step_target(row["steps"])
                if target <= row["steps"]:
                    continue
                base_name = row["name"]
                for m in state.manifests:
                    out_path = Path(f"manifests/auto_{base_name}_s{target}.jsonl")
                    if out_path.exists():
                        break
                    try:
                        rows = load_and_transform(
                            Path(m), name_pattern=f"{base_name.rsplit('-', 1)[0]}*",
                            steps=target, output_path=out_path,
                        )
                        if rows and str(out_path) not in state.manifests:
                            state.manifests.append(str(out_path))
                            state.db.ingest_manifest(str(out_path))
                            state.db.record_event("auto_deepen", source=base_name, target_steps=target)
                    except Exception:
                        pass
                    break

        time.sleep(state.poll_interval)


def _make_handler(state: RuntimeState, tool_server: Any):
    """Create HTTP handler with shared state and tool server."""
    from chronohorn.observe.serve import _build_api_data_from_db, _HTML, Handler

    class RuntimeHandler(Handler):
        def do_GET(self):
            if self.path == "/" or self.path == "/index.html":
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(_HTML.encode())
            elif self.path.startswith("/api/status"):
                data = _build_api_data_from_db(state.db)
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
    return RuntimeHandler


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    from chronohorn.observe.serve import _find_chrome, _launch_chrome_app
    from chronohorn.mcp import ToolServer

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
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--width", type=int, default=420)
    parser.add_argument("--height", type=int, default=520)
    args = parser.parse_args(argv)

    # Initialize DB
    db_path = args.result_dir.replace("out/results", "out/chronohorn.db")
    if db_path == args.result_dir:
        # result_dir didn't contain "out/results", fall back to default
        db_path = "out/chronohorn.db"
    state = RuntimeState(db_path=db_path)
    state.manifests = list(args.manifest)
    state.result_dir = args.result_dir
    state.poll_interval = args.poll
    state.auto_deepen = args.auto_deepen
    state.max_steps = args.max_steps

    # One-time rebuild from existing JSON archive
    count = state.db.rebuild_from_archive(args.result_dir)
    state.db.record_event("init", results=count)

    # Ingest manifests into jobs table
    for manifest in args.manifest:
        state.db.ingest_manifest(manifest)

    # Create shared MCP tool server with runtime DB connection
    tool_server = ToolServer(db=state.db)

    # Start background threads
    threading.Thread(target=_fleet_probe_loop, args=(state,), daemon=True).start()
    threading.Thread(target=_result_watcher_loop, args=(state,), daemon=True).start()
    state.db.record_event("started", component="fleet_probe")
    state.db.record_event("started", component="result_watcher")
    print(f"fleet probe: started", file=sys.stderr)
    print(f"result watcher: started ({count} initial results)", file=sys.stderr)

    if state.manifests:
        threading.Thread(target=_drain_loop, args=(state,), daemon=True).start()
        state.db.record_event("started", component="drain",
            manifests=[Path(m).name for m in state.manifests],
            auto_deepen=state.auto_deepen)
        mode = "auto-deepen" if state.auto_deepen else "manual"
        print(f"drain: started ({len(state.manifests)} manifests, {mode})", file=sys.stderr)

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
