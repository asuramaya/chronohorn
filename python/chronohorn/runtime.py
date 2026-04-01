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

from chronohorn.runtime_store import IncrementalStore


class RuntimeState:
    """Thread-safe shared state for all runtime services."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._fleet: dict[str, Any] = {}
        self._drain_status: dict[str, Any] = {}
        self._events: list[dict[str, Any]] = []
        self.store = IncrementalStore()
        self.manifests: list[str] = []
        self.result_dir: str = "out/results"
        self.poll_interval: int = 60
        self.auto_deepen: bool = False
        self.max_steps: int = 10000

    @property
    def fleet(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._fleet)

    def update_fleet(self, fleet: dict[str, Any]) -> None:
        with self._lock:
            self._fleet = fleet

    @property
    def drain_status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._drain_status)

    def update_drain(self, status: dict[str, Any]) -> None:
        with self._lock:
            self._drain_status = status

    def add_event(self, event: str, **kwargs: Any) -> None:
        with self._lock:
            entry = {"t": time.time(), "event": event, **kwargs}
            self._events.append(entry)
            if len(self._events) > 200:
                self._events = self._events[-200:]

    @property
    def events(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events[-30:])


def _fleet_probe_loop(state: RuntimeState) -> None:
    """Background thread: probe fleet via SSH every 30s."""
    from chronohorn.observe.serve import _probe_fleet

    while True:
        try:
            fleet = _probe_fleet()
            state.update_fleet(fleet)
        except Exception:
            pass
        time.sleep(30)


def _result_watcher_loop(state: RuntimeState) -> None:
    """Background thread: poll result dir every 10s for new results."""
    while True:
        try:
            new = state.store.refresh()
            if new:
                state.add_event("new_results", names=new, count=len(new))
        except Exception:
            pass
        time.sleep(10)


def _drain_loop(state: RuntimeState) -> None:
    """Background thread: drain manifests + auto-deepen."""
    from chronohorn.fleet.drain import drain_tick
    from chronohorn.fleet.auto_deepen import should_deepen, next_step_target
    from chronohorn.fleet.manifest_transform import load_and_transform

    while True:
        # Snapshot manifest list (can grow from auto-deepen)
        manifests = list(state.manifests)

        for manifest in manifests:
            try:
                tick = drain_tick(
                    manifest,
                    result_out_dir=Path(state.result_dir),
                )
                state.update_drain({
                    "manifest": Path(manifest).name,
                    "pending": tick.pending,
                    "running": tick.running,
                    "completed": tick.completed,
                    "blocked": tick.blocked,
                    "launched": tick.launched,
                    "pulled": tick.pulled,
                    "done": tick.is_done,
                })
                if tick.launched > 0:
                    state.add_event("launched", count=tick.launched, manifest=Path(manifest).name)
                if tick.pulled > 0:
                    state.add_event("pulled", count=tick.pulled)
            except Exception as exc:
                state.add_event("drain_error", error=str(exc)[:200])

        # Check new results for auto-deepen
        if state.auto_deepen:
            new = state.store.refresh()
            for name in new:
                result = state.store.get(name)
                if not result:
                    continue
                probes = result.get("training", {}).get("probes", [])
                cfg = result.get("config", {})
                train = cfg.get("train", {}) if isinstance(cfg.get("train"), dict) else cfg
                steps = train.get("steps", 0)
                if not steps or not probes:
                    continue
                if should_deepen(probes, current_steps=steps, max_steps=state.max_steps):
                    target = next_step_target(steps)
                    # Find which manifest this came from
                    source = None
                    base_name = name
                    # Strip step suffixes to find the base config
                    for pat in ("-1000s", "-5000s", "-10000s"):
                        base_name = base_name.replace(pat, "")
                    for m in manifests:
                        source = m
                        break  # use first manifest as source
                    if source:
                        out_path = Path(f"manifests/auto_{base_name}_s{target}.jsonl")
                        try:
                            rows = load_and_transform(
                                Path(source),
                                name_pattern=f"{base_name}*",
                                steps=target,
                                output_path=out_path,
                            )
                            if rows and str(out_path) not in state.manifests:
                                state.manifests.append(str(out_path))
                                state.add_event("auto_deepen",
                                    source=name, target_steps=target,
                                    manifest=out_path.name, rows=len(rows))
                        except Exception as exc:
                            state.add_event("auto_deepen_error", error=str(exc)[:200])

        time.sleep(state.poll_interval)


def _make_handler(state: RuntimeState, tool_server: Any):
    """Create HTTP handler with shared state and tool server."""
    from chronohorn.observe.serve import _build_api_data, _HTML, Handler

    class RuntimeHandler(Handler):
        def do_GET(self):
            if self.path == "/" or self.path == "/index.html":
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(_HTML.encode())
            elif self.path.startswith("/api/status"):
                data = _build_api_data(state.result_dir, skip_fleet_probe=True)
                data["fleet"] = state.fleet
                data["drain"] = state.drain_status
                data["events"] = state.events
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
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(state.events).encode())
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

    # Initialize shared state
    state = RuntimeState()
    state.manifests = list(args.manifest)
    state.result_dir = args.result_dir
    state.poll_interval = args.poll
    state.auto_deepen = args.auto_deepen
    state.max_steps = args.max_steps
    state.store = IncrementalStore(result_dir=args.result_dir)
    state.store.refresh()
    state.add_event("init", results=state.store.result_count)

    # Create shared MCP tool server
    tool_server = ToolServer()

    # Start background threads
    threading.Thread(target=_fleet_probe_loop, args=(state,), daemon=True).start()
    threading.Thread(target=_result_watcher_loop, args=(state,), daemon=True).start()
    state.add_event("started", component="fleet_probe")
    state.add_event("started", component="result_watcher")
    print(f"fleet probe: started", file=sys.stderr)
    print(f"result watcher: started ({state.store.result_count} initial results)", file=sys.stderr)

    if state.manifests:
        threading.Thread(target=_drain_loop, args=(state,), daemon=True).start()
        state.add_event("started", component="drain",
            manifests=[Path(m).name for m in state.manifests],
            auto_deepen=state.auto_deepen)
        mode = "auto-deepen" if state.auto_deepen else "manual"
        print(f"drain: started ({len(state.manifests)} manifests, {mode})", file=sys.stderr)

    # Start HTTP server with action endpoint
    handler = _make_handler(state, tool_server)
    handler.result_dir = args.result_dir
    server = HTTPServer(("127.0.0.1", args.port), handler)
    state.add_event("started", component="http", port=args.port)

    chrome_proc = None
    if not args.no_browser:
        chrome_proc = _launch_chrome_app(args.port, args.width, args.height)
        if chrome_proc:
            print(f"chrome: pid {chrome_proc.pid}", file=sys.stderr)

    best = state.store.best_bpb
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
