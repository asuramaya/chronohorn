"""Chronohorn unified runtime daemon.

Runs all three services in one process:
  1. Fleet drain loop (dispatch + result pull-back)
  2. Fleet probe cache (async SSH, non-blocking)
  3. HTTP visualization server

Usage:
    chronohorn runtime --manifest <path> [--port 7070] [--poll 60]
    chronohorn runtime --port 7070  # viz only, no drain

This replaces running drain + serve separately.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from http.server import HTTPServer
from pathlib import Path
from typing import Any, Sequence


class RuntimeState:
    """Shared state between drain thread, probe thread, and HTTP server."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._fleet: dict[str, Any] = {}
        self._fleet_ts: float = 0
        self._drain_status: dict[str, Any] = {}
        self._last_result_count: int = 0
        self._events: list[dict[str, Any]] = []
        self.manifests: list[str] = []
        self.result_dir: str = "out/results"
        self.poll_interval: int = 60

    @property
    def fleet(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._fleet)

    def update_fleet(self, fleet: dict[str, Any]) -> None:
        with self._lock:
            self._fleet = fleet
            self._fleet_ts = time.time()

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
            if len(self._events) > 100:
                self._events = self._events[-100:]

    @property
    def events(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events[-20:])


# Global state shared between threads
_state = RuntimeState()


def _fleet_probe_loop(state: RuntimeState) -> None:
    """Background thread: probe fleet via SSH every 30s, update shared state."""
    from chronohorn.observe.serve import _probe_fleet

    while True:
        try:
            fleet = _probe_fleet()
            state.update_fleet(fleet)
            running = sum(len(h.get("containers", [])) for h in fleet.values())
            if running:
                state.add_event("fleet_probe", gpus=running)
        except Exception as exc:
            state.add_event("fleet_probe_error", error=str(exc))
        time.sleep(30)


def _drain_loop(state: RuntimeState) -> None:
    """Background thread: drain manifests, pull results."""
    from chronohorn.fleet.drain import drain_tick

    while True:
        for manifest in state.manifests:
            try:
                tick_state = drain_tick(
                    manifest,
                    result_out_dir=Path(state.result_dir),
                )
                status = {
                    "manifest": manifest,
                    "pending": tick_state.pending,
                    "running": tick_state.running,
                    "completed": tick_state.completed,
                    "blocked": tick_state.blocked,
                    "launched": tick_state.launched,
                    "pulled": tick_state.pulled,
                    "done": tick_state.is_done,
                }
                state.update_drain(status)

                if tick_state.launched > 0:
                    state.add_event("launched", count=tick_state.launched, manifest=Path(manifest).name)
                if tick_state.pulled > 0:
                    state.add_event("pulled", count=tick_state.pulled)
                if tick_state.is_done:
                    state.add_event("drain_done", manifest=Path(manifest).name)

            except Exception as exc:
                state.add_event("drain_error", error=str(exc)[:200], manifest=Path(manifest).name)

        time.sleep(state.poll_interval)


def _make_handler(state: RuntimeState):
    """Create HTTP handler that uses shared state instead of per-request SSH."""
    from chronohorn.observe.serve import _build_api_data, _HTML, Handler

    class RuntimeHandler(Handler):
        def do_GET(self):
            if self.path == "/" or self.path == "/index.html":
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(_HTML.encode())
            elif self.path.startswith("/api/status"):
                # Use cached fleet instead of per-request SSH
                data = _build_api_data(state.result_dir, skip_fleet_probe=True)
                data["fleet"] = state.fleet  # override with cached probe
                data["drain"] = state.drain_status
                data["events"] = state.events
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())
            elif self.path.startswith("/api/events"):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(state.events).encode())
            else:
                self.send_error(404)

    return RuntimeHandler


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    from chronohorn.observe.serve import _find_chrome, _launch_chrome_app

    parser = argparse.ArgumentParser(
        prog="chronohorn runtime",
        description="Unified runtime: drain + fleet probe + visualization.",
    )
    parser.add_argument("--manifest", action="append", default=[], help="Manifest to drain (repeatable)")
    parser.add_argument("--port", type=int, default=7070, help="HTTP port (default 7070)")
    parser.add_argument("--poll", type=int, default=60, help="Drain poll interval seconds (default 60)")
    parser.add_argument("--result-dir", default="out/results", help="Local result directory")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open Chrome")
    parser.add_argument("--width", type=int, default=420)
    parser.add_argument("--height", type=int, default=520)
    args = parser.parse_args(argv)

    state = _state
    state.manifests = args.manifest
    state.result_dir = args.result_dir
    state.poll_interval = args.poll

    # Start fleet probe thread (always)
    probe_thread = threading.Thread(target=_fleet_probe_loop, args=(state,), daemon=True)
    probe_thread.start()
    state.add_event("started", component="fleet_probe")
    print("fleet probe: started", file=sys.stderr)

    # Start drain thread (if manifests provided)
    if state.manifests:
        drain_thread = threading.Thread(target=_drain_loop, args=(state,), daemon=True)
        drain_thread.start()
        state.add_event("started", component="drain", manifests=[Path(m).name for m in state.manifests])
        print(f"drain: started ({len(state.manifests)} manifests)", file=sys.stderr)
    else:
        print("drain: skipped (no --manifest)", file=sys.stderr)

    # Start HTTP server
    handler_class = _make_handler(state)
    handler_class.result_dir = args.result_dir
    server = HTTPServer(("127.0.0.1", args.port), handler_class)
    state.add_event("started", component="http", port=args.port)

    chrome_proc = None
    if not args.no_browser:
        chrome_proc = _launch_chrome_app(args.port, args.width, args.height)
        if chrome_proc:
            print(f"chrome: app window opened (pid {chrome_proc.pid})", file=sys.stderr)

    print(f"chronohorn runtime: http://127.0.0.1:{args.port}", file=sys.stderr)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down...", file=sys.stderr)
    finally:
        if chrome_proc:
            chrome_proc.terminate()
    return 0
