from __future__ import annotations

import json
import threading

from chronohorn.fleet.drain import DrainState


class _FakeRuntimeDB:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def record_event(self, event: str, **data: object) -> None:
        self.events.append((event, data))

    def drain_status(self) -> dict[str, object]:
        return {"pending": 1, "running": 2, "completed": 3}


def test_runtime_drain_loop_uses_resolved_manifest_paths(tmp_path, monkeypatch):
    from chronohorn.runtime import RuntimeState, _drain_loop

    manifest_path = tmp_path / "nested" / "scan.jsonl"
    manifest_path.parent.mkdir()
    manifest_path.write_text("", encoding="utf-8")
    manifest_rel = manifest_path.relative_to(tmp_path)
    manifest_abs = str(manifest_path.resolve())

    seen: dict[str, object] = {}

    def _fake_drain_db_tick(*, db, manifests, result_out_dir, dispatch):
        seen["manifests"] = list(manifests)
        seen["dispatch"] = dispatch
        return DrainState(
            manifest_path=",".join(manifests),
            pending=0,
            running=0,
            completed=0,
            blocked=0,
            launched=0,
            pulled=0,
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("chronohorn.fleet.drain.drain_db_tick", _fake_drain_db_tick)
    monkeypatch.setattr("chronohorn.runtime._wait_or_stop", lambda state, timeout: state.stop_event.set() or True)

    state = RuntimeState(db_path=tmp_path / "test.db")
    state.db.close()
    state.db = _FakeRuntimeDB()
    state.manifests = [str(manifest_rel)]
    state.dispatch = False

    _drain_loop(state)

    assert seen["manifests"] == [manifest_abs]
    assert state.db.events == [
        (
            "drain_tick",
            {
                "manifests": [manifest_abs],
                "pending": 0,
                "running": 0,
                "completed": 0,
                "launched": 0,
                "pulled": 0,
            },
        )
    ]


def test_runtime_health_payload_reports_component_state(tmp_path):
    from chronohorn.runtime import RuntimeState, _runtime_health_payload

    stop_event = threading.Event()

    def _worker():
        stop_event.wait()

    state = RuntimeState(db_path=tmp_path / "test.db")
    state.db.close()
    state.db = _FakeRuntimeDB()
    thread = threading.Thread(target=_worker, name="probe")
    state.register_thread("fleet_probe", thread)
    state.mark_component_started("fleet_probe", interval_sec=30)
    thread.start()
    try:
        state.mark_component_ok("fleet_probe", hosts=2)
        payload = _runtime_health_payload(state)
    finally:
        stop_event.set()
        thread.join(timeout=1)

    component = payload["components"]["fleet_probe"]
    assert payload["drain"] == {"pending": 1, "running": 2, "completed": 3}
    assert component["status"] == "ok"
    assert component["thread_alive"] is True
    assert component["details"] == {"hosts": 2}


def test_runtime_status_payload_includes_health(tmp_path):
    from chronohorn.runtime import RuntimeState, _make_handler

    class _FakeHandler:
        def __init__(self) -> None:
            self.path = "/api/status"
            self.headers = {}
            self.rfile = None
            self.status_code = None
            self.body = b""

        def send_response(self, code: int) -> None:
            self.status_code = code

        def send_header(self, key: str, value: str) -> None:
            return None

        def end_headers(self) -> None:
            return None

        @property
        def wfile(self):
            class _Writer:
                def __init__(self, outer):
                    self._outer = outer

                def write(self, data: bytes) -> None:
                    self._outer.body += data

            return _Writer(self)

        def send_error(self, code: int) -> None:
            raise AssertionError(f"unexpected send_error({code})")

    state = RuntimeState(db_path=tmp_path / "test.db")
    state.mark_component_ok("http", port=7070)
    handler_cls = _make_handler(state, tool_server=None)
    handler = _FakeHandler()
    try:
        handler_cls.do_GET(handler)
    finally:
        state.db.close()

    payload = json.loads(handler.body)
    assert handler.status_code == 200
    assert "health" in payload
    assert payload["health"]["components"]["http"]["details"] == {"port": 7070}


def test_runtime_drain_loop_marks_component_error(tmp_path, monkeypatch):
    from chronohorn.runtime import RuntimeState, _drain_loop

    monkeypatch.setattr("chronohorn.fleet.drain.drain_db_tick", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr("chronohorn.runtime._wait_or_stop", lambda state, timeout: state.stop_event.set() or True)

    state = RuntimeState(db_path=tmp_path / "test.db")
    state.db.close()
    state.db = _FakeRuntimeDB()

    _drain_loop(state)

    drain = state.health_snapshot()["components"]["drain"]
    assert drain["status"] == "error"
    assert "boom" in str(drain["last_error"])
    assert ("drain_error", {"error": "boom"}) in state.db.events
