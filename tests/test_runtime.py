from __future__ import annotations

from chronohorn.fleet.drain import DrainState


class _FakeRuntimeDB:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def record_event(self, event: str, **data: object) -> None:
        self.events.append((event, data))


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
