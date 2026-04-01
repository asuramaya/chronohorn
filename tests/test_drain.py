from __future__ import annotations

from chronohorn.fleet.drain import DrainState


def test_drain_state_done_when_no_pending_no_running():
    state = DrainState(
        manifest_path="/fake/manifest.jsonl",
        pending=0, running=0, completed=5, blocked=0, launched=0, pulled=0,
    )
    assert state.is_done is True


def test_drain_state_not_done_when_pending():
    state = DrainState(
        manifest_path="/fake/manifest.jsonl",
        pending=3, running=2, completed=5, blocked=0, launched=0, pulled=0,
    )
    assert state.is_done is False


def test_drain_state_not_done_when_running():
    state = DrainState(
        manifest_path="/fake/manifest.jsonl",
        pending=0, running=2, completed=5, blocked=0, launched=0, pulled=0,
    )
    assert state.is_done is False


def test_drain_state_stalled_detection():
    state = DrainState(
        manifest_path="/fake/manifest.jsonl",
        pending=0, running=0, completed=2, blocked=3, launched=0, pulled=0,
    )
    assert state.is_done is False  # blocked jobs remain
