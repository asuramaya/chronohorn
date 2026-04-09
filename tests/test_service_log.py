from __future__ import annotations

import json
from io import StringIO

from chronohorn.service_log import configure_service_log, format_service_log_line, service_log


def test_service_log_formats_and_persists(tmp_path, monkeypatch):
    log_path = tmp_path / "chronohorn.service.jsonl"
    configure_service_log(log_path)
    stream = StringIO()
    try:
        payload = service_log(
            "runtime.drain",
            "tick",
            level="info",
            stream=stream,
            pending=1,
            running=2,
        )
    finally:
        configure_service_log(None)

    assert payload["component"] == "runtime.drain"
    assert payload["fields"] == {"pending": 1, "running": 2}
    assert "[runtime.drain] INFO tick" in stream.getvalue()

    rows = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 1
    decoded = json.loads(rows[0])
    assert decoded["message"] == "tick"
    assert decoded["fields"] == {"pending": 1, "running": 2}


def test_format_service_log_line_sorts_fields():
    line = format_service_log_line(
        {
            "component": "observe.serve",
            "level": "warning",
            "message": "server ready",
            "fields": {"url": "http://127.0.0.1:7070", "port": 7070},
        }
    )

    assert line.startswith("[observe.serve] WARNING server ready")
    assert 'port=7070' in line
    assert 'url="http://127.0.0.1:7070"' in line
