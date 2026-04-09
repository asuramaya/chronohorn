from __future__ import annotations

import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

_ENV_KEY = "CHRONOHORN_SERVICE_LOG"


def configure_service_log(path: str | Path | None) -> str | None:
    if path is None:
        os.environ.pop(_ENV_KEY, None)
        return None
    resolved = str(Path(path).expanduser().resolve())
    Path(resolved).parent.mkdir(parents=True, exist_ok=True)
    os.environ[_ENV_KEY] = resolved
    return resolved


def _resolve_service_log_path(path: str | Path | None = None) -> Path | None:
    raw = str(path) if path is not None else os.environ.get(_ENV_KEY, "")
    raw = raw.strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def _coerce_field(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_coerce_field(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _coerce_field(v) for k, v in value.items()}
    return str(value)


def _service_log_payload(component: str, level: str, message: str, **fields: Any) -> dict[str, Any]:
    return {
        "ts": time.time(),
        "ts_iso": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "component": component,
        "level": level,
        "message": message,
        "fields": {str(k): _coerce_field(v) for k, v in fields.items()},
    }


def format_service_log_line(payload: dict[str, Any]) -> str:
    component = str(payload.get("component") or "chronohorn")
    level = str(payload.get("level") or "info").upper()
    message = str(payload.get("message") or "")
    fields = payload.get("fields") or {}
    pieces = [f"[{component}]", level, message]
    if isinstance(fields, dict):
        field_text = " ".join(f"{key}={json.dumps(value, sort_keys=True)}" for key, value in sorted(fields.items()))
        if field_text:
            pieces.append(field_text)
    return " ".join(piece for piece in pieces if piece)


def service_log(
    component: str,
    message: str,
    *,
    level: str = "info",
    stream: TextIO | None = None,
    log_path: str | Path | None = None,
    **fields: Any,
) -> dict[str, Any]:
    payload = _service_log_payload(component, level, message, **fields)
    target = stream if stream is not None else sys.stderr
    print(format_service_log_line(payload), file=target, flush=True)

    resolved_log_path = _resolve_service_log_path(log_path)
    if resolved_log_path is not None:
        try:
            with resolved_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
        except OSError:
            pass
    return payload
