from __future__ import annotations

import re
from pathlib import PurePosixPath

_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_env_key(key: str) -> str:
    if not _ENV_KEY_RE.match(key):
        raise ValueError(
            f"Invalid environment variable key: {key!r}. "
            f"Must match [A-Za-z_][A-Za-z0-9_]*"
        )
    return key


def validate_relative_posix_subpath(path: str, *, field_name: str = "path") -> str:
    raw = str(path).strip()
    if raw in {"", ".", "./"}:
        return "."
    pure = PurePosixPath(raw)
    if pure.is_absolute():
        raise ValueError(f"{field_name} must be a relative subpath, got absolute path: {raw!r}")
    cleaned_parts: list[str] = []
    for part in pure.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            raise ValueError(f"{field_name} must not contain parent traversal: {raw!r}")
        cleaned_parts.append(part)
    return PurePosixPath(*cleaned_parts).as_posix() if cleaned_parts else "."


def validate_posix_path_within_root(
    path: str,
    *,
    root: str,
    field_name: str = "path",
) -> str:
    root_path = PurePosixPath(root)
    raw = str(path).strip()
    if not raw:
        raise ValueError(f"{field_name} must not be empty")
    pure = PurePosixPath(raw)
    if pure.is_absolute():
        if pure.parts[: len(root_path.parts)] != root_path.parts:
            raise ValueError(f"{field_name} must stay within {root_path.as_posix()}: {raw!r}")
        tail_parts = pure.parts[len(root_path.parts) :]
    else:
        tail_parts = pure.parts
    cleaned_parts: list[str] = []
    for part in tail_parts:
        if part in {"", "."}:
            continue
        if part == "..":
            raise ValueError(f"{field_name} must not escape {root_path.as_posix()}: {raw!r}")
        cleaned_parts.append(part)
    return root_path.joinpath(*cleaned_parts).as_posix()


def validate_job_name(name: str) -> str:
    text = str(name)
    if not text.strip():
        raise ValueError("job name must be a non-empty name")
    if "/" in text or "\\" in text or "\x00" in text:
        raise ValueError(f"job name must not contain path separators or NUL bytes: {text!r}")
    return text
