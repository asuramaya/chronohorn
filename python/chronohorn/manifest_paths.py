from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


def manifest_identity_values(paths: Iterable[str]) -> set[str]:
    identities: set[str] = set()
    for value in paths:
        raw = str(value or "").strip()
        if not raw:
            continue
        identities.add(raw)
        path = Path(raw).expanduser()
        identities.add(path.name)
        if path.is_absolute() or path.exists():
            try:
                identities.add(str(path.resolve()))
            except OSError:
                identities.add(str(path))
    return identities


def manifest_matches(candidate: str, manifest_filters: Iterable[str]) -> bool:
    candidate_raw = str(candidate or "").strip()
    if not candidate_raw:
        return False
    candidate_path = Path(candidate_raw).expanduser()
    candidate_name = candidate_path.name
    candidate_exact = {candidate_raw}
    if candidate_path.is_absolute() or candidate_path.exists():
        try:
            candidate_exact.add(str(candidate_path.resolve()))
        except OSError:
            candidate_exact.add(str(candidate_path))

    for value in manifest_filters:
        raw = str(value or "").strip()
        if not raw:
            continue
        requested_path = Path(raw).expanduser()
        is_path_filter = requested_path.is_absolute() or requested_path.name != raw
        if is_path_filter:
            requested_exact = {raw}
            if requested_path.is_absolute() or requested_path.exists():
                try:
                    requested_exact.add(str(requested_path.resolve()))
                except OSError:
                    requested_exact.add(str(requested_path))
            if candidate_exact & requested_exact:
                return True
            continue
        if candidate_name == requested_path.name:
            return True
    return False
