"""Incremental result store with filesystem watching.

Maintains a hot cache of all result JSONs. On refresh(), only reads
new files since the last scan. All queries read from the cache.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

class IncrementalStore:
    def __init__(self, result_dir: Path | str = "out/results") -> None:
        self._dir = Path(result_dir)
        self._results: dict[str, dict[str, Any]] = {}
        self._seen: set[str] = set()

    @property
    def result_count(self) -> int:
        return len(self._results)

    @property
    def best_bpb(self) -> float | None:
        bpbs = [r.get("model", {}).get("test_bpb") for r in self._results.values()]
        valid = [b for b in bpbs if b is not None]
        return min(valid) if valid else None

    def refresh(self) -> list[str]:
        if not self._dir.is_dir():
            return []
        current = {p.stem: p for p in self._dir.glob("*.json")}
        new_names = []
        for name, path in current.items():
            if name in self._seen:
                continue
            try:
                data = json.loads(path.read_text())
                if isinstance(data, dict):
                    self._results[name] = data
                    self._seen.add(name)
                    new_names.append(name)
            except (json.JSONDecodeError, OSError):
                pass
        return new_names

    def get(self, name: str) -> dict[str, Any] | None:
        return self._results.get(name)

    def all_results(self) -> dict[str, dict[str, Any]]:
        return dict(self._results)

    def leaderboard(self, top_k: int = 20) -> list[dict[str, Any]]:
        rows = []
        for name, r in self._results.items():
            bpb = r.get("model", {}).get("test_bpb")
            if bpb is not None:
                rows.append({"name": name, "bpb": round(bpb, 4)})
        rows.sort(key=lambda x: x["bpb"])
        return rows[:top_k]

    def names(self) -> set[str]:
        return set(self._results.keys())
