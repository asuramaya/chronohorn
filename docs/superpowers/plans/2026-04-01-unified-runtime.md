# Chronohorn Unified Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge the MCP server, fleet drain, fleet probe, visualization, and auto-deepen into a single runtime process with incremental state and interactive control.

**Architecture:** One process, three interfaces (MCP stdio, HTTP viz, drain loop), one shared RuntimeState backed by an incremental RunStore. The viz becomes a control surface, not just a display. The drain auto-deepens based on forecaster signals.

**Tech Stack:** Python 3.9+, threading, http.server, watchdog (optional, filesystem polling fallback)

---

## File Structure

| Action | Path | Responsibility |
|--------|------|---------------|
| Rewrite | `python/chronohorn/runtime.py` | Unified daemon: MCP + HTTP + drain + probe in one process |
| Create | `python/chronohorn/runtime_store.py` | Incremental RunStore wrapper with filesystem watching |
| Modify | `python/chronohorn/mcp.py` | ToolServer uses shared RuntimeState instead of per-call state |
| Modify | `python/chronohorn/observe/serve.py` | HTTP handler uses shared state, add /api/action endpoint |
| Create | `python/chronohorn/fleet/auto_deepen.py` | Auto-deepen logic: forecaster → manifest → dispatch |
| Modify | `python/chronohorn/cli.py` | Wire runtime command |
| Create | `tests/test_runtime_store.py` | Incremental store tests |
| Create | `tests/test_auto_deepen.py` | Auto-deepen logic tests |

---

### Task 1: Incremental RunStore

The core data layer. Wraps RunStore with filesystem-watching and incremental updates.

**Files:**
- Create: `python/chronohorn/runtime_store.py`
- Create: `tests/test_runtime_store.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_runtime_store.py
from __future__ import annotations

import json
from pathlib import Path

from chronohorn.runtime_store import IncrementalStore


def test_initial_scan(tmp_path: Path):
    rd = tmp_path / "results"
    rd.mkdir()
    (rd / "job1.json").write_text(json.dumps({"model": {"test_bpb": 2.05}}))
    (rd / "job2.json").write_text(json.dumps({"model": {"test_bpb": 1.95}}))

    store = IncrementalStore(result_dir=rd)
    store.refresh()
    assert store.result_count == 2
    assert store.best_bpb == 1.95


def test_incremental_detects_new_file(tmp_path: Path):
    rd = tmp_path / "results"
    rd.mkdir()
    (rd / "job1.json").write_text(json.dumps({"model": {"test_bpb": 2.05}}))

    store = IncrementalStore(result_dir=rd)
    store.refresh()
    assert store.result_count == 1

    (rd / "job2.json").write_text(json.dumps({"model": {"test_bpb": 1.85}}))
    new = store.refresh()
    assert store.result_count == 2
    assert len(new) == 1
    assert new[0] == "job2"
    assert store.best_bpb == 1.85


def test_no_change_returns_empty(tmp_path: Path):
    rd = tmp_path / "results"
    rd.mkdir()
    (rd / "job1.json").write_text(json.dumps({"model": {"test_bpb": 2.05}}))

    store = IncrementalStore(result_dir=rd)
    store.refresh()
    new = store.refresh()
    assert len(new) == 0


def test_results_sorted_by_bpb(tmp_path: Path):
    rd = tmp_path / "results"
    rd.mkdir()
    for i, bpb in enumerate([2.1, 1.9, 2.0]):
        (rd / f"job{i}.json").write_text(json.dumps({"model": {"test_bpb": bpb}}))

    store = IncrementalStore(result_dir=rd)
    store.refresh()
    board = store.leaderboard(3)
    assert board[0]["bpb"] == 1.9
    assert board[2]["bpb"] == 2.1
```

- [ ] **Step 2: Implement IncrementalStore**

```python
# python/chronohorn/runtime_store.py
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
        self._results: dict[str, dict[str, Any]] = {}  # name -> parsed JSON
        self._seen: set[str] = set()  # filenames already loaded

    @property
    def result_count(self) -> int:
        return len(self._results)

    @property
    def best_bpb(self) -> float | None:
        bpbs = [r.get("model", {}).get("test_bpb") for r in self._results.values()]
        valid = [b for b in bpbs if b is not None]
        return min(valid) if valid else None

    def refresh(self) -> list[str]:
        """Scan result dir, load new files. Returns list of new result names."""
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

    def all_results(self) -> list[tuple[str, dict[str, Any]]]:
        return list(self._results.items())

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
```

- [ ] **Step 3: Run tests**
- [ ] **Step 4: Commit**

---

### Task 2: Auto-Deepen Logic

When a pilot completes and the learning curve slope is alive, automatically generate a deepening run.

**Files:**
- Create: `python/chronohorn/fleet/auto_deepen.py`
- Create: `tests/test_auto_deepen.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_auto_deepen.py
from __future__ import annotations

from chronohorn.fleet.auto_deepen import should_deepen, deepen_config


def test_should_deepen_alive_slope():
    probes = [
        {"step": 400, "bpb": 2.2},
        {"step": 800, "bpb": 2.1},
        {"step": 1000, "bpb": 2.05},
    ]
    assert should_deepen(probes, current_steps=1000, max_steps=10000) is True


def test_should_not_deepen_flat_slope():
    probes = [
        {"step": 400, "bpb": 2.05},
        {"step": 800, "bpb": 2.05},
        {"step": 1000, "bpb": 2.05},
    ]
    assert should_deepen(probes, current_steps=1000, max_steps=10000) is False


def test_should_not_deepen_at_max():
    probes = [
        {"step": 8000, "bpb": 1.90},
        {"step": 10000, "bpb": 1.85},
    ]
    assert should_deepen(probes, current_steps=10000, max_steps=10000) is False


def test_deepen_config_step_progression():
    cfg = deepen_config(current_steps=1000)
    assert cfg["steps"] == 5000

    cfg = deepen_config(current_steps=5000)
    assert cfg["steps"] == 10000
```

- [ ] **Step 2: Implement**

```python
# python/chronohorn/fleet/auto_deepen.py
"""Auto-deepen: when a pilot's slope is alive, generate the next horizon."""
from __future__ import annotations

from typing import Any

# Step progression: 1k -> 5k -> 10k
_STEP_LADDER = [1000, 5000, 10000]


def should_deepen(
    probes: list[dict[str, Any]],
    *,
    current_steps: int,
    max_steps: int = 10000,
    min_improvement: float = 0.005,
) -> bool:
    """Check if the learning curve slope is alive and worth deepening."""
    if current_steps >= max_steps:
        return False
    if len(probes) < 2:
        return False
    # Check last two probes
    p1 = probes[-2]
    p2 = probes[-1]
    b1 = p1.get("bpb") or p1.get("test_bpb") or 0
    b2 = p2.get("bpb") or p2.get("test_bpb") or 0
    return (b1 - b2) > min_improvement


def deepen_config(current_steps: int) -> dict[str, Any]:
    """Return the next step count in the progression."""
    for target in _STEP_LADDER:
        if target > current_steps:
            return {"steps": target}
    return {"steps": current_steps}  # already at max
```

- [ ] **Step 3: Run tests**
- [ ] **Step 4: Commit**

---

### Task 3: HTTP Action Endpoint

Add `/api/action` to the viz server so the dashboard can trigger MCP tool calls.

**Files:**
- Modify: `python/chronohorn/observe/serve.py`

- [ ] **Step 1: Add action endpoint to Handler**

In the HTTP handler's `do_GET`/`do_POST`, add:

```python
    def do_POST(self):
        if self.path == "/api/action":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                req = json.loads(body)
                tool_name = req.get("tool", "")
                tool_args = req.get("args", {})
                # Call through the shared ToolServer
                if self._tool_server:
                    result = self._tool_server.call_tool(tool_name, tool_args)
                else:
                    result = {"error": "no tool server attached"}
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
```

Add `_tool_server = None` as a class attribute on Handler, set it from the runtime.

- [ ] **Step 2: Commit**

---

### Task 4: Unified Runtime with All Services

Rewrite runtime.py to integrate: MCP ToolServer, IncrementalStore, auto-deepen, HTTP with actions.

**Files:**
- Rewrite: `python/chronohorn/runtime.py`

- [ ] **Step 1: Rewrite runtime.py**

The runtime should:

1. Create a shared `IncrementalStore`
2. Create a shared `ToolServer` (from mcp.py)
3. Start fleet probe thread (SSH every 30s, updates cached state)
4. Start result watcher thread (polls result_dir every 10s, calls store.refresh())
5. Start drain thread (if manifests provided, polls every N seconds)
6. In the drain thread, after each tick:
   - Call store.refresh() to pick up new results
   - For each new result, check should_deepen()
   - If yes, generate deepening manifest via load_and_transform, add to drain list
7. Start HTTP server with action endpoint connected to ToolServer
8. The ToolServer reads from the same IncrementalStore

Key changes from current runtime.py:
- IncrementalStore replaces per-request file scanning
- ToolServer is shared (not created per MCP connection)
- Auto-deepen is integrated into drain
- HTTP handler has action endpoint

- [ ] **Step 2: Wire auto-deepen into drain thread**

After each drain tick, check completed results:

```python
for name in new_results:
    result = store.get(name)
    if not result:
        continue
    probes = result.get("training", {}).get("probes", [])
    steps = result.get("config", {}).get("train", {}).get("steps", 0)
    if should_deepen(probes, current_steps=steps, max_steps=max_steps):
        cfg = deepen_config(steps)
        # Generate deepening manifest
        deepened = load_and_transform(
            Path(manifest), name_pattern=name.rsplit("-", 1)[0] + "*",
            steps=cfg["steps"],
            output_path=Path(f"manifests/auto_deepen_{name}_s{cfg['steps']}.jsonl"),
        )
        if deepened:
            state.manifests.append(str(output_path))
            state.add_event("auto_deepen", name=name, target_steps=cfg["steps"])
```

- [ ] **Step 3: Test the full runtime**

```bash
PYTHONPATH=python:../decepticons/src python3 -m chronohorn runtime \
  --manifest manifests/frontier_learned_recurrence.jsonl \
  --port 7070 --poll 90 --auto-deepen --max-steps 10000
```

- [ ] **Step 4: Commit**

---

### Task 5: MCP ToolServer Shared State

Make the MCP ToolServer use the shared IncrementalStore and RuntimeState.

**Files:**
- Modify: `python/chronohorn/mcp.py`

- [ ] **Step 1: Add shared store support to ToolServer**

Add an optional `store` parameter to ToolServer.__init__:

```python
class ToolServer:
    def __init__(self, *, shared_store: IncrementalStore | None = None) -> None:
        self._store = RunStore()
        self._shared_store = shared_store
        self._stages_run: list[str] = []
```

Tools like `chronohorn_learning_curve`, `chronohorn_marginal_rank`, `chronohorn_compare` should read from `_shared_store` when available, falling back to disk reads.

- [ ] **Step 2: Commit**

---

## Execution Order

```
Task 1 (IncrementalStore) — independent, foundation
Task 2 (auto-deepen) — independent, foundation
Task 3 (action endpoint) — independent
Task 4 (unified runtime) — depends on 1, 2, 3
Task 5 (MCP shared state) — depends on 1
```

Parallelizable: Tasks 1, 2, 3
Sequential: Task 4 (after 1-3), Task 5 (after 1)
