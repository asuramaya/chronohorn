# Chronohorn — Developer Guide

Architecture search runtime and experiment tracker. **Not a training framework.** Chronohorn is family-agnostic: it tracks results from any model family, provides saturation/forecast analysis, manages fleet dispatch, and exposes everything via MCP tools for AI-assisted research.

## Key Abstractions

**`ChronohornDB`** (`python/chronohorn/db.py`) — SQLite single source of truth. Schema v3. Single-writer discipline: all mutations serialize through a dedicated writer thread via a queue. Reads use a separate connection in autocommit mode with WAL journaling. Thread-safe reads via `_read_lock`.

**`FamilyTrainingAdapter`** protocol (`python/chronohorn/families/adapter.py`) — how model families plug into the runtime. Defines architecture aliases, illegal detection, config summaries, training entrypoints, and more.

**`ToolServer`** (`python/chronohorn/mcp.py`) — 27 MCP tools, all DB-backed. No RunStore dependency. Tools cover manifests, results, forecasting, fleet control, learning curves, saturation analysis, and frontier tracking.

**`serve.py`** (`python/chronohorn/observe/serve.py`) — HTTP visualization dashboard with 6 tabs: curves, frontier, fleet, bpb/tf, config, manifests. Always reads from the DB.

**`runtime.py`** (`python/chronohorn/runtime.py`) — unified daemon combining drain + fleet probe + viz + MCP in a single long-running process.

## How to Add a New Model Family

1. Create `python/chronohorn/families/<name>/` as a Python package
2. In `__init__.py`, export a `<UPPER_NAME>_TRAINING_ADAPTER` singleton (e.g. `POLYHASH_TRAINING_ADAPTER`)
3. Implement `adapter.py` following the `FamilyTrainingAdapter` protocol

The registry (`python/chronohorn/families/registry.py`) auto-discovers families via `pkgutil.iter_modules` — no manual registration needed.

**Required protocol methods:**
- `architecture_aliases()` — strings that route to this family (e.g. `["polyhash", "polyhash_v6"]`)
- `detect_illegal(payload)` — flag future-leakage or invalid results
- `config_summary(result_json)` — extract key config fields for display
- `training_entrypoints()` — CLI entrypoints for `chronohorn train`

**Optional:** Export a `<UPPER_NAME>_FRONTIER_EMITTER` implementing `FamilyFrontierEmitter` for automated scan generation.

**Key rule:** Family-specific code lives only in `families/<name>/`. Core infra (db, mcp, serve, runtime) must never import family modules directly — always use the registry.

## Result JSON Format

What chronohorn expects when ingesting experiment results:

```json
{
  "model": {
    "test_bpb": 1.75,
    "architecture": "my_model",
    "params": 10000000,
    "test_bits_per_token": 4.27,
    "train_bpb": 1.80
  },
  "config": {
    "train": {
      "steps": 10000,
      "seq_len": 512,
      "batch_size": 64,
      "learning_rate": 0.005
    }
  },
  "training": {
    "performance": {
      "tokens_per_second": 350000,
      "elapsed_sec": 900,
      "steps_completed": 10000,
      "estimated_sustained_tflops": 1.2
    },
    "probes": [
      {"step": 100, "bpb": 2.5},
      {"step": 1000, "bpb": 2.0},
      {"step": 10000, "bpb": 1.75}
    ]
  }
}
```

`test_bpb` is required. `architecture` is recommended for family detection. `probes` enable learning curve and saturation analysis.

## How to Run

```bash
# Install (editable)
pip install -e .

# Dashboard
chronohorn observe serve --result-dir out/results

# Full daemon (drain + fleet probe + viz + MCP)
chronohorn runtime --manifest manifests/my_scan.jsonl --port 7070

# MCP stdio transport
chronohorn mcp

# From source without install
PYTHONPATH=python python -m chronohorn --help
```

MCP configuration for Claude Code — add to `.mcp.json`:
```json
{
  "mcpServers": {
    "chronohorn": {
      "command": "python",
      "args": ["-m", "chronohorn.mcp_transport"],
      "env": { "PYTHONPATH": "python" }
    }
  }
}
```

## Code Conventions

- All DB reads go through `_read()` / `_read_one()` (thread-safe via `_read_lock`)
- All DB writes go through `_write()` / `_write_many()` (queued to writer thread)
- Family-specific code lives only in `families/<name>/`
- Core infra must never import family modules directly — use `registry.py`
- No hardcoded family names in core infra
- The DB is the live truth; JSON files are archives

## Current Families

- **`causal-bank`** — OPC kernel models (substrate modes, selective scan, patch encoding, memory attachment)
- **`polyhash`** — Hash-embedding models (polysemy-inspired learned lookup tables)

Both are examples — the runtime does not depend on either.
