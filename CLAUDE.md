# Chronohorn — Developer Guide

Family-agnostic experiment tracker and architecture-search runtime for predictive descendants. Tracks results from any model family, provides saturation/forecast analysis, manages fleet dispatch, and exposes everything via 64 MCP tools for AI-assisted research.

**Not a training framework.** Model code lives in [decepticons](https://github.com/asuramaya/decepticons). Chronohorn runs experiments and measures results.

## Ecosystem

```
decepticons  → kernel (model primitives, substrates, readouts)
chronohorn   → runtime (training, tracking, fleet, MCP)
heinrich     → forensics (model geometry, activation traces)
```

Dependency direction: `chronohorn → decepticons`. Never the reverse.

## Key Abstractions

**`ChronohornDB`** (`python/chronohorn/db.py`) — SQLite single source of truth. Single-writer discipline: all mutations serialize through a dedicated writer thread via a queue. Reads use a separate connection in autocommit mode with WAL journaling.

**`FamilyTrainingAdapter`** protocol (`python/chronohorn/families/adapter.py`) — how model families plug into the runtime. Defines architecture aliases, illegal detection, config summaries, `infer_from_config()` for family detection, and training entrypoints.

**`ToolServer`** (`python/chronohorn/mcp.py`) — 64 MCP tools, all DB-backed. Covers manifests, results, forecasting, fleet control, learning curves, saturation analysis, frontier tracking, job registration, and fleet sync.

**`serve.py`** (`python/chronohorn/observe/serve.py`) — HTTP visualization dashboard. Always reads from the DB.

**`runtime.py`** (`python/chronohorn/runtime.py`) — unified daemon combining drain + fleet probe + viz in a single long-running process. Use `--no-dispatch` for monitor-only mode.

## How to Add a New Model Family

1. Create `python/chronohorn/families/<name>/` as a Python package
2. In `__init__.py`, export a `<UPPER_NAME>_TRAINING_ADAPTER` singleton
3. Implement `adapter.py` following the `FamilyTrainingAdapter` protocol
4. Implement `infer_from_config()` to detect the family from config fields

The registry (`python/chronohorn/families/registry.py`) auto-discovers families via `pkgutil.iter_modules` — no manual registration needed.

**Key rule:** Family-specific code lives only in `families/<name>/`. Core infra (db, mcp, serve, runtime) must never import family modules directly — always use the registry.

## Measurement Methodology

**bpb (bits per byte):** `bpt × tokens_per_byte` where `tokens_per_byte` comes from sentencepiece on the actual test data. NOT shard file bytes — text bytes. For sp1024: `tokens_per_byte ≈ 0.411`, `text_bytes_per_token ≈ 2.436`.

**Probes vs Finals:**
- Probes use 2-8 eval batches (65K-262K tokens). For monitoring trends only.
- Final eval uses 200 batches (6.5M tokens). For claims and comparisons.
- Probes carry `eval_batches` field to prevent confusion.
- Never compare probe numbers across experiments. Only compare 200-batch finals.

**Eval stream reset:** The test stream resets to position 0 before every eval, ensuring every probe and final measures the same data.

**Causality:** Verified by `decepticons/tests/test_causality.py`. Feed identical sequences up to position t, different after t. If logits at t differ, causality is violated.

## How to Run

```bash
# Install (Python >=3.11)
pip install -e .

# Dashboard
chronohorn observe serve --result-dir out/results

# Runtime daemon (monitor-only, no auto-dispatch)
chronohorn runtime --port 7071 --no-dispatch

# Runtime daemon (with dispatch)
chronohorn runtime --manifest manifests/my_scan.jsonl --port 7070

# MCP stdio transport
chronohorn mcp
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

## Fleet Operations

Requires: Docker, NVIDIA drivers, Linux, SSH on each GPU host.

```bash
# Register a manually-launched run (via MCP)
chronohorn_register_run name=my-run host=slop-01 steps=50000 seed=42

# Sync fleet: pull results, probe hosts, show frontier
chronohorn_fleet_sync hosts=["slop-01","slop-02"]

# Clean poisoned DB (purge unregistered results)
chronohorn_reset
```

Always pass `-e PYTHONUNBUFFERED=1` and `--export-dir` when launching Docker training containers.

## Data Provisioning

Nodes self-provision training data from HuggingFace. K8s jobs include an init container that downloads shards automatically.

```bash
# Provision sp1024 data on a node
chronohorn data provision

# Provision sp4096 variant
chronohorn data provision --variant sp4096

# Verify without downloading
chronohorn data provision --verify
```

## Substrate Transforms (Session 8)

Six primitives between the substrate and readout, controlled by config flags. All off by default.

**Write-side** (before/during EMA):
- `--substrate-bank-router`: Route tokens to mode banks by type (~2k params)
- `--overwrite-gate`: Per-mode gate that erases stale state (~130k params)

**Read-side** (after EMA, before readout):
- `--magnitude-normalize`: Kill position counter via L2 norm (1 param)
- `--mode-selector`: Per-token soft attention over modes (~130k params)
- `--temporal-attention`: Cross-attention over substrate snapshots (~50k params)

**Readout**:
- `--linear-readout-kind tied_embed_readout`: Experts project to embed space, logits via shared embed.T. 73% param savings at sp8192.

## Substrate Rotations (Session 9)

Noncommutative rotation primitives in the gated delta scan. Controlled by config flags. All off by default.

**The lasso** (`--lasso-rotation`): 2×2 matrix transition per mode pair. Noncommutative — "A B" ≠ "B A". -0.109 bpb at 316k tok/s. 16k params. **Best substrate primitive.** Uses parallel Hillis-Steele prefix scan with `torch.roll`+`torch.where` (dynamo-compatible).

**Complex rotation** (`--complex-rotation`): SO(2) input-dependent phase rotation. Commutative — can't encode order. -0.008 bpb. Historical only.

**Quaternion** (`--quaternion-rotation`): SO(3) Hamilton product. Matches lasso bpb but 7% slower. Norm constraint doesn't help.

**SO(5)** (`--so5-rotation`): Lie algebra → `matrix_exp` → guaranteed SO(5) rotation. Non-solvable compact group. Stable (can't diverge). Requires `--state-dim 40 --num-heads 8` (head_dim=5). ~2.7× slower than lasso due to matrix_exp.

**Quintic** (`--quintic-rotation`): GL(5) unconstrained 5×5 matrix. **UNSTABLE (NaN)**. Use `--so5-rotation` instead.

**Position signal** (`--position-signal`): Feeds `log(1+t)` into gate/lasso projections. Breaks shift invariance. Available but bpb-neutral at 10k steps.

**Key finding:** The wall is content capacity, not order. Position R²=0.001 across all models. The lasso doubled content R² (0.101→0.209). All bpb improvement comes from richer content coupling.

**Speed defaults (auto):** `batch_size` auto-scales to fill 60% VRAM. `torch.compile` auto-enables for ≥5000 steps. `--linear-modes` is PRE-scale (multiplied by `--scale` in `apply_variant`).

## Pre-launch Validation

`fleet/preflight.py` runs before every k8s job submit:
- Command safety gates (catches `tied_recursive` without override flag)
- Remote data path verification via SSH

## Code Conventions

- All DB reads go through `_read()` / `_read_one()` (thread-safe via `_read_lock`)
- All DB writes go through `_write()` / `_write_many()` (queued to writer thread)
- Family-specific code lives only in `families/<name>/`
- Core infra must never import family modules directly — use `registry.py`
- No hardcoded family names in core infra
- The DB is the live truth; JSON files are archives
- `safe_float()` lives in `engine/results.py` — one copy, not seven
- `bits_per_token_from_loss()` lives in `engine/performance.py` — one copy
- Silent `except Exception: pass` is forbidden — log to stderr

## Current Families

- **`causal-bank`** — Decepticons kernel models. Frozen linear substrate, local conv, MLP/expert readout. Model code in `decepticons.models`.
- **`polyhash`** — Hash-embedding models (O(1) lookup tables + gated scan + PKM). Independent of decepticons.
- **`transformer`** — Adapter only. External training.

## Result JSON Format

```json
{
  "model": {
    "test_bpb": 1.75,
    "architecture": "my_model",
    "params": 10000000
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
    "final_eval_batches": 200,
    "performance": {
      "tokens_per_second": 350000,
      "elapsed_sec": 900,
      "steps_completed": 10000
    },
    "probes": [
      {"step": 100, "bpb": 2.5, "eval_batches": 2},
      {"step": 1000, "bpb": 2.0, "eval_batches": 8},
      {"step": 10000, "bpb": 1.75, "eval_batches": 16}
    ]
  },
  "dataset": {
    "test_tokens_per_byte": 0.4105,
    "test_bytes_per_token": 2.436
  }
}
```

`test_bpb` is required. `architecture` is recommended for family detection. `probes` enable learning curve and saturation analysis. `eval_batches` on probes distinguishes monitoring from claims.
