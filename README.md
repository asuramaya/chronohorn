# Chronohorn

<p align="center">
  <img src="docs/chronohorn.jpg" alt="Chronohorn" width="520">
</p>

> Mascot: Chronohorn is the final boss of Blinx: The Time Sweeper, a massive Time Monster that Blinx must defeat using every Time Control ability.

Runtime engine for `open-predictive-coder` descendants. Built for
non-transformer frontier work, agent-driven experiment control, and the real
hardware shape of one laptop plus a couple of slop boxes.

## What It Does

`Chronohorn` turns `opc` model families into runnable systems:

- backend-specific training on MLX/Metal and Torch/CUDA
- multi-host placement across Apple, CPU, and NVIDIA lanes
- runtime observation through a normalized run store
- closed-loop frontier control and action planning
- budget forecasting against competition constraints
- export bundle emission
- Rust replay, scoring, and packed-artifact evaluation

Built for agents:

- terminal observer surface: `chronohorn observe ...`
- MCP server: `chronohorn mcp`
- console-script alias: `chronohorn-mcp`
- runtime state is normalized instead of scattered across manifests, launch
  records, result JSONs, forecast output, and tracked frontier side-state

## Install

Python:

```bash
python3 -m pip install -e .[train]
```

Optional extras:

- `.[torch]` for Torch/CUDA
- `.[metal]` for MLX/Metal

For sibling-repo work:

```bash
python3 -m pip install -e ../open-predictive-coder -e .[train]
```

Rust:

```bash
cargo build
```

From source:

```bash
PYTHONPATH=python python3 -m chronohorn --help
```

Validated package smoke:

```bash
uv venv .venv
uv pip install --python .venv/bin/python --no-deps -e .
.venv/bin/chronohorn --help
.venv/bin/chronohorn fleet --help
.venv/bin/chronohorn control --help
.venv/bin/chronohorn observe --help
.venv/bin/chronohorn train --help
.venv/bin/chronohorn export --help
.venv/bin/chronohorn mcp --help
.venv/bin/chronohorn-mcp --help
```

## CLI

```bash
chronohorn train train-causal-bank-mlx
chronohorn train train-causal-bank-torch
chronohorn train measure-backend-parity

chronohorn fleet dispatch --manifest manifests/frontier_long_slop_matrix.jsonl
chronohorn fleet drain --manifest manifests/frontier_long_slop_matrix.jsonl
chronohorn fleet queue --manifest manifests/frontier_long_slop_matrix.jsonl
chronohorn fleet transform --manifest manifests/frontier_long_slop_matrix.jsonl --filter 'ex-j-*' --steps 5200 --output out.jsonl
chronohorn fleet forecast-results --path <result.json>
chronohorn fleet emit-causal-bank-matrix --regime exotic-16mb
chronohorn control recommend --manifest manifests/frontier_long_slop_matrix.jsonl
chronohorn control act --manifest manifests/frontier_long_slop_matrix.jsonl

chronohorn observe pipeline --manifest manifests/frontier_long_slop_matrix.jsonl
chronohorn observe status --manifest manifests/frontier_long_slop_matrix.jsonl --probe-runtime
chronohorn observe frontier --manifest manifests/frontier_long_slop_matrix.jsonl --probe-runtime
chronohorn observe query-records --kind runtime_state

chronohorn runtime  # unified daemon: drain + fleet probe + viz + auto-deepen

chronohorn export --help
chronohorn mcp
chronohorn-mcp

cargo run -p chronohorn -- --help
cargo run -p chronohorn-cli -- --help
```

## Unified Runtime Daemon

`chronohorn runtime` combines drain, fleet probing, visualization, and
auto-deepen in a single long-running process:

```bash
chronohorn runtime --manifest manifests/frontier_exotic_deepen.jsonl
```

It runs a poll loop that:

1. probes fleet state and pulls completed result JSONs
2. re-dispatches eligible pending jobs
3. checks learning-curve slopes and auto-generates next-horizon deepening rows
4. serves an HTTP visualization UI on `http://localhost:7878`

The visualization server has 6 tabs: curves, frontier, fleet, bpb/tf, config, manifests.
It auto-launches in Chrome app mode on start.

## MCP Integration

Add to your client settings:

```json
{
  "mcpServers": {
    "chronohorn": {
      "command": "chronohorn-mcp",
      "args": []
    }
  }
}
```

Canonical CLI entrypoint: `chronohorn mcp`

Console-script alias for MCP clients that want a single executable:
`chronohorn-mcp`

21 tools are exposed:

- `chronohorn_manifests`
- `chronohorn_runtime_status`
- `chronohorn_launches`
- `chronohorn_results`
- `chronohorn_forecast`
- `chronohorn_records`
- `chronohorn_status`
- `chronohorn_frontier`
- `chronohorn_pipeline`
- `chronohorn_control_recommend`
- `chronohorn_control_act`
- `chronohorn_reset`
- `chronohorn_fleet_dispatch`
- `chronohorn_fleet_drain_tick`
- `chronohorn_fleet_status`
- `chronohorn_learning_curves`
- `chronohorn_compare`
- `chronohorn_marginal_rank`
- `chronohorn_auto_deepen`
- `chronohorn_artifact_check`
- `chronohorn_subscribe`

The `/api/action` endpoint on the visualization server accepts dashboard control
commands (stop, promote, deepen) from agents or the UI.

## Architecture

`Chronohorn` sits between the shared kernel and the externalized audit system:

```text
open-predictive-coder -> chronohorn -> heinrich
kernel                 runtime       evidence / audit
```

Ownership is simple:

- `open-predictive-coder`
  - model-family semantics
  - reusable substrates, routing, and readouts
  - backend-neutral export meaning
- `chronohorn`
  - training execution
  - fleet orchestration
  - runtime observation, forecasting, and control
  - replay, scoring, and artifact economics
- `heinrich`
  - external evidence packaging
  - validation bundles
  - public-facing audit compression

## Runtime Record Pipeline

The observer layer follows the same small-stage style as Heinrich, but for
runtime state instead of evidence:

```text
tracked_state -> manifest -> runtime_state -> live_log -> launch -> result -> forecast
```

Every stage writes normalized `RunRecord` rows into a shared `RunStore`. The
observer then merges those into run-level snapshots for agents and CLIs. That
includes the tracked `state/frontier_status.json` references, feasible small
artifact baselines, and frontier notes, not just live slop runs.

## Modules

Python:

- [`python/chronohorn/engine`](./python/chronohorn/engine)
  - budgets, performance accounting, result summaries, forecasting, probes
- [`python/chronohorn/families`](./python/chronohorn/families)
  - family adapters and scan policy
- [`python/chronohorn/observe`](./python/chronohorn/observe)
  - observer CLI over the run store
  - `serve.py` — HTTP visualization server with 6-tab UI
- [`python/chronohorn/control`](./python/chronohorn/control)
  - closed-loop frontier controller over the store and planner
- [`python/chronohorn/store.py`](./python/chronohorn/store.py)
  - normalized runtime record schema and merged run snapshots
  - `IncrementalStore` for hot-cached results
- [`python/chronohorn/pipeline.py`](./python/chronohorn/pipeline.py)
  - stage-oriented runtime observer pipeline with local result cache
- [`python/chronohorn/runtime.py`](./python/chronohorn/runtime.py)
  - unified runtime daemon combining drain + fleet probe + viz + auto-deepen
- [`python/chronohorn/mcp.py`](./python/chronohorn/mcp.py)
  - MCP tool registry and stateful runtime server (21 tools)
- [`python/chronohorn/fleet`](./python/chronohorn/fleet)
  - placement, queueing, telemetry, and forecast wrappers
  - `drain.py` — unattended manifest execution with poll loop and re-dispatch
  - `results.py` — SSH-based result pull-back from remote containers
  - `auto_deepen.py` — slope-based auto-generation of next-horizon rows
  - `manifest_transform.py` — filter and mutate manifest rows without editing scan code
- [`python/chronohorn/export`](./python/chronohorn/export)
  - export bundle ABI and CLI

Rust:

- [`crates/chronohorn-core`](./crates/chronohorn-core)
  - generic runtime infrastructure
- [`crates/chronohorn-causal-bank`](./crates/chronohorn-causal-bank)
  - live causal-bank runtime family
- [`crates/chronohorn-runtime`](./crates/chronohorn-runtime)
  - typed export-bundle loading
- [`crates/chronohorn-cli`](./crates/chronohorn-cli)
  - export inspection and bundle probing
- [`src/archive`](./src/archive)
  - quarantined historical bridge/runtime families

## Experiment Matrix System

The scan emitter produces structured JSONL manifests from a single family scan
module. Supported regimes:

```bash
chronohorn fleet emit-causal-bank-matrix --regime current       # short pilot ablation
chronohorn fleet emit-causal-bank-matrix --regime long-slop     # two-slop deep matrix
chronohorn fleet emit-causal-bank-matrix --regime exotic-16mb   # artifact-viable mutations
```

The `exotic-16mb` regime covers the full architectural knob space with estimated
int6 artifact sizes in every row (`artifact_mb_est`). The scan enforces the 16MB
golf budget and warns on any overrun.

The spec-to-command derivation is a single source of truth: the same config dict
drives both the manifest metadata and the trainer CLI string.

Illegal result detection flags future-leakage at result load time. Results from
runs with illegal forward-looking context are quarantined and excluded from the
frontier.

## Architectural Variants (OPC Kernel Surface)

The causal-bank family now exposes a significantly expanded config surface. Key
variants explored this session:

### Substrate Modes (`substrate_mode`)

- `frozen` — fixed random projection (baseline)
- `learnable_decays` — decay rates are gradient-tracked
- `learnable_mixing` — mixing weights are gradient-tracked
- `learned_recurrence` — full learned recurrence (Mamba-style B/C projections with selective scan); chunked parallel scan replaces sequential for-loop

### Memory Attachment (`memory_kind`)

- `none` — no auxiliary memory
- `ngram` — n-gram prior lookup
- `exact_context` — exact context cache
- `statistical_backoff` — backoff hierarchy over n-gram and exact-context layers

`OnlineCausalMemory` provides a runtime n-gram accumulator with a 7-feature
query interface that is updated incrementally during training.

### Stacked Substrate Blocks

- `num_blocks` — number of stacked substrate blocks
- `block_mixing_ratio` — bottleneck mixing ratio between blocks
- `block_stride` — multi-timescale striding across blocks

### Selective Scan (`state_dim`, `num_heads`)

- `state_dim` — inner state dimension for B/C projections
- `num_heads` — number of selective scan heads

### Byte-to-Patch Encoding (`patch_size`, `patch_causal_decoder`)

- `patch_size` — number of raw bytes per patch
- `patch_causal_decoder` — `none` / `autoregressive` / `mlp_factored` / `hybrid`
  - `hybrid` runs a global SSM over patches plus a local window over raw bytes; best legal hybrid result: **1.909 bpb** at 5k steps

### Fast/Slow State Splitting

- `num_hemispheres` — number of state hemispheres
- `fast_hemisphere_ratio` — fraction of state dedicated to fast updates
- `fast_lr_mult` — learning rate multiplier for the fast hemisphere

### Polynomial Expansion

- `local_poly_order` — NVAR polynomial feature expansion on local window
- `substrate_poly_order` — polynomial expansion on substrate output

### Stability Controls

- `training_noise` — noise injected during forward pass for regularization
- `adaptive_reg` — automatically scaled regularization based on gradient statistics
- Decay regularization added to loss

### Readout

- Recurrent readout (GRU) added to `CAUSAL_BANK_READOUT_KINDS`

### Validation Helpers

- `learnable_substrate_keys()` — returns the set of config keys that are
  gradient-tracked under each `substrate_mode`
- Period and half-life range validation with descriptive error messages

## Current Runtime Path

The promoted live path today is:

1. train a causal-bank descendant
2. export an `opc` bundle
3. probe and replay it in Rust
4. build offline packed artifacts
5. score held-out tokens prequentially

The main Rust commands are:

```bash
cargo run -p chronohorn -- \
  run-causal-bank-checkpoint <checkpoint-or-bundle> <summary.json> @fineweb [val_tokens]

cargo run -p chronohorn -- \
  run-causal-bank-ngram-bulk-from-table <checkpoint-or-bundle> <summary.json> @fineweb <artifact.bin> [val_tokens] [report_every]

cargo run -p chronohorn-cli -- \
  probe-causal-bank-export-bundle <export-root>
```

## Docs

- [docs/REPO_BOUNDARY.md](./docs/REPO_BOUNDARY.md)
  - ownership split between `opc`, `chronohorn`, and `heinrich`
- [docs/FLEET.md](./docs/FLEET.md)
  - placement, queueing, observer layer, and slop execution model
- [docs/FRONTIER_QUEUE.md](./docs/FRONTIER_QUEUE.md)
  - live frontier order and active manifests
- [docs/STACK.md](./docs/STACK.md)
  - current runtime and artifact path
- [docs/CRATE_MAP.md](./docs/CRATE_MAP.md)
  - Rust workspace map
- [docs/ARCHIVE.md](./docs/ARCHIVE.md)
  - historical surface index

## License

MIT
