# Chronohorn

<p align="center">
  <img src="docs/chronohorn.jpg" alt="Chronohorn" width="520">
</p>

> Mascot: Chronohorn is the stone timekeeper.

Runtime engine for `open-predictive-coder` descendants. Built for
non-transformer frontier work, agent-driven experiment control, and the real
hardware shape of one laptop plus a couple of slop boxes.

## What It Does

`Chronohorn` turns `opc` model families into runnable systems:

- backend-specific training on MLX/Metal and Torch/CUDA
- multi-host placement across Apple, CPU, and NVIDIA lanes
- runtime observation through a normalized run store
- budget forecasting against competition constraints
- export bundle emission
- Rust replay, scoring, and packed-artifact evaluation

Built for agents:

- terminal observer surface: `chronohorn observe ...`
- MCP server: `chronohorn-mcp`
- runtime state is normalized instead of scattered across manifests, launch
  records, result JSONs, and forecast output

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
chronohorn fleet queue --manifest manifests/frontier_long_slop_matrix.jsonl
chronohorn fleet forecast-results --path <result.json>

chronohorn observe pipeline --manifest manifests/frontier_long_slop_matrix.jsonl
chronohorn observe status --manifest manifests/frontier_long_slop_matrix.jsonl --probe-runtime
chronohorn observe query-records --kind runtime_state

chronohorn export --help
chronohorn mcp
chronohorn-mcp

cargo run -p chronohorn -- --help
cargo run -p chronohorn-cli -- --help
```

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

9 tools are exposed:

- `chronohorn_manifests`
- `chronohorn_runtime_status`
- `chronohorn_launches`
- `chronohorn_results`
- `chronohorn_forecast`
- `chronohorn_records`
- `chronohorn_status`
- `chronohorn_pipeline`
- `chronohorn_reset`

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
  - runtime observation and forecasting
  - replay, scoring, and artifact economics
- `heinrich`
  - external evidence packaging
  - validation bundles
  - public-facing audit compression

## Runtime Record Pipeline

The observer layer follows the same small-stage style as Heinrich, but for
runtime state instead of evidence:

```text
manifest -> runtime_state -> launch -> result -> forecast
```

Every stage writes normalized `RunRecord` rows into a shared `RunStore`. The
observer then merges those into run-level snapshots for agents and CLIs.

## Modules

Python:

- [`python/chronohorn/engine`](./python/chronohorn/engine)
  - budgets, performance accounting, result summaries, forecasting, probes
- [`python/chronohorn/families`](./python/chronohorn/families)
  - family adapters and scan policy
- [`python/chronohorn/observe`](./python/chronohorn/observe)
  - observer CLI over the run store
- [`python/chronohorn/store.py`](./python/chronohorn/store.py)
  - normalized runtime record schema and merged run snapshots
- [`python/chronohorn/pipeline.py`](./python/chronohorn/pipeline.py)
  - stage-oriented runtime observer pipeline
- [`python/chronohorn/mcp.py`](./python/chronohorn/mcp.py)
  - MCP tool registry and stateful runtime server
- [`python/chronohorn/fleet`](./python/chronohorn/fleet)
  - placement, queueing, telemetry, and forecast wrappers
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
