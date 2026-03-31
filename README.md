# Chronohorn

`Chronohorn` is the standalone execution repo for descendants built from
`open-predictive-coder`.

The public boundary this repo cares about today is simple:

1. `open-predictive-coder` provides the shared Python kernel
2. `chronohorn` turns descendants into trainable, exportable, replayable systems

External audit and evidence packaging are intentionally out of scope here.

It owns three first-class public surfaces:

1. Python train/export code under [`python/chronohorn`](./python/chronohorn)
2. Python fleet launch/status code under [`python/chronohorn/fleet`](./python/chronohorn/fleet)
3. Rust runtime/compiler inspection under [`src/`](./src) and [`crates/`](./crates)

Inside the Rust workspace, the promoted split is now explicit:

- [`crates/chronohorn-core`](./crates/chronohorn-core): generic runtime infrastructure
- [`crates/chronohorn-causal-bank`](./crates/chronohorn-causal-bank): live causal-bank runtime family
- [`src/archive/`](./src/archive): historical bridge and exploratory grouping

The root Rust shell is no longer a single flat command file:

- [`src/main.rs`](./src/main.rs): binary entry point and top-level dispatch
- [`src/shell_core.rs`](./src/shell_core.rs): core inspection, doctrine, and audit utility commands
- [`src/shell_causal_bank.rs`](./src/shell_causal_bank.rs): promoted causal-bank command group
- [`src/shell_archive.rs`](./src/shell_archive.rs): quarantined archive bridge-family command group
- [`src/shell_usage.rs`](./src/shell_usage.rs): public and archive help text
- [`src/shell_support.rs`](./src/shell_support.rs): shared shell parsing and summary helpers

The promoted system is no longer the old `match+skip` bridge family. The live path today is:

1. load a trained causal-bank checkpoint
2. replay it causally in Rust
3. build frozen train-side table artifacts from `@fineweb` train shards
4. evaluate on held-out val shards with the same causal runtime
5. keep oracle structure offline and out of runtime

In other words, `Chronohorn` is where kernel ideas become execution speed,
artifact economics, replay parity, and fleet-scale measurement.

## Install

Python:

```bash
python3 -m pip install -e .[train]
```

Use `.[torch]` for the Torch/CUDA surface or `.[metal]` for MLX/Metal.

Standalone installs expect the public `open-predictive-coder` package to be
available as a dependency. For sibling-repo work in a shared workspace, an
editable install of the kernel repo keeps the boundary explicit:

```bash
python3 -m pip install -e ../open-predictive-coder -e .[train]
```

Rust:

```bash
cargo build
```

The promoted Python commands are then available either through the installed
entry point:

```bash
chronohorn --help
```

or directly from source:

```bash
PYTHONPATH=python python3 -m chronohorn --help
```

Validated isolated package smoke:

```bash
uv venv .venv
uv pip install --python .venv/bin/python --no-deps -e .
.venv/bin/chronohorn --help
.venv/bin/chronohorn fleet --help
.venv/bin/chronohorn train --help
.venv/bin/chronohorn export --help
```

That smoke only validates the package surface and entrypoints. Training extras
still depend on the `open-predictive-coder` kernel package plus local backend
installs such as `sentencepiece`, `torch`, or `mlx`.

## Repo Surfaces

Python surface:

- `python -m chronohorn train ...`
- `python -m chronohorn export ...`
- descendant model-family code and bridge training live under [`python/chronohorn`](./python/chronohorn)
- train and parity summaries now include:
  - analytical causal-bank FLOP estimates
  - observed token throughput
  - estimated sustained and interval TFLOPs

Rust surface:

- `cargo run -p chronohorn -- ...` for the promoted runtime/scoring commands
- `cargo run -p chronohorn-cli -- ...` for export-bundle inspection and replay-prep
- `python -m chronohorn fleet ...` for manifest-driven launch/status
- fleet planning now uses measured throughput and estimated TFLOPs from real Chronohorn result JSONs
- runtime/compiler code lives under [`src/`](./src)
- typed export inspection lives under [`crates/chronohorn-cli`](./crates/chronohorn-cli) and [`crates/chronohorn-runtime`](./crates/chronohorn-runtime)
- bundle boundary commands now include:
  - `inspect-export`
  - `inspect-inventory`
  - `verify-probe`
  - `probe-causal-bank-export-bundle`

## Live Stack

Core infrastructure:

- [crates/chronohorn-core/src/runtime.rs](./crates/chronohorn-core/src/runtime.rs): process-level parallel runtime setup, thread-pool reporting, `CHRONOHORN_THREADS`
- [crates/chronohorn-core/src/data.rs](./crates/chronohorn-core/src/data.rs): token shard loading and root resolution
- [crates/chronohorn-core/src/checkpoint.rs](./crates/chronohorn-core/src/checkpoint.rs): `.npz` / `.npy` checkpoint inspection
- [crates/chronohorn-core/src/protocol.rs](./crates/chronohorn-core/src/protocol.rs): scorer contract
- [crates/chronohorn-core/src/audit.rs](./crates/chronohorn-core/src/audit.rs): internal runtime-check battery
- [crates/chronohorn-core/src/bridge.rs](./crates/chronohorn-core/src/bridge.rs): oracle/compressor/bridge doctrine

Promoted runtime path:

- [crates/chronohorn-causal-bank/src/lib.rs](./crates/chronohorn-causal-bank/src/lib.rs): promoted family grouping for the live causal-bank line
- [crates/chronohorn-causal-bank/src/checkpoint.rs](./crates/chronohorn-causal-bank/src/checkpoint.rs): current Rust replay implementation for the promoted causal-bank line
- [crates/chronohorn-causal-bank/src/exact_experts.rs](./crates/chronohorn-causal-bank/src/exact_experts.rs): causal exact-table helpers
- [crates/chronohorn-causal-bank/src/ngram_bulk.rs](./crates/chronohorn-causal-bank/src/ngram_bulk.rs): frozen table builders and bulk eval
- [crates/chronohorn-causal-bank/src/exact_ngram_checkpoint.rs](./crates/chronohorn-causal-bank/src/exact_ngram_checkpoint.rs): merged checkpoint probes
- [crates/chronohorn-causal-bank/src/oracle.rs](./crates/chronohorn-causal-bank/src/oracle.rs): token-level oracle support used offline
- [crates/chronohorn-causal-bank/src/ranked_teacher.rs](./crates/chronohorn-causal-bank/src/ranked_teacher.rs): ranked-teacher loaders

## Artifact Flow

The current artifact flow is explicit:

1. build offline artifacts from train data
2. freeze them to disk
3. load the checkpoint and artifact into the causal runtime
4. score val tokens prequentially
5. keep runtime-check pressure on the runtime/artifact boundary

The most important commands are:

```bash
cargo run -p chronohorn -- print-parallel-runtime

cargo run -p chronohorn -- \
  run-causal-bank-checkpoint <checkpoint-path|bundle-dir> <summary.json> @fineweb [val_tokens]

cargo run -p chronohorn-cli -- \
  probe-causal-bank-export-bundle <export-root>

cargo run -p chronohorn -- \
  audit-causal-bank-checkpoint <checkpoint-path|bundle-dir> <summary.json> @fineweb [val_tokens] [chunk_size] [max_chunks]

cargo run -p chronohorn -- \
  build-causal-bank-ngram-oracle-budgeted-table @fineweb <artifact.bin> [train_tokens] [report_every] [profile] [oracle_stride]

cargo run -p chronohorn -- \
  run-causal-bank-ngram-bulk-from-table <checkpoint-path|bundle-dir> <summary.json> @fineweb <artifact.bin> [val_tokens] [report_every]
```

Compatibility note:

- `audit-*` command names still exist in the root Rust CLI
- inside `Chronohorn`, they mean internal runtime checks and invariants
- they are not a claim that this repo is its own external auditor
- some internal archive/runtime implementation names still reflect older family labels, but the promoted public surface uses causal-bank names
- archive families are now grouped under `chronohorn::archive` instead of appearing as flat top-level modules in the public story

For full command coverage, run:

```bash
cargo run -p chronohorn -- --help
cargo run -p chronohorn -- help-archive
cargo run -p chronohorn-cli -- --help
```

## Data Split

On the local stored `@fineweb` root currently used by this repo:

- train shard: `100,000,000` tokens
- val shard: `62,021,846` tokens

So in current usage:

- `100M` means offline artifact build on train
- `62M` means the full held-out validation side

## Parallel Runtime

`Chronohorn` now treats CPU/core delegation as runtime infrastructure.

- default: use all available threads
- override: `CHRONOHORN_THREADS=<n>`
- inspect:

```bash
cargo run -p chronohorn -- print-parallel-runtime
```

The current builder and bulk scorer use this runtime instead of hiding ad hoc thread choices inside individual experiments.

## Docs

- [docs/REPO_BOUNDARY.md](./docs/REPO_BOUNDARY.md): public repo boundary and ownership rules
- [docs/DOCTRINE.md](./docs/DOCTRINE.md): project boundary
- [docs/STACK.md](./docs/STACK.md): current live runtime/artifact stack
- [docs/CRATE_MAP.md](./docs/CRATE_MAP.md): Rust workspace surface
- [docs/FLEET.md](./docs/FLEET.md): backend-agnostic `cpu` / `metal` / `cuda` launcher surface
- [docs/ARCHIVE.md](./docs/ARCHIVE.md): archive index for older bridge families and migration notes
- [docs/SCALER_FRAMEWORK.md](./docs/SCALER_FRAMEWORK.md): parent-line interpretation

Historical plans and retired branch notes now live under
[`docs/archive/`](./docs/archive/). They are preserved for archaeology, not as
the current public contract for `Chronohorn`.

## Fleet

`Chronohorn` now has a manifest-driven hardware surface for mixed local and remote execution.

- `cpu`: remote Linux snapshot jobs for Rust builders and eval
- `metal`: local MLX descendant jobs on this Mac
- `cuda`: remote GPU container jobs on the slop boxes

Launch through the package surface:

```bash
PYTHONPATH=python python3 -m chronohorn fleet \
  --manifest manifests/fleet_example.jsonl
```

The legacy wrapper script at [scripts/dispatch_experiment.py](./scripts/dispatch_experiment.py)
still exists for source-tree convenience, but the package command above is the
promoted entrypoint. The fleet model and manifest schema are documented in
[docs/FLEET.md](./docs/FLEET.md).

The checked-in `fleet_example.jsonl` is a portable local smoke. Machine-local
lab manifests live alongside it under [`manifests/`](./manifests/), but they are
not treated as the public starting point.

## Archive Policy

`Chronohorn` still keeps a large amount of early bridge and probe code under `src/`.

That code is not deleted because it still matters for:

- archaeology
- falsification
- runtime boundary pressure
- recovering ideas that survive cleanly under the newer runtime

But it is no longer the repo's promoted stack. The live system is the checkpoint runtime plus offline-built artifacts. Older bridge families are documented in [docs/ARCHIVE.md](./docs/ARCHIVE.md).
