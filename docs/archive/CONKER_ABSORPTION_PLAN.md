# Conker Absorption Plan

Status: historical migration record. `Chronohorn` is now the main repo surface, and this document exists to preserve the intended boundary and cleanup order rather than to describe a speculative future split.

This document defines how `Chronohorn` should absorb the current `Conker` and `conker-rs` code without creating a second entangled runtime. The goal is to make `Chronohorn` the main line, with `Conker` reduced to an internal model family and `conker-rs` reduced to an internal Rust crate set.

## Target Shape

`Chronohorn` should own four layers:

1. `opc` or `opc-like` Python kernel/core layer, used as the shared mechanism library for reusable substrates, memory primitives, routing, readouts, and export ABI generation.
2. `chronohorn` Python training/search/export layer, used for model-family code, mutation search, checkpoint export, and short probes.
3. `chronohorn` Rust runtime/compiler layer, used for replay, packed-memory compilation, runtime checks, and bulk evaluation.
4. fleet/orchestration scripts, used only as control-plane glue.

The `Conker` family should become an internal namespace, not a top-level repo boundary.

## Proposed Package And Crate Layout

### Python

Recommended package layout:

- `chronohorn/python/chronohorn/models/`
  - `conker3.py`
  - `conker7.py`
  - `readouts.py`
  - model-family adapters
- `chronohorn/python/chronohorn/train/`
  - training loops
  - bridge entrypoints
  - dataset glue
  - export helpers
- `chronohorn/python/chronohorn/experiments/`
  - mutation search
  - ablation runners
  - checkpoint sweep scripts
- `chronohorn/python/chronohorn/fleet/`
  - manifest helpers
  - status probes
  - launch wrappers

### Rust

Recommended crate layout:

- `chronohorn/crates/chronohorn-schema/`
  - export ABI
  - artifact metadata
  - provenance fields
- `chronohorn/crates/chronohorn-runtime/`
  - checkpoint replay
  - deterministic substrate reconstruction
  - bulk scoring
- `chronohorn/crates/chronohorn-compiler/`
  - row-stats extraction
  - packed artifact repacking
  - byte-budgeted layout
- `chronohorn/crates/chronohorn-cli/`
  - end-user command surface

## Exact Source Groups To Move Or Retire

### Move Into Chronohorn Python

Move or absorb the current `conker` Python training/search code into `chronohorn/python/chronohorn/`:

- `conker/conker/src/conker3.py`
- `conker/conker/src/conker3_torch.py`
- `conker/conker/src/models.py`
- `conker/conker/src/models_torch.py`
- `conker/conker/src/golf_data.py`
- `conker/conker/src/golf_data_torch.py`
- `conker/conker/src/quantize.py`
- `conker/conker/scripts/run_conker3_golf_bridge.py`
- `conker/conker/scripts/run_conker3_golf_bridge_torch.py`
- `conker/conker/scripts/run_conker3_staticgate_plateau_sweep.py`
- `chronohorn/scripts/run_local_conker3_staticgate_frontier.zsh`
- `chronohorn/manifests/*conker*`

The intent is not to preserve every old file path. The intent is to preserve the stable behavior under a new root package and new command names.

### Move Into Chronohorn Rust

Absorb the current `conker-rs` prototype into `chronohorn/crates/`:

- `conker-rs/src/main.rs`
- `conker-rs/README.md`
- `conker-rs/Cargo.toml`
- `conker-rs/Cargo.lock`

This code should become the starting point for `chronohorn-runtime` and `chronohorn-schema`, not a separate long-term repo.

### Retire Or Rename

These are not new public top-level systems after the migration:

- `conker` becomes a model-family namespace under `chronohorn`
- `conker-rs` becomes internal crate history under `chronohorn`
- duplicate bridge scripts should be collapsed into `chronohorn` training/export entrypoints

## Command Surface Changes

The public interface should be normalized around `chronohorn` commands, not sibling repo names.

Planned commands:

- `chronohorn train conker3 ...`
- `chronohorn train conker7 ...`
- `chronohorn export ...`
- `chronohorn replay ...`
- `chronohorn build-row-stats ...`
- `chronohorn pack ...`
- `chronohorn check-runtime ...`
- `chronohorn fleet ...`

Existing `conker` bridge scripts should become thin wrappers or be retired once the new `chronohorn` entrypoints are stable.

The Rust command surface in `chronohorn` should keep the current behavior but be renamed around the new repo identity:

- checkpoint replay stays
- oracle/table build stays
- pack/repack stays
- internal runtime checks stay
- bulk eval stays

Only the naming should change from `conker` family wording to `chronohorn` family wording.

## Migration Phases

### Phase 0: Boundary Freeze

Freeze the roles:

- `chronohorn` owns runtime/compiler/search
- `conker` is an internal family label
- `conker-rs` is an internal Rust starting point

Deliverables:

- one export ABI draft
- one directory/package layout draft
- one command-surface map

### Phase 1: Python Absorption

Move the `conker` Python family into `chronohorn/python/chronohorn/`.

Success criteria:

- the current `Conker-3` and related bridge jobs still run
- the new package path is the canonical one
- old repo-specific imports are gone or isolated

### Phase 2: Rust Absorption

Move the `conker-rs` replay prototype into `chronohorn/crates/`.

Success criteria:

- Rust can load the export ABI
- deterministic substrate replay works under the new crate identity
- the checkpoint/runtime boundary is no longer tied to the old repo name

### Phase 3: Export ABI

Define the stable artifact export contract between Python and Rust.

Success criteria:

- one exported artifact schema
- one Rust loader
- one Python exporter
- parity tests for a short shard

### Phase 4: Command Unification

Replace old entrypoints with `chronohorn` entrypoints.

Success criteria:

- train/export/replay/pack/runtime-checks all live under `chronohorn`
- old `conker` command names are only compatibility shims or archived

### Phase 5: Cleanup

Remove duplicate paths, stale wrappers, and repo-name drift.

Success criteria:

- `chronohorn` is the main line
- `conker` is only a model family name
- `conker-rs` is only a historical prototype lineage

## Non-Goals

Do not do these during the absorption:

- do not merge external audit/evidence tooling into the system runtime
- do not rewrite all training code in Rust first
- do not introduce a second schema for validity bundles if an export ABI already exists
- do not keep separate top-level repo identities for the same model family

## Guiding Rule

The absorbing principle is:

- `chronohorn` owns execution
- `opc` owns the reusable Python kernel/core
- `heinrich` owns independent audit and evidence packaging

`Conker` should disappear as a repo boundary, not as a model-family label.
