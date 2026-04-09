# Stack

This document describes the current live `Chronohorn` stack.

## Topology

`Chronohorn` now has a real Rust runtime stack instead of a loose collection of bridge probes.

The current layers are:

1. runtime infrastructure
2. data + checkpoint boundary
3. causal checkpoint replay
4. offline artifact builders
5. runtime checks
6. fleet launch surface

## Runtime Infrastructure

Parallel runtime setup lives in [crates/chronohorn-core/src/runtime.rs](../crates/chronohorn-core/src/runtime.rs).

It owns:

- global Rayon pool setup
- default thread-width selection
- `CHRONOHORN_THREADS`
- runtime reporting through `print-parallel-runtime`

This is process-wide infrastructure, not experiment-local plumbing.

## Data And Artifact Boundary

These files define the honest I/O boundary:

- [crates/chronohorn-core/src/data.rs](../crates/chronohorn-core/src/data.rs)
- [crates/chronohorn-core/src/checkpoint.rs](../crates/chronohorn-core/src/checkpoint.rs)
- [crates/chronohorn-core/src/bridge.rs](../crates/chronohorn-core/src/bridge.rs)
- [crates/chronohorn-runtime/src/loader.rs](../crates/chronohorn-runtime/src/loader.rs)

Current local `@fineweb` layout:

- train: `100,000,000` tokens
- val: `62,021,846` tokens

That means artifact build and held-out eval are already separated at the data-root level.

Important detail:

- root replay code now consumes the typed export-bundle loader in `chronohorn-runtime`
- manifest, learned-state index, checksum, and sidecar reference resolution are no longer implemented twice

## Causal Runtime

The promoted family namespace is [crates/chronohorn-causal-bank/src/lib.rs](../crates/chronohorn-causal-bank/src/lib.rs).

The current replay implementation lives in [crates/chronohorn-causal-bank/src/checkpoint.rs](../crates/chronohorn-causal-bank/src/checkpoint.rs).

It owns:

- checkpoint loading
- recurrent causal-bank state updates for the current implementation line
- batched readout
- checkpoint audit entrypoints

Important detail:

- the recurrent state update is still sequential
- the heavy readout path is block-batched
- dense batch math uses hardware BLAS on macOS via Accelerate

This is now the main causal scorer, not a side experiment.

## Offline Artifact Builders

The promoted artifact-builder namespace is [crates/chronohorn-causal-bank/src/lib.rs](../crates/chronohorn-causal-bank/src/lib.rs).

The current artifact path lives in [crates/chronohorn-causal-bank/src/ngram_bulk.rs](../crates/chronohorn-causal-bank/src/ngram_bulk.rs).

It owns:

- frozen raw fingerprinted tables
- oracle-edited row selection
- prebuilt artifact serialization
- bulk scalar eval over held-out val

The intended split is:

1. build artifact from train tokens
2. save artifact
3. reuse artifact across checkpoint evals

This keeps long train-side scans out of the eval loop.

## Runtime Checks

The internal runtime-check battery remains in [crates/chronohorn-core/src/audit.rs](../crates/chronohorn-core/src/audit.rs).

These checks are still the gate inside `Chronohorn`:

- normalization
- repeatability
- future-suffix invariance
- answer-mask invariance
- prefix-truncation parity
- stream-rechunk parity
- sample-set invariance
- gold-logprob consistency

Bulk scalar research scorers are allowed for measurement, but promoted runtime
claims still have to survive this path.

Boundary note:

- `Chronohorn` owns internal execution invariants and replay parity
- external audit and evidence packaging are intentionally out of scope here
- legacy `audit-*` names remain for compatibility with the current CLI surface

## Fleet Launch Surface

Mixed-machine execution is documented in [FLEET.md](./FLEET.md).

The backend model is:

- `cpu` for remote Rust / artifact work
- `metal` for local MLX descendant jobs
- `cuda` for remote GPU container work

The key point is that this is one manifest and multiple honest backends, not a fake homogeneous cluster.

The currently promoted causal-bank loop is:

1. cheap `10k` O(n) architecture ablations
2. scale/context-survival rows for winners
3. deeper frontier or replication follow-up only after the cheap lanes separate

That policy is emitted from the family-owned scan regimes under
[`python/chronohorn/families/causal_bank/scan.py`](../python/chronohorn/families/causal_bank/scan.py),
including `breakthrough-10k`, `toward-one`, `toward-one-next`, and `gated-retention`.

## Current Promoted Commands

Checkpoint runtime:

```bash
cargo run -p chronohorn -- \
  run-causal-bank-checkpoint <checkpoint-path|bundle-dir> <summary.json> @fineweb [val_tokens]
```

Checkpoint audit:

```bash
cargo run -p chronohorn -- \
  audit-causal-bank-checkpoint <checkpoint-path|bundle-dir> <summary.json> @fineweb [val_tokens] [chunk_size] [max_chunks]
```

Export bundle replay probe:

```bash
cargo run -p chronohorn-cli -- \
  probe-causal-bank-export-bundle <export-root>
```

Artifact build:

```bash
cargo run -p chronohorn -- \
  build-causal-bank-ngram-oracle-budgeted-table @fineweb <artifact.bin> [train_tokens] [report_every] [profile] [oracle_stride]
```

Artifact eval:

```bash
cargo run -p chronohorn -- \
  run-causal-bank-ngram-bulk-from-table <checkpoint-path|bundle-dir> <summary.json> @fineweb <artifact.bin> [val_tokens] [report_every]
```

Runtime report:

```bash
cargo run -p chronohorn -- print-parallel-runtime
```

## Archive Boundary

Older bridge families now sit behind [src/archive/](../src/archive), and they are no longer the default reading of the repo.

Use [ARCHIVE.md](./ARCHIVE.md) for those lines.
