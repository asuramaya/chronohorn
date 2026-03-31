# Table Compiler

This document proposes a two-stage table compiler for `Chronohorn`.

The goal is to separate:

1. a single train-data scan that extracts row statistics once
2. a later repack step that turns those statistics into byte-budgeted artifacts

This keeps the expensive scan offline, keeps the causal runtime clean, and makes it easier to produce multiple packed variants from the same research corpus.

## Current Repo Shape

The compiler proposal has to fit the code that already exists:

- [crates/chronohorn-causal-bank/src/checkpoint.rs](../crates/chronohorn-causal-bank/src/checkpoint.rs): causal-bank checkpoint replay and runtime scoring for the current implementation line
- [crates/chronohorn-causal-bank/src/ngram_bulk.rs](../crates/chronohorn-causal-bank/src/ngram_bulk.rs): offline train-side table builder and artifact loader
- [crates/chronohorn-core/src/runtime.rs](../crates/chronohorn-core/src/runtime.rs): process-wide Rayon configuration and thread reporting

The current stack already separates the causal scorer from the offline table path. This proposal makes that split explicit at the artifact level.

## Why A Compiler

The existing offline builder already has two different jobs:

- extract train-side row statistics
- serialize a frozen artifact for later reuse

Those jobs should not be fused into one opaque build pass. A compiler shape is better because it allows:

- one train scan for many packings
- deterministic rebuilds from the same research artifact
- explicit byte budgeting
- cleaner audits of what came from data and what came from packing policy

## Stage 1: Research Artifact

Stage 1 scans the train root once and emits a research artifact.

This artifact should store row statistics, not a submission-shaped packed table.

Minimum contents:

- order-specific row counts
- support sizes and totals
- row fingerprints or stable row keys
- oracle-trust / pruning metadata when available
- provenance: data root, token budget, build time, runtime settings

The point is to preserve the useful table facts in a form that can be repacked later without rescanning the shard root.

This stage is the natural home for the current offline scan logic in [crates/chronohorn-causal-bank/src/ngram_bulk.rs](../crates/chronohorn-causal-bank/src/ngram_bulk.rs).

## Stage 2: Packed Artifact

Stage 2 takes the research artifact and repacks it into a byte-budgeted artifact.

This is the artifact that should look like a submission-shaped bundle:

- fixed binary layout
- bounded size
- stable load path
- minimal runtime metadata

The packed artifact should contain only what the runtime needs for scoring or evaluation.

The current `save_ngram_table_artifact` / `load_ngram_table_artifact` path in [crates/chronohorn-causal-bank/src/ngram_bulk.rs](../crates/chronohorn-causal-bank/src/ngram_bulk.rs) is already close to this role. The compiler proposal is to make that final packing phase explicit instead of implicit.

## Proposed Flow

1. Scan `@fineweb` train shards once.
2. Extract row statistics into a research artifact.
3. Run one or more packers over that artifact.
4. Emit submission-shaped packed artifacts with different byte budgets or profiles.
5. Load the packed artifact later for checkpoint eval or bulk scoring.

That means one research build can feed several packed outputs:

- conservative budget
- medium budget
- large budget
- oracle-pruned variant
- oracle-budgeted variant

## Parallelism

The compiler should use the existing process-level runtime in [crates/chronohorn-core/src/runtime.rs](../crates/chronohorn-core/src/runtime.rs) rather than managing threads locally.

That means:

- Rayon pool setup stays global
- `CHRONOHORN_THREADS` still controls width
- scan and repack phases can both use the shared runtime

If a pass can be parallelized, it should use the same infrastructure the rest of `Chronohorn` already uses.

## Alignment With Current Causal-Bank Path

The compiler is not a new runtime.

It is a way to support the current causal path in [crates/chronohorn-causal-bank/src/checkpoint.rs](../crates/chronohorn-causal-bank/src/checkpoint.rs) and the current offline builder in [crates/chronohorn-causal-bank/src/ngram_bulk.rs](../crates/chronohorn-causal-bank/src/ngram_bulk.rs) without mixing research data extraction with final artifact packing.

That keeps the live scorer focused on:

- checkpoint replay
- causal evaluation
- audit compatibility

And keeps the compiler focused on:

- train-side table extraction
- byte-budgeted packing
- artifact reuse

## Artifact Distinction

Use two different artifact meanings:

- research artifact: rich, replayable, possibly larger than any final submission target
- packed artifact: byte-budgeted, submission-shaped, directly loadable by the runtime

That distinction matters because it keeps the compiler honest:

- research artifacts can be inspected and repacked
- packed artifacts are the actual deployable outputs

## Suggested Implementation Boundary

The compiler should stay conceptually separate from the causal runtime and from CLI wiring.

The clean boundary is:

- extract row statistics once from train data
- pack later into one or more artifacts
- keep the runtime as a consumer, not a builder

That matches the current repo direction and gives `Chronohorn` a clearer table pipeline without changing the live scorer model.
