# Chronohorn

`Chronohorn` is a clean Rust reset for the post-`Conker` / post-`BLINX` stack.

The point is not to port the contaminated Python architecture line-for-line.
The point is to separate concerns hard enough that weird runtime behavior, hidden score paths, and accidental artifact leakage have nowhere to hide.

Current scope:

- inspect `.npz` checkpoints outside Python
- define a small causal scoring trait
- run legality attacks against a scorer
- include a built-in cheating demo that proves the auditor can catch more than future leakage

Mental model:

- `oracle/` will eventually be the noncausal analysis layer
- `runtime/` will be the deployed causal scorer
- `audit/` attacks the runtime
- `checkpoint/` owns the artifact boundary

This first crate keeps those ideas small and explicit:

- [checkpoint.rs](./src/checkpoint.rs): `.npz` / `.npy` inspection
- [data.rs](./src/data.rs): parameter-golf shard loading outside Python
- [oracle.rs](./src/oracle.rs): cleaned oracle summaries from BLINX attack outputs
- [packed_memory.rs](./src/packed_memory.rs): packed unigram/bigram/trigram scorer
- [protocol.rs](./src/protocol.rs): runtime scorer contract
- [audit.rs](./src/audit.rs): legality probes
- [demo.rs](./src/demo.rs): legal and cheating toy scorers

## Commands

Inspect a checkpoint:

```bash
cargo run --manifest-path chronohorn/Cargo.toml -- inspect-npz path/to/checkpoint.npz
```

Run the built-in legality demo:

```bash
cargo run --manifest-path chronohorn/Cargo.toml -- audit-demo legal
cargo run --manifest-path chronohorn/Cargo.toml -- audit-demo length-peek
cargo run --manifest-path chronohorn/Cargo.toml -- audit-demo boundary-double-update
cargo run --manifest-path chronohorn/Cargo.toml -- audit-demo reported-gold-cheat
```

Audit a real packed-memory scorer built directly from FineWeb shards:

```bash
cargo run --manifest-path chronohorn/Cargo.toml -- \
  audit-packed-memory conker-standalone/conker/data/datasets/fineweb10B_sp1024 200000 2048
```

Diff packed tables stored in a saved checkpoint against a Rust rebuild from shards:

```bash
cargo run --manifest-path chronohorn/Cargo.toml -- \
  compare-packed-memory \
  conker-standalone/conker/out/conker10_giddyup_fineweb_peak_candidate4_seed42_2026-03-28.npz \
  conker-standalone/conker/out/conker10_giddyup_fineweb_peak_candidate4_seed42_2026-03-28.json \
  conker-standalone/conker/data/datasets/fineweb10B_sp1024
```

Turn BLINX oracle-attack outputs into cleaned bridge targets:

```bash
cargo run --manifest-path chronohorn/Cargo.toml -- \
  oracle-clean-summary \
  blinx/conker/out/blinx_oracle_attack_2026-03-28.json \
  8
```

Run the first byte-level bridge prototype:

```bash
cargo run --manifest-path chronohorn/Cargo.toml -- \
  train-byte-bridge \
  4 \
  16 \
  80
```

Print the reset rationale:

```bash
cargo run --manifest-path chronohorn/Cargo.toml -- design
```

## Current Read

The important new attack surface is not just future leakage.
It is hidden scalar scoring.

A scorer can:

- return a clean causal full distribution
- pass suffix and answer perturbation checks
- still score the gold token through a different internal path

`Chronohorn` bakes that attack into the base legality audit through `gold_logprob_consistency`.

The next attack surface is chunk-shape leakage.

A scorer can:

- ignore future token values entirely
- still depend on full chunk length or batch shape
- pass suffix perturbation and answer-mask checks
- yet disagree with strict prefix-only replay

`Chronohorn` now attacks that through `prefix_truncation_parity`.

The next attack surface after that is chunk-boundary state leakage.

A scorer can:

- be perfectly prefix-causal inside each chunk
- keep normalization and gold-logprob accounting clean
- still update state incorrectly at chunk boundaries
- and only reveal that when the exact same token stream is rechunked

`Chronohorn` now attacks that through `stream_rechunk_parity`.

The first real-data result is already useful:

- the saved `Conker-10` FineWeb checkpoint carries packed memory as an explicit artifact surface
- `Chronohorn` can rebuild those tables from shards in Rust
- the saved tables match the Rust rebuild to numerical identity
- the checkpoint-loaded memory scorer passes the same sampled legality checks outside Python

The first oracle-hygiene result is also useful:

- Chronohorn ingests BLINX attack output instead of trusting raw bidirectional oracle numbers
- it ranks bridge targets by `left_leaveout_candidate4 - self_inclusion_uplift`
- on the March 28 BLINX attack bundle, radius `4` is the best bridge surface by that criterion
- the strongest bridgeable rows are repetitive run-script surfaces, not docs with obvious self-inclusion contamination

The first bridge-head result is also useful:

- Chronohorn now has a tiny byte-level bridge head trained on cleaned oracle labels from local files
- it uses leave-one-out bidirectional `candidate <= 4` as the target and left-only n-gram features as inputs
- on an `80` file split with stride `16`:
  - radius `2`: eval accuracy `0.9057` vs majority `0.9038`
  - radius `3`: eval accuracy `0.8494` vs majority `0.8311`
  - radius `4`: eval accuracy `0.8072` vs majority `0.7527`
- so radius `4` gives the strongest real bridge lift over the class-prior baseline, matching the cleaned-oracle summary

The first chunk-shape result is also useful:

- the built-in `length-peek` cheat passes normalization, repeatability, future-suffix invariance, answer-mask invariance, and gold-logprob consistency
- it fails `prefix_truncation_parity` and `stream_rechunk_parity`
- the built-in `boundary-double-update` cheat passes `prefix_truncation_parity` and fails only `stream_rechunk_parity`
- while adding `stream_rechunk_parity`, Chronohorn found and fixed a real bug in its own packed-memory runtime: chunk-boundary context had been resetting
- the real packed-memory FineWeb runner now passes both `prefix_truncation_parity` and `stream_rechunk_parity`

That is the right foundation for the next generation of experiments.
