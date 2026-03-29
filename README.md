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

That is the right foundation for the next generation of experiments.
