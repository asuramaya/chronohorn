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

That is the right foundation for the next generation of experiments.

