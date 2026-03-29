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

Inspect a data root and its claim tier:

```bash
cargo run --manifest-path chronohorn/Cargo.toml -- inspect-data-root @replay
cargo run --manifest-path chronohorn/Cargo.toml -- inspect-data-root @fineweb
cargo run --manifest-path chronohorn/Cargo.toml -- inspect-data-root /path/to/data-root
```

Show the Chronohorn data home, built-in aliases, and local registry location:

```bash
cargo run --manifest-path chronohorn/Cargo.toml -- print-data-home
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

## Data Tiers

`Chronohorn` treats data provenance as part of the result.

- `target_eval`: real shard root with train and val shards; promotion-ready
- `architecture_only`: replay or local synthetic root; useful for architecture search, not leaderboard claims
- `blocked`: missing or broken root; promotion cannot run

This is machine-readable through `inspect-data-root` and carried into JSON run bundles.

## Data Home

`Chronohorn` now gives dataset roots a canonical place to live.

- default data home: [chronohorn/data](./data)
- built-in aliases:
  - `@replay`
  - `@local-code`
  - `@fineweb`
- optional local override registry: `chronohorn/data/roots.json`
- local stored roots should live under [chronohorn/data/roots](./data/roots)

The intended workflow is:

1. keep architecture-only roots explicit through `@replay` or `@local-code`
2. if a real shard root exists, point `@fineweb` or a custom alias at it through `roots.json`
3. let every run bundle carry the resolved root and claim tier

This avoids `/tmp` folklore and makes “missing”, “architecture_only”, and “target_eval” part of the machine-readable output.

Important:

- the `openai/parameter-golf` GitHub repo provides downloader scripts, not the shard files themselves
- a neat local setup stores downloaded roots under `chronohorn/data/roots/`
- `@fineweb` should be treated as a stored-root alias, not as a promise that the legacy symlink still exists

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

The first bridge-to-codec result is better:

- `run-byte-bridge-codec` keeps the same file split but evaluates actual byte-level bits-per-byte
- it compares four paths:
  - base left-only n-gram model
  - heuristic top-4 gate
  - oracle-trained bridge gate
  - direct compression gate trained on NLL
- on the same `80` file split with stride `16`, the direct gate is best at every tested radius:
  - radius `2`: base `3.4268`, heuristic `2.9649`, oracle `3.0575`, direct `2.9111`
  - radius `3`: base `3.8002`, heuristic `2.3037`, oracle `2.4066`, direct `2.2862`
  - radius `4`: base `4.3620`, heuristic `2.2785`, oracle `2.4248`, direct `2.2598`
- that means the cleaned-oracle bridge is useful, but the stronger immediate result is that the same causal features can train directly against compression loss and beat both the oracle gate and the heuristic

The first token-level bridge result is sharper:

- `run-token-bridge` lifts the same idea onto packed token memory built from FineWeb shards
- it compares:
  - base packed-memory model
  - heuristic top-k gate
  - direct NLL-trained gate
  - oracle top-k upper bound
- on real token data, the intervention has persistent headroom but the current gate class cannot use it
- with `candidate_k = 4`, `train_stride = 4`, and increasing train memory:
  - `500k / 2048`: base `9.5080`, oracle `9.1960`
  - `1M / 2048`: base `9.0750`, oracle `8.7450`
  - `1M / 4096`: base `8.9015`, oracle `8.5620`
  - `2M / 4096`: base `8.4272`, oracle `8.0670`
  - `2M / 8192`: base `8.2155`, oracle `7.8529`
- widening the gate family did not change that:
  - linear direct gate: `0.0`
  - tiny MLP gate: `0.0`
  - trigram-bucket gate: `0.0`
- so the token-level result is:
  - the packed-memory intervention is real
  - top-4 candidate restriction has consistent oracle value
  - but that oracle value is not obviously causally exploitable from the packed-memory posterior alone
  - richer gates over the same posterior still collapse back to the unbiased base model
- the best audited token path so far is `2M / 8192 / k=4`, and it passes the full current legality suite

The next token-level result is the real mutation:

- the project is no longer limited to posterior-only gates
- two genuinely different causal channels now exist:
  - `run-token-copy-bridge`: recent-copy / retrieval signal
  - `run-token-skip-bridge`: gapped skip-context signal
- in the current workspace the original FineWeb train shards are missing, so these were run on a replay root synthesized from the saved `fineweb_val_tokens_000000_2026-03-28.npy` stream:
  - first `N` tokens used as pseudo-train memory
  - next held-out slice used for tune/eval
  - these are replay experiments, not leaderboard claims
- replay results:
  - copy, `500k / 2048`: base `8.3544`, heuristic `8.2681`, direct `8.2751`, oracle `7.9311`
  - copy, `1M / 4096`: base `7.9849`, heuristic `7.9653`, direct `7.9684`, oracle `7.7994`
  - copy, `2M / 8192`: base `7.3646`, heuristic `7.3534`, direct `7.3571`, oracle `7.1952`
  - skip, `500k / 2048`: base `11.0991`, heuristic `10.8726`, direct `10.8676`, oracle `10.3864`
  - skip, `1M / 4096`: base `11.0318`, heuristic `10.8539`, direct `10.8312`, oracle `10.4536`
  - skip, `2M / 8192`: base `10.7188`, heuristic `10.4626`, direct `10.4362`, oracle `10.0066`
- the scaling pattern is informative:
  - copy is real and audited, but its gain shrinks as packed memory gets stronger
  - skip gets stronger as packed memory grows, and is currently the better orthogonal channel
- audited replay checkpoints so far:
  - copy, `500k / 2048`: passes normalization, repeatability, future suffix invariance, answer mask invariance, prefix truncation parity, stream rechunk parity, and gold-logprob consistency
  - copy, `2M / 8192`: passes the same suite
  - skip, `500k / 2048`: passes the same suite
- the state change is simple:
  - posterior-only gates still collapse
  - orthogonal causal channels do not

The first chunk-shape result is also useful:

- the built-in `length-peek` cheat passes normalization, repeatability, future-suffix invariance, answer-mask invariance, and gold-logprob consistency
- it fails `prefix_truncation_parity` and `stream_rechunk_parity`
- the built-in `boundary-double-update` cheat passes `prefix_truncation_parity` and fails only `stream_rechunk_parity`
- while adding `stream_rechunk_parity`, Chronohorn found and fixed a real bug in its own packed-memory runtime: chunk-boundary context had been resetting
- the real packed-memory FineWeb runner now passes both `prefix_truncation_parity` and `stream_rechunk_parity`

That is the right foundation for the next generation of experiments.
