# Archive

This document marks the parts of `Chronohorn` that are still useful for archaeology, falsification, or historical context, but are not the promoted runtime stack.

## Why Keep Archive Code

Older lines still matter because they preserve:

- failed interfaces worth avoiding
- audit failures worth remembering
- bridge targets that may later survive in cleaner form
- exact numbers and wiring from past branches

Archive does not mean useless. It means not promoted.

## Promoted Stack

If you are trying to understand the current system, start here instead:

- [README.md](../README.md)
- [STACK.md](./STACK.md)
- [DOCTRINE.md](./DOCTRINE.md)
- [crates/chronohorn-core/src/runtime.rs](../crates/chronohorn-core/src/runtime.rs)
- [crates/README.md](../crates/README.md)

Current causal-bank runtime implementation details still live in:

- [crates/chronohorn-causal-bank/src/checkpoint.rs](../crates/chronohorn-causal-bank/src/checkpoint.rs)
- [crates/chronohorn-causal-bank/src/ngram_bulk.rs](../crates/chronohorn-causal-bank/src/ngram_bulk.rs)

## Archived Docs

- [archive/ATTACK_LOOP.md](./archive/ATTACK_LOOP.md): legacy `match+skip` loop
- [SCALER_FRAMEWORK.md](./SCALER_FRAMEWORK.md): parent-line interpretation, still useful
- [archive/C3_C7_NEXT.md](./archive/C3_C7_NEXT.md): future ranked-teacher direction, not the current artifact frontier
- [archive/CONKER_ABSORPTION_PLAN.md](./archive/CONKER_ABSORPTION_PLAN.md): historical migration record for the `Conker` absorption wave

## Archived Module Families

Early bridge families:

- `token_bridge.rs`
- `token_copy_bridge.rs`
- `token_skip_bridge.rs`
- `token_skipcopy_bridge.rs`
- `token_match_bridge.rs`
- `token_matchcopy_bridge.rs`
- `token_matchskip_bridge.rs`
- `token_matchskipcopy_bridge.rs`
- `token_word_bridge.rs`
- `token_column_bridge.rs`
- `token_decay_bridge.rs`

Older runtime probes:

- `packed_memory.rs`
- `token_ngram_checkpoint.rs`
- `src/archive/token_conker3.rs`
- `token_c3c7.rs`
- `byte_bridge.rs`
- `token_experiment_matrix.rs`

These files are still available because they capture real experiments and attack surfaces, but they should not be mistaken for the current stack.

## Reading Rule

If a file or command is not part of the checkpoint runtime, offline artifact builder, or audit path, treat it as archive until proven otherwise.
