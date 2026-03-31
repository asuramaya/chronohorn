# Legacy Attack Loop

`run_attack_loop.zsh` is the old long-running `Chronohorn` worker loop for the `match+skip` bridge family.

It is kept for archaeology and falsification, not as the promoted runtime path.

The current live stack is:

- checkpoint replay in [crates/chronohorn-causal-bank/src/checkpoint.rs](../crates/chronohorn-causal-bank/src/checkpoint.rs)
- offline artifact build in [crates/chronohorn-causal-bank/src/ngram_bulk.rs](../crates/chronohorn-causal-bank/src/ngram_bulk.rs)
- runtime configuration in [crates/chronohorn-core/src/runtime.rs](../crates/chronohorn-core/src/runtime.rs)
- audit in [crates/chronohorn-core/src/audit.rs](../crates/chronohorn-core/src/audit.rs)

## What This Loop Was For

The loop operationalized the older four-track bridge search:

1. `oracle`
2. `compressor`
3. `bridge`
4. `audit`

Specifically, it swept audited `match+skip` bundles on the active shard root.

## Historical Default Cycle

Each cycle used to run these workers in parallel:

- `matchskip_k12`
- `matchskip_k16`
- `matchskip_k20`
- `matchskip_depth12_k16`

And every `N` cycles, a heavier worker:

- `matchskip_heavy_k16`

Outputs were written under:

- `chronohorn/out/attack_loops/<run_id>/`

## Launch

Foreground:

```bash
zsh chronohorn/scripts/run_attack_loop.zsh --data-root @fineweb --cycles 1
```

Detached:

```bash
nohup zsh chronohorn/scripts/run_attack_loop.zsh \
  --data-root @fineweb \
  --sleep-sec 30 \
  > chronohorn/out/attack_loop.log 2>&1 &
```

## Current Reading

Use this loop only if you are:

- reproducing the old `match+skip` line
- testing audit regressions against legacy bridge code
- comparing newer runtime work against older archived searches

Do not treat it as the current `Chronohorn` frontier.
