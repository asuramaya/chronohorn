# Attack Loop

`run_attack_loop.zsh` is the long-running `Chronohorn` worker loop.

It operationalizes the four-track build path:

1. `oracle`
- refresh cleaned oracle summaries from BLINX attack output

2. `compressor`
- run real audited `match+skip` bundles on the active shard root

3. `bridge`
- sweep `candidate_k`
- compare deeper `match_depth`
- periodically run a heavier train-budget probe

4. `audit`
- use `run-token-matchskip-bundle-json`, so every promoted result carries the full legality bundle

## Default Cycle

Each cycle currently runs these workers in parallel:

- `matchskip_k12`
- `matchskip_k16`
- `matchskip_k20`
- `matchskip_depth12_k16`

And every `N` cycles, a heavier scale worker:

- `matchskip_heavy_k16`

The script keeps:

- `summary.tsv`
- `latest_cycle.txt`
- `best.txt`
- `best.json`

under `chronohorn/out/attack_loops/<run_id>/`.

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

## Intent

The loop is not just a benchmark queue.
It is a standing attack on the remaining oracle gap:

- keep the compressor causal
- keep the oracle offline
- make the bridge fight for legal gain
- keep every promoted number audited
