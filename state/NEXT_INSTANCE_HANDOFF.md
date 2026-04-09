# Next Instance Handoff

Snapshot taken: 2026-04-09 14:31:30 CDT

## Current Truth

- Branch: `main`
- Published base HEAD: `f987dec` (`feat: publish frontier tooling and repo guides`)
- Intended repo state after this handoff refresh is clean
- Active drain daemon:
  - pid file: `out/drain_14.pid`
  - live pid: `8770`
  - manifest: `manifests/frontier_toward_14_methodical.jsonl`
  - log: `out/drain_14.log`
- Important:
  - the drain truth fix is now committed in [drain.py](/Users/asuramaya/Code/carving_machine_v3/chronohorn/python/chronohorn/fleet/drain.py)
  - the expanded `cb-14` emitter/test/manifests are now committed
  - tracked repo organization guides now exist under [docs/README.md](/Users/asuramaya/Code/carving_machine_v3/chronohorn/docs/README.md), [manifests/README.md](/Users/asuramaya/Code/carving_machine_v3/chronohorn/manifests/README.md), [scripts/README.md](/Users/asuramaya/Code/carving_machine_v3/chronohorn/scripts/README.md), and [state/README.md](/Users/asuramaya/Code/carving_machine_v3/chronohorn/state/README.md)

## Frontier

### Best overall provisional frontier

- `cb-s8-experts-50k`: `1.7295840075 bpb`, `12.39 MiB int6`, `65510 tok/s`
- `cb-s12-base-50k`: `1.7542252121 bpb`, `9.99 MiB int6`, `58494 tok/s`
- `cb-s8-bands4-50k`: `1.7609855762 bpb`, `6.38 MiB int6`, `86924 tok/s`

### Best current scaling O(n) gated-delta rows

- `cb-delta-s12-h8-s32-50k-cos`: `1.8069302145 bpb`, `10.95 MiB int6`, `174108 tok/s`
- `cb-delta-s12-h8-s32-bands4-50k-cos`: `1.8074844278 bpb`, `13.34 MiB int6`, `184982 tok/s`
- `cb-14-delta-s12-h8-s32-split-50k-cos`: `1.8079578743 bpb`, `10.95 MiB int6`, `180018 tok/s`

### Trust state

- Result count: `97`
- Controlled legal results: `95`
- Admissible: `0`
- Provisional: `95` controlled legal, `97` total
- Metric state split:
  - `dataset_anchored`: `8`
  - `legacy_result_schema`: `87` controlled legal

Interpretation:
- The frontier is still soft because trust promotion has not happened yet.
- The old leaders are still mostly single-seed.
- Most newer causal-bank rows are still tagged `legacy_result_schema`.

## Live Queue

Active manifest:
- `manifests/frontier_toward_14_methodical.jsonl`

DB state for this manifest at snapshot:
- `24 completed`
- `7 pending`
- `3 running`

Running jobs:
- `cb-14-scan-s12-h4-s64-25k` on `slop-01`
  - pod: `ch-cb-14-scan-s12-h4-s64-25k-1ea49ccb-vglvv`
- `cb-14-scan-s12-h8-s32-bands4-seq512-25k` on `slop-02`
  - pod: `ch-cb-14-scan-s12-h8-s32-bands4-seq512-25k-00500b98-v4hz4`
- `cb-14-scan-s8-h8-s32-bands4-25k-home` on `slop-home`
  - pod: `ch-cb-14-scan-s8-h8-s32-bands4-25k-home-b43a9a35-nb6q4`

Useful monitors:
- `out/drain_14.log`
- `out/results/`
- `kubectl get pods -A -o wide | rg 'cb-14-|cb-delta-|cb-hunt-|cb-frontier-'`

Interpretation:
- The matrix has moved off the main delta tranche and into the matched scan tranche.
- The current comparison pressure is delta speed/quality vs slower scan baselines under scale and context stress.

## What We Learned

### Main architecture read

- `gated_delta` is the current lead substrate.
- `scan` is much slower than `gated_delta`, but not catastrophically worse on quality.
- The big gain is the substrate redesign itself, not readout-only tweaks stacked on top.

### Concrete comparisons

- `gated_delta` long-run leader:
  - `cb-14-delta-s12-h8-s32-split-50k-cos`: `1.80796 bpb`, `180018 tok/s`
- matched scan baselines:
  - `cb-14-scan-s12-h8-s32-bands4-25k`: `1.84423 bpb`, `34964 tok/s`
  - `cb-14-scan-s12-h8-s32-25k`: `1.85478 bpb`, `34622 tok/s`

Implication:
- delta is buying a very large speed advantage and still beating scan on the best completed rows.

### Mutation reads inside the current matrix

- `bands4` is no longer a major quality lever on top of primary `gated_delta`
  - it is mostly a speed tradeoff now
- `split_banks` is mixed
  - it helped the long `50k-cos` run
  - it lost at `seq512`
  - treat it as a duration/optimization aid, not a clean universal memory fix
- `hemi2` is mildly promising
  - `cb-14-delta-s12-hemi2-fast025-25k`: `1.87927 bpb`
  - `cb-14-delta-s12-hemi2-fast050-25k`: `1.87961 bpb`
- `local4` is mildly promising and faster
  - `cb-14-delta-s12-local4-25k`: `1.88179 bpb`, `194708 tok/s`
- `local16` looks worse and slower
  - `cb-14-delta-s12-local16-25k`: `1.89163 bpb`, `145047 tok/s`
- head/state factorization is mostly flat so far
  - `cb-14-delta-s12-h4-s64-25k`: `1.88744 bpb`
  - `cb-14-delta-s12-h16-s16-25k`: `1.88688 bpb`
- patch hybrids are speed tricks, not frontier candidates in current form
  - `cb-14-delta-s12-p2hybrid-25k`: `1.93207 bpb`, `183627 tok/s`, `16.22 MiB int6`
  - `cb-14-delta-s12-p4hybrid-25k`: `1.91379 bpb`, `204751 tok/s`, `21.50 MiB int6`
  - both are over the `16 MiB` artifact budget

## Tooling / Control-Plane State

- The drain truth lag fix is committed and live:
  - jobs are now marked `completed` in SQLite as soon as executor state says they finished
  - result ingest remains a separate later step
- The active `cb-14` manifests are now tracked in-repo:
  - [frontier_toward_14_methodical.jsonl](/Users/asuramaya/Code/carving_machine_v3/chronohorn/manifests/frontier_toward_14_methodical.jsonl)
  - [frontier_slop_home_fill.jsonl](/Users/asuramaya/Code/carving_machine_v3/chronohorn/manifests/frontier_slop_home_fill.jsonl)
- Repo polish/docs are now tracked:
  - [docs/README.md](/Users/asuramaya/Code/carving_machine_v3/chronohorn/docs/README.md)
  - [docs/REPO_HYGIENE_PLAN.md](/Users/asuramaya/Code/carving_machine_v3/chronohorn/docs/REPO_HYGIENE_PLAN.md)
  - [scripts/export_excel_snapshot.py](/Users/asuramaya/Code/carving_machine_v3/chronohorn/scripts/export_excel_snapshot.py)

## History / Recent Commit Chain

- `f987dec` feat: publish frontier tooling and repo guides
- `c6f217a` fix: stabilize ruff ci surface
- `fa7ba39` feat: add gated-delta saturation slab
- `e4573ab` fix: harden gated-delta launch path
- `253cdf8` feat: queue gated-delta breakthrough slab
- `10222c4` feat: add causal-bank benchmark harness

## Files Worth Opening First

- [state/NEXT_INSTANCE_HANDOFF.md](/Users/asuramaya/Code/carving_machine_v3/chronohorn/state/NEXT_INSTANCE_HANDOFF.md)
- [state/next_instance_handoff.json](/Users/asuramaya/Code/carving_machine_v3/chronohorn/state/next_instance_handoff.json)
- [state/frontier_status.json](/Users/asuramaya/Code/carving_machine_v3/chronohorn/state/frontier_status.json)
- [manifests/frontier_toward_14_methodical.jsonl](/Users/asuramaya/Code/carving_machine_v3/chronohorn/manifests/frontier_toward_14_methodical.jsonl)
- [out/drain_14.log](/Users/asuramaya/Code/carving_machine_v3/chronohorn/out/drain_14.log)
- [out/results/cb-14-delta-s12-h8-s32-split-50k-cos.json](/Users/asuramaya/Code/carving_machine_v3/chronohorn/out/results/cb-14-delta-s12-h8-s32-split-50k-cos.json)
- [out/results/cb-14-scan-s12-h8-s32-bands4-25k.json](/Users/asuramaya/Code/carving_machine_v3/chronohorn/out/results/cb-14-scan-s12-h8-s32-bands4-25k.json)
- [out/results/cb-14-scan-s12-h8-s32-25k.json](/Users/asuramaya/Code/carving_machine_v3/chronohorn/out/results/cb-14-scan-s12-h8-s32-25k.json)

## Immediate Next Actions

1. Let the three live scan rows finish and refresh the tracked state snapshot again.
2. Compare the active scan tranche directly against the best completed delta rows on both quality and throughput.
3. Keep prioritizing:
   - primary `gated_delta`
   - `hemi2`
   - `local4`
   - matched `scan` comparisons as the slower baseline
4. Deprioritize:
   - `local16`
   - patch-hybrid rows as frontier candidates until they recover quality and fit under the artifact budget
