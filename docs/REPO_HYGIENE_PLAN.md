# Repo Hygiene Plan

This is the current cleanup plan for making the repo easier to navigate without
breaking live queues or active runtime tooling.

## Phase 1: Clarify The Surface

Goal: make the existing layout understandable before moving anything.

Done in the current pass:

- add [README.md](../README.md) links to repo-organization guides
- add [docs/README.md](./README.md) to separate live docs from historical material
- add [manifests/README.md](../manifests/README.md) to explain durable vs generated queue files
- add [state/README.md](../state/README.md) to define tracked snapshot files
- add [scripts/README.md](../scripts/README.md) to explain wrapper vs maintenance scripts
- remove misleading claims about a top-level `chronohorn.export` package from:
  - [python/README.md](../python/README.md)
  - [python/chronohorn/__init__.py](../python/chronohorn/__init__.py)
- document the repo-root package shim in [chronohorn/__init__.py](../chronohorn/__init__.py)
- remove redundant files:
  - [tests/__init__.py](../tests/__init__.py)
  - [manifests/.gitkeep](../manifests/.gitkeep)

## Phase 2: Stabilize Truth Surfaces

Goal: stop splitting operational truth across too many files.

Actions:

1. Decide whether `state/` is the canonical tracked snapshot surface.
2. If yes:
   - keep `frontier_status.json`
   - keep `NEXT_INSTANCE_HANDOFF.md`
   - keep `next_instance_handoff.json`
   - refresh them in place
3. If no:
   - move session handoffs under `out/`
   - stop referencing `state/frontier_status.json` from live docs

Files involved:

- [state/frontier_status.json](../state/frontier_status.json)
- [state/NEXT_INSTANCE_HANDOFF.md](../state/NEXT_INSTANCE_HANDOFF.md)
- [state/next_instance_handoff.json](../state/next_instance_handoff.json)
- [docs/FLEET.md](./FLEET.md)
- [docs/FRONTIER_QUEUE.md](./FRONTIER_QUEUE.md)

## Phase 3: Split Live vs Historical Queue Material

Goal: make `manifests/` readable without breaking live drains.

Actions after active queues are rotated:

1. Move historical manifests under one archive tree, for example:
   - `manifests/archive/2026-04-05/...`
2. Split current live manifests into:
   - `manifests/regimes/`
   - `manifests/generated/`
3. Update docs and tools to treat generated manifests as operational state, not long-term API.

Do not do this while current daemons still reference live flat paths.

Files involved:

- [manifests/](../manifests)
- [manifests.archive-2026-04-05](/Users/asuramaya/Code/carving_machine_v3/chronohorn/manifests.archive-2026-04-05)

## Phase 4: Reduce Package And CLI Ambiguity

Goal: make the code layout match what docs claim.

Actions:

1. Decide whether the repo-root `chronohorn/` shim stays.
2. If it stays, document it as dev-only compatibility.
3. If it goes, rely on editable install and `PYTHONPATH=python`.
4. Normalize script wrappers vs real package entrypoints.
5. Decide whether the promoted Rust operator CLI is the root shell or `crates/chronohorn-cli`.

Files involved:

- [chronohorn/__init__.py](../chronohorn/__init__.py)
- [chronohorn/__main__.py](../chronohorn/__main__.py)
- [python/chronohorn/cli.py](../python/chronohorn/cli.py)
- [python/chronohorn/_entrypoints.py](../python/chronohorn/_entrypoints.py)
- [scripts/dispatch_experiment.py](../scripts/dispatch_experiment.py)
- [scripts/train_polyhash.py](../scripts/train_polyhash.py)
- [scripts/backfill_db.py](../scripts/backfill_db.py)
- [src/main.rs](../src/main.rs)
- [crates/chronohorn-cli/src/main.rs](../crates/chronohorn-cli/src/main.rs)

## Phase 5: Archive Historical Model And Wrapper Material

Goal: stop mixing promoted code with experiment archaeology.

Actions:

1. Move superseded polyhash model versions into an archive subtree.
2. Move old planning docs and stale queue docs into clearer historical locations.
3. Keep promoted package roots focused on active code only.

Files involved:

- [python/chronohorn/families/polyhash/models](/Users/asuramaya/Code/carving_machine_v3/chronohorn/python/chronohorn/families/polyhash/models)
- [src/archive](../src/archive)
- [docs/FRONTIER_QUEUE.md](./FRONTIER_QUEUE.md)
- [docs/GPU_ROADMAP.md](./GPU_ROADMAP.md)
- [docs/superpowers/plans](/Users/asuramaya/Code/carving_machine_v3/chronohorn/docs/superpowers/plans)

## Phase 6: Reorganize Tests By Subsystem

Goal: make the test tree navigable.

Suggested target layout:

- `tests/db/`
- `tests/fleet/`
- `tests/observe/`
- `tests/mcp/`
- `tests/families/`
- `tests/engine/`

This is valuable, but it is intentionally later because it touches many imports
and test discovery paths at once.

