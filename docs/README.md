# Docs Guide

This directory has both live documentation and historical design material.

## Canonical Live Docs

Start here when orienting to the current repo:

- [REPO_BOUNDARY.md](./REPO_BOUNDARY.md): what belongs in `decepticons`, `chronohorn`, and outside the runtime path
- [STACK.md](./STACK.md): current promoted stack and runtime layers
- [FLEET.md](./FLEET.md): fleet/runtime/control-plane behavior
- [REPO_HYGIENE_PLAN.md](./REPO_HYGIENE_PLAN.md): cleanup phases for making the repo easier to navigate
- [ARCHIVE.md](./ARCHIVE.md): what has been intentionally retired or demoted

## Operational State

The repo’s tracked state surfaces live under [`state/`](../state/README.md), not
under `docs/`.

If you need live runtime truth, prefer:

- the DB in `out/chronohorn.db`
- the active drain/runtime logs in `out/`
- the tracked state snapshot files in `state/`

## Historical Docs

These are not the primary source of current operational truth:

- [FRONTIER_QUEUE.md](./FRONTIER_QUEUE.md)
- [GPU_ROADMAP.md](./GPU_ROADMAP.md)
- everything under [archive/](./archive)
- planning material under [superpowers/](./superpowers)

Those files are still useful for context, but they should be read as historical
or planning material unless another live doc explicitly promotes them.

## Hygiene Direction

The intended direction is:

- keep `docs/` for canonical live docs and clearly marked historical material
- keep `state/` for tracked runtime snapshot/handoff surfaces
- keep `out/` for generated runtime output only
- keep `manifests/` for live queue definitions, with archive separation handled there
