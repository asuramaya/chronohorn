# Chronohorn Data Home

This directory is the canonical home for Chronohorn dataset resolution.

Use it for three things:

- `roots.json`: optional local alias overrides
- `roots/`: stable local storage for dataset roots when you intentionally keep them
- provenance: a predictable place that commands can report in bundles and audits

Chronohorn ships with built-in aliases for:

- `@replay`
- `@local-code`
- `@fineweb`

Those built-ins are only defaults. A local `roots.json` can override them.

The missing FineWeb case that motivated this layer was a temp-backed checkout of `parameter-golf/data` under `/private/tmp`. This directory exists so future roots have a stable, explicit home instead.

## Important Distinction

The `openai/parameter-golf` GitHub repo does not contain the shard files directly.
It contains the download helpers under `data/`, notably:

- `data/cached_challenge_fineweb.py`
- `data/README.md`

So the neat workflow is:

1. use the official downloader to materialize a real shard root locally
2. store that root somewhere stable, preferably under `chronohorn/data/roots/`
3. point `chronohorn/data/roots.json` at it
4. let Chronohorn resolve it through `@fineweb` or another alias

## Recommended Layout

Example stable local layout:

- `chronohorn/data/roots/fineweb10B_sp1024/`
- `chronohorn/data/roots/replay_root/`
- `chronohorn/data/roots.json`

Until a real shard root exists, `@fineweb` is only a placeholder alias and will usually resolve to a blocked legacy path.
