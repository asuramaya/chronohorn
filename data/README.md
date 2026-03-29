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
