# Chronohorn Crates

This directory holds the scoped Rust crates that support the Chronohorn export,
runtime, and CLI boundary.

Current crates:

- `chronohorn-core`
- `chronohorn-causal-bank`
- `chronohorn-schema`
- `chronohorn-runtime`
- `chronohorn-cli`

Planned later:

- `chronohorn-compiler`

`Chronohorn` remains the main system repo and the place where export bundles
become fast replay, compilation, and scoring.

Current serious split:

- `chronohorn-core`
  - generic runtime infrastructure shared across families
- `chronohorn-causal-bank`
  - promoted causal-bank family runtime
- `chronohorn-schema`
  - typed `opc-export` manifest and learned-state index types
- `chronohorn-runtime`
  - export bundle inspection, blob loading, probe verification, and canonical export-bundle loading for Rust consumers
- `chronohorn-cli`
  - thin Rust CLI over runtime inspection
  - installed binary name: `chronohorn-cli`
  - intentionally separate from the Python `chronohorn` command

The root Rust runtime now acts as the public shell and archive host. Promoted
generic and family-specific code should land in these crates rather than
growing back into a flat root source tree.

External audit and evidence packaging are not staged here. That work belongs in
`heinrich`, not in a parallel Rust self-audit crate inside `chronohorn`.
