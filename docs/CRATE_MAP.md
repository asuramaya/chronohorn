# Crate Map

The Rust side of `Chronohorn` now has a deliberate split.

## Current crates

- `chronohorn-core`
  - generic runtime infrastructure
  - bridge doctrine, checkpoint loading, data-root handling, runtime checks, protocol traits, and process runtime setup
- `chronohorn-causal-bank`
  - promoted causal-bank replay and artifact family
  - checkpoint replay, exact experts, offline oracle helpers, ranked-teacher loaders, and n-gram artifact builders
- `chronohorn-schema`
  - typed `opc-export` manifest and learned-state index structures
  - checksum sidecar types
- `chronohorn-runtime`
  - export bundle inspection
  - manifest/index/checksum validation
  - typed learned-state tensor inventory loading
  - runtime replay-prep load plans
  - tensor-probe verification against bundle notes
  - canonical export-bundle manifest/index/path loading for Rust consumers
- `chronohorn-cli`
  - thin CLI over ABI/runtime inspection
  - installed binary name: `chronohorn-cli`
  - manifest, inventory, replay-prep, tensor-probe verification, and export-bundle replay probe commands

## Root crate

The root `chronohorn` crate is now the public shell over:

- `chronohorn-core`
- `chronohorn-causal-bank`
- `chronohorn-runtime`
- `chronohorn-schema`

Within the root crate, the promoted family structure is now explicit:

- `chronohorn::causal_bank`
  - re-export of the live causal-bank family crate
- `chronohorn::archive`
  - retained historical bridge and exploratory families

The root crate keeps the public command surface and archive modules, but the
promoted generic and family-specific runtime code now lives in scoped crates.
The root shell itself is also split by responsibility:

- `src/main.rs`
  - thin top-level binary entry point
  - dispatches to shell modules
- `src/shell_core.rs`
  - core inspection, doctrine, and audit utility commands
- `src/shell_causal_bank.rs`
  - promoted causal-bank command group
- `src/shell_archive.rs`
  - quarantined archive bridge-family commands
- `src/shell_usage.rs`
  - public and archive help text
- `src/shell_support.rs`
  - shared shell parsing and summary helpers

## Next staged crate

- `chronohorn-compiler`
  - packed-memory compile and repack surface

External audit and evidence packaging are intentionally not part of this crate
map. That role belongs to `heinrich`.
