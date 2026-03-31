# chronohorn-cli

This crate is the thin Rust CLI for the Chronohorn export/runtime path.

Installed binary name:

- `chronohorn-cli`

This is intentionally distinct from the Python `chronohorn` package entrypoint.

Current scope:

- print ABI constants
- inspect a full export bundle directory
- inspect a manifest directly when needed
- inspect learned-state inventory
- verify the Python-written tensor probe against the actual blob files
- run `probe-causal-bank-export-bundle` against a real exported causal-bank bundle

The generic bundle boundary is owned by `chronohorn-runtime`.
The causal-bank replay probe is the current family-specific layer on top of that
boundary.

This crate is intentionally thin and should stay that way.
