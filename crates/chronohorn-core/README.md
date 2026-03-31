# chronohorn-core

Generic Rust execution infrastructure for `Chronohorn`.

This crate owns the family-neutral runtime units:

- audit
- bridge doctrine
- checkpoint and tensor loading
- data-root and token-shard loading
- protocol traits
- process runtime configuration

Promoted family runtimes should depend on this crate instead of reaching back
into the root `chronohorn` crate.
