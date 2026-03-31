# chronohorn-runtime

This crate is the Rust runtime-side export loader and verifier.

Current scope:

- load and inspect typed `opc-export` bundle metadata
- validate manifest/index/checksum consistency
- load learned-state tensor blobs for inventory/probe validation
- verify Python-written tensor probe notes against the actual exported blobs
- become the eventual home for deterministic checkpoint replay on top of the same bundle contract

This crate intentionally starts narrow. Real replay logic lands on top of the
same typed export bundle contract so `Chronohorn` can optimize execution,
replay, and artifact throughput without owning external audit authority.
