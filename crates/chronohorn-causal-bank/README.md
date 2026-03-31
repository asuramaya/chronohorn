# chronohorn-causal-bank

Promoted causal-bank family runtime for `Chronohorn`.

This crate owns the live family-specific Rust path:

- causal-bank checkpoint replay
- exact expert helpers
- exact-ngram checkpoint probes
- token-level oracle support used offline
- ranked-teacher loaders
- n-gram artifact builders and bulk evaluation

The root `chronohorn` crate re-exports this crate as `chronohorn::causal_bank`.
