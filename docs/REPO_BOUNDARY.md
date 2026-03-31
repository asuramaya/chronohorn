# Repo Boundary

`Chronohorn` is the system repo built on top of `open-predictive-coder`.

This document describes the public boundary that matters right now:

1. `open-predictive-coder` is the shared kernel
2. `chronohorn` is the execution repo

External audit and evidence packaging are intentionally out of scope for this
repo and are not part of the public `Chronohorn` contract.

## OPC

`open-predictive-coder` is the shared Python kernel.

It owns:

- family-neutral predictive mechanisms
- reusable substrate and memory primitives
- routing and readout abstractions when they are genuinely generic
- export ABI helpers used by descendants
- backend-neutral causal-bank family metadata and deterministic substrate rules

It does not own:

- backend-specific MLX or Torch/NVIDIA model implementations
- replay/runtime optimization
- full-val scoring
- fleet execution
- external evidence or audit packaging

## Chronohorn

`Chronohorn` is the system repo.

It owns:

- backend-specific descendant implementations
- training and frontier search
- export bundle emission
- Rust replay
- packed-memory compilation
- held-out scoring
- fleet placement and snapshot discipline
- runtime-speed and artifact-economics work

This is where an `opc`-descended model becomes a real runnable system.

`Chronohorn` may keep internal runtime checks, parity probes, and replay guards.
Those checks exist to keep execution honest and reproducible. They are not a
replacement for external audit.

## Practical Rule

If a change is about one of these, it belongs in `Chronohorn`:

- faster replay
- smaller/faster snapshots
- CUDA or Metal training surfaces
- packed-table compilation
- export-to-runtime parity
- artifact throughput on real hardware

If a change is about one of these, it belongs in `opc`:

- reusable predictive abstractions
- kernel-level readouts and routing ideas
- family-neutral export helpers
- backend-neutral causal-bank config, variant, and substrate logic

## Legacy Naming Note

Some `Chronohorn` files still use legacy implementation-detail names, for
example `crates/chronohorn-core/src/audit.rs`, `audit-*` command names, and
compatibility family-specific Rust implementation paths. In the current boundary model, those are
internal runtime-check or implementation labels. They are not a claim that
`Chronohorn` is its own independent auditor, and they are not the promoted
public family name.
