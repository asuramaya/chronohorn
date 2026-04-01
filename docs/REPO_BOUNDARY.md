# Repo Boundary

`Chronohorn` is the system repo built on top of `open-predictive-coder`.

This document describes the public boundary that matters right now:

1. `open-predictive-coder` is the shared kernel
2. `chronohorn` is the execution repo

External audit and evidence packaging are intentionally out of scope for this
repo and are not part of the public `Chronohorn` contract.

`heinrich` is the companion system for those externalized tasks. `chronohorn`
may now expose Heinrich-shaped observer and MCP surfaces for runtime state, but
those are explicitly runtime-control tools, not evidence-packaging or public
audit tools.

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
- runtime observation, run records, and MCP control surfaces
- closed-loop frontier control and action planning
- Rust replay
- packed-memory compilation
- held-out scoring
- fleet placement and snapshot discipline
- runtime-speed and artifact-economics work

This is where an `opc`-descended model becomes a real runnable system.

`Chronohorn` may keep internal runtime checks, parity probes, and replay guards.
Those checks exist to keep execution honest and reproducible. They are not a
replacement for external audit.

## Heinrich

`heinrich` is not the runtime system. It owns:

- external validation and evidence packaging
- claim/bundle compression for agent context
- public-facing audit posture

`chronohorn` should reuse Heinrich's shape where it helps:

- tiny store layer
- tiny stage pipeline
- tiny MCP transport

But it should not duplicate Heinrich's evidence bundle logic. The observer
pipeline in `chronohorn` is for tracked frontier state, manifests, launch
records, result summaries, budget forecasts, and runtime decisions.

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
- input projection schemes (`random`, `orthogonal_rows`, `split_banks`, `kernel_energy`)
- oscillatory scheduling algorithms (`logspace`, `mincorr_greedy`, `period_bucket_greedy`)

Note: `Chronohorn` owns the training CLI that threads OPC config knobs through
to the fleet scan system. When a new OPC knob is added (like `input_proj_scheme`),
the chronohorn side must wire it through `_training_spec()`, `_torch_train_command()`,
and the CLI argument parser. A consistency test in `tests/test_scan_consistency.py`
validates this wiring.

## Legacy Naming Note

Some `Chronohorn` files still use legacy implementation-detail names, for
example `crates/chronohorn-core/src/audit.rs`, `audit-*` command names, and
compatibility family-specific Rust implementation paths. In the current boundary model, those are
internal runtime-check or implementation labels. They are not a claim that
`Chronohorn` is its own independent auditor, and they are not the promoted
public family name.
