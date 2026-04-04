"""Causal-bank CLI/validation constants.

These are copies of the string-tuple constants defined in
``decepticons.causal_bank`` so that CLI argument parsers can
reference them without importing OPC at module load time.
"""

from __future__ import annotations

CAUSAL_BANK_VARIANTS: tuple[str, ...] = (
    "base",
    "linear_only",
    "local_only",
    "gated",
    "window4",
    "window16",
    "shared_embedding",
)

CAUSAL_BANK_OSCILLATORY_SCHEDULES: tuple[str, ...] = (
    "logspace",
    "mincorr_greedy",
    "period_bucket_greedy",
)

CAUSAL_BANK_READOUT_KINDS: tuple[str, ...] = (
    "mlp",
    "tied_recursive",
    "routed_sqrelu_experts",
    "recurrent",
)

CAUSAL_BANK_INPUT_PROJ_SCHEMES: tuple[str, ...] = (
    "random",
    "orthogonal_rows",
    "split_banks",
    "kernel_energy",
)
