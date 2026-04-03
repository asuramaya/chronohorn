"""Shared constants for the PolyHash family."""
from __future__ import annotations

POLYHASH_FAMILY_ID = "polyhash"

# Architecture versions that have been used in experiments.
KNOWN_VERSIONS = ("v1", "v2", "v3", "v4", "v5", "v6", "v7")

# Default config fields common across all PolyHash versions.
DEFAULT_NUM_TABLES = 8
DEFAULT_BUCKETS_PER_TABLE = 65536
DEFAULT_EMBED_PER_TABLE = 16
DEFAULT_HIDDEN_DIM = 512
DEFAULT_NUM_LAYERS = 2
DEFAULT_CONV_KERNEL = 8
DEFAULT_SCAN_DIM = 256
DEFAULT_MATCH_OFFSETS = (1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32)

# Config keys that are always extracted for summaries.
SUMMARY_KEYS = (
    "arch_version",
    "scan_dim",
    "num_tables",
    "buckets_per_table",
    "hidden_dim",
    "num_layers",
    "activation",
    "conv_kernel",
    "match_offsets",
)

# SAM-related keys (v7+).
SAM_SUMMARY_KEYS = (
    "sam_enabled",
    "sam_buckets",
    "sam_embed_dim",
    "sam_heads",
    "sam_quant_bits",
    "sam_straight_through",
    "sam_soft_temp",
)
