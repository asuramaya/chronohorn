"""Model implementations have moved to their family packages.

Polyhash models: ``chronohorn.families.polyhash.models.*``
Causal-bank models: ``chronohorn.families.causal_bank.models.*``

These shims exist for backward compatibility with saved configs, manifests,
and remote containers that reference the old import paths.
"""
from __future__ import annotations

import importlib
import sys

_REDIRECT = {}

for _name in (
    "polyhash_v2", "polyhash_v3", "polyhash_v4", "polyhash_v5",
    "polyhash_v6", "polyhash_v7", "polyhash_v8", "polyhash_v8h",
    "polyhash_v8m", "polyhash_v10", "polyhash_v11", "polyhash_v12",
    "polyhash_model", "hash_embed_model", "ngram_table",
):
    _REDIRECT[_name] = f"chronohorn.families.polyhash.models.{_name}"

for _name in (
    "causal_bank_core", "causal_bank_mlx", "causal_bank_torch",
    "readouts_mlx", "readouts_torch", "quantize", "common",
):
    _REDIRECT[_name] = f"chronohorn.families.causal_bank.models.{_name}"

# Eagerly register redirects so `from chronohorn.models.X import Y` works
for _old_name, _new_path in _REDIRECT.items():
    _old_path = f"chronohorn.models.{_old_name}"
    if _old_path not in sys.modules:
        try:
            _mod = importlib.import_module(_new_path)
            sys.modules[_old_path] = _mod
        except ImportError:
            pass  # family not installed — skip silently

__all__: list[str] = []
