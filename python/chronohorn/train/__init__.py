"""Chronohorn training surface.

Shared training infra lives here (datasets, runtime, CLI).
Family-specific training code has moved to ``chronohorn.families.<name>.training.*``.

Import shims exist for backward compatibility with old paths.
"""
from __future__ import annotations

import importlib
import sys

from .cli import main

_REDIRECT = {}

for _name in (
    "causal_bank_training_primitives",
    "causal_bank_training_stack",
    "causal_bank_training_support",
    "train_causal_bank_mlx",
    "train_causal_bank_torch",
    "measure_backend_parity",
    "queue_static_bank_gate",
    "sweep_static_bank_gate",
):
    _REDIRECT[_name] = f"chronohorn.families.causal_bank.training.{_name}"

for _old_name, _new_path in _REDIRECT.items():
    _old_path = f"chronohorn.train.{_old_name}"
    if _old_path not in sys.modules:
        try:
            _mod = importlib.import_module(_new_path)
            sys.modules[_old_path] = _mod
        except ImportError:
            pass

__all__ = ["main"]
