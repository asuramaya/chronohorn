"""Chronohorn training and short-probe surface.

The promoted entrypoints are:

- ``python -m chronohorn.train``
- ``python -m chronohorn train ...``

Import dataset helpers from concrete modules when needed:

- ``chronohorn.train.token_shard_dataset``
- ``chronohorn.train.token_shard_dataset_torch``

Formal training/model surfaces for the promoted causal-bank line live in:

- ``chronohorn.train.causal_bank_training_stack``
- ``chronohorn.train.causal_bank_training_primitives``
"""

from __future__ import annotations

from .cli import main

__all__ = ["main"]
