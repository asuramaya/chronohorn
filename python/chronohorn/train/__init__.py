"""Chronohorn training surface.

Shared training infra lives here (datasets, runtime, CLI).
Family-specific training code lives in ``chronohorn.families.<name>.training.*``.
"""
from __future__ import annotations

from .cli import main

__all__ = ["main"]
