"""Chronohorn Python descendant package.

This package is intentionally lightweight at import time.
Implementation-heavy training and export code lives in subpackages.
The promoted CLI entrypoints are:

- ``python -m chronohorn``
- ``python -m chronohorn.observe``
- ``python -m chronohorn.train``
- ``python -m chronohorn.export``
- ``python -m chronohorn.mcp_transport``
"""

from __future__ import annotations

from importlib import metadata

__all__ = ["__version__"]


def _detect_version() -> str:
    try:
        return metadata.version("chronohorn")
    except metadata.PackageNotFoundError:
        return "0.0.0"


__version__ = _detect_version()
