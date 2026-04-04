"""Export surface has moved to ``chronohorn.families.causal_bank.export``.

This shim exists for backward compatibility.
"""
from __future__ import annotations

import importlib
import sys

_REDIRECT = {}
for _name in ("abi", "bundle", "cli", "schema"):
    _REDIRECT[_name] = f"chronohorn.families.causal_bank.export.{_name}"

for _old_name, _new_path in _REDIRECT.items():
    _old_path = f"chronohorn.export.{_old_name}"
    if _old_path not in sys.modules:
        try:
            _mod = importlib.import_module(_new_path)
            sys.modules[_old_path] = _mod
        except ImportError:
            pass
