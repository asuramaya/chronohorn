from __future__ import annotations

import importlib


def import_symbol(module_name: str, attr: str):
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ImportError(f"Could not resolve {attr!r} from {module_name!r}.") from exc
