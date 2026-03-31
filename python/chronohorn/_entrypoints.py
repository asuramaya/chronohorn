from __future__ import annotations

import importlib
import importlib.util
from typing import Sequence


def dispatch_module(module_name: str, argv: Sequence[str], *, require_installed: bool = True) -> int:
    if require_installed and importlib.util.find_spec(module_name) is None:
        raise SystemExit(f"entrypoint module {module_name!r} is not installed")
    module = importlib.import_module(module_name)
    main = getattr(module, "main", None)
    if main is None:
        raise SystemExit(f"entrypoint module {module_name!r} does not expose main()")
    result = main(list(argv))
    return 0 if result is None else int(result)
