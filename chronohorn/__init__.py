"""Repository-root development shim for the real ``python/chronohorn`` package.

Editable installs and the published package resolve from ``python/`` via
``pyproject.toml``. This shim only exists so ``python -m chronohorn`` works
cleanly from the repo root without installation.
"""

from __future__ import annotations

from importlib import util
from pathlib import Path

_SHIM_DIR = Path(__file__).resolve().parent
_REAL_PACKAGE_DIR = _SHIM_DIR.parent / "python" / "chronohorn"
__path__ = [str(_SHIM_DIR), str(_REAL_PACKAGE_DIR)]

_real_init = _REAL_PACKAGE_DIR / "__init__.py"
_spec = util.spec_from_file_location("_chronohorn_real_init", _real_init)
if _spec is None or _spec.loader is None:
    raise ImportError(f"unable to load Chronohorn package from {_real_init}")
_module = util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

__doc__ = _module.__doc__
__all__ = list(getattr(_module, "__all__", []))
for _name in __all__:
    globals()[_name] = getattr(_module, _name)
