from __future__ import annotations

from pathlib import Path
import sys


def ensure_open_predictive_coder_importable() -> None:
    try:
        import open_predictive_coder  # noqa: F401
        return
    except ImportError:
        pass

    sibling_src = Path(__file__).resolve().parents[3] / "open-predictive-coder" / "src"
    if sibling_src.exists():
        sibling_str = str(sibling_src)
        if sibling_str not in sys.path:
            sys.path.insert(0, sibling_str)
        try:
            import open_predictive_coder  # noqa: F401
            return
        except ImportError:
            pass

    raise ImportError(
        "Chronohorn requires the open_predictive_coder package for backend-neutral kernel "
        "surfaces. Install the sibling opc repo or make open_predictive_coder importable."
    )
