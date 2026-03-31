#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
CHRONOHORN_ROOT = SCRIPT_PATH.parents[1]
PYTHON_ROOT = CHRONOHORN_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from chronohorn.fleet.dispatch import main


if __name__ == "__main__":
    raise SystemExit(main())
