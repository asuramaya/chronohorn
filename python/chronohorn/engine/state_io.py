from __future__ import annotations

from pathlib import Path

import numpy as np


def save_state_npz(path: Path, state: dict[str, object]) -> None:
    arrays = {name: np.array(value) for name, value in state.items()}
    np.savez(path, **arrays)
