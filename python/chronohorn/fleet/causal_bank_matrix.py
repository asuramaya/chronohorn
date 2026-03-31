from __future__ import annotations

from typing import Sequence

from .family_matrix import main as emit_family_matrix_main


def main(argv: Sequence[str] | None = None) -> int:
    args = ["--family", "causal-bank", *(list(argv or []))]
    return emit_family_matrix_main(args)
