from __future__ import annotations

import sys
from typing import Sequence

from .causal_bank_matrix import main as emit_causal_bank_matrix_main
from .dispatch import main as dispatch_main
from .queue import main as queue_main


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "queue":
        return queue_main(args[1:])
    if args and args[0] == "emit-causal-bank-matrix":
        return emit_causal_bank_matrix_main(args[1:])
    return dispatch_main(args)
