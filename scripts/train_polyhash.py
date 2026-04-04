#!/usr/bin/env python3
"""Wrapper — the trainer moved to chronohorn.families.polyhash.training.train_polyhash."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
from chronohorn.families.polyhash.training.train_polyhash import main

if __name__ == "__main__":
    main()
