"""RETIRED (2026-07-09): superseded by the polyhash family package.

ShardedDataset, load_val, and the training loop that lived here were promoted
into the package during the family-isolation refactor (ae3a74d) and now live at:
    chronohorn/families/polyhash/models/hash_embed_model.py   (HashEmbedModel)
    chronohorn/families/polyhash/training/train_polyhash.py    (its own ShardedDataset)

This standalone copy was a stale duplicate (it imported the package model but
re-defined its own dataloader). Train through the family runtime instead. Kept
as a loud redirect rather than silently running divergent code.
"""
import sys


def main() -> int:
    sys.stderr.write(__doc__ + "\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
