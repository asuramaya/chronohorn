# Scripts

This directory is a mixed surface today.

## Canonical Preference

When possible, prefer package CLIs:

- `chronohorn ...`
- `python -m chronohorn ...`

Those are the promoted entrypoints.

## What Is Here Today

- thin wrappers preserved for convenience
  - `dispatch_experiment.py`
  - `train_polyhash.py`
- maintenance / export utilities
  - `backfill_db.py`
  - `export_excel_snapshot.py`
- older one-off research or operational helpers

## Cleanup Direction

The intended direction is:

- keep true maintenance utilities together
- move thin wrappers to a clearly marked legacy/admin area or remove them if redundant
- avoid growing `scripts/` as a junk drawer when a package CLI or documented tool fits better

