# State

This directory contains tracked runtime snapshot surfaces.

It is intentionally separate from `out/`:

- `state/` holds small, human-readable or machine-readable tracked snapshots
- `out/` holds generated runtime output, logs, DB files, results, and launch records

## Files

- `frontier_status.json`
  - current tracked frontier/runtime snapshot for humans and docs
- `NEXT_INSTANCE_HANDOFF.md`
  - current human-readable handoff for the next coding instance
- `next_instance_handoff.json`
  - machine-readable version of the same handoff

## Policy

- these files are refreshed in place
- they are not historical logs
- if you need history, use git or archive the snapshot elsewhere

If a snapshot is purely local or ephemeral, it should go under `out/`, not here.

