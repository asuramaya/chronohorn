# Manifests

This directory holds live queue definitions used by Chronohorn’s fleet tooling.

## What Belongs Here

- durable named regimes that are part of the current search program
- currently active generated queue files that the running drain daemons depend on

Examples of durable regime manifests:

- `frontier_gated_delta.jsonl`
- `frontier_gated_delta_saturation.jsonl`
- `frontier_breakthrough_hunt.jsonl`

Examples of session-shaped generated manifests that are currently live:

- `frontier_toward_14_methodical.jsonl`
- `frontier_slop_home_fill.jsonl`

## Archive Boundary

Historical manifests already live in the sibling archive directory:

- `manifests.archive-2026-04-05/`

The next cleanup phase should fold archive material under a single
`manifests/archive/` tree, but this repo currently keeps the live manifest paths
stable to avoid breaking active daemons and existing launch records.

## Current Rule

Until the live queues are rotated:

- do not rename or move active manifest files casually
- treat files referenced by running drains as operational state
- add new generated queue files here only when they are actually live

