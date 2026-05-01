# Chronohorn

<p align="center">
  <img src="docs/chronohorn.jpg" alt="Chronohorn" width="520">
</p>

<p align="center">
  <a href="https://pypi.org/project/chronohorn/"><img alt="PyPI" src="https://img.shields.io/pypi/v/chronohorn.svg"></a>
  <a href="https://github.com/asuramaya/chronohorn/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/asuramaya/chronohorn/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/asuramaya/chronohorn/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue">
  <img alt="Status" src="https://img.shields.io/badge/status-alpha-orange">
</p>

<p align="center">
  <a href="https://chronohorn.com"><b>Website</b></a> ·
  <a href="https://github.com/asuramaya/chronohorn/blob/main/CLAUDE.md">Developer guide</a> ·
  <a href="https://github.com/asuramaya/chronohorn/tree/main/docs">Docs</a> ·
  <a href="https://github.com/asuramaya/chronohorn/blob/main/CHANGELOG.md">Changelog</a>
</p>

> Family-agnostic experiment tracker and architecture-search runtime for predictive descendants. 
> Built on the shared [`decepticons`](https://github.com/asuramaya/decepticons) kernel.


## What It Does

- **Tracks experiments** from any model family, stores them in SQLite, and keeps legality/trust state attached to results
- **Analyzes curves and frontiers** with saturation, marginal ranking, velocity, and ablation-board views
- **Runs the search loop** through manifest-driven fleet dispatch, drain, result pull-back, and auto-deepen/control surfaces
- **Exposes runtime state to agents** through 64 MCP tools, a terminal observer, and an HTTP runtime dashboard

Chronohorn is family-agnostic at the runtime layer. Family-specific mutation policy lives under `python/chronohorn/families/<name>/`; the shared mechanism layer stays in the [`decepticons`](https://github.com/asuramaya/decepticons) kernel.

## Quick Start

```bash
# Install from PyPI (decepticons kernel pulled automatically)
python3 -m pip install chronohorn

# Or monorepo dev install: shared kernel first, then runtime
python3 -m pip install -e ../decepticons
python3 -m pip install -e .

# Ingest results and view the observer/dashboard
chronohorn observe serve --result-dir out/results

# Emit a family-owned scan manifest
chronohorn fleet emit-family-matrix --family causal-bank --regime gated-retention

# Full daemon: drain + fleet probe + observer + MCP
chronohorn runtime --manifest manifests/frontier_gated_retention.jsonl

# CLI help
chronohorn --help
```

## MCP Integration

Chronohorn exposes a stateful MCP surface for experiment querying, frontier analysis, ablation tracking, fleet control, saturation detection, learning-curve comparison, and manifest/runtime management. The exact tool set changes with the runtime; the live registry is in [`python/chronohorn/mcp.py`](./python/chronohorn/mcp.py). Run `chronohorn mcp` for stdio transport. See [`.mcp.json`](./.mcp.json) for a client configuration example.

## Repo Boundary

The intended split is:

```text
decepticons -> chronohorn -> heinrich
kernel         runtime       evidence / audit
```

- `decepticons` owns reusable mechanisms, config validation, and export-friendly kernel surfaces
- `chronohorn` owns training, replay, scoring, scan emission, fleet execution, and runtime control
- `heinrich` is outside the runtime path and owns external evidence packaging

See [docs/REPO_BOUNDARY.md](./docs/REPO_BOUNDARY.md) and [docs/STACK.md](./docs/STACK.md) for the promoted boundary.

## Repo Guide

The repo has a few different kinds of material that matter for different reasons:

- [docs/README.md](./docs/README.md) points to canonical live docs vs historical docs
- [manifests/README.md](./manifests/README.md) explains named regimes, generated queue files, and archive intent
- [state/README.md](./state/README.md) explains the tracked runtime snapshot and handoff files
- [scripts/README.md](./scripts/README.md) explains which scripts are wrappers vs maintenance utilities

## Adding a Model Family

Create a package at `python/chronohorn/families/<name>/` implementing the `FamilyTrainingAdapter` protocol. The registry auto-discovers it — no manual registration. See `CLAUDE.md` for the full protocol reference and conventions.

## Current Focus

The active causal-bank search is organized around cheap O(n) architecture screening before promotion:

- `10k` rapid ablation lanes for mechanism screening
- scale/context survival rows aimed at pushing the frontier toward `1.0`
- primary learned-substrate experiments around `gated_delta`
- VRAM-tier-aware fleet placement so small CUDA rows can prefer the smallest sufficient GPU lane

Current manifests live under [`manifests/`](./manifests/), and current results/launch state live under [`out/results/`](./out/results/) and [`out/fleet/`](./out/fleet/).

## License

MIT — see [LICENSE](LICENSE).
