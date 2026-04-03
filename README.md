# Chronohorn

Architecture search runtime for next-token prediction models.

<p align="center">
  <img src="docs/chronohorn.jpg" alt="Chronohorn" width="520">
</p>

## What It Does

- **Tracks experiments** from any model family — ingests result JSONs, stores in SQLite, detects illegal runs automatically
- **Analyzes learning curves** — saturation detection, power-law forecasting, frontier ranking across architectures
- **Manages fleet dispatch** — manifest-driven job scheduling, SSH result pull-back, auto-deepening of promising runs

Chronohorn is family-agnostic. You plug in transformers, SSMs, hash-embedding models, or anything else by implementing a simple adapter protocol. The runtime handles DB storage, learning curves, frontier analysis, fleet management, and visualization.

## Quick Start

```bash
# Install
pip install -e .

# Ingest results and view dashboard
chronohorn observe serve --result-dir out/results

# Full daemon: drain + fleet probe + viz + MCP
chronohorn runtime --manifest manifests/my_scan.jsonl

# CLI help
chronohorn --help
```

## MCP Integration

Chronohorn exposes 27 tools via the Model Context Protocol for AI-assisted architecture search. Tools cover experiment querying, frontier analysis, fleet control, saturation detection, learning curve comparison, and manifest management. Configure in your MCP client settings or run `chronohorn mcp` for stdio transport. See `.mcp.json` for a Claude Code configuration example.

## Adding a Model Family

Create a package at `python/chronohorn/families/<name>/` implementing the `FamilyTrainingAdapter` protocol. The registry auto-discovers it — no manual registration. See `CLAUDE.md` for the full protocol reference and conventions.

## Current Results

| Family | Best bpb | Steps | Notes |
|--------|----------|-------|-------|
| causal-bank | 1.909 | 5K | hybrid patch decoder, OPC kernel |
| polyhash | 1.978 | 10K | polysemy-inspired hash embeddings |

165+ experiments tracked across two families. Frontier analysis, saturation detection, and auto-deepening run continuously.

## License

MIT — see [LICENSE](LICENSE).
