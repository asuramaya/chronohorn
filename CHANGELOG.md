# Changelog

All notable changes to chronohorn will be documented here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/), and the project tries
to follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.1] - 2026-05-01

### Added
- GitHub Pages site at [chronohorn.com](https://chronohorn.com/) with deploy
  workflow. Naoto-Ohshima dream aesthetic — Blinx time-crystal motifs,
  NiGHTS paraloop animation, kindred panel pairing chronohorn with
  decepticons.
- PyPI release workflow using Trusted Publishing on tag push.
- Project URL metadata: documentation, issues, changelog.

### Changed
- `decepticons` dep is a normal PyPI pin (`>=0.1.1`) instead of a git URL,
  unblocking PyPI publishing.
- Canonical tagline unified across README, pyproject, CLAUDE.md, site, and
  the package docstring: "Family-agnostic experiment tracker and
  architecture-search runtime for predictive descendants."
- CLAUDE.md MCP tool count corrected (55 → 64) to match the live registry.
- Release workflow simplified to publish directly to PyPI (TestPyPI step
  removed — re-add when a TestPyPI Trusted Publisher is configured).

## [0.1.0] - withdrawn

Tagged but never published. The release workflow's TestPyPI step failed
because only the PyPI Trusted Publisher was configured. Replaced by 0.1.1.

- Family-agnostic experiment tracker (SQLite, single-writer discipline).
- 64-tool MCP surface for AI-agent integration.
- Multi-backend fleet dispatcher (CPU / Metal / CUDA) with planner placement.
- HTTP runtime dashboard.
- Family adapter protocol with auto-discovery.
- Built-in saturation, frontier, and forecast analysis.
- `causal-bank`, `polyhash`, and `transformer` shipped families.
