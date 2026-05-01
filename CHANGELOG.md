# Changelog

All notable changes to chronohorn will be documented here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/), and the project tries
to follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- GitHub Pages site at [chronohorn.com](https://chronohorn.com/) with deploy workflow.
- PyPI release workflow using Trusted Publishing (TestPyPI → PyPI on tag).
- Project URL metadata: documentation, issues, changelog.

### Changed
- `decepticons` dep is now a normal PyPI pin (`>=0.1.1`) instead of a git URL,
  unblocking PyPI publishing.
- Canonical tagline unified across README, pyproject, CLAUDE.md, site, and the
  package docstring: "Family-agnostic experiment tracker and architecture-search
  runtime for predictive descendants."
- CLAUDE.md MCP tool count corrected (55 → 64) to match the live registry.

## [0.1.0] - TBD

Initial public release.

- Family-agnostic experiment tracker (SQLite, single-writer discipline).
- 64-tool MCP surface for AI-agent integration.
- Multi-backend fleet dispatcher (CPU / Metal / CUDA) with planner placement.
- HTTP runtime dashboard.
- Family adapter protocol with auto-discovery.
- Built-in saturation, frontier, and forecast analysis.
- `causal-bank`, `polyhash`, and `transformer` shipped families.
