# Changelog

All notable changes to chronohorn will be documented here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/), and the project tries
to follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `chronohorn.train.async_checkpoint` — background training-state writer that
  overlaps the periodic-checkpoint disk write with training (pipeline overlap).
  Snapshots the resume point synchronously (a consistent CPU clone) then writes
  it off-thread, so the loop no longer stalls on `torch.save`. Covenant-safe by
  construction: the writer only reads a snapshot and does pure I/O — it never
  touches the training RNG, optimizer, or data stream, so the trajectory and
  resume point are unchanged. Opt-in via `CHRONOHORN_ASYNC_CHECKPOINT=1` (default
  off = byte-identical to the synchronous path). prev-then-new ordering preserved
  so a crash never drops the last resume point. 8-test covenant battery.
- `chronohorn.data.build_pentad_shards` — the multi-modal (pentad) shard builder
  promoted from `data/staging/pentad_shard.py`. `write_interleaved_shards()` is
  the parameterized, tested core (round-robin fold interleave + held-out per-fold
  val split); `load_folds()` / `byte_coverage()` wrap the enwik9 + raw sources.
- `val_large` enwik8 split `(95_000_000, 4_194_304)` — 8x `val` (2048 slots of
  2048), same held-out region. The pool that makes deep-bucket effctx SEMs
  honest: single-seed shallow buckets over the 256-slot `val` had seed_spread
  0.27-0.38 (never measurements). Kept SEPARATE from `val` so prior ledger
  numbers stay comparable. Shard: `data/roots/diet_text/enwik_val_large_000000.bin`
  (gitignored; regenerable via `write_byte_shard(path, load_split("val_large"))`).
- `chronohorn.eval.knn_datastore` — the census state-kNN eval promoted from the
  `experiments/harnesses/knn_*.py` scripts to package architecture, composing
  the decepticons kernel organs (`LinearStateStreamer` + `StateKNNMemory`). Store
  size and the RAM store tier (`--store-device cpu`) are config; reproduces the
  former harness to the 4th decimal. `--states-mode {stream,windowed}` and
  `--arm pca128 jl128 ...` fold in the last harness (`knn_sweep.py`): windowed
  per-forward substrate states and the key-transform sweep. `--seeds 0 1 2`
  (`run_seeds`) reports per-seed + pooled mean/SEM across query draws — honest
  error bars for the scaling figure (sequential; the eval is GPU-bound). Shared
  eval helpers in `chronohorn.eval.harness_util`.
- RAM shard cache in `train/token_shard_dataset.py` — bounded LRU cache of
  decoded shards, opt-in via `CHRONOHORN_SHARD_CACHE_BYTES` (default `0` =
  unchanged). Removes the round-robin disk re-read when the dataset fits in RAM;
  hard byte budget with LRU eviction so tight cgroup/k8s jobs can't OOM.
- `CONTRIBUTING.md` and `MANIFEST.in` for parity with the decepticons repo.
- `scripts/release.sh` — one-command bump → tag → push.
- Footer version chip on the site.
- GitHub repo description, homepage URL, and topics aligned with canonical
  copy.

### Changed
- Retired stale loose scripts to redirects: `scripts/train_hash_embed.py` (its
  ShardedDataset/loop already live in `families/polyhash/training/`) and
  `data/staging/pentad_shard.py` (folded into `data.build_pentad_shards`).
- `release.yml` rebuilt on the decepticons template: tag-vs-pyproject
  version check, smoke-install of the wheel, automatic GitHub Release
  creation with CHANGELOG-extracted notes and dist artifacts attached.
- `decepticons` dep bumped to `>=0.1.3`.
- Cross-repo lint hygiene: ignore-list additions in `pyproject.toml` for
  established try/except/pass patterns; real fix for unused `os` import in
  `observe/serve.py` and try/except/pass blocks in `runtime.py`.

## [0.1.1] - 2026-05-01

### Added
- GitHub Pages site at [chronohorn.com](https://chronohorn.com/) with deploy
  workflow. Crystal motifs, animated hero portrait, sibling-project panel
  pairing chronohorn with decepticons.
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

[Unreleased]: https://github.com/asuramaya/chronohorn/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/asuramaya/chronohorn/releases/tag/v0.1.1
[0.1.0]: https://github.com/asuramaya/chronohorn/releases/tag/v0.1.0
