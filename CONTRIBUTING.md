# Contributing

Thanks for your interest in `chronohorn`. This is an alpha-stage runtime —
the surface will change. Bug reports, small fixes, and discussions are
welcome; please file an issue before starting larger changes so we can talk
about scope and the runtime/kernel boundary.

## Quickstart for contributors

```bash
git clone https://github.com/asuramaya/chronohorn
cd chronohorn
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
pytest -v
```

For the model backends or the rust workspace:

```bash
pip install -e ".[torch]"        # torch families (causal-bank training, polyhash)
pip install -e ".[cuda]"         # adds mamba-ssm, causal-conv1d
pip install -e ".[metal]"        # Apple Silicon / MLX
cargo check --workspace          # rust replay path
```

## What belongs in chronohorn

`chronohorn` is the **runtime** — training, tracking, fleet, MCP, replay.
Reusable mechanisms (substrates, gating, routing, readouts) live in the
[`decepticons`](https://github.com/asuramaya/decepticons) kernel. Read
[`docs/REPO_BOUNDARY.md`](./docs/REPO_BOUNDARY.md) and the kernel-side
[`chronohorn_boundary.md`](https://github.com/asuramaya/decepticons/blob/main/docs/chronohorn_boundary.md)
before adding cross-cutting surfaces.

If a primitive could be described without naming a specific descendant family
and reused unchanged by more than one downstream — it probably belongs in
decepticons, not here.

If it's an experiment policy, scan regime, family-specific training recipe,
fleet placement rule, MCP tool, or anything that touches the SQLite DB or
result archive — it belongs here.

## Adding a new model family

Drop a package at `python/chronohorn/families/<name>/` implementing the
`FamilyTrainingAdapter` protocol. The registry auto-discovers it via
`pkgutil.iter_modules` — no manual registration. Concretely:

- [ ] `__init__.py` exports a `<UPPER_NAME>_TRAINING_ADAPTER` singleton
- [ ] `adapter.py` implements architecture aliases, illegal detection,
      config summaries, `infer_from_config()`, and training entrypoints
- [ ] family-specific code stays under `families/<name>/` — core infra
      (`db.py`, `mcp.py`, `runtime.py`, `observe/serve.py`) never imports
      family modules directly

See [`CLAUDE.md`](./CLAUDE.md) for the full developer guide.

## Style

- Python ≥ 3.11. Type-annotated.
- `ruff` is the only linter — config in `pyproject.toml`. `ruff check .` should
  pass before you push.
- Line length 120, but format is handled by ruff — don't fight it.
- All DB reads go through `_read()` / `_read_one()`; all DB writes through
  `_write()` / `_write_many()`. Single-writer discipline is non-negotiable.
- Silent `except Exception: pass` is forbidden — use `contextlib.suppress`
  or log to stderr.

## Tests

```bash
pytest -v                                  # full suite
pytest tests/test_drain.py -v              # fleet drain logic
pytest tests/test_engine_core.py -v        # measurement primitives
cargo test --workspace                     # rust replay
```

Backend-specific tests need the relevant extra:

```bash
pip install -e ".[torch]" && pytest tests/test_patch_readout_loss.py -v
```

## Pull requests

- One concern per PR. Architectural cleanups and bug fixes shouldn't ride together.
- Reference the issue you're closing.
- If your PR changes a public API or MCP tool surface, update `CHANGELOG.md`.
- A passing CI run is required.

## Releasing

Releases are one command. The script in [`scripts/release.sh`](./scripts/release.sh)
bumps the version everywhere, commits, tags, and pushes. The push triggers
[`.github/workflows/release.yml`](./.github/workflows/release.yml) which builds,
publishes to PyPI, and creates a GitHub Release — all automatic.

### Before tagging

While you work, log changes under `## [Unreleased]` in [`CHANGELOG.md`](./CHANGELOG.md).
This is what becomes the release notes for the next version. Group entries by
`### Added`, `### Changed`, `### Fixed`, `### Removed`.

### Cut a release

```bash
scripts/release.sh 0.1.2     # bump patch
scripts/release.sh 0.2.0     # bump minor
scripts/release.sh 1.0.0     # first stable
scripts/release.sh 1.1.0-rc1 # pre-release
```

The script:

1. Checks the working tree is clean and `main` is in sync with `origin/main`.
2. Bumps the version in `pyproject.toml`, `CHANGELOG.md` (moves `[Unreleased]`
   entries under a new `[X.Y.Z] - <today>` header), and `site/index.html`
   (footer version chip).
3. Shows the diff and asks for confirmation.
4. Commits as `chore(release): vX.Y.Z`, tags `vX.Y.Z`, pushes both.

### What happens after the push

Pushing the tag fires `release.yml`:

| Job | Does |
| --- | --- |
| `build` | Verifies the tag matches `pyproject.toml` version, builds sdist + wheel, runs `twine check`, smoke-imports the wheel in a fresh venv, uploads `dist/` as an artifact. |
| `publish` | Publishes to PyPI via OIDC trusted publishing. No tokens required — the `pypi` GitHub environment authorizes the run. |
| `github-release` | Extracts the matching section from `CHANGELOG.md`, creates a GitHub Release at `vX.Y.Z` with those notes, and attaches the wheel + sdist. |

In parallel, the push of the bump commit (which touched `site/index.html`)
fires [`.github/workflows/pages.yml`](./.github/workflows/pages.yml) and
redeploys <https://chronohorn.com> with the new version visible in the footer.

### Versioning rules (semver)

While `< 1.0.0`:
- `0.1.0 → 0.1.1` — bug fix only, no API change
- `0.1.x → 0.2.0` — breaking changes are allowed pre-1.0
- `0.x.x → 1.0.0` — first stable API. After this, breaking changes need a major bump.

### If something goes wrong

| Failure | Recovery |
| --- | --- |
| `build` fails on `twine check` | Fix locally, delete the bad tag (`git tag -d vX.Y.Z && git push origin :refs/tags/vX.Y.Z`), bump to the next patch, run `scripts/release.sh` again. |
| `publish` fails | Re-run the failed job from the GitHub Actions UI. The build artifact is still there. |
| Released a broken version | You cannot re-upload to PyPI. Yank with `twine yank chronohorn==X.Y.Z -m "reason"` and ship `X.Y.(Z+1)` with the fix. |

## Reporting bugs

Open an issue at <https://github.com/asuramaya/chronohorn/issues>. A minimal
reproduction (a `python` snippet, a manifest fragment, or a failing test) is
worth more than a long description.

## License

By contributing, you agree your contributions are licensed under the MIT
License — see [`LICENSE`](./LICENSE).
