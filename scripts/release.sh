#!/usr/bin/env bash
# scripts/release.sh — bump version everywhere, commit, tag, push.
#
# One-shot release flow. See CONTRIBUTING.md → Releasing for the full cycle.
#
# Usage:
#   scripts/release.sh X.Y.Z
#
# What it does, in order:
#   1. Sanity checks: clean main, in sync with origin
#   2. Updates version in:
#        - pyproject.toml
#        - CHANGELOG.md (moves [Unreleased] entries to [X.Y.Z] dated today)
#        - site/index.html (footer version chip)
#   3. Shows the diff, asks for confirmation
#   4. Commits as `chore(release): vX.Y.Z`
#   5. Tags `vX.Y.Z`
#   6. Pushes commit + tag → triggers .github/workflows/release.yml

set -euo pipefail

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ] || [ -z "${1:-}" ]; then
  cat <<'USAGE'
usage: scripts/release.sh X.Y.Z

  X.Y.Z   semver version, e.g. 0.1.1 or 0.2.0 or 1.0.0-rc1

The script bumps version in pyproject.toml, CHANGELOG.md, and site/index.html
(footer chip), commits, tags, and pushes. Pushing the tag triggers PyPI publish
+ GitHub Release creation via .github/workflows/release.yml.
USAGE
  exit 0
fi

NEW_VERSION="$1"
TODAY="$(date -u +%Y-%m-%d)"
TAG="v${NEW_VERSION}"
REPO="asuramaya/chronohorn"

# ── validate ──────────────────────────────────────────────────────────────
if ! [[ "$NEW_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+([-+][0-9A-Za-z.-]+)?$ ]]; then
  echo "error: '$NEW_VERSION' is not a valid semver string" >&2
  exit 2
fi

if [ "$(git rev-parse --abbrev-ref HEAD)" != "main" ]; then
  echo "error: must be on 'main' branch" >&2
  exit 2
fi

if [ -n "$(git status --porcelain)" ]; then
  echo "error: working tree is not clean — commit or stash first" >&2
  git status --short
  exit 2
fi

git fetch --quiet origin
if [ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]; then
  echo "error: local main is not in sync with origin/main — pull first" >&2
  exit 2
fi

if git rev-parse "${TAG}" >/dev/null 2>&1; then
  echo "error: tag ${TAG} already exists" >&2
  exit 2
fi

CURRENT_VERSION="$(grep -E '^version = ' pyproject.toml | sed -E 's/version = "(.+)"/\1/')"
if [ "$CURRENT_VERSION" = "$NEW_VERSION" ]; then
  echo "error: pyproject.toml already at ${NEW_VERSION}" >&2
  exit 2
fi

# ── bump pyproject.toml ───────────────────────────────────────────────────
sed -i.bak -E "s/^version = \"[^\"]+\"/version = \"${NEW_VERSION}\"/" pyproject.toml
rm -f pyproject.toml.bak

# ── bump CHANGELOG.md ─────────────────────────────────────────────────────
python3 - "$NEW_VERSION" "$CURRENT_VERSION" "$TODAY" "$REPO" <<'PYEOF'
import re
import sys
from pathlib import Path

new_version, current_version, today, repo = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
path = Path("CHANGELOG.md")
text = path.read_text()

# Insert a new [X.Y.Z] section right after [Unreleased].
# Whatever was under [Unreleased] becomes the new release's notes.
text = re.sub(
    r"## \[Unreleased\]\n",
    f"## [Unreleased]\n\n## [{new_version}] - {today}\n",
    text,
    count=1,
)

path.write_text(text)
PYEOF

# ── bump site/index.html ──────────────────────────────────────────────────
# Footer version chip. The hero is intentionally not version-tagged.
sed -i.bak -E "s|(<span class=\"version-chip\">v)[0-9]+\.[0-9]+\.[0-9]+([0-9A-Za-z.-]*)|\1${NEW_VERSION}|" site/index.html
rm -f site/index.html.bak

# ── show diff and confirm ─────────────────────────────────────────────────
echo
echo "=== changes for ${TAG} ==="
git --no-pager diff --stat
echo
git --no-pager diff
echo
read -r -p "Commit, tag, and push ${TAG}? [y/N] " ans
if [ "$ans" != "y" ] && [ "$ans" != "Y" ]; then
  echo "aborted; changes left uncommitted (run 'git checkout .' to revert)"
  exit 1
fi

# ── commit, tag, push ────────────────────────────────────────────────────
git add pyproject.toml CHANGELOG.md site/index.html
git commit -m "chore(release): ${TAG}"
git tag -a "${TAG}" -m "${TAG}"
git push origin main
git push origin "${TAG}"

echo
echo "✓ ${TAG} pushed"
echo "  release workflow: https://github.com/${REPO}/actions/workflows/release.yml"
echo "  pypi (when green): https://pypi.org/project/chronohorn/${NEW_VERSION}/"
echo "  github release:   https://github.com/${REPO}/releases/tag/${TAG}"
