# Chronohorn — Public Copy Style

This file locks the canonical copy that should appear consistently across
public surfaces (README, pyproject, site, CLAUDE.md, package docstring,
CHANGELOG, social blurbs). Update this file before changing any of those
surfaces, so the canonical block stays the source of truth.

Internal-facing docs under `docs/*.md` have their own established voice and
are not bound by this file.

## Tagline (one line, ~12 words)

> Family-agnostic experiment tracker and architecture-search runtime for predictive descendants.

This is the line every public surface should lead with. PyPI description,
GitHub repo description, README opening, site `<title>` and `og:description`,
package docstring, CHANGELOG release-note opener.

## Pitch (two sentences, technical surfaces)

> Chronohorn is a family-agnostic experiment tracker and architecture-search
> runtime. SQLite-backed truth, 64 MCP tools, multi-backend fleet dispatch
> (CPU/Metal/CUDA), saturation and frontier analysis, and a plug-in family
> adapter protocol.

## Site hero (poetic — site only)

> Sweep the clock. Track the frontier.

This is *only* for the site hero. Don't let it leak into pyproject, the GitHub
repo description, or the README.

## Naming

- **Chronohorn** — capitalized in prose ("Chronohorn is a runtime…")
- **chronohorn** — lowercase as the package name, the CLI binary, the brand
  glyph in the site nav, and inside code blocks

## Hyphenation

- **architecture-search runtime** — hyphenated. It's a compound modifier in
  front of a noun. Never "architecture search runtime."
- **family-agnostic** — hyphenated.
- **multi-backend** — hyphenated.
- **frontier-control surface** — hyphenated. (Used in some doctrine docs.)

## Key facts (numbers)

- **64** MCP tools (verify against `python/chronohorn/mcp.py`'s `TOOLS` dict
  before bumping; grep `'^    "chronohorn_'`)
- **3** backends — CPU, Metal, CUDA
- **3** families shipped — `causal-bank`, `polyhash`, `transformer`
- **Python ≥ 3.11**
- **SQLite-backed**, single-writer discipline, WAL reads
- **MIT** license

When any of these change, update STYLE.md first, then propagate.

## Sibling project: `decepticons`

- **Technical surfaces** (README, pyproject, CLAUDE.md, docs):
  *"shared decepticons kernel"* / *"the decepticons kernel"* /
  *"built on the shared decepticons kernel"*. Always link to
  <https://github.com/asuramaya/decepticons> on first mention.
- **Site only**: the "kindred" panel is allowed to use the chrome-vs-dream
  framing because it's an explicit visual rhetorical device. Don't let that
  language travel to other surfaces.

## Don't

- Don't say "predictive-coder descendants" — that's the legacy term. Use
  "predictive descendants."
- Don't write "Chronohorn is a framework." It is a runtime / tracker, not a
  training framework. Model code lives in decepticons.
- Don't use emojis in technical copy (README, pyproject, CLAUDE.md, docs).
  The site is allowed atmospheric SVG glyphs. CHANGELOG is plain text.
- Don't drift the tagline mid-document. If a surface needs additional framing,
  the canonical tagline should still appear once, near the top.

## Surfaces this file governs

| Surface | Canonical line lives at |
|---|---|
| `README.md` | Line 3 (immediately after `# Chronohorn`) |
| `pyproject.toml` | `description = ...` |
| `CLAUDE.md` | First paragraph |
| `site/index.html` | `<title>`, `<meta name="description">`, `og:description` |
| `python/chronohorn/__init__.py` | First docstring line |
| `CHANGELOG.md` | Release-note opener for any new minor/major version |

## Surfaces this file does NOT govern

- `docs/REPO_BOUNDARY.md`, `docs/STACK.md`, `docs/FLEET.md`,
  `docs/DOCTRINE.md`, `docs/CRATE_MAP.md` — internal doctrine, own voice.
- Manifest READMEs, scripts READMEs — internal operational notes.
- Code comments, docstrings on internal modules.
