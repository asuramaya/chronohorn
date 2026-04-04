"""Family registry — auto-discovers family packages under chronohorn.families.*.

Each family package must expose at minimum a TRAINING_ADAPTER singleton
(naming convention: ``<UPPER_FAMILY>_TRAINING_ADAPTER``), and optionally a
``<UPPER_FAMILY>_FRONTIER_EMITTER``.  The package is imported lazily on first
access.

The registry builds an *architecture alias map* by calling
``adapter.architecture_aliases()`` on each discovered adapter, so that
architecture strings like ``"polyhash_v6"`` or ``"opc"`` can be resolved to the
correct family without any hardcoded if/elif chains in core infra.
"""
from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chronohorn.families.adapter import FamilyFrontierEmitter, FamilyTrainingAdapter


# ---------------------------------------------------------------------------
# Auto-discovery of family sub-packages
# ---------------------------------------------------------------------------

def _discover_family_packages() -> dict[str, str]:
    """Scan ``chronohorn/families/*/`` for sub-packages, return {family_id: module_path}.

    A sub-package qualifies if it contains an ``__init__.py`` (is a package) and
    is not a private module (no leading underscore).  The family_id is derived
    from the directory name with underscores replaced by hyphens.
    """
    families_dir = Path(__file__).resolve().parent
    found: dict[str, str] = {}
    for info in pkgutil.iter_modules([str(families_dir)]):
        if info.ispkg and not info.name.startswith("_"):
            family_id = info.name.replace("_", "-")
            module_path = f"chronohorn.families.{info.name}"
            found[family_id] = module_path
    return found


# Lazy-populated on first access.
_family_packages: dict[str, str] | None = None


def _get_family_packages() -> dict[str, str]:
    global _family_packages
    if _family_packages is None:
        _family_packages = _discover_family_packages()
    return _family_packages


# ---------------------------------------------------------------------------
# Adapter caches + alias map
# ---------------------------------------------------------------------------

_training_adapter_cache: dict[str, "FamilyTrainingAdapter"] = {}
_frontier_emitter_cache: dict[str, "FamilyFrontierEmitter"] = {}
_adapter_load_failures: set[str] = set()  # family_ids that failed to load — don't retry
_alias_map: dict[str, str] | None = None  # architecture string → family_id


def _load_attr(module_path: str, attr_name: str) -> object:
    mod = importlib.import_module(module_path)
    return getattr(mod, attr_name)


def _adapter_attr_name(family_id: str) -> str:
    """Convention: CAUSAL_BANK_TRAINING_ADAPTER for family 'causal-bank'."""
    return family_id.replace("-", "_").upper() + "_TRAINING_ADAPTER"


def _emitter_attr_name(family_id: str) -> str:
    return family_id.replace("-", "_").upper() + "_FRONTIER_EMITTER"


def _build_alias_map() -> dict[str, str]:
    """Build a map of architecture strings → family_id from all adapters."""
    amap: dict[str, str] = {}
    for fid in _get_family_packages():
        # family_id itself is always an alias
        amap[fid] = fid
        amap[fid.replace("-", "_")] = fid
        # Load adapter and ask for its aliases
        try:
            adapter = resolve_training_adapter(fid)
            for alias in adapter.architecture_aliases():
                amap[alias.lower()] = fid
        except Exception as exc:
            _adapter_load_failures.add(fid)
            import sys
            print(f"chronohorn registry: adapter {fid!r} failed to load: {exc}", file=sys.stderr)
    return amap


def _get_alias_map() -> dict[str, str]:
    global _alias_map
    if _alias_map is None:
        _alias_map = _build_alias_map()
    return _alias_map


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def available_family_ids() -> list[str]:
    """Return sorted list of discovered family IDs."""
    return sorted(_get_family_packages())


def resolve_training_adapter(family_id: str) -> "FamilyTrainingAdapter":
    """Get a family's training adapter (lazy-loaded and cached)."""
    if family_id in _training_adapter_cache:
        return _training_adapter_cache[family_id]
    packages = _get_family_packages()
    module_path = packages.get(family_id)
    if module_path is None:
        known = ", ".join(available_family_ids()) or "none"
        raise KeyError(f"unknown Chronohorn family {family_id!r}; known families: {known}")
    adapter = _load_attr(module_path, _adapter_attr_name(family_id))
    _training_adapter_cache[family_id] = adapter  # type: ignore[assignment]
    return adapter  # type: ignore[return-value]


def resolve_frontier_emitter(family_id: str) -> "FamilyFrontierEmitter":
    """Get a family's frontier emitter (lazy-loaded and cached)."""
    if family_id in _frontier_emitter_cache:
        return _frontier_emitter_cache[family_id]
    packages = _get_family_packages()
    module_path = packages.get(family_id)
    if module_path is None:
        known = ", ".join(sorted(packages)) or "none"
        raise KeyError(f"unknown Chronohorn frontier family {family_id!r}; known emitters: {known}")
    emitter = _load_attr(module_path, _emitter_attr_name(family_id))
    _frontier_emitter_cache[family_id] = emitter  # type: ignore[assignment]
    return emitter  # type: ignore[return-value]


def get_adapter(family_id: str) -> "FamilyTrainingAdapter":
    """Convenience alias for resolve_training_adapter."""
    return resolve_training_adapter(family_id)


def resolve_family_id(architecture: str) -> str | None:
    """Map an architecture string (e.g. ``"polyhash_v6"``, ``"opc"``) to a family ID.

    Returns ``None`` if the architecture doesn't match any known family.
    Performs prefix matching as a fallback (e.g. ``"polyhash_v99"`` matches ``"polyhash"``).
    """
    key = architecture.lower().strip()
    amap = _get_alias_map()

    # Exact match
    if key in amap:
        return amap[key]

    # Prefix match: e.g. "polyhash_v99" → check if any alias is a prefix
    for alias, fid in amap.items():
        if key.startswith(alias) or alias.startswith(key):
            return fid

    return None


def detect_family(cfg: dict) -> str | None:
    """Detect family from a config/payload dict.  Returns family_id or None.

    Checks ``family``, ``architecture`` fields, then infers from config shape.
    """
    explicit = cfg.get("family") or cfg.get("architecture")
    if explicit:
        return resolve_family_id(str(explicit))

    # Infer from config shape by asking each adapter
    for fid in _get_family_packages():
        if fid in _adapter_load_failures:
            continue
        try:
            adapter = resolve_training_adapter(fid)
            if hasattr(adapter, "infer_from_config") and adapter.infer_from_config(cfg):
                return fid
        except (KeyError, ImportError):
            continue

    return None


def detect_illegal(payload: dict, family_id: str | None = None) -> bool:
    """Delegate illegal detection to the appropriate family adapter.

    If family_id is not provided, attempts to detect it from the payload.
    Returns False for unknown families.
    """
    if family_id is None:
        cfg = payload.get("config", {})
        train = cfg.get("train", {}) if isinstance(cfg.get("train"), dict) else cfg
        model = payload.get("model") or {}
        arch = (
            model.get("architecture")
            or train.get("architecture")
            or cfg.get("architecture")
            or train.get("family")
            or cfg.get("family")
        )
        if arch:
            family_id = resolve_family_id(str(arch))
    if family_id is None:
        return False
    try:
        adapter = resolve_training_adapter(family_id)
        return adapter.detect_illegal(payload)
    except (KeyError, ImportError):
        return False
