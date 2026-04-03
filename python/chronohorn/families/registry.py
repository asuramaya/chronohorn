from __future__ import annotations

from chronohorn.families.adapter import FamilyFrontierEmitter, FamilyTrainingAdapter


def _training_adapters() -> dict[str, FamilyTrainingAdapter]:
    """Build the training-adapter registry on first use.

    Importing here rather than at module level prevents the causal_bank adapter
    from pulling the full training backend (torch/mlx/open-predictive-coder) into
    fleet and control surfaces that only need the lightweight protocol types.
    """
    # Lazy import: avoids pulling open_predictive_coder into fleet/control surfaces
    from chronohorn.families.causal_bank import CAUSAL_BANK_TRAINING_ADAPTER

    return {
        CAUSAL_BANK_TRAINING_ADAPTER.family_id: CAUSAL_BANK_TRAINING_ADAPTER,
    }


def _frontier_emitters() -> dict[str, FamilyFrontierEmitter]:
    """Build the frontier-emitter registry on first use.

    Same lazy-import rationale as ``_training_adapters``.
    """
    # Lazy import: avoids pulling open_predictive_coder into fleet/control surfaces
    from chronohorn.families.causal_bank import CAUSAL_BANK_FRONTIER_EMITTER

    return {
        CAUSAL_BANK_FRONTIER_EMITTER.family_id: CAUSAL_BANK_FRONTIER_EMITTER,
    }


def available_family_ids() -> list[str]:
    """Return sorted list of all registered family IDs."""
    return sorted(set(_training_adapters()) | set(_frontier_emitters()))


def resolve_training_adapter(family_id: str) -> FamilyTrainingAdapter:
    """Return the ``FamilyTrainingAdapter`` for *family_id*.

    Raises ``KeyError`` with a descriptive message listing known families when
    *family_id* is not registered.
    """
    adapters = _training_adapters()
    try:
        return adapters[family_id]
    except KeyError as exc:
        known = ", ".join(available_family_ids()) or "none"
        raise KeyError(f"unknown Chronohorn family {family_id!r}; known families: {known}") from exc


def resolve_frontier_emitter(family_id: str) -> FamilyFrontierEmitter:
    """Return the ``FamilyFrontierEmitter`` for *family_id*.

    Raises ``KeyError`` with a descriptive message listing known emitters when
    *family_id* is not registered.
    """
    emitters = _frontier_emitters()
    try:
        return emitters[family_id]
    except KeyError as exc:
        known = ", ".join(sorted(emitters)) or "none"
        raise KeyError(f"unknown Chronohorn frontier family {family_id!r}; known emitters: {known}") from exc
