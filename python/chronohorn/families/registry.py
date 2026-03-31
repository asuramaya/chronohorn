from __future__ import annotations

from chronohorn.families.adapter import FamilyFrontierEmitter, FamilyTrainingAdapter
from chronohorn.families.causal_bank import CAUSAL_BANK_FRONTIER_EMITTER, CAUSAL_BANK_TRAINING_ADAPTER


TRAINING_ADAPTERS: dict[str, FamilyTrainingAdapter] = {
    CAUSAL_BANK_TRAINING_ADAPTER.family_id: CAUSAL_BANK_TRAINING_ADAPTER,
}

FRONTIER_EMITTERS: dict[str, FamilyFrontierEmitter] = {
    CAUSAL_BANK_FRONTIER_EMITTER.family_id: CAUSAL_BANK_FRONTIER_EMITTER,
}


def available_family_ids() -> list[str]:
    return sorted(set(TRAINING_ADAPTERS) | set(FRONTIER_EMITTERS))


def resolve_training_adapter(family_id: str) -> FamilyTrainingAdapter:
    try:
        return TRAINING_ADAPTERS[family_id]
    except KeyError as exc:
        known = ", ".join(available_family_ids()) or "none"
        raise KeyError(f"unknown Chronohorn family {family_id!r}; known families: {known}") from exc


def resolve_frontier_emitter(family_id: str) -> FamilyFrontierEmitter:
    try:
        return FRONTIER_EMITTERS[family_id]
    except KeyError as exc:
        known = ", ".join(sorted(FRONTIER_EMITTERS)) or "none"
        raise KeyError(f"unknown Chronohorn frontier family {family_id!r}; known emitters: {known}") from exc
