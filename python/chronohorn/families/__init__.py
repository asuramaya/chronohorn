"""Chronohorn descendant family packages."""

from .adapter import FamilyFrontierEmitter, FamilyTrainingAdapter, FrontierTopology
from .registry import (
    available_family_ids,
    detect_family,
    detect_illegal,
    get_adapter,
    resolve_family_id,
    resolve_frontier_emitter,
    resolve_training_adapter,
)

__all__ = [
    "FamilyFrontierEmitter",
    "FamilyTrainingAdapter",
    "FrontierTopology",
    "available_family_ids",
    "detect_family",
    "detect_illegal",
    "get_adapter",
    "resolve_family_id",
    "resolve_frontier_emitter",
    "resolve_training_adapter",
]
