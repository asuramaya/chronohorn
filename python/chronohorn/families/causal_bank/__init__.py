"""Causal-bank family package for Chronohorn scans and family-specific policy."""

__all__ = [
    "CAUSAL_BANK_FRONTIER_EMITTER",
    "CAUSAL_BANK_TRAINING_ADAPTER",
    "CausalBankFrontierEmitter",
    "CausalBankTrainingAdapter",
    "build_current_regime_scan",
    "main",
]


def __getattr__(name: str):
    if name in ("CAUSAL_BANK_TRAINING_ADAPTER", "CausalBankTrainingAdapter"):
        from .adapter import CAUSAL_BANK_TRAINING_ADAPTER, CausalBankTrainingAdapter
        return {"CAUSAL_BANK_TRAINING_ADAPTER": CAUSAL_BANK_TRAINING_ADAPTER,
                "CausalBankTrainingAdapter": CausalBankTrainingAdapter}[name]
    if name in ("CAUSAL_BANK_FRONTIER_EMITTER", "CausalBankFrontierEmitter",
                "build_current_regime_scan", "main"):
        from .scan import (
            CAUSAL_BANK_FRONTIER_EMITTER,
            CausalBankFrontierEmitter,
            build_current_regime_scan,
            main,
        )
        return {
            "CAUSAL_BANK_FRONTIER_EMITTER": CAUSAL_BANK_FRONTIER_EMITTER,
            "CausalBankFrontierEmitter": CausalBankFrontierEmitter,
            "build_current_regime_scan": build_current_regime_scan,
            "main": main,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
