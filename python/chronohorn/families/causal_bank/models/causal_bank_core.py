from __future__ import annotations

_OPC_NAMES = (
    "CAUSAL_BANK_DETERMINISTIC_SUBSTRATE_SEED",
    "CAUSAL_BANK_FAMILY",
    "CAUSAL_BANK_FAMILY_ID",
    "CAUSAL_BANK_INPUT_PROJ_SCHEMES",
    "CAUSAL_BANK_OSCILLATORY_SCHEDULES",
    "CAUSAL_BANK_READOUT_KINDS",
    "CAUSAL_BANK_VARIANTS",
    "CausalBankConfig",
    "CausalBankFamilySpec",
    "apply_variant",
    "build_linear_bank",
    "osc_pair_count",
    "scale_config",
    "validate_config",
)

_opc_cache: dict | None = None


def _get_opc():
    global _opc_cache
    if _opc_cache is not None:
        return _opc_cache
    try:
        from decepticons.causal_bank import (
            CAUSAL_BANK_DETERMINISTIC_SUBSTRATE_SEED,
            CAUSAL_BANK_FAMILY,
            CAUSAL_BANK_FAMILY_ID,
            CAUSAL_BANK_INPUT_PROJ_SCHEMES,
            CAUSAL_BANK_OSCILLATORY_SCHEDULES,
            CAUSAL_BANK_READOUT_KINDS,
            CAUSAL_BANK_VARIANTS,
            CausalBankConfig,
            CausalBankFamilySpec,
            apply_variant,
            build_linear_bank,
            osc_pair_count,
            scale_config,
            validate_config,
        )
    except ImportError as exc:
        raise ImportError(
            "decepticons is required for causal-bank models. "
            "Install the decepticons package or make it importable."
        ) from exc
    _opc_cache = {
        "CAUSAL_BANK_DETERMINISTIC_SUBSTRATE_SEED": CAUSAL_BANK_DETERMINISTIC_SUBSTRATE_SEED,
        "CAUSAL_BANK_FAMILY": CAUSAL_BANK_FAMILY,
        "CAUSAL_BANK_FAMILY_ID": CAUSAL_BANK_FAMILY_ID,
        "CAUSAL_BANK_INPUT_PROJ_SCHEMES": CAUSAL_BANK_INPUT_PROJ_SCHEMES,
        "CAUSAL_BANK_OSCILLATORY_SCHEDULES": CAUSAL_BANK_OSCILLATORY_SCHEDULES,
        "CAUSAL_BANK_READOUT_KINDS": CAUSAL_BANK_READOUT_KINDS,
        "CAUSAL_BANK_VARIANTS": CAUSAL_BANK_VARIANTS,
        "CausalBankConfig": CausalBankConfig,
        "CausalBankFamilySpec": CausalBankFamilySpec,
        "apply_variant": apply_variant,
        "build_linear_bank": build_linear_bank,
        "osc_pair_count": osc_pair_count,
        "scale_config": scale_config,
        "validate_config": validate_config,
    }
    return _opc_cache


def __getattr__(name: str):
    if name in _OPC_NAMES:
        return _get_opc()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
