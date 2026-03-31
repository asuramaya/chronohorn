from __future__ import annotations

from chronohorn._opc import ensure_open_predictive_coder_importable

ensure_open_predictive_coder_importable()

from open_predictive_coder.causal_bank import (  # noqa: E402,F401
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
