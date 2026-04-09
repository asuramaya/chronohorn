from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "decepticons" / "src"))

from chronohorn.families.causal_bank.adapter import CausalBankTrainingAdapter


def test_config_summary_falls_back_to_model_section():
    adapter = CausalBankTrainingAdapter()

    summary = adapter.config_summary(
        {
            "config": {},
            "model": {
                "variant": "base",
                "scale": 8.0,
                "learning_rate": 0.0015,
                "local_window": 8,
                "local_scale": 0.25,
                "oscillatory_frac": 0.5,
                "oscillatory_schedule": "mincorr_greedy",
                "input_proj_scheme": "split_banks",
                "substrate_mode": "learnable_mixing",
                "state_dim": 16,
                "state_impl": "retention",
                "num_heads": 4,
                "block_mixing_ratio": 0.25,
                "linear_readout_kind": "routed_sqrelu_experts",
                "linear_readout_num_experts": 4,
                "readout_bands": 4,
                "bank_gate_span": 0.5,
                "static_bank_gate": False,
            },
        }
    )

    assert summary["variant"] == "base"
    assert summary["linear_readout_kind"] == "routed_sqrelu_experts"
    assert summary["readout_bands"] == 4
    assert summary["substrate_mode"] == "learnable_mixing"
    assert summary["input_proj_scheme"] == "split_banks"
    assert summary["state_impl"] == "retention"
    assert summary["num_heads"] == 4
