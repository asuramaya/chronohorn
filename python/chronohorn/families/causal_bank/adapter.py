from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from chronohorn.families.adapter import FamilyTrainingAdapter
from chronohorn.train.causal_bank_training_primitives import (
    add_causal_bank_training_arguments,
    assert_safe_model_config,
    assert_safe_readout_compute,
    build_causal_bank_variant_config,
)
from chronohorn.train.causal_bank_training_support import (
    attach_replay_fixture_logprob_reference,
    build_replay_parity_fixture,
    estimate_causal_bank_training_performance,
    write_causal_bank_export_bundle,
)


@dataclass(frozen=True)
class CausalBankTrainingAdapter(FamilyTrainingAdapter):
    family_id: str = "causal-bank"

    def add_training_arguments(self, parser: Any, *, backend: str) -> Any:
        return add_causal_bank_training_arguments(parser, backend=backend)

    def build_variant_config(
        self,
        args: Any,
        *,
        ConfigClass: Any,
        scale_config: Any,
        seq_len: int,
        vocab_size: int,
    ) -> tuple[Any, tuple[int, ...]]:
        return build_causal_bank_variant_config(
            args,
            ConfigClass=ConfigClass,
            scale_config=scale_config,
            seq_len=seq_len,
            vocab_size=vocab_size,
        )

    def validate_config(
        self,
        args: Any,
        config: Any,
        *,
        baseline_linear_hidden: tuple[int, ...],
        out_dim: int,
    ) -> None:
        assert_safe_model_config(args, config)
        assert_safe_readout_compute(
            args,
            config,
            baseline_linear_hidden=baseline_linear_hidden,
            out_dim=out_dim,
        )

    def estimate_training_performance(
        self,
        *,
        config: Any,
        vocab_size: int,
        batch_size: int,
        seq_len: int,
        trainable_param_count: int,
    ) -> dict[str, Any]:
        return estimate_causal_bank_training_performance(
            config=config,
            vocab_size=vocab_size,
            batch_size=batch_size,
            seq_len=seq_len,
            trainable_param_count=trainable_param_count,
        )

    def build_replay_fixture(
        self,
        dataset: Any,
        *,
        split: str,
        sequence_length: int,
    ) -> dict[str, Any]:
        return build_replay_parity_fixture(
            dataset,
            split=split,
            sequence_length=sequence_length,
        )

    def attach_replay_reference(
        self,
        replay_fixture: dict[str, Any],
        logits: Any,
    ) -> dict[str, Any]:
        return attach_replay_fixture_logprob_reference(replay_fixture, logits)

    def write_export_bundle(self, **kwargs: Any) -> Any:
        return write_causal_bank_export_bundle(**kwargs)


CAUSAL_BANK_TRAINING_ADAPTER = CausalBankTrainingAdapter()
