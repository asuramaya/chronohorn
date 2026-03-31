from __future__ import annotations

from typing import Any, Protocol


class FamilyTrainingAdapter(Protocol):
    family_id: str

    def add_training_arguments(self, parser: Any, *, backend: str) -> Any: ...

    def build_variant_config(
        self,
        args: Any,
        *,
        ConfigClass: Any,
        scale_config: Any,
        seq_len: int,
        vocab_size: int,
    ) -> tuple[Any, tuple[int, ...]]: ...

    def validate_config(
        self,
        args: Any,
        config: Any,
        *,
        baseline_linear_hidden: tuple[int, ...],
        out_dim: int,
    ) -> None: ...

    def estimate_training_performance(
        self,
        *,
        config: Any,
        vocab_size: int,
        batch_size: int,
        seq_len: int,
        trainable_param_count: int,
    ) -> dict[str, Any]: ...

    def build_replay_fixture(
        self,
        dataset: Any,
        *,
        split: str,
        sequence_length: int,
    ) -> dict[str, Any]: ...

    def attach_replay_reference(
        self,
        replay_fixture: dict[str, Any],
        logits: Any,
    ) -> dict[str, Any]: ...

    def write_export_bundle(self, **kwargs: Any) -> Any: ...
