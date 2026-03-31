from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class FrontierTopology:
    source_dir: str
    remote_cwd_rel: str = "chronohorn"
    hosts: tuple[str, ...] = ("slop-01", "slop-02")
    image: str = "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime"
    snapshot_paths: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=lambda: {"PYTHONUNBUFFERED": "1"})
    remote_data_root: str = "/snapshot/chronohorn/data/roots/fineweb10B_sp1024"


class FamilyFrontierEmitter(Protocol):
    family_id: str

    def supported_regimes(self) -> Sequence[str]: ...

    def build_scan_rows(self, *, regime: str, topology: FrontierTopology) -> list[dict[str, object]]: ...

    def default_output_path(self, *, regime: str) -> str: ...


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

    def build_table_eval_argv(
        self,
        *,
        job: Mapping[str, Any],
        chronohorn_root: Path,
    ) -> list[str]: ...

    def score_frontier_job(
        self,
        *,
        job: Mapping[str, Any],
        completed_runs: Sequence[Any],
    ) -> float: ...

    def write_export_bundle(self, **kwargs: Any) -> Any: ...
