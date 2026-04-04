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

    def architecture_aliases(self) -> Sequence[str]:
        """Return architecture strings that map to this family.

        Used by the registry to route unknown architecture names to the correct
        adapter.  E.g. ``["causal_bank", "opc"]`` or ``["polyhash", "polyhash_v6"]``.
        The family_id itself is always included automatically.
        """
        return ()

    def training_entrypoints(self) -> Mapping[str, tuple[str, str]]:
        """Return CLI entrypoints this family contributes to ``chronohorn train``.

        Returns a dict of ``{command_name: (module_path, help_text)}``.
        The default implementation returns no entrypoints.
        """
        return {}

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

    def infer_from_config(self, cfg: dict) -> bool:
        """Return True if the config dict looks like it belongs to this family.

        Called by the registry's ``detect_family`` when no explicit family or
        architecture field is present.  Each adapter checks for its own
        distinctive config fields.  The default returns False.
        """
        return False

    def detect_illegal(self, payload: dict) -> bool:
        """Check if a result payload is illegal (e.g. future leakage)."""
        ...

    def estimate_artifact_mb(self, config: dict) -> float:
        """Estimate int6 artifact size in megabytes from a config dict."""
        ...

    def config_summary(self, result_json: dict) -> dict:
        """Extract key config fields from a result JSON for display."""
        ...
