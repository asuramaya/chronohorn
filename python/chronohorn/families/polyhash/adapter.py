"""PolyHash family adapter — implements FamilyTrainingAdapter and FamilyFrontierEmitter."""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chronohorn.families.adapter import FamilyFrontierEmitter, FamilyTrainingAdapter, FrontierTopology
from chronohorn.families.polyhash.constants import (
    POLYHASH_FAMILY_ID,
    SAM_SUMMARY_KEYS,
    SUMMARY_KEYS,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VERSION_RE = re.compile(r"polyhash_?(v\d+)", re.IGNORECASE)


def _detect_arch_version(payload: dict) -> str:
    """Best-effort extraction of architecture version from a result payload."""
    cfg = payload.get("config", {})
    train = cfg.get("train", {}) if isinstance(cfg.get("train"), dict) else cfg

    # Explicit field
    ver = train.get("arch_version") or train.get("version")
    if ver:
        return str(ver)

    # Infer from family id string (e.g. "polyhash_v7b")
    family = train.get("family", "") or payload.get("family", "")
    m = _VERSION_RE.search(family)
    if m:
        return m.group(1)

    # Infer from script path
    script = train.get("script", "")
    m = _VERSION_RE.search(script)
    if m:
        return m.group(1)

    return "unknown"


def _extract_train(payload: dict) -> dict:
    cfg = payload.get("config", {})
    return cfg.get("train", {}) if isinstance(cfg.get("train"), dict) else cfg


# ---------------------------------------------------------------------------
# Training adapter
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PolyHashTrainingAdapter(FamilyTrainingAdapter):
    family_id: str = POLYHASH_FAMILY_ID

    def architecture_aliases(self) -> Sequence[str]:
        return (
            "polyhash", "polyhash_v1", "polyhash_v2", "polyhash_v3",
            "polyhash_v4", "polyhash_v5", "polyhash_v6", "polyhash_v7",
            "hash_embed",  # legacy name for early experiments
        )

    # -- protocol methods ----------------------------------------------------

    def detect_illegal(self, payload: dict) -> bool:
        """PolyHash has no illegal config patterns (no future-leakage risk)."""
        return False

    def estimate_artifact_mb(self, config: dict) -> float:
        """Estimate int6 artifact size: params * 6 bits / 8 / 1024^2."""
        params = config.get("trainable_params") or config.get("total_params") or 0
        if not params:
            params = self._estimate_params_from_config(config)
        return params * 6 / 8 / 1024 / 1024

    def config_summary(self, result_json: dict) -> dict:
        """Extract key config fields for display."""
        train = _extract_train(result_json)
        summary: dict[str, Any] = {}

        summary["arch_version"] = _detect_arch_version(result_json)

        for key in SUMMARY_KEYS:
            if key == "arch_version":
                continue  # already set
            val = train.get(key)
            if val is not None:
                summary[key] = val

        # SAM config (v7+)
        sam_cfg: dict[str, Any] = {}
        for key in SAM_SUMMARY_KEYS:
            val = train.get(key)
            if val is not None:
                sam_cfg[key] = val
        if sam_cfg:
            summary["sam"] = sam_cfg

        return summary

    # -- stubs for methods not yet wired for polyhash -----------------------

    def add_training_arguments(self, parser: Any, *, backend: str) -> Any:
        raise NotImplementedError("PolyHash training arguments are script-local for now")

    def build_variant_config(
        self,
        args: Any,
        *,
        ConfigClass: Any,
        scale_config: Any,
        seq_len: int,
        vocab_size: int,
    ) -> tuple[Any, tuple[int, ...]]:
        raise NotImplementedError("PolyHash variant config not yet wired")

    def validate_config(
        self,
        args: Any,
        config: Any,
        *,
        baseline_linear_hidden: tuple[int, ...],
        out_dim: int,
    ) -> None:
        pass  # No safety constraints for polyhash configs

    def estimate_training_performance(
        self,
        *,
        config: Any,
        vocab_size: int,
        batch_size: int,
        seq_len: int,
        trainable_param_count: int,
    ) -> dict[str, Any]:
        return {"family": self.family_id, "note": "perf estimation not yet implemented"}

    def build_replay_fixture(
        self,
        dataset: Any,
        *,
        split: str,
        sequence_length: int,
    ) -> dict[str, Any]:
        raise NotImplementedError("PolyHash replay not yet wired")

    def attach_replay_reference(
        self,
        replay_fixture: dict[str, Any],
        logits: Any,
    ) -> dict[str, Any]:
        raise NotImplementedError("PolyHash replay not yet wired")

    def build_table_eval_argv(
        self,
        *,
        job: Mapping[str, Any],
        chronohorn_root: Path,
    ) -> list[str]:
        raise NotImplementedError("PolyHash table eval not yet wired")

    def score_frontier_job(
        self,
        *,
        job: Mapping[str, Any],
        completed_runs: Sequence[Any],
    ) -> float:
        return 0.0  # neutral scoring until polyhash-specific heuristic is built

    def write_export_bundle(self, **kwargs: Any) -> Any:
        raise NotImplementedError("PolyHash export not yet wired")

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _estimate_params_from_config(config: dict) -> int:
        """Rough parameter count from architecture config fields."""
        num_tables = config.get("num_tables", 8)
        buckets = config.get("buckets_per_table", 65536)
        embed = config.get("embed_per_table", 16)
        hidden = config.get("hidden_dim", 512)
        layers = config.get("num_layers", 2)
        scan_dim = config.get("scan_dim", 256)
        byte_embed = config.get("byte_embed_dim", 128)
        vocab = config.get("vocab_size", 1024)

        # Hash tables
        params = num_tables * buckets * embed
        # Byte embedding
        params += vocab * byte_embed
        # Scan (gate + input projections, rough)
        params += 4 * scan_dim * scan_dim
        # MLP layers (SwiGLU: 3*d*h per layer)
        for _ in range(layers):
            params += 3 * hidden * hidden
        # Output head
        params += hidden * vocab

        # SAM tables if present
        sam_buckets = config.get("sam_buckets", 0)
        sam_embed = config.get("sam_embed_dim", 0)
        sam_heads = config.get("sam_heads", 0)
        if sam_buckets and sam_embed and sam_heads:
            params += sam_heads * sam_buckets * sam_embed

        return params


# ---------------------------------------------------------------------------
# Frontier emitter (minimal — scan rows are currently built by hand)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PolyHashFrontierEmitter(FamilyFrontierEmitter):
    family_id: str = POLYHASH_FAMILY_ID

    def supported_regimes(self) -> Sequence[str]:
        return ("polyhash",)

    def default_topology(self) -> FrontierTopology:
        return FrontierTopology(source_dir=str(Path(__file__).resolve().parents[5]))

    def build_scan_rows(self, *, regime: str, topology: FrontierTopology) -> list[dict[str, object]]:
        if regime != "polyhash":
            raise ValueError(f"unsupported polyhash frontier regime: {regime}")
        # PolyHash manifests are currently hand-crafted JSONL files.
        return []

    def default_output_path(self, *, regime: str) -> str:
        root = Path(__file__).resolve().parents[5]
        return str(root / "chronohorn" / "manifests" / "frontier_polyhash.jsonl")


# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

POLYHASH_TRAINING_ADAPTER = PolyHashTrainingAdapter()
POLYHASH_FRONTIER_EMITTER = PolyHashFrontierEmitter()
