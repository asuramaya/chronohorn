"""Transformer family adapter -- implements FamilyTrainingAdapter.

This is a *bridge* adapter.  It tells Chronohorn how to recognise, validate,
and summarise results from external transformer training runs (nanoGPT,
minGPT, custom GPT/LLaMA code, etc.).  It does NOT provide training
infrastructure -- transformer training is managed outside Chronohorn.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chronohorn.families.adapter import FamilyTrainingAdapter
from chronohorn.families.transformer.constants import (
    KEY_ALIASES,
    KNOWN_ARCHITECTURES,
    SUMMARY_KEYS,
    TRANSFORMER_FAMILY_ID,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXTERNAL_MSG = (
    "Transformer training is managed externally. "
    "Use chronohorn to track results, not to train."
)

_EXPORT_MSG = (
    "Transformer models are trained and exported externally. "
    "Use chronohorn to track results, not to export artifacts. "
    "To export, use your training framework's export tools."
)

_TRAINING_ARGS_MSG = (
    "Transformer training arguments are defined externally. "
    "Chronohorn does not control transformer training CLI flags. "
    "Configure your training framework directly."
)

_VARIANT_MSG = (
    "Transformer variant configs are built externally. "
    "Chronohorn tracks results from any transformer config "
    "but does not generate configs. Use your training framework."
)

_PERF_MSG = (
    "Transformer performance estimation is not available in Chronohorn. "
    "Use your training framework's profiler or benchmarking tools."
)

_REPLAY_MSG = (
    "Transformer replay fixtures are managed by the training framework. "
    "Chronohorn does not build or attach replay data for transformers."
)

_TABLE_EVAL_MSG = (
    "Table evaluation is not applicable to transformer models. "
    "Transformer evaluation is done externally via the training framework."
)


def _extract_sections(payload: dict) -> tuple[dict, dict]:
    """Return (train_or_toplevel_config, model_section) from a result payload."""
    cfg = payload.get("config", {})
    train = cfg.get("train", {}) if isinstance(cfg.get("train"), dict) else cfg
    model = payload.get("model", {})
    if not isinstance(model, dict):
        model = {}
    return train, model


# ---------------------------------------------------------------------------
# Training adapter
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TransformerTrainingAdapter(FamilyTrainingAdapter):
    family_id: str = TRANSFORMER_FAMILY_ID

    # -- architecture routing -----------------------------------------------

    def architecture_aliases(self) -> Sequence[str]:
        return KNOWN_ARCHITECTURES

    # -- result analysis (the core bridge functionality) --------------------

    def detect_illegal(self, payload: dict) -> bool:
        """Bidirectional attention is illegal for causal next-token prediction."""
        train, model = _extract_sections(payload)

        # Check both train config and model config for bidirectional flags.
        if train.get("bidirectional") or model.get("bidirectional"):
            return True
        if train.get("causal") is False or model.get("causal") is False:
            return True

        return False

    def estimate_artifact_mb(self, config: dict) -> float:
        """Estimate int6 artifact size: params * 6 bits / 8 / 1024^2."""
        params = config.get("trainable_params") or config.get("total_params") or 0
        if not params:
            params = config.get("num_params") or 0
        if not params:
            params = self._estimate_params_from_config(config)
        return params * 6 / 8 / 1024 / 1024

    def config_summary(self, result_json: dict) -> dict:
        """Extract key config fields for display.

        Checks canonical SUMMARY_KEYS first, then tries KEY_ALIASES so that
        results from different frameworks (HuggingFace, nanoGPT, etc.) all
        produce a normalised summary.
        """
        train, model = _extract_sections(result_json)
        # Merge model fields into train with lower precedence
        merged = {**model, **train}

        summary: dict[str, Any] = {}

        # Architecture name if present
        arch = merged.get("architecture") or merged.get("model_type")
        if arch:
            summary["architecture"] = arch

        # Param count
        num_params = merged.get("num_params") or merged.get("trainable_params") or merged.get("total_params")
        if num_params:
            summary["num_params"] = num_params

        # Canonical keys
        for key in SUMMARY_KEYS:
            val = merged.get(key)
            if val is not None:
                summary[key] = val

        # Aliased keys -- only fill in if canonical form is missing
        for alias, canonical in KEY_ALIASES.items():
            if canonical not in summary:
                val = merged.get(alias)
                if val is not None:
                    summary[canonical] = val

        return summary

    def training_entrypoints(self) -> Mapping[str, tuple[str, str]]:
        """Transformer training is external -- no chronohorn CLI entrypoints."""
        return {}

    # -- stubs: methods that don't apply to a bridge adapter ----------------

    def add_training_arguments(self, parser: Any, *, backend: str) -> Any:
        raise NotImplementedError(_TRAINING_ARGS_MSG)

    def build_variant_config(
        self,
        args: Any,
        *,
        ConfigClass: Any,
        scale_config: Any,
        seq_len: int,
        vocab_size: int,
    ) -> tuple[Any, tuple[int, ...]]:
        raise NotImplementedError(_VARIANT_MSG)

    def validate_config(
        self,
        args: Any,
        config: Any,
        *,
        baseline_linear_hidden: tuple[int, ...],
        out_dim: int,
    ) -> None:
        pass  # All configs are accepted -- validation is the user's responsibility.

    def estimate_training_performance(
        self,
        *,
        config: Any,
        vocab_size: int,
        batch_size: int,
        seq_len: int,
        trainable_param_count: int,
    ) -> dict[str, Any]:
        raise NotImplementedError(_PERF_MSG)

    def build_replay_fixture(
        self,
        dataset: Any,
        *,
        split: str,
        sequence_length: int,
    ) -> dict[str, Any]:
        raise NotImplementedError(_REPLAY_MSG)

    def attach_replay_reference(
        self,
        replay_fixture: dict[str, Any],
        logits: Any,
    ) -> dict[str, Any]:
        raise NotImplementedError(_REPLAY_MSG)

    def build_table_eval_argv(
        self,
        *,
        job: Mapping[str, Any],
        chronohorn_root: Path,
    ) -> list[str]:
        raise NotImplementedError(_TABLE_EVAL_MSG)

    def score_frontier_job(
        self,
        *,
        job: Mapping[str, Any],
        completed_runs: Sequence[Any],
    ) -> float:
        return 0.0  # Neutral scoring -- frontier is not managed by this bridge.

    def write_export_bundle(self, **kwargs: Any) -> Any:
        raise NotImplementedError(_EXPORT_MSG)

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _estimate_params_from_config(config: dict) -> int:
        """Rough parameter count from architecture config fields.

        Uses standard transformer parameter formula:
          - Embedding:    vocab_size * d_model
          - Per layer:    4 * d_model^2 (attention) + 2 * d_model * d_ff (MLP)
          - Output head:  d_model * vocab_size  (often tied with embedding)
        """
        d_model = (
            config.get("n_embd")
            or config.get("d_model")
            or config.get("hidden_size")
            or config.get("embed_dim")
            or 0
        )
        n_layers = (
            config.get("n_layers")
            or config.get("num_layers")
            or 0
        )
        vocab_size = config.get("vocab_size", 0)
        d_ff = (
            config.get("d_ff")
            or config.get("intermediate_size")
            or config.get("ffn_dim")
            or (4 * d_model if d_model else 0)
        )

        if not (d_model and n_layers):
            return 0

        # Token + position embeddings
        max_seq = (
            config.get("max_seq_len")
            or config.get("context_length")
            or config.get("block_size")
            or config.get("max_position_embeddings")
            or 1024
        )
        params = vocab_size * d_model  # token embedding
        params += max_seq * d_model    # position embedding

        # Transformer layers
        per_layer = (
            4 * d_model * d_model  # Q, K, V, output projections
            + 2 * d_model * d_ff   # MLP up + down
            + 4 * d_model          # layer norms (2 per layer, weight + bias each)
        )
        params += n_layers * per_layer

        # Check for mixture-of-experts (params dominated by expert FFN)
        num_experts = config.get("num_experts") or config.get("num_local_experts") or 1
        if num_experts > 1:
            # MoE: FFN is replicated per expert
            # Adjust: base_params + (num_experts - 1) * ffn_params_per_layer * n_layers
            ffn_per_layer = 2 * d_model * d_ff  # up + down projections
            expert_params = (num_experts - 1) * ffn_per_layer * n_layers
            params += expert_params

        # Final layer norm + output head (if not weight-tied)
        params += d_model          # final LN
        params += d_model * vocab_size  # unembedding (may be tied)

        return params


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

TRANSFORMER_TRAINING_ADAPTER = TransformerTrainingAdapter()
