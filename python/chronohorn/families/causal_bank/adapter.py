from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Sequence

from chronohorn.families.adapter import FamilyTrainingAdapter
from chronohorn.store import RunSnapshot
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

    @staticmethod
    def _job_spec(payload: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "variant": payload.get("variant"),
            "scale": payload.get("scale"),
            "learning_rate": payload.get("learning_rate"),
            "weight_decay": payload.get("weight_decay"),
            "local_window": payload.get("local_window"),
            "oscillatory_frac": payload.get("oscillatory_frac"),
            "linear_readout_kind": payload.get("linear_readout_kind"),
            "linear_readout_num_experts": payload.get("linear_readout_num_experts"),
            "bank_gate_span": payload.get("bank_gate_span"),
            "static_bank_gate": payload.get("static_bank_gate"),
        }

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

    def build_table_eval_argv(
        self,
        *,
        job: Mapping[str, Any],
        chronohorn_root: Path,
    ) -> list[str]:
        script = chronohorn_root / "scripts" / "slop_run_causal_bank_from_table.zsh"
        checkpoint_path = str(job.get("checkpoint_path", job.get("checkpoint_npz")))
        summary_path = str(job.get("summary_path", job.get("checkpoint_json")))
        if checkpoint_path in {"None", ""}:
            raise ValueError(f"{job['name']}: family table eval requires checkpoint_path")
        if summary_path in {"None", ""}:
            raise ValueError(f"{job['name']}: family table eval requires summary_path")
        artifact_bin = str(job.get("artifact_bin", "")).strip()
        if not artifact_bin:
            raise ValueError(f"{job['name']}: family table eval requires artifact_bin")
        host = str(job.get("host", "")).strip()
        if not host:
            raise ValueError(f"{job['name']}: family table eval requires host")
        return [
            "zsh",
            str(script),
            host,
            str(job["name"]),
            checkpoint_path,
            summary_path,
            artifact_bin,
            str(job.get("threads", 12)),
            str(job.get("val_tokens", 62021846)),
            str(job.get("report_every", 1000000)),
        ]

    def score_frontier_job(
        self,
        *,
        job: Mapping[str, Any],
        completed_runs: Sequence[RunSnapshot],
    ) -> float:
        job_spec = self._job_spec(job)
        peers: list[tuple[float, Mapping[str, Any]]] = []
        for run in completed_runs:
            if run.family != self.family_id:
                continue
            metric = run.forecast_metric_value if run.forecast_metric_value is not None else run.metric_value
            if metric is None:
                continue
            manifest = run.metadata.get("manifest")
            if not isinstance(manifest, Mapping):
                continue
            peers.append((float(metric), manifest))
        if not peers:
            return 0.0
        peers.sort(key=lambda item: item[0])
        score = 0.0
        for rank, (metric, peer_manifest) in enumerate(peers[:3], start=1):
            peer_spec = self._job_spec(peer_manifest)
            weight = 1.0 / rank
            score += -metric * weight
            if job_spec["variant"] == peer_spec["variant"]:
                score += 2.5 * weight
            if job_spec["linear_readout_kind"] == peer_spec["linear_readout_kind"]:
                score += 2.0 * weight
            if job_spec["linear_readout_num_experts"] == peer_spec["linear_readout_num_experts"]:
                score += 1.0 * weight
            if job_spec["static_bank_gate"] == peer_spec["static_bank_gate"]:
                score += 0.75 * weight
            if job_spec["learning_rate"] == peer_spec["learning_rate"]:
                score += 1.5 * weight
            if job_spec["weight_decay"] == peer_spec["weight_decay"]:
                score += 0.5 * weight
            if job_spec["local_window"] == peer_spec["local_window"]:
                score += 1.0 * weight
            if (
                job_spec["scale"] is not None
                and peer_spec["scale"] is not None
            ):
                score += max(0.0, 1.5 - abs(float(job_spec["scale"]) - float(peer_spec["scale"]))) * weight
            if (
                job_spec["oscillatory_frac"] is not None
                and peer_spec["oscillatory_frac"] is not None
            ):
                score += max(
                    0.0,
                    1.0 - abs(float(job_spec["oscillatory_frac"]) - float(peer_spec["oscillatory_frac"])) * 2.0,
                ) * weight
            if (
                job_spec["bank_gate_span"] is not None
                and peer_spec["bank_gate_span"] is not None
            ):
                score += max(
                    0.0,
                    0.75 - abs(float(job_spec["bank_gate_span"]) - float(peer_spec["bank_gate_span"])),
                ) * weight
        return score

    def write_export_bundle(self, **kwargs: Any) -> Any:
        return write_causal_bank_export_bundle(**kwargs)


CAUSAL_BANK_TRAINING_ADAPTER = CausalBankTrainingAdapter()
