from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
import shlex
from typing import Any


MANIFEST_METADATA_KEYS = {
    "name",
    "command",
    "executor_kind",
    "executor_name",
    "goal",
    "host",
    "hosts",
    "image",
    "gpu",
    "source_dir",
    "snapshot_paths",
    "remote_cwd_rel",
    "env",
    "launcher",
    "backend",
    "resource_class",
    "workload_kind",
    "work_tokens",
    "family",
    "manifest_path",
    "run_id",
    "requested_launcher",
    "config",
    "manifest",
    "parent",
    "state",
    "placement_cores",
    "threads",
    "report_every",
    "checkpoint_path",
    "artifact_bin",
    "summary_path",
    "val_tokens",
    "train_tokens",
    "sync_paths",
    "rsync_excludes",
    "cwd",
    "argv",
    "log_path",
    "cluster_gateway_host",
    "remote_source_dir",
    "container",
    "remote_run",
    "remote_snapshot",
    "remote_assets",
    "runtime_namespace",
    "runtime_job_name",
    "runtime_pod_name",
    "runtime_node_name",
    "launched_at",
    "completed_at",
    "generated_by",
}

SCRIPT_ARCHITECTURES = {
    "train_hash_embed.py": "hash_embed",
    "measure-backend-parity": "causal-bank",
    "train_polyhash.py": "polyhash",
    "train_polyhash_v2.py": "polyhash_v2",
    "train_polyhash_v3.py": "polyhash_v3",
    "train_polyhash_v4.py": "polyhash_v4",
    "train_polyhash_v5.py": "polyhash_v5",
    "train_polyhash_v6.py": "polyhash_v6",
    "train_polyhash_v7.py": "polyhash_v7",
    "train_polyhash_v8.py": "polyhash_v8",
    "train_polyhash_v8h.py": "polyhash_v8h",
    "train_polyhash_v8m.py": "polyhash_v8m",
    "train_polyhash_v10.py": "polyhash_v10",
    "train_polyhash_v11.py": "polyhash_v11",
    "train-causal-bank-mix": "causal-bank",
    "train-causal-bank-mlx": "causal-bank",
    "train-causal-bank-torch": "causal-bank",
    "train_causal_bank_torch.py": "causal-bank",
    "train_causal_bank_mlx.py": "causal-bank",
}

FLAG_KEY_ALIASES = {
    "lr": "learning_rate",
    "seq_len": "seq_len",
    "batch_size": "batch_size",
    "buckets": "buckets_per_table",
    "embed_dim": "embed_per_table",
    "hidden": "hidden_dim",
    "mlp_layers": "num_layers",
    "byte_embed": "byte_embed_dim",
    "sam_embed": "sam_embed_dim",
    "sam_bits": "sam_quant_bits",
    "sam_ste": "sam_straight_through",
    "scan_chunk": "scan_chunk_size",
}

NEGATED_FLAG_KEYS = {
    "no_sam": ("sam_enabled", False),
    "no_residual": ("residual", False),
}

NON_CONFIG_FLAG_KEYS = {"json"}

_INT_RE = re.compile(r"^[+-]?\d+$")


@dataclass(frozen=True)
class ParsedTrainingCommand:
    trainer: str | None
    architecture: str | None
    family: str | None
    config: dict[str, Any]
    steps: int | None
    seed: int | None
    learning_rate: float | None
    batch_size: int | None


def _maybe_number(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if _INT_RE.match(value):
        try:
            return int(value)
        except ValueError:
            return value
    try:
        parsed = float(value)
    except ValueError:
        return value
    if parsed.is_integer() and value.strip().isdigit():
        return int(parsed)
    return parsed


def _normalize_flag_key(flag: str) -> str:
    key = flag[2:] if flag.startswith("--") else flag
    key = key.replace("-", "_")
    return FLAG_KEY_ALIASES.get(key, key)


def _coerce_positive_int(*values: Any) -> int | None:
    for value in values:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


def _coerce_optional_int(*values: Any) -> int | None:
    for value in values:
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _coerce_positive_float(*values: Any) -> float | None:
    for value in values:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


def _resolve_family_id(alias: Any) -> str | None:
    if not alias:
        return None
    from chronohorn.families.registry import resolve_family_id

    text = str(alias).strip()
    for candidate in (text, text.replace("-", "_"), text.replace("_", "-")):
        resolved = resolve_family_id(candidate)
        if resolved:
            return resolved
    return None


def _normalized_architecture(alias: Any) -> str | None:
    if not alias:
        return None
    return str(alias).strip().replace("-", "_")


def _find_training_invocation(command: str) -> tuple[str | None, list[str]]:
    if not command.strip():
        return None, []
    segments = [segment.strip() for segment in command.split(";") if segment.strip()]
    for segment in reversed(segments):
        try:
            tokens = shlex.split(segment, posix=True)
        except ValueError:
            continue
        for index, token in enumerate(tokens):
            if token.endswith(".py"):
                return token.split("/")[-1], tokens[index + 1 :]
            if token == "-m" and index + 3 < len(tokens):
                if tokens[index + 1] == "chronohorn" and tokens[index + 2] == "train":
                    return tokens[index + 3], tokens[index + 4 :]
    return None, []


def parse_training_command(command: str, *, family_hint: Any = None) -> ParsedTrainingCommand:
    trainer, arg_tokens = _find_training_invocation(command)
    parsed_config: dict[str, Any] = {}
    if trainer:
        parsed_config["script"] = trainer

    architecture = _normalized_architecture(family_hint)
    if architecture is None and trainer:
        architecture = SCRIPT_ARCHITECTURES.get(trainer.split("/")[-1])
    if architecture:
        parsed_config["architecture"] = architecture

    canonical_family = _resolve_family_id(family_hint) or _resolve_family_id(architecture)
    if canonical_family:
        parsed_config["family"] = canonical_family

    index = 0
    while index < len(arg_tokens):
        token = arg_tokens[index]
        if not token.startswith("--"):
            index += 1
            continue
        if "=" in token:
            flag, raw_value = token.split("=", 1)
            value: Any = _maybe_number(raw_value)
            index += 1
        elif index + 1 < len(arg_tokens) and not arg_tokens[index + 1].startswith("--"):
            flag = token
            value = _maybe_number(arg_tokens[index + 1])
            index += 2
        else:
            flag = token
            value = True
            index += 1
        raw_key = flag[2:].replace("-", "_")
        key = _normalize_flag_key(flag)
        if key in NON_CONFIG_FLAG_KEYS or raw_key in NON_CONFIG_FLAG_KEYS:
            continue
        parsed_config[key] = value
        if raw_key != key:
            parsed_config.setdefault(raw_key, value)
        negated = NEGATED_FLAG_KEYS.get(key)
        if negated is not None and bool(value):
            parsed_config[negated[0]] = negated[1]

    if "sam_enabled" not in parsed_config and any(key.startswith("sam_") for key in parsed_config):
        parsed_config["sam_enabled"] = not bool(parsed_config.get("no_sam"))

    return ParsedTrainingCommand(
        trainer=trainer,
        architecture=architecture,
        family=canonical_family,
        config=parsed_config,
        steps=_coerce_positive_int(parsed_config.get("steps")),
        seed=_coerce_optional_int(parsed_config.get("seed")),
        learning_rate=_coerce_positive_float(parsed_config.get("learning_rate")),
        batch_size=_coerce_positive_int(parsed_config.get("batch_size")),
    )


def _explicit_manifest_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    nested = payload.get("config")
    if isinstance(nested, Mapping):
        config.update(dict(nested))
    for key, value in payload.items():
        if key in MANIFEST_METADATA_KEYS:
            continue
        config[key] = value
    for key in NON_CONFIG_FLAG_KEYS:
        config.pop(key, None)
    return config


def normalize_manifest_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    family_hint = (
        normalized.get("family")
        or (normalized.get("config", {}) if isinstance(normalized.get("config"), Mapping) else {}).get("architecture")
        or (normalized.get("config", {}) if isinstance(normalized.get("config"), Mapping) else {}).get("family")
    )
    parsed = parse_training_command(str(normalized.get("command") or ""), family_hint=family_hint)
    explicit_config = _explicit_manifest_config(normalized)

    merged_config: dict[str, Any] = {}
    merged_config.update(explicit_config)
    merged_config.update(parsed.config)

    architecture = (
        _normalized_architecture(explicit_config.get("architecture"))
        or _normalized_architecture(family_hint)
        or parsed.architecture
    )
    if architecture:
        merged_config["architecture"] = architecture
        normalized["architecture"] = architecture

    canonical_family = (
        _resolve_family_id(explicit_config.get("family"))
        or _resolve_family_id(family_hint)
        or parsed.family
    )
    if canonical_family:
        merged_config["family"] = canonical_family
        normalized["family"] = canonical_family

    steps = _coerce_positive_int(
        parsed.steps,
        normalized.get("steps"),
        explicit_config.get("steps"),
    )
    seed = _coerce_optional_int(
        parsed.seed,
        normalized.get("seed"),
        explicit_config.get("seed"),
    )
    learning_rate = _coerce_positive_float(
        parsed.learning_rate,
        normalized.get("learning_rate"),
        normalized.get("lr"),
        explicit_config.get("learning_rate"),
        explicit_config.get("lr"),
    )
    batch_size = _coerce_positive_int(
        parsed.batch_size,
        normalized.get("batch_size"),
        explicit_config.get("batch_size"),
    )

    if steps is not None:
        normalized["steps"] = steps
        merged_config["steps"] = steps
    if seed is not None:
        normalized["seed"] = seed
        merged_config["seed"] = seed
    if learning_rate is not None:
        normalized["learning_rate"] = learning_rate
        merged_config["learning_rate"] = learning_rate
    if batch_size is not None:
        normalized["batch_size"] = batch_size
        merged_config["batch_size"] = batch_size

    for key in NON_CONFIG_FLAG_KEYS:
        merged_config.pop(key, None)

    normalized["config"] = merged_config
    if parsed.trainer:
        normalized.setdefault("trainer", parsed.trainer)
    return normalized
