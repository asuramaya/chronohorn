from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "ChecksumSet",
    "ExportBundleSpec",
    "ExportManifest",
    "ExportPaths",
    "LearnedStateIndex",
    "LearnedStateRef",
    "LearnedTensorEntry",
]


@dataclass(frozen=True)
class ExportPaths:
    root: Path
    manifest: Path = field(init=False)
    learned_state_index: Path = field(init=False)
    checksums: Path = field(init=False)
    notes: Path = field(init=False)
    learned_state_dir: Path = field(init=False)
    learned_state_blobs: Path = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest", self.root / "manifest.json")
        object.__setattr__(self, "learned_state_index", self.root / "learned_state" / "index.json")
        object.__setattr__(self, "checksums", self.root / "checksums.json")
        object.__setattr__(self, "notes", self.root / "notes.json")
        object.__setattr__(self, "learned_state_dir", self.root / "learned_state")
        object.__setattr__(self, "learned_state_blobs", self.root / "learned_state" / "blobs")


@dataclass(frozen=True)
class LearnedTensorEntry:
    name: str
    shape: tuple[int, ...]
    dtype: str
    storage: str
    blob: str
    checksum: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": [int(dim) for dim in self.shape],
            "dtype": self.dtype,
            "storage": self.storage,
            "blob": self.blob,
            "checksum": self.checksum,
        }


@dataclass(frozen=True)
class LearnedStateIndex:
    tensor_format: str
    tensor_count: int
    tensor_index: tuple[LearnedTensorEntry, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tensor_format": self.tensor_format,
            "tensor_count": int(self.tensor_count),
            "tensor_index": [entry.to_dict() for entry in self.tensor_index],
        }


@dataclass(frozen=True)
class LearnedStateRef:
    tensor_format: str
    tensor_count: int
    tensor_index_ref: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "tensor_format": self.tensor_format,
            "tensor_count": int(self.tensor_count),
            "tensor_index_ref": self.tensor_index_ref,
        }


@dataclass(frozen=True)
class ChecksumSet:
    algorithm: str
    manifest_body: str | None
    learned_state_index: str
    blobs: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "manifest_body": self.manifest_body,
            "learned_state_index": self.learned_state_index,
            "blobs": dict(self.blobs),
        }


@dataclass(frozen=True)
class ExportManifest:
    abi_name: str
    abi_version: str
    exporter_version: str
    exported_utc: str
    model_family_id: str
    model_variant_id: str
    kernel_version: str
    tokenizer_id: str
    data_root_id: str
    artifact_role: str
    deterministic_substrate: Mapping[str, Any]
    learned_state: LearnedStateRef
    checksums: ChecksumSet
    source_commit: str | None = None
    train_step: int | None = None
    train_wallclock_s: float | None = None
    sequence_length: int | None = None
    vocab_size: int | None = None
    dtype_policy: str | None = None
    quantization_policy: str | None = None
    export_notes: str | None = None
    packed_memory: Mapping[str, Any] | None = None
    notes_ref: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "abi_name": self.abi_name,
            "abi_version": self.abi_version,
            "exporter_version": self.exporter_version,
            "exported_utc": self.exported_utc,
            "model_family_id": self.model_family_id,
            "model_variant_id": self.model_variant_id,
            "kernel_version": self.kernel_version,
            "tokenizer_id": self.tokenizer_id,
            "data_root_id": self.data_root_id,
            "artifact_role": self.artifact_role,
            "deterministic_substrate": dict(self.deterministic_substrate),
            "learned_state": self.learned_state.to_dict(),
            "checksums": self.checksums.to_dict(),
        }
        if self.source_commit is not None:
            data["source_commit"] = self.source_commit
        if self.train_step is not None:
            data["train_step"] = int(self.train_step)
        if self.train_wallclock_s is not None:
            data["train_wallclock_s"] = float(self.train_wallclock_s)
        if self.sequence_length is not None:
            data["sequence_length"] = int(self.sequence_length)
        if self.vocab_size is not None:
            data["vocab_size"] = int(self.vocab_size)
        if self.dtype_policy is not None:
            data["dtype_policy"] = self.dtype_policy
        if self.quantization_policy is not None:
            data["quantization_policy"] = self.quantization_policy
        if self.export_notes is not None:
            data["export_notes"] = self.export_notes
        if self.packed_memory is not None:
            data["packed_memory"] = dict(self.packed_memory)
        if self.notes_ref is not None:
            data["notes_ref"] = self.notes_ref
        return data


@dataclass(frozen=True)
class ExportBundleSpec:
    manifest: ExportManifest
    tensor_blobs: Mapping[str, bytes]
    learned_state_index: LearnedStateIndex
    checksums: ChecksumSet
    notes: Any | None = None
