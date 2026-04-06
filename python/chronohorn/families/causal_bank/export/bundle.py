from __future__ import annotations

import hashlib
import io
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .abi import (
    ABI_NAME,
    ABI_VERSION,
    DEFAULT_CHECKSUM_ALGORITHM,
    DEFAULT_EXPORTER_VERSION,
    DEFAULT_LEARNED_STATE_FORMAT,
)
from .schema import (
    ChecksumSet,
    ExportManifest,
    ExportPaths,
    LearnedStateIndex,
    LearnedStateRef,
    LearnedTensorEntry,
)


def _utc_now_rfc3339() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _checksum_bytes(data: bytes, algorithm: str = DEFAULT_CHECKSUM_ALGORITHM) -> str:
    if algorithm == "blake2b":
        digest = hashlib.blake2b(data, digest_size=32).hexdigest()
    elif algorithm == "sha256":
        digest = hashlib.sha256(data).hexdigest()
    else:
        raise ValueError(f"Unsupported checksum algorithm: {algorithm}")
    return f"{algorithm}:{digest}"


def _slugify_tensor_name(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return slug or "tensor"


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach") and callable(value.detach):
        tensor = value.detach()
        if hasattr(tensor, "cpu") and callable(tensor.cpu):
            tensor = tensor.cpu()
        if hasattr(tensor, "numpy") and callable(tensor.numpy):
            return np.asarray(tensor.numpy())
    if hasattr(value, "numpy") and callable(value.numpy):
        try:
            return np.asarray(value.numpy())
        except TypeError:
            pass
    return np.asarray(value)


def _encode_numpy_blob(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _build_tensor_probe_entry(name: str, array: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    flat = contiguous.reshape(-1)
    flat64 = flat.astype(np.float64, copy=False)
    sample = [float(value) for value in flat64[: min(4, flat64.size)]]
    return {
        "name": name,
        "dtype": str(contiguous.dtype),
        "shape": [int(dim) for dim in contiguous.shape],
        "elem_count": int(flat64.size),
        "sum64": float(flat64.sum(dtype=np.float64)),
        "absmax": float(np.max(np.abs(flat64))) if flat64.size else 0.0,
        "l2_norm": float(np.linalg.norm(flat64)) if flat64.size else 0.0,
        "sample": sample,
    }


def _merge_notes_payload(notes: Any | None, tensor_probe: dict[str, Any]) -> dict[str, Any]:
    if notes is None:
        payload: dict[str, Any] = {}
    elif isinstance(notes, str):
        payload = {"notes": notes}
    elif isinstance(notes, Mapping):
        payload = dict(notes)
    else:
        payload = {"notes": notes}
    payload["tensor_probe_v1"] = tensor_probe
    return payload


def _tensor_entries_from_state(
    learned_state: Mapping[str, Any],
    *,
    checksum_algorithm: str,
) -> tuple[tuple[tuple[str, str, bytes, LearnedTensorEntry], ...], dict[str, str], tuple[dict[str, Any], ...]]:
    tensor_records: list[tuple[str, str, bytes, LearnedTensorEntry]] = []
    blob_checksums: dict[str, str] = {}
    tensor_probes: list[dict[str, Any]] = []
    for ordinal, (tensor_name, tensor_value) in enumerate(learned_state.items()):
        array = _as_numpy(tensor_value)
        if array.dtype == object:
            raise TypeError(f"Tensor {tensor_name!r} has object dtype, which is not supported.")
        contiguous = np.ascontiguousarray(array)
        blob_name = f"{ordinal:04d}__{_slugify_tensor_name(tensor_name)}.npy"
        blob_relpath = str(Path("learned_state") / "blobs" / blob_name)
        blob_bytes = _encode_numpy_blob(contiguous)
        blob_checksum = _checksum_bytes(blob_bytes, algorithm=checksum_algorithm)
        blob_checksums[tensor_name] = blob_checksum
        tensor_probes.append(_build_tensor_probe_entry(tensor_name, contiguous))
        tensor_records.append(
            (
                tensor_name,
                blob_name,
                blob_bytes,
                LearnedTensorEntry(
                    name=tensor_name,
                    shape=tuple(int(dim) for dim in contiguous.shape),
                    dtype=str(contiguous.dtype),
                    storage="blob_ref",
                    blob=blob_relpath,
                    checksum=blob_checksum,
                ),
            )
        )
    return tuple(tensor_records), blob_checksums, tuple(tensor_probes)


@dataclass(frozen=True)
class ExportBundleWriter:
    output_dir: Path
    checksum_algorithm: str = DEFAULT_CHECKSUM_ALGORITHM

    @property
    def paths(self) -> ExportPaths:
        return ExportPaths(self.output_dir)

    def write(
        self,
        *,
        model_family_id: str,
        model_variant_id: str,
        kernel_version: str,
        tokenizer_id: str,
        data_root_id: str,
        deterministic_substrate: Mapping[str, Any],
        learned_state: Mapping[str, Any],
        artifact_role: str = "replay",
        exporter_version: str = DEFAULT_EXPORTER_VERSION,
        exported_utc: str | None = None,
        source_commit: str | None = None,
        train_step: int | None = None,
        train_wallclock_s: float | None = None,
        sequence_length: int | None = None,
        vocab_size: int | None = None,
        dtype_policy: str | None = None,
        quantization_policy: str | None = None,
        export_notes: str | None = None,
        notes: Any | None = None,
        packed_memory: Mapping[str, Any] | None = None,
    ) -> Path:
        root = self.output_dir.expanduser()
        paths = ExportPaths(root)
        root.mkdir(parents=True, exist_ok=True)

        if not learned_state:
            raise ValueError("learned_state must contain at least one tensor entry.")

        tensor_records, blob_checksums, tensor_probes = _tensor_entries_from_state(
            learned_state,
            checksum_algorithm=self.checksum_algorithm,
        )

        tensor_entries: list[LearnedTensorEntry] = []
        for _, blob_name, blob_bytes, entry in tensor_records:
            tensor_entries.append(entry)
            _write_bytes(paths.learned_state_blobs / blob_name, blob_bytes)

        learned_state_index = LearnedStateIndex(
            tensor_format=DEFAULT_LEARNED_STATE_FORMAT,
            tensor_count=len(tensor_entries),
            tensor_index=tuple(tensor_entries),
        )
        learned_state_ref = LearnedStateRef(
            tensor_format=DEFAULT_LEARNED_STATE_FORMAT,
            tensor_count=len(tensor_entries),
            tensor_index_ref=str(paths.learned_state_index.relative_to(root)),
        )
        learned_state_index_bytes = _canonical_json_bytes(learned_state_index.to_dict())
        learned_state_index_checksum = _checksum_bytes(learned_state_index_bytes, algorithm=self.checksum_algorithm)
        _write_bytes(paths.learned_state_index, learned_state_index_bytes)

        base_checksums = ChecksumSet(
            algorithm=self.checksum_algorithm,
            manifest_body=None,
            learned_state_index=learned_state_index_checksum,
            blobs=blob_checksums,
        )

        manifest = ExportManifest(
            abi_name=ABI_NAME,
            abi_version=ABI_VERSION,
            exporter_version=exporter_version,
            exported_utc=exported_utc or _utc_now_rfc3339(),
            model_family_id=model_family_id,
            model_variant_id=model_variant_id,
            kernel_version=kernel_version,
            tokenizer_id=tokenizer_id,
            data_root_id=data_root_id,
            artifact_role=artifact_role,
            deterministic_substrate=dict(deterministic_substrate),
            learned_state=learned_state_ref,
            checksums=base_checksums,
            source_commit=source_commit,
            train_step=train_step,
            train_wallclock_s=train_wallclock_s,
            sequence_length=sequence_length,
            vocab_size=vocab_size,
            dtype_policy=dtype_policy,
            quantization_policy=quantization_policy,
            export_notes=export_notes,
            packed_memory=packed_memory,
            notes_ref=str(Path("notes.json")),
        )

        manifest_for_checksum = manifest.to_dict()
        manifest_for_checksum["checksums"] = base_checksums.to_dict()
        manifest_body_bytes = _canonical_json_bytes(manifest_for_checksum)
        manifest_body_checksum = _checksum_bytes(manifest_body_bytes, algorithm=self.checksum_algorithm)
        manifest = replace(
            manifest,
            checksums=replace(base_checksums, manifest_body=manifest_body_checksum),
        )
        manifest_bytes = _canonical_json_bytes(manifest.to_dict())
        _write_bytes(paths.manifest, manifest_bytes)

        notes_payload = _merge_notes_payload(
            notes,
            {
                "tensor_count": len(tensor_probes),
                "tensors": list(tensor_probes),
            },
        )
        _write_bytes(paths.notes, _canonical_json_bytes(notes_payload))

        checksums_payload = {
            "algorithm": self.checksum_algorithm,
            "manifest_body": manifest_body_checksum,
            "learned_state_index": learned_state_index_checksum,
            "blobs": blob_checksums,
        }
        _write_bytes(paths.checksums, _canonical_json_bytes(checksums_payload))
        return root


def write_opc_export_bundle(output_dir: str | Path, **kwargs: Any) -> Path:
    writer = ExportBundleWriter(Path(output_dir))
    return writer.write(**kwargs)
