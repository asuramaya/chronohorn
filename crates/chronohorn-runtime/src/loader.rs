use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use chronohorn_schema::{
    CHRONOHORN_EXPORT_REQUIRED_FIELDS, ChronohornExportChecksums,
    ChronohornExportLearnedStateIndex, ChronohornExportManifest,
};
use serde::Deserialize;
use serde::de::DeserializeOwned;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChronohornExportInspectRequest {
    pub export_root: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChronohornExportInspectReport {
    pub export_root: String,
    pub manifest_path: String,
    pub learned_state_index_path: String,
    pub checksums_path: String,
    pub abi_name: String,
    pub abi_version: String,
    pub exporter_version: String,
    pub model_family_id: String,
    pub model_variant_id: String,
    pub kernel_version: String,
    pub tokenizer_id: String,
    pub data_root_id: String,
    pub artifact_role: String,
    pub learned_state_tensor_count: usize,
    pub learned_state_blob_count: usize,
    pub packed_memory_attached: bool,
    pub notes_path: Option<String>,
}

impl ChronohornExportInspectReport {
    pub fn render(&self) -> String {
        let notes_path = self.notes_path.as_deref().unwrap_or("none");
        format!(
            "chronohorn_export_inspect_report\nexport_root: {}\nmanifest_path: {}\nlearned_state_index_path: {}\nchecksums_path: {}\nabi_name: {}\nabi_version: {}\nexporter_version: {}\nmodel_family_id: {}\nmodel_variant_id: {}\nkernel_version: {}\ntokenizer_id: {}\ndata_root_id: {}\nartifact_role: {}\nlearned_state_tensor_count: {}\nlearned_state_blob_count: {}\npacked_memory_attached: {}\nnotes_path: {}\nrequired_fields: {}\n",
            self.export_root,
            self.manifest_path,
            self.learned_state_index_path,
            self.checksums_path,
            self.abi_name,
            self.abi_version,
            self.exporter_version,
            self.model_family_id,
            self.model_variant_id,
            self.kernel_version,
            self.tokenizer_id,
            self.data_root_id,
            self.artifact_role,
            self.learned_state_tensor_count,
            self.learned_state_blob_count,
            self.packed_memory_attached,
            notes_path,
            CHRONOHORN_EXPORT_REQUIRED_FIELDS.join(", "),
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChronohornTensorInventoryInspectRequest {
    pub export_root: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChronohornTensorInventoryInspectReport {
    pub export_root: String,
    pub manifest_path: String,
    pub learned_state_index_path: String,
    pub abi_name: String,
    pub abi_version: String,
    pub exporter_version: String,
    pub model_family_id: String,
    pub model_variant_id: String,
    pub kernel_version: String,
    pub tensor_format: String,
    pub tensor_count: usize,
    pub checksum_algorithm: String,
    pub blob_checksum_count: usize,
    pub tensor_entries: Vec<chronohorn_schema::ChronohornExportTensorEntry>,
    pub storage_counts: BTreeMap<String, usize>,
    pub packed_memory_attached: bool,
    pub notes_path: Option<String>,
}

impl ChronohornTensorInventoryInspectReport {
    pub fn render(&self) -> String {
        let notes_path = self.notes_path.as_deref().unwrap_or("none");
        let mut out = format!(
            "chronohorn_tensor_inventory_inspect_report\nexport_root: {}\nmanifest_path: {}\nlearned_state_index_path: {}\nabi_name: {}\nabi_version: {}\nexporter_version: {}\nmodel_family_id: {}\nmodel_variant_id: {}\nkernel_version: {}\ntensor_format: {}\ntensor_count: {}\nchecksum_algorithm: {}\nblob_checksum_count: {}\npacked_memory_attached: {}\nnotes_path: {}\n",
            self.export_root,
            self.manifest_path,
            self.learned_state_index_path,
            self.abi_name,
            self.abi_version,
            self.exporter_version,
            self.model_family_id,
            self.model_variant_id,
            self.kernel_version,
            self.tensor_format,
            self.tensor_count,
            self.checksum_algorithm,
            self.blob_checksum_count,
            self.packed_memory_attached,
            notes_path,
        );
        out.push_str("storage_counts:\n");
        for (storage, count) in &self.storage_counts {
            out.push_str(&format!("  {:<18} {}\n", storage, count));
        }
        out.push_str("tensor_entries:\n");
        for entry in &self.tensor_entries {
            out.push_str(&format!(
                "  {:<40} {:<10} {:<12} {:<18} {:<18} {}\n",
                entry.name,
                entry.dtype,
                entry.storage,
                format!("{:?}", entry.shape),
                entry.blob,
                entry.checksum,
            ));
        }
        out.push_str(&format!(
            "required_fields: {}\n",
            CHRONOHORN_EXPORT_REQUIRED_FIELDS.join(", ")
        ));
        out
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChronohornTensorProbeVerifyRequest {
    pub export_root: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChronohornReplayPrepRequest {
    pub export_root: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChronohornReplayTensorPlan {
    pub name: String,
    pub dtype: String,
    pub storage: String,
    pub shape: Vec<usize>,
    pub blob: String,
    pub checksum: String,
    pub file_bytes: usize,
    pub payload_bytes: usize,
    pub data_offset: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChronohornReplayPrepReport {
    pub export_root: String,
    pub manifest_path: String,
    pub learned_state_index_path: String,
    pub notes_path: Option<String>,
    pub abi_name: String,
    pub abi_version: String,
    pub exporter_version: String,
    pub model_family_id: String,
    pub model_variant_id: String,
    pub kernel_version: String,
    pub tokenizer_id: String,
    pub data_root_id: String,
    pub artifact_role: String,
    pub tensor_format: String,
    pub tensor_count: usize,
    pub tensor_plan_count: usize,
    pub total_file_bytes: usize,
    pub total_payload_bytes: usize,
    pub storage_counts: BTreeMap<String, usize>,
    pub dtype_counts: BTreeMap<String, usize>,
    pub tensor_plan: Vec<ChronohornReplayTensorPlan>,
    pub packed_memory_attached: bool,
    pub runtime_ready: bool,
}

impl ChronohornReplayPrepReport {
    pub fn render(&self) -> String {
        let notes_path = self.notes_path.as_deref().unwrap_or("none");
        let mut out = format!(
            "chronohorn_replay_prep_report\nexport_root: {}\nmanifest_path: {}\nlearned_state_index_path: {}\nabi_name: {}\nabi_version: {}\nexporter_version: {}\nmodel_family_id: {}\nmodel_variant_id: {}\nkernel_version: {}\ntokenizer_id: {}\ndata_root_id: {}\nartifact_role: {}\ntensor_format: {}\ntensor_count: {}\ntensor_plan_count: {}\ntotal_file_bytes: {}\ntotal_payload_bytes: {}\npacked_memory_attached: {}\nruntime_ready: {}\nnotes_path: {}\n",
            self.export_root,
            self.manifest_path,
            self.learned_state_index_path,
            self.abi_name,
            self.abi_version,
            self.exporter_version,
            self.model_family_id,
            self.model_variant_id,
            self.kernel_version,
            self.tokenizer_id,
            self.data_root_id,
            self.artifact_role,
            self.tensor_format,
            self.tensor_count,
            self.tensor_plan_count,
            self.total_file_bytes,
            self.total_payload_bytes,
            self.packed_memory_attached,
            self.runtime_ready,
            notes_path,
        );
        out.push_str("storage_counts:\n");
        for (storage, count) in &self.storage_counts {
            out.push_str(&format!("  {:<18} {}\n", storage, count));
        }
        out.push_str("dtype_counts:\n");
        for (dtype, count) in &self.dtype_counts {
            out.push_str(&format!("  {:<18} {}\n", dtype, count));
        }
        out.push_str("tensor_plan:\n");
        for entry in &self.tensor_plan {
            out.push_str(&format!(
                "  {:<40} {:<10} {:<12} {:<18} {:<12} {:<12} {}\n",
                entry.name,
                entry.dtype,
                entry.storage,
                format!("{:?}", entry.shape),
                entry.file_bytes,
                entry.payload_bytes,
                entry.blob,
            ));
        }
        out
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChronohornTensorProbeVerifyReport {
    pub export_root: String,
    pub manifest_path: String,
    pub learned_state_index_path: String,
    pub notes_path: Option<String>,
    pub tensor_count: usize,
    pub loaded_blob_count: usize,
    pub total_blob_bytes: usize,
    pub notes_probe_present: bool,
    pub probe_match: bool,
    pub mismatches: Vec<String>,
}

impl ChronohornTensorProbeVerifyReport {
    pub fn render(&self) -> String {
        let notes_path = self.notes_path.as_deref().unwrap_or("none");
        let mut out = format!(
            "chronohorn_tensor_probe_verify_report\nexport_root: {}\nmanifest_path: {}\nlearned_state_index_path: {}\nnotes_path: {}\ntensor_count: {}\nloaded_blob_count: {}\ntotal_blob_bytes: {}\nnotes_probe_present: {}\nprobe_match: {}\n",
            self.export_root,
            self.manifest_path,
            self.learned_state_index_path,
            notes_path,
            self.tensor_count,
            self.loaded_blob_count,
            self.total_blob_bytes,
            self.notes_probe_present,
            self.probe_match,
        );
        if self.mismatches.is_empty() {
            out.push_str("mismatches: none\n");
        } else {
            out.push_str("mismatches:\n");
            for mismatch in &self.mismatches {
                out.push_str(&format!("  - {}\n", mismatch));
            }
        }
        out
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChronohornExportBundlePaths {
    pub export_root: PathBuf,
    pub manifest_path: PathBuf,
    pub learned_state_index_path: PathBuf,
    pub checksums_path: PathBuf,
    pub notes_path: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq)]
struct ChronohornLoadedTensorProbe {
    name: String,
    dtype: String,
    shape: Vec<usize>,
    elem_count: usize,
    data_bytes: usize,
    sum64: f64,
    absmax: f64,
    l2_norm: f64,
    sample: Vec<f64>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
struct ChronohornNotesPayload {
    #[serde(default)]
    tensor_probe_v1: Option<ChronohornNotesTensorProbe>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
struct ChronohornNotesTensorProbe {
    tensor_count: usize,
    tensors: Vec<ChronohornNotesTensorProbeEntry>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
struct ChronohornNotesTensorProbeEntry {
    name: String,
    dtype: String,
    shape: Vec<usize>,
    elem_count: usize,
    sum64: f64,
    absmax: f64,
    l2_norm: f64,
    sample: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct NpyMeta {
    dtype: String,
    shape: Vec<usize>,
    data_offset: usize,
}

impl ChronohornExportBundlePaths {
    fn from_root(export_root: impl Into<PathBuf>) -> Self {
        let export_root = export_root.into();
        Self {
            manifest_path: export_root.join("manifest.json"),
            learned_state_index_path: export_root.join("learned_state").join("index.json"),
            checksums_path: export_root.join("checksums.json"),
            notes_path: None,
            export_root,
        }
    }

    fn from_manifest_path(manifest_path: impl Into<PathBuf>) -> Result<Self, String> {
        let manifest_path = manifest_path.into();
        let export_root = manifest_path
            .parent()
            .ok_or_else(|| format!("manifest path has no parent: {}", manifest_path.display()))?
            .to_path_buf();
        let mut paths = Self::from_root(export_root);
        paths.manifest_path = manifest_path;
        Ok(paths)
    }

    fn attach_notes_path(&mut self, notes_ref: Option<&str>) -> Result<(), String> {
        self.notes_path = notes_ref
            .map(|notes| resolve_export_reference_path(&self.export_root, notes))
            .transpose()?;
        Ok(())
    }
}

pub fn inspect_export_bundle(
    request: &ChronohornExportInspectRequest,
) -> Result<ChronohornExportInspectReport, String> {
    let bundle = ChronohornExportBundlePaths::from_root(&request.export_root);
    inspect_bundle_paths(bundle)
}

pub fn inspect_manifest_json(
    manifest_path: impl Into<PathBuf>,
) -> Result<ChronohornExportInspectReport, String> {
    let bundle = ChronohornExportBundlePaths::from_manifest_path(manifest_path)?;
    inspect_bundle_paths(bundle)
}

pub fn prepare_replay_bundle(
    request: &ChronohornReplayPrepRequest,
) -> Result<ChronohornReplayPrepReport, String> {
    let mut bundle = ChronohornExportBundlePaths::from_root(&request.export_root);
    let ChronohornLoadedBundle {
        manifest,
        learned_state_index,
        checksums: _checksums,
        ..
    } = load_export_bundle(&bundle)?;
    bundle.attach_notes_path(manifest.notes_ref.as_deref())?;
    let notes_path = bundle
        .notes_path
        .as_ref()
        .filter(|path| path.is_file())
        .map(|path| path.display().to_string());

    let tensor_plan = load_tensor_layouts(&bundle, &learned_state_index)?;
    let mut storage_counts = BTreeMap::new();
    let mut dtype_counts = BTreeMap::new();
    let mut total_file_bytes = 0usize;
    let mut total_payload_bytes = 0usize;
    for entry in &tensor_plan {
        *storage_counts.entry(entry.storage.clone()).or_insert(0) += 1;
        *dtype_counts.entry(entry.dtype.clone()).or_insert(0) += 1;
        total_file_bytes += entry.file_bytes;
        total_payload_bytes += entry.payload_bytes;
    }

    Ok(ChronohornReplayPrepReport {
        export_root: bundle.export_root.display().to_string(),
        manifest_path: bundle.manifest_path.display().to_string(),
        learned_state_index_path: bundle.learned_state_index_path.display().to_string(),
        notes_path,
        abi_name: manifest.abi_name,
        abi_version: manifest.abi_version,
        exporter_version: manifest.exporter_version,
        model_family_id: manifest.model_family_id,
        model_variant_id: manifest.model_variant_id,
        kernel_version: manifest.kernel_version,
        tokenizer_id: manifest.tokenizer_id,
        data_root_id: manifest.data_root_id,
        artifact_role: manifest.artifact_role,
        tensor_format: learned_state_index.tensor_format,
        tensor_count: learned_state_index.tensor_count,
        tensor_plan_count: tensor_plan.len(),
        total_file_bytes,
        total_payload_bytes,
        storage_counts,
        dtype_counts,
        tensor_plan,
        packed_memory_attached: manifest.packed_memory.is_some(),
        runtime_ready: true,
    })
}

fn inspect_bundle_paths(
    mut bundle: ChronohornExportBundlePaths,
) -> Result<ChronohornExportInspectReport, String> {
    let ChronohornLoadedBundle {
        manifest,
        learned_state_index,
        checksums: _checksums,
        ..
    } = load_export_bundle(&bundle)?;

    bundle.attach_notes_path(manifest.notes_ref.as_deref())?;

    let blob_count = learned_state_index.tensor_index.len();
    let notes_path = bundle
        .notes_path
        .as_ref()
        .filter(|path| path.is_file())
        .map(|path| path.display().to_string());

    Ok(ChronohornExportInspectReport {
        export_root: bundle.export_root.display().to_string(),
        manifest_path: bundle.manifest_path.display().to_string(),
        learned_state_index_path: bundle.learned_state_index_path.display().to_string(),
        checksums_path: bundle.checksums_path.display().to_string(),
        abi_name: manifest.abi_name,
        abi_version: manifest.abi_version,
        exporter_version: manifest.exporter_version,
        model_family_id: manifest.model_family_id,
        model_variant_id: manifest.model_variant_id,
        kernel_version: manifest.kernel_version,
        tokenizer_id: manifest.tokenizer_id,
        data_root_id: manifest.data_root_id,
        artifact_role: manifest.artifact_role,
        learned_state_tensor_count: manifest.learned_state.tensor_count,
        learned_state_blob_count: blob_count,
        packed_memory_attached: manifest.packed_memory.is_some(),
        notes_path,
    })
}

pub fn inspect_tensor_inventory(
    request: &ChronohornTensorInventoryInspectRequest,
) -> Result<ChronohornTensorInventoryInspectReport, String> {
    let mut bundle = ChronohornExportBundlePaths::from_root(&request.export_root);
    let ChronohornLoadedBundle {
        manifest,
        learned_state_index,
        checksums,
        ..
    } = load_export_bundle(&bundle)?;
    bundle.attach_notes_path(manifest.notes_ref.as_deref())?;
    let notes_path = bundle
        .notes_path
        .as_ref()
        .filter(|path| path.is_file())
        .map(|path| path.display().to_string());

    let mut storage_counts = BTreeMap::new();
    for entry in &learned_state_index.tensor_index {
        *storage_counts.entry(entry.storage.clone()).or_insert(0) += 1;
    }

    Ok(ChronohornTensorInventoryInspectReport {
        export_root: bundle.export_root.display().to_string(),
        manifest_path: bundle.manifest_path.display().to_string(),
        learned_state_index_path: bundle.learned_state_index_path.display().to_string(),
        abi_name: manifest.abi_name,
        abi_version: manifest.abi_version,
        exporter_version: manifest.exporter_version,
        model_family_id: manifest.model_family_id,
        model_variant_id: manifest.model_variant_id,
        kernel_version: manifest.kernel_version,
        tensor_format: learned_state_index.tensor_format,
        tensor_count: learned_state_index.tensor_count,
        checksum_algorithm: checksums.algorithm,
        blob_checksum_count: checksums.blobs.len(),
        tensor_entries: learned_state_index.tensor_index,
        storage_counts,
        packed_memory_attached: manifest.packed_memory.is_some(),
        notes_path,
    })
}

pub fn verify_tensor_probe(
    request: &ChronohornTensorProbeVerifyRequest,
) -> Result<ChronohornTensorProbeVerifyReport, String> {
    let mut bundle = ChronohornExportBundlePaths::from_root(&request.export_root);
    let ChronohornLoadedBundle {
        manifest,
        learned_state_index,
        checksums: _checksums,
        ..
    } = load_export_bundle(&bundle)?;
    bundle.attach_notes_path(manifest.notes_ref.as_deref())?;

    let loaded = load_tensor_probes(&bundle, &learned_state_index)?;
    let notes_probe = load_notes_probe(&bundle)?;
    let notes_path = bundle
        .notes_path
        .as_ref()
        .filter(|path| path.is_file())
        .map(|path| path.display().to_string());

    let mut mismatches = Vec::new();
    let notes_probe_present = notes_probe.is_some();
    if let Some(notes_probe) = notes_probe {
        compare_tensor_probe(&loaded, &notes_probe, &mut mismatches);
    } else {
        mismatches.push("notes.json does not contain tensor_probe_v1".to_string());
    }

    let total_blob_bytes = loaded.iter().map(|entry| entry.data_bytes).sum();
    Ok(ChronohornTensorProbeVerifyReport {
        export_root: bundle.export_root.display().to_string(),
        manifest_path: bundle.manifest_path.display().to_string(),
        learned_state_index_path: bundle.learned_state_index_path.display().to_string(),
        notes_path,
        tensor_count: learned_state_index.tensor_count,
        loaded_blob_count: loaded.len(),
        total_blob_bytes,
        notes_probe_present,
        probe_match: mismatches.is_empty(),
        mismatches,
    })
}

pub fn load_json_file<T: DeserializeOwned>(path: &Path) -> Result<T, String> {
    let raw = fs::read_to_string(path).map_err(|err| format!("read {}: {err}", path.display()))?;
    serde_json::from_str(&raw).map_err(|err| format!("parse json {}: {err}", path.display()))
}

/// Resolve a reference string from an export bundle to an absolute path.
///
/// Returns `Err` if the reference is an absolute path or contains any parent-directory (`..`)
/// component, both of which would allow a malformed bundle to escape the export root.
pub fn resolve_export_reference_path(
    export_root: &Path,
    reference: &str,
) -> Result<PathBuf, String> {
    let path = Path::new(reference);
    if path.is_absolute() {
        return Err(format!(
            "absolute paths not allowed in export bundles: {reference}"
        ));
    }
    for component in path.components() {
        if matches!(component, std::path::Component::ParentDir) {
            return Err(format!(
                "path traversal not allowed in export bundles: {reference}"
            ));
        }
    }
    Ok(export_root.join(path))
}

fn load_export_bundle(
    bundle: &ChronohornExportBundlePaths,
) -> Result<ChronohornLoadedBundle, String> {
    let manifest: ChronohornExportManifest = load_json_file(&bundle.manifest_path)?;
    manifest.validate_minimal()?;

    let learned_state_index: ChronohornExportLearnedStateIndex =
        load_json_file(&bundle.learned_state_index_path)?;
    learned_state_index.validate_minimal()?;

    let checksums_sidecar: ChronohornExportChecksums = load_json_file(&bundle.checksums_path)?;
    validate_checksums_consistency(&manifest, &learned_state_index, &checksums_sidecar)?;
    // TODO: verify blob checksums by recomputing from file contents.
    // Currently only cross-references between manifest/index/sidecar JSON files are checked;
    // the actual blob bytes on disk are not verified against the recorded checksums.
    // Requires adding a sha2 (or equivalent) dependency to chronohorn-runtime.

    Ok(ChronohornLoadedBundle {
        bundle: bundle.clone(),
        manifest,
        learned_state_index,
        checksums: checksums_sidecar,
    })
}

fn load_notes_probe(
    bundle: &ChronohornExportBundlePaths,
) -> Result<Option<ChronohornNotesTensorProbe>, String> {
    let Some(notes_path) = bundle.notes_path.as_ref() else {
        return Ok(None);
    };
    if !notes_path.is_file() {
        return Ok(None);
    }
    let notes: ChronohornNotesPayload = load_json_file(notes_path)?;
    Ok(notes.tensor_probe_v1)
}

fn load_tensor_probes(
    bundle: &ChronohornExportBundlePaths,
    learned_state_index: &ChronohornExportLearnedStateIndex,
) -> Result<Vec<ChronohornLoadedTensorProbe>, String> {
    let tensor_layouts = load_tensor_layouts(bundle, learned_state_index)?;
    let mut out = Vec::with_capacity(tensor_layouts.len());
    for layout in tensor_layouts {
        let blob_path = resolve_export_reference_path(&bundle.export_root, &layout.blob)?;
        let bytes =
            fs::read(&blob_path).map_err(|err| format!("read {}: {err}", blob_path.display()))?;
        let meta = NpyMeta {
            dtype: layout.dtype.clone(),
            shape: layout.shape.clone(),
            data_offset: layout.data_offset,
        };
        let probe = load_f32_tensor_probe(layout.name, &bytes, meta)?;
        out.push(probe);
    }
    Ok(out)
}

fn load_tensor_layouts(
    bundle: &ChronohornExportBundlePaths,
    learned_state_index: &ChronohornExportLearnedStateIndex,
) -> Result<Vec<ChronohornReplayTensorPlan>, String> {
    let mut out = Vec::with_capacity(learned_state_index.tensor_index.len());
    for entry in &learned_state_index.tensor_index {
        let blob_path = resolve_export_reference_path(&bundle.export_root, &entry.blob)?;
        let bytes =
            fs::read(&blob_path).map_err(|err| format!("read {}: {err}", blob_path.display()))?;
        let meta = parse_npy_meta(&bytes)?;
        let normalized_dtype = normalize_npy_dtype(&meta.dtype);
        if normalized_dtype != entry.dtype {
            return Err(format!(
                "tensor {} dtype mismatch: index={} blob={}",
                entry.name, entry.dtype, normalized_dtype
            ));
        }
        if meta.shape != entry.shape {
            return Err(format!(
                "tensor {} shape mismatch: index={:?} blob={:?}",
                entry.name, entry.shape, meta.shape
            ));
        }
        let payload_bytes = shape_elem_count(&meta.shape)?
            .checked_mul(dtype_byte_width(&normalized_dtype).ok_or_else(|| {
                format!(
                    "tensor {} uses unsupported dtype {}",
                    entry.name, normalized_dtype
                )
            })?)
            .ok_or_else(|| format!("payload byte overflow for tensor {}", entry.name))?;
        out.push(ChronohornReplayTensorPlan {
            name: entry.name.clone(),
            dtype: normalized_dtype,
            storage: entry.storage.clone(),
            shape: entry.shape.clone(),
            blob: entry.blob.clone(),
            checksum: entry.checksum.clone(),
            file_bytes: bytes.len(),
            payload_bytes,
            data_offset: meta.data_offset,
        });
    }
    Ok(out)
}

fn compare_tensor_probe(
    loaded: &[ChronohornLoadedTensorProbe],
    expected: &ChronohornNotesTensorProbe,
    mismatches: &mut Vec<String>,
) {
    if expected.tensor_count != loaded.len() {
        mismatches.push(format!(
            "tensor_count mismatch: notes={} loaded={}",
            expected.tensor_count,
            loaded.len()
        ));
    }
    let expected_by_name: BTreeMap<&str, &ChronohornNotesTensorProbeEntry> = expected
        .tensors
        .iter()
        .map(|entry| (entry.name.as_str(), entry))
        .collect();
    for loaded_entry in loaded {
        let Some(expected_entry) = expected_by_name.get(loaded_entry.name.as_str()) else {
            mismatches.push(format!(
                "missing tensor probe entry for {}",
                loaded_entry.name
            ));
            continue;
        };
        if expected_entry.dtype != loaded_entry.dtype {
            mismatches.push(format!(
                "dtype mismatch for {}: notes={} loaded={}",
                loaded_entry.name, expected_entry.dtype, loaded_entry.dtype
            ));
        }
        if expected_entry.shape != loaded_entry.shape {
            mismatches.push(format!(
                "shape mismatch for {}: notes={:?} loaded={:?}",
                loaded_entry.name, expected_entry.shape, loaded_entry.shape
            ));
        }
        if expected_entry.elem_count != loaded_entry.elem_count {
            mismatches.push(format!(
                "elem_count mismatch for {}: notes={} loaded={}",
                loaded_entry.name, expected_entry.elem_count, loaded_entry.elem_count
            ));
        }
        compare_float(
            loaded_entry.name.as_str(),
            "sum64",
            expected_entry.sum64,
            loaded_entry.sum64,
            mismatches,
        );
        compare_float(
            loaded_entry.name.as_str(),
            "absmax",
            expected_entry.absmax,
            loaded_entry.absmax,
            mismatches,
        );
        compare_float(
            loaded_entry.name.as_str(),
            "l2_norm",
            expected_entry.l2_norm,
            loaded_entry.l2_norm,
            mismatches,
        );
        if expected_entry.sample.len() != loaded_entry.sample.len() {
            mismatches.push(format!(
                "sample length mismatch for {}: notes={} loaded={}",
                loaded_entry.name,
                expected_entry.sample.len(),
                loaded_entry.sample.len()
            ));
        }
        for (idx, (expected_value, loaded_value)) in expected_entry
            .sample
            .iter()
            .zip(loaded_entry.sample.iter())
            .enumerate()
        {
            compare_float(
                loaded_entry.name.as_str(),
                &format!("sample[{idx}]"),
                *expected_value,
                *loaded_value,
                mismatches,
            );
        }
    }
    for expected_entry in &expected.tensors {
        if !loaded.iter().any(|entry| entry.name == expected_entry.name) {
            mismatches.push(format!(
                "notes contain tensor probe for missing tensor {}",
                expected_entry.name
            ));
        }
    }
}

fn compare_float(
    tensor_name: &str,
    field: &str,
    expected: f64,
    loaded: f64,
    mismatches: &mut Vec<String>,
) {
    let tolerance = 1e-9_f64.max(1e-7 * expected.abs().max(loaded.abs()).max(1.0));
    if (expected - loaded).abs() > tolerance {
        mismatches.push(format!(
            "{} {} mismatch: notes={} loaded={} tol={}",
            tensor_name, field, expected, loaded, tolerance
        ));
    }
}

fn load_f32_tensor_probe(
    name: String,
    bytes: &[u8],
    meta: NpyMeta,
) -> Result<ChronohornLoadedTensorProbe, String> {
    if normalize_npy_dtype(&meta.dtype) != "float32" {
        return Err(format!(
            "tensor {name} uses unsupported dtype {} for tensor probe",
            meta.dtype
        ));
    }
    let elem_count = shape_elem_count(&meta.shape)?;
    let expected_bytes = elem_count
        .checked_mul(4)
        .ok_or_else(|| format!("byte overflow for {name}"))?;
    let payload = bytes
        .get(meta.data_offset..meta.data_offset + expected_bytes)
        .ok_or_else(|| format!("truncated payload for {name}: expected {expected_bytes} bytes"))?;

    let mut sum64 = 0.0_f64;
    let mut absmax = 0.0_f64;
    let mut l2_acc = 0.0_f64;
    let mut sample = Vec::new();
    for chunk in payload.chunks_exact(4) {
        let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f64;
        if sample.len() < 4 {
            sample.push(value);
        }
        sum64 += value;
        absmax = absmax.max(value.abs());
        l2_acc += value * value;
    }
    Ok(ChronohornLoadedTensorProbe {
        name,
        dtype: "float32".to_string(),
        shape: meta.shape,
        elem_count,
        data_bytes: expected_bytes,
        sum64,
        absmax,
        l2_norm: l2_acc.sqrt(),
        sample,
    })
}

fn parse_npy_meta(bytes: &[u8]) -> Result<NpyMeta, String> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err("invalid npy magic".to_string());
    }
    let major = bytes[6];
    let minor = bytes[7];
    let (header_len, data_offset) = match (major, minor) {
        (1, _) => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        (2, _) | (3, _) => (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        ),
        _ => return Err(format!("unsupported npy version {major}.{minor}")),
    };
    let header_end = data_offset + header_len;
    if header_end > bytes.len() {
        return Err("truncated npy header".to_string());
    }
    let header = std::str::from_utf8(&bytes[data_offset..header_end])
        .map_err(|err| format!("header utf8: {err}"))?;
    let dtype = extract_between(header, "'descr':", ",")
        .or_else(|| extract_between(header, "\"descr\":", ","))
        .unwrap_or_else(|| "?".to_string())
        .trim()
        .trim_matches('\'')
        .trim_matches('"')
        .to_string();
    let shape_block =
        extract_paren_block(header, "shape").ok_or_else(|| "missing shape".to_string())?;
    let shape = parse_shape(&shape_block)?;
    Ok(NpyMeta {
        dtype,
        shape,
        data_offset: header_end,
    })
}

fn extract_between(text: &str, start_pat: &str, end_pat: &str) -> Option<String> {
    let start = text.find(start_pat)? + start_pat.len();
    let rest = &text[start..];
    let end = rest.find(end_pat)?;
    Some(rest[..end].to_string())
}

fn extract_paren_block(text: &str, key: &str) -> Option<String> {
    let key_pos = text.find(key)?;
    let open_rel = text[key_pos..].find('(')?;
    let open = key_pos + open_rel;
    let close_rel = text[open..].find(')')?;
    let close = open + close_rel;
    Some(text[open + 1..close].to_string())
}

fn parse_shape(shape_block: &str) -> Result<Vec<usize>, String> {
    let trimmed = shape_block.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }
    let mut shape = Vec::new();
    for part in trimmed.split(',') {
        let token = part.trim();
        if token.is_empty() {
            continue;
        }
        let value = token
            .parse::<usize>()
            .map_err(|err| format!("parse shape component {token:?}: {err}"))?;
        shape.push(value);
    }
    Ok(shape)
}

fn shape_elem_count(shape: &[usize]) -> Result<usize, String> {
    let mut count = 1usize;
    for &dim in shape {
        count = count
            .checked_mul(dim)
            .ok_or_else(|| format!("shape overflow for {:?}", shape))?;
    }
    Ok(count)
}

fn normalize_npy_dtype(dtype: &str) -> String {
    match dtype {
        "<f4" | "|f4" => "float32".to_string(),
        "<f8" | "|f8" => "float64".to_string(),
        "<i8" | "|i8" => "int64".to_string(),
        "<i4" | "|i4" => "int32".to_string(),
        "<u8" | "|u8" => "uint64".to_string(),
        "<u4" | "|u4" => "uint32".to_string(),
        other => other.to_string(),
    }
}

fn dtype_byte_width(dtype: &str) -> Option<usize> {
    match dtype {
        "float32" | "int32" | "uint32" => Some(4),
        "float64" | "int64" | "uint64" => Some(8),
        _ => None,
    }
}

fn validate_checksums_consistency(
    manifest: &ChronohornExportManifest,
    learned_state_index: &ChronohornExportLearnedStateIndex,
    checksums_sidecar: &ChronohornExportChecksums,
) -> Result<(), String> {
    if manifest.checksums.algorithm != checksums_sidecar.algorithm {
        return Err(format!(
            "checksum algorithm mismatch: manifest={} sidecar={}",
            manifest.checksums.algorithm, checksums_sidecar.algorithm
        ));
    }
    if manifest.checksums.learned_state_index != checksums_sidecar.learned_state_index {
        return Err(
            "learned_state_index checksum mismatch between manifest and sidecar".to_string(),
        );
    }
    if manifest.checksums.manifest_body != checksums_sidecar.manifest_body {
        return Err("manifest_body checksum mismatch between manifest and sidecar".to_string());
    }
    if manifest.checksums.blobs.len() != learned_state_index.tensor_index.len() {
        return Err(format!(
            "blob checksum count mismatch: checksums={} tensor_index={}",
            manifest.checksums.blobs.len(),
            learned_state_index.tensor_index.len()
        ));
    }
    for tensor in &learned_state_index.tensor_index {
        let expected =
            manifest.checksums.blobs.get(&tensor.name).ok_or_else(|| {
                format!("missing manifest blob checksum for tensor {}", tensor.name)
            })?;
        if expected != &tensor.checksum {
            return Err(format!(
                "tensor checksum mismatch for {}: manifest={} index={}",
                tensor.name, expected, tensor.checksum
            ));
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChronohornLoadedBundle {
    pub bundle: ChronohornExportBundlePaths,
    pub manifest: ChronohornExportManifest,
    pub learned_state_index: ChronohornExportLearnedStateIndex,
    pub checksums: ChronohornExportChecksums,
}

pub fn load_export_bundle_material(
    export_root: impl Into<PathBuf>,
) -> Result<ChronohornLoadedBundle, String> {
    let bundle = ChronohornExportBundlePaths::from_root(export_root);
    load_export_bundle(&bundle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "chronohorn_runtime_{name}_{}_{}",
            std::process::id(),
            nanos
        ));
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    fn write_json(path: &Path, value: &serde_json::Value) {
        fs::write(path, serde_json::to_vec(value).expect("serialize json")).expect("write json");
    }

    fn encode_npy_f32_v1(values: &[f32], shape: &[usize]) -> Vec<u8> {
        let shape_repr = if shape.is_empty() {
            "()".to_string()
        } else if shape.len() == 1 {
            format!("({},)", shape[0])
        } else {
            let dims = shape
                .iter()
                .map(|dim| dim.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            format!("({dims},)")
        };
        let mut header = format!(
            "{{'descr': '<f4', 'fortran_order': False, 'shape': {}, }}",
            shape_repr
        );
        let preamble_len = 10usize;
        let total_without_padding = preamble_len + header.len() + 1;
        let pad_len = (16 - (total_without_padding % 16)) % 16;
        header.push_str(&" ".repeat(pad_len));
        header.push('\n');

        let mut out = Vec::new();
        out.extend_from_slice(b"\x93NUMPY");
        out.push(1);
        out.push(0);
        out.extend_from_slice(&(header.len() as u16).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        for value in values {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out
    }

    fn write_test_bundle(root: &Path, bad_probe: bool) {
        let learned_state_dir = root.join("learned_state");
        let blobs_dir = learned_state_dir.join("blobs");
        fs::create_dir_all(&blobs_dir).expect("create blob dir");

        let blob_rel = "learned_state/blobs/0000__toy.weight.npy";
        let blob_bytes = encode_npy_f32_v1(&[1.0, -2.0], &[2]);
        fs::write(root.join(blob_rel), &blob_bytes).expect("write blob");

        write_json(
            &learned_state_dir.join("index.json"),
            &serde_json::json!({
                "tensor_format": "npy",
                "tensor_count": 1,
                "tensor_index": [{
                    "name": "toy.weight",
                    "shape": [2],
                    "dtype": "float32",
                    "storage": "blob_ref",
                    "blob": blob_rel,
                    "checksum": "blake2b:test-blob",
                }]
            }),
        );

        write_json(
            &root.join("checksums.json"),
            &serde_json::json!({
                "algorithm": "blake2b",
                "manifest_body": "blake2b:test-manifest",
                "learned_state_index": "blake2b:test-index",
                "blobs": {
                    "toy.weight": "blake2b:test-blob"
                }
            }),
        );

        let sum64 = if bad_probe { 0.0 } else { -1.0 };
        write_json(
            &root.join("notes.json"),
            &serde_json::json!({
                "tensor_probe_v1": {
                    "tensor_count": 1,
                    "tensors": [{
                        "name": "toy.weight",
                        "dtype": "float32",
                        "shape": [2],
                        "elem_count": 2,
                        "sum64": sum64,
                        "absmax": 2.0,
                        "l2_norm": 5.0_f64.sqrt(),
                        "sample": [1.0, -2.0]
                    }]
                }
            }),
        );

        write_json(
            &root.join("manifest.json"),
            &serde_json::json!({
                "abi_name": "opc-export",
                "abi_version": "1.0.0",
                "exporter_version": "chronohorn-export-test",
                "exported_utc": "2026-03-30T00:00:00Z",
                "model_family_id": "toy",
                "model_variant_id": "toy-v1",
                "kernel_version": "opc-kernel-test",
                "tokenizer_id": "toy-tokenizer",
                "data_root_id": "toy-data",
                "artifact_role": "replay",
                "deterministic_substrate": {"kind": "toy"},
                "learned_state": {
                    "tensor_format": "npy",
                    "tensor_count": 1,
                    "tensor_index_ref": "learned_state/index.json"
                },
                "checksums": {
                    "algorithm": "blake2b",
                    "manifest_body": "blake2b:test-manifest",
                    "learned_state_index": "blake2b:test-index",
                    "blobs": {
                        "toy.weight": "blake2b:test-blob"
                    }
                },
                "notes_ref": "notes.json"
            }),
        );
    }

    #[test]
    fn verify_probe_passes_on_matching_bundle() {
        let root = unique_temp_dir("verify_probe_ok");
        write_test_bundle(&root, false);

        let report = verify_tensor_probe(&ChronohornTensorProbeVerifyRequest {
            export_root: root.clone(),
        })
        .expect("verify tensor probe");

        assert!(report.probe_match, "expected probe to match: {:?}", report);
        assert!(report.mismatches.is_empty());
        assert_eq!(report.tensor_count, 1);
        assert_eq!(report.loaded_blob_count, 1);
        assert!(report.notes_probe_present);
    }

    #[test]
    fn verify_probe_reports_mismatch() {
        let root = unique_temp_dir("verify_probe_bad");
        write_test_bundle(&root, true);

        let report = verify_tensor_probe(&ChronohornTensorProbeVerifyRequest {
            export_root: root.clone(),
        })
        .expect("verify tensor probe");

        assert!(!report.probe_match, "expected probe mismatch");
        assert!(!report.mismatches.is_empty());
        assert!(
            report
                .mismatches
                .iter()
                .any(|line| line.contains("sum64 mismatch")),
            "expected sum64 mismatch, got {:?}",
            report.mismatches
        );
    }

    #[test]
    fn prepare_replay_reports_tensor_plan() {
        let root = unique_temp_dir("prepare_replay");
        write_test_bundle(&root, false);
        let blob_len = fs::read(root.join("learned_state/blobs/0000__toy.weight.npy"))
            .expect("read blob")
            .len();

        let report = prepare_replay_bundle(&ChronohornReplayPrepRequest {
            export_root: root.clone(),
        })
        .expect("prepare replay");

        assert!(report.runtime_ready);
        assert_eq!(report.tensor_count, 1);
        assert_eq!(report.tensor_plan_count, 1);
        assert_eq!(report.total_file_bytes, blob_len);
        assert_eq!(report.total_payload_bytes, 8);
        assert_eq!(report.storage_counts.get("blob_ref"), Some(&1));
        assert_eq!(report.dtype_counts.get("float32"), Some(&1));
        assert_eq!(report.tensor_plan[0].name, "toy.weight");
        assert_eq!(report.tensor_plan[0].payload_bytes, 8);
        assert_eq!(report.tensor_plan[0].file_bytes, blob_len);
        assert_eq!(
            report.tensor_plan[0].blob,
            "learned_state/blobs/0000__toy.weight.npy"
        );
    }
}
