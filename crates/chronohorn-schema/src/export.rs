use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

pub const CHRONOHORN_EXPORT_ABI_NAME: &str = "opc-export";
pub const CHRONOHORN_EXPORT_ABI_VERSION: &str = "1.0.0";

pub const CHRONOHORN_EXPORT_REQUIRED_FIELDS: &[&str] = &[
    "abi_name",
    "abi_version",
    "exporter_version",
    "exported_utc",
    "model_family_id",
    "model_variant_id",
    "kernel_version",
    "tokenizer_id",
    "data_root_id",
    "deterministic_substrate",
    "learned_state",
    "checksums",
    "artifact_role",
];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChronohornExportLearnedStateRef {
    pub tensor_format: String,
    pub tensor_count: usize,
    pub tensor_index_ref: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChronohornExportChecksums {
    pub algorithm: String,
    pub manifest_body: String,
    pub learned_state_index: String,
    pub blobs: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChronohornExportManifest {
    pub abi_name: String,
    pub abi_version: String,
    pub exporter_version: String,
    pub exported_utc: String,
    pub model_family_id: String,
    pub model_variant_id: String,
    pub kernel_version: String,
    pub tokenizer_id: String,
    pub data_root_id: String,
    pub deterministic_substrate: serde_json::Value,
    pub learned_state: ChronohornExportLearnedStateRef,
    pub checksums: ChronohornExportChecksums,
    pub artifact_role: String,
    #[serde(default)]
    pub packed_memory: Option<serde_json::Value>,
    #[serde(default)]
    pub notes_ref: Option<String>,
    #[serde(default)]
    pub dtype_policy: Option<String>,
    #[serde(default)]
    pub quantization_policy: Option<String>,
    #[serde(default)]
    pub sequence_length: Option<usize>,
    #[serde(default)]
    pub vocab_size: Option<usize>,
    #[serde(default)]
    pub train_step: Option<usize>,
    #[serde(default)]
    pub train_wallclock_s: Option<f64>,
}

impl ChronohornExportManifest {
    pub fn validate_minimal(&self) -> Result<(), String> {
        if self.abi_name != CHRONOHORN_EXPORT_ABI_NAME {
            return Err(format!(
                "unsupported abi_name: {} (expected {})",
                self.abi_name, CHRONOHORN_EXPORT_ABI_NAME
            ));
        }
        if self.abi_version.is_empty() {
            return Err("abi_version must not be empty".to_string());
        }
        if self.exporter_version.is_empty() {
            return Err("exporter_version must not be empty".to_string());
        }
        if self.model_family_id.is_empty() {
            return Err("model_family_id must not be empty".to_string());
        }
        if self.model_variant_id.is_empty() {
            return Err("model_variant_id must not be empty".to_string());
        }
        if self.kernel_version.is_empty() {
            return Err("kernel_version must not be empty".to_string());
        }
        if self.learned_state.tensor_count == 0 {
            return Err("learned_state.tensor_count must be > 0".to_string());
        }
        if self.learned_state.tensor_index_ref.is_empty() {
            return Err("learned_state.tensor_index_ref must not be empty".to_string());
        }
        if self.checksums.algorithm.is_empty() {
            return Err("checksums.algorithm must not be empty".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChronohornExportTensorEntry {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub storage: String,
    pub blob: String,
    pub checksum: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChronohornExportLearnedStateIndex {
    pub tensor_format: String,
    pub tensor_count: usize,
    pub tensor_index: Vec<ChronohornExportTensorEntry>,
}

impl ChronohornExportLearnedStateIndex {
    pub fn validate_minimal(&self) -> Result<(), String> {
        if self.tensor_format.is_empty() {
            return Err("tensor_format must not be empty".to_string());
        }
        if self.tensor_count != self.tensor_index.len() {
            return Err(format!(
                "tensor_count mismatch: manifest says {}, index has {} entries",
                self.tensor_count,
                self.tensor_index.len()
            ));
        }
        for entry in &self.tensor_index {
            if entry.name.is_empty() {
                return Err("tensor entry name must not be empty".to_string());
            }
            if entry.blob.is_empty() {
                return Err(format!(
                    "tensor entry {} blob must not be empty",
                    entry.name
                ));
            }
            if entry.checksum.is_empty() {
                return Err(format!(
                    "tensor entry {} checksum must not be empty",
                    entry.name
                ));
            }
        }
        Ok(())
    }
}
