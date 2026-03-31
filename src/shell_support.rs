use std::fs;
use std::path::Path;

use serde::Deserialize;

pub(crate) fn parse_usize_flag(
    value: Option<String>,
    label: &str,
    default: usize,
) -> Result<usize, String> {
    match value {
        Some(raw) => raw
            .parse::<usize>()
            .map_err(|err| format!("invalid {label} {raw}: {err}")),
        None => Ok(default),
    }
}

pub(crate) fn parse_f64_flag(
    value: Option<String>,
    label: &str,
    default: f64,
) -> Result<f64, String> {
    match value {
        Some(raw) => raw
            .parse::<f64>()
            .map_err(|err| format!("invalid {label} {raw}: {err}")),
        None => Ok(default),
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct SummaryModel {
    pub(crate) packed_tokens: usize,
    pub(crate) trigram_buckets: usize,
    pub(crate) alpha_bigram: f64,
    pub(crate) alpha_trigram: f64,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SummaryDataset {
    pub(crate) source_path: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SummaryRoot {
    pub(crate) model: SummaryModel,
    pub(crate) dataset: SummaryDataset,
}

pub(crate) fn load_summary(path: &str) -> Result<SummaryRoot, String> {
    let raw = fs::read_to_string(path).map_err(|err| format!("read {path}: {err}"))?;
    serde_json::from_str(&raw).map_err(|err| format!("parse summary {path}: {err}"))
}

pub(crate) fn infer_data_root_from_summary(path: &str) -> Result<String, String> {
    let summary = load_summary(path)?;
    let first = summary
        .dataset
        .source_path
        .split("::")
        .next()
        .ok_or_else(|| format!("could not parse source_path in {path}"))?
        .trim();
    let cleaned = first.trim_end_matches('*').trim_end_matches('/');
    let parent = Path::new(cleaned)
        .parent()
        .ok_or_else(|| format!("could not infer data root from {cleaned}"))?;
    Ok(parent.display().to_string())
}

pub(crate) fn indent_block(text: &str, prefix: &str) -> String {
    let mut out = String::new();
    for line in text.lines() {
        out.push_str(prefix);
        out.push_str(line);
        out.push('\n');
    }
    out
}
