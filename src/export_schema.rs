use serde::Serialize;

use chronohorn::data::ResolvedDataRoot;

pub const TOKEN_MATCHSKIP_ALIASES: &[&str] = &[
    "token_matchskip_bridge",
    "matchskip_bridge",
    "matchskip",
];

#[derive(Debug, Clone, Serialize)]
pub struct ChronohornExportManifest {
    pub schema: &'static str,
    pub schema_version: &'static str,
    pub family: &'static str,
    pub aliases: &'static [&'static str],
    pub source_command: &'static str,
    pub data_root_spec: String,
    pub data_root_resolution: ResolvedDataRoot,
    pub runner_name: &'static str,
    pub selected_runtime_gate: String,
    pub selected_runtime_lambda: Option<f64>,
    pub chunk_size: usize,
    pub max_chunks: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChronohornExportBundle<'a, R, A> {
    pub manifest: ChronohornExportManifest,
    pub report: &'a R,
    pub audit: &'a A,
}
