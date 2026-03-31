//! Legacy matchskip-family bridge report bundle support.
//!
//! This is not the promoted `opc-export` artifact boundary. It remains only to
//! support older bridge-family report exports that still live in the root Rust
//! command surface.

use serde::Serialize;

use chronohorn_core::data::ResolvedDataRoot;

pub const TOKEN_SEQUENCE_SKIP_BRIDGE_ALIASES: &[&str] =
    &["token_matchskip_bridge", "matchskip_bridge", "matchskip"];

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize)]
pub struct DeclaredBridgeClaim {
    pub declared: bool,
    pub basis: &'static str,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize)]
pub struct BridgeLegalityClaims {
    pub oracle_artifact_runtime_exclusion: DeclaredBridgeClaim,
    pub target_side_lookup_absence_at_eval: DeclaredBridgeClaim,
    pub oracle_supervision_offline_only: DeclaredBridgeClaim,
}

#[derive(Debug, Clone, Serialize)]
pub struct BridgeReportManifest {
    pub schema: &'static str,
    pub schema_version: &'static str,
    pub family: &'static str,
    pub aliases: &'static [&'static str],
    pub oracle_contract_declared: bool,
    pub source_command: &'static str,
    pub data_root_spec: String,
    pub data_root_resolution: ResolvedDataRoot,
    pub runner_name: &'static str,
    pub selected_runtime_gate: String,
    pub selected_runtime_lambda: Option<f64>,
    pub chunk_size: usize,
    pub max_chunks: usize,
}

#[allow(dead_code)]
impl BridgeReportManifest {
    pub fn bridge_legality_claims(&self) -> BridgeLegalityClaims {
        BridgeLegalityClaims {
            oracle_artifact_runtime_exclusion: DeclaredBridgeClaim {
                declared: self.oracle_contract_declared,
                basis: "declared-only: oracle artifacts are kept out of the runtime runner path",
            },
            target_side_lookup_absence_at_eval: DeclaredBridgeClaim {
                declared: true,
                basis: "declared-only: the exported audit path only exercises local score_chunk/adapt_chunk calls",
            },
            oracle_supervision_offline_only: DeclaredBridgeClaim {
                declared: self.oracle_contract_declared,
                basis: "declared-only: oracle supervision is confined to offline target generation and bridge training",
            },
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct BridgeReportBundle<'a, R, A> {
    pub manifest: BridgeReportManifest,
    pub report: &'a R,
    pub audit: &'a A,
}
