use serde::{Deserialize, Serialize};

pub const CHRONOHORN_DOCTRINE_VERSION: &str = "1";

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SystemRole {
    Oracle,
    Compressor,
    Bridge,
    Audit,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OracleTargetKind {
    CandidateSetLeq4,
    CandidateSetLeq8,
    CleanBridgeScore,
    MemoryTrust,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleTargetProvenance {
    pub source_family: String,
    pub source_path: Option<String>,
    pub leave_one_out: bool,
    pub contamination_adjusted: bool,
    pub future_blind_at_runtime: bool,
    pub offline_only: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleTargetStats {
    pub bidirectional_fraction: f64,
    pub left_only_fraction: f64,
    pub self_inclusion_uplift: f64,
    pub future_context_uplift: f64,
    pub clean_bridge_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleBridgeTarget {
    pub kind: OracleTargetKind,
    pub radius: usize,
    pub candidate_k: usize,
    pub stats: OracleTargetStats,
    pub provenance: OracleTargetProvenance,
}

#[derive(Debug, Clone, Serialize)]
pub struct BridgeDoctrine {
    pub version: &'static str,
    pub doctrine: &'static str,
    pub roles: [SystemRole; 4],
    pub oracle_contract: &'static [&'static str],
    pub compressor_contract: &'static [&'static str],
    pub bridge_contract: &'static [&'static str],
    pub audit_contract: &'static [&'static str],
}

pub fn bridge_doctrine() -> BridgeDoctrine {
    BridgeDoctrine {
        version: CHRONOHORN_DOCTRINE_VERSION,
        doctrine:
            "chronohorn is a causal compressor trained and shaped by a noncausal oracle, with the oracle excluded from runtime and the runtime attacked until the gain survives",
        roles: [
            SystemRole::Oracle,
            SystemRole::Compressor,
            SystemRole::Bridge,
            SystemRole::Audit,
        ],
        oracle_contract: &[
            "offline only",
            "leave-one-out structural analysis",
            "candidate-set labels",
            "contamination audits",
            "never in eval path",
        ],
        compressor_contract: &[
            "packed memory plus orthogonal experts",
            "actual scorer",
            "actual bpb",
            "fully causal",
        ],
        bridge_contract: &[
            "oracle signal becomes legal runtime leverage",
            "target generation",
            "feature prediction",
            "gating and control learning",
            "the seam that matters most",
        ],
        audit_contract: &[
            "prove the bridge did not smuggle the oracle into runtime",
            "prove the scorer is still one causal prequential process",
        ],
    }
}

impl OracleBridgeTarget {
    pub fn runtime_legal_by_construction(&self) -> bool {
        self.provenance.leave_one_out
            && self.provenance.contamination_adjusted
            && self.provenance.future_blind_at_runtime
            && self.provenance.offline_only
    }
}

pub fn clean_candidate4_target(
    radius: usize,
    left_only_fraction: f64,
    bidirectional_fraction: f64,
    self_inclusion_uplift: f64,
    future_context_uplift: f64,
    source_family: impl Into<String>,
    source_path: Option<String>,
) -> OracleBridgeTarget {
    OracleBridgeTarget {
        kind: OracleTargetKind::CandidateSetLeq4,
        radius,
        candidate_k: 4,
        stats: OracleTargetStats {
            bidirectional_fraction,
            left_only_fraction,
            self_inclusion_uplift,
            future_context_uplift,
            clean_bridge_score: left_only_fraction - self_inclusion_uplift,
        },
        provenance: OracleTargetProvenance {
            source_family: source_family.into(),
            source_path,
            leave_one_out: true,
            contamination_adjusted: true,
            future_blind_at_runtime: true,
            offline_only: true,
        },
    }
}

pub fn render_bridge_doctrine() -> String {
    let doctrine = bridge_doctrine();
    let mut out = String::new();
    out.push_str("chronohorn_doctrine\n");
    out.push_str(&format!("version: {}\n", doctrine.version));
    out.push_str(&format!("doctrine: {}\n", doctrine.doctrine));
    out.push_str("roles:\n");
    out.push_str("  oracle:\n");
    for item in doctrine.oracle_contract {
        out.push_str(&format!("    - {}\n", item));
    }
    out.push_str("  compressor:\n");
    for item in doctrine.compressor_contract {
        out.push_str(&format!("    - {}\n", item));
    }
    out.push_str("  bridge:\n");
    for item in doctrine.bridge_contract {
        out.push_str(&format!("    - {}\n", item));
    }
    out.push_str("  audit:\n");
    for item in doctrine.audit_contract {
        out.push_str(&format!("    - {}\n", item));
    }
    out
}
