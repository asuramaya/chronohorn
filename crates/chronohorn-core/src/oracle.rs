use crate::bridge::{
    OracleBridgeTarget, OracleTargetKind, OracleTargetProvenance, OracleTargetStats,
    clean_candidate4_target,
};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fs;

#[derive(Debug, Clone, Deserialize)]
pub struct OracleAttackCorpus {
    pub file_count: usize,
    pub total_bytes: usize,
    pub radii: Vec<usize>,
    pub files: Vec<OracleAttackFile>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OracleAttackFile {
    pub path: String,
    pub size: usize,
    pub radii: Vec<OracleAttackRadius>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OracleAttackRadius {
    pub radius: usize,
    pub positions: usize,
    pub bidi_leaveout_candidate4_fraction: f64,
    pub left_leaveout_candidate4_fraction: f64,
    pub self_inclusion_candidate4_uplift: f64,
    pub future_context_candidate4_uplift: f64,
    pub naive_net_removed_bytes: i64,
}

#[derive(Debug, Clone)]
pub struct RadiusSummary {
    pub radius: usize,
    pub file_count: usize,
    pub mean_bidi_leaveout_candidate4: f64,
    pub mean_left_leaveout_candidate4: f64,
    pub mean_self_inclusion_uplift: f64,
    pub mean_future_context_uplift: f64,
    pub mean_naive_net_removed_bytes: f64,
    pub causalizable_share: f64,
    pub clean_bridge_score: f64,
}

#[derive(Debug, Clone)]
pub struct BridgeableRow {
    pub path: String,
    pub size: usize,
    pub radius: usize,
    pub positions: usize,
    pub bidi_leaveout_candidate4_fraction: f64,
    pub left_leaveout_candidate4_fraction: f64,
    pub self_inclusion_candidate4_uplift: f64,
    pub future_context_candidate4_uplift: f64,
    pub naive_net_removed_bytes: i64,
    pub clean_bridge_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlinxOraclePositionLabel {
    pub path: String,
    pub size: usize,
    pub radius: usize,
    pub position: usize,
    pub center: u8,
    pub center_hex: String,
    pub left_context_hex: String,
    pub right_context_hex: String,
    pub bidi_inclusive_support_size: usize,
    pub bidi_leaveout_support_size: usize,
    pub left_leaveout_support_size: usize,
    pub bidi_inclusive_candidate4: bool,
    pub bidi_leaveout_candidate4: bool,
    pub left_leaveout_candidate4: bool,
    pub self_inclusion_support_changed: bool,
    pub future_context_support_changed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlinxOracleTargetRow {
    pub label: BlinxOraclePositionLabel,
    pub targets: Vec<OracleBridgeTarget>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlinxOracleTargetDataset {
    pub schema: String,
    pub source_family: String,
    pub rows: Vec<BlinxOracleTargetRow>,
}

pub fn load_oracle_attack(path: &str) -> Result<OracleAttackCorpus, String> {
    let raw = fs::read_to_string(path).map_err(|err| format!("read {path}: {err}"))?;
    serde_json::from_str(&raw).map_err(|err| format!("parse oracle attack {path}: {err}"))
}

pub fn load_blinx_oracle_position_labels(
    path: &str,
) -> Result<Vec<BlinxOraclePositionLabel>, String> {
    let raw = fs::read_to_string(path).map_err(|err| format!("read {path}: {err}"))?;
    parse_blinx_oracle_position_labels(&raw, path)
}

pub fn load_blinx_oracle_target_dataset(path: &str) -> Result<BlinxOracleTargetDataset, String> {
    let labels = load_blinx_oracle_position_labels(path)?;
    Ok(blinx_oracle_target_dataset_from_labels(labels))
}

pub fn blinx_oracle_target_dataset_from_labels(
    labels: Vec<BlinxOraclePositionLabel>,
) -> BlinxOracleTargetDataset {
    let rows = labels.into_iter().map(blinx_oracle_target_row).collect();
    BlinxOracleTargetDataset {
        schema: "chronohorn.blinx.oracle.position_targets".to_string(),
        source_family: "blinx_oracle_position_export".to_string(),
        rows,
    }
}

pub fn blinx_oracle_target_row(label: BlinxOraclePositionLabel) -> BlinxOracleTargetRow {
    let targets = blinx_oracle_targets_for_label(&label);
    BlinxOracleTargetRow { label, targets }
}

pub fn blinx_oracle_targets_for_label(label: &BlinxOraclePositionLabel) -> Vec<OracleBridgeTarget> {
    vec![
        candidate_set_target_from_label(label, 2, OracleTargetKind::CandidateSetLeq2),
        candidate_set_target_from_label(label, 4, OracleTargetKind::CandidateSetLeq4),
        candidate_set_target_from_label(label, 8, OracleTargetKind::CandidateSetLeq8),
        memory_trust_target_from_label(label),
        bridge_confidence_target_from_label(label),
        clean_bridge_score_target_from_label(label),
    ]
}

fn parse_blinx_oracle_position_labels(
    raw: &str,
    source_path: &str,
) -> Result<Vec<BlinxOraclePositionLabel>, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }

    if let Ok(rows) = serde_json::from_str::<Vec<BlinxOraclePositionLabel>>(trimmed) {
        return Ok(rows);
    }

    let mut rows = Vec::new();
    for (line_no, line) in trimmed.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let row = serde_json::from_str::<BlinxOraclePositionLabel>(line)
            .map_err(|err| format!("parse blinx oracle label {source_path}:{line_no}: {err}"))?;
        rows.push(row);
    }
    if rows.is_empty() {
        return Err(format!("no blinx oracle labels found in {source_path}"));
    }
    Ok(rows)
}

fn candidate_set_target_from_label(
    label: &BlinxOraclePositionLabel,
    candidate_k: usize,
    kind: OracleTargetKind,
) -> OracleBridgeTarget {
    let provenance = blinx_oracle_target_provenance(label);
    let score = if candidate_set_matches(label, candidate_k) {
        1.0
    } else {
        0.0
    };
    OracleBridgeTarget {
        kind,
        radius: label.radius,
        candidate_k,
        score,
        stats: blinx_oracle_target_stats(label),
        provenance,
    }
}

fn memory_trust_target_from_label(label: &BlinxOraclePositionLabel) -> OracleBridgeTarget {
    let provenance = blinx_oracle_target_provenance(label);
    let mut stats = blinx_oracle_target_stats(label);
    let score = normalized_support_score(label.left_leaveout_support_size);
    stats.left_only_fraction = score;
    stats.memory_trust = score;
    OracleBridgeTarget {
        kind: OracleTargetKind::MemoryTrust,
        radius: label.radius,
        candidate_k: 0,
        score,
        stats,
        provenance,
    }
}

fn bridge_confidence_target_from_label(label: &BlinxOraclePositionLabel) -> OracleBridgeTarget {
    let provenance = blinx_oracle_target_provenance(label);
    let mut stats = blinx_oracle_target_stats(label);
    let score = normalized_support_score(label.bidi_leaveout_support_size);
    stats.bidirectional_fraction = score;
    stats.bridge_confidence = score;
    OracleBridgeTarget {
        kind: OracleTargetKind::BridgeConfidence,
        radius: label.radius,
        candidate_k: 0,
        score,
        stats,
        provenance,
    }
}

fn clean_bridge_score_target_from_label(label: &BlinxOraclePositionLabel) -> OracleBridgeTarget {
    let provenance = blinx_oracle_target_provenance(label);
    let mut stats = blinx_oracle_target_stats(label);
    let memory_trust = normalized_support_score(label.left_leaveout_support_size);
    let bridge_confidence = normalized_support_score(label.bidi_leaveout_support_size);
    let contamination_penalty = 0.25 * f64::from(label.self_inclusion_support_changed as u8)
        + 0.25 * f64::from(label.future_context_support_changed as u8);
    let score = memory_trust.min(bridge_confidence) - contamination_penalty;
    stats.left_only_fraction = memory_trust;
    stats.bidirectional_fraction = bridge_confidence;
    stats.memory_trust = memory_trust;
    stats.bridge_confidence = bridge_confidence;
    stats.self_inclusion_uplift = if label.self_inclusion_support_changed {
        (bridge_confidence - memory_trust).abs()
    } else {
        0.0
    };
    stats.future_context_uplift = if label.future_context_support_changed {
        (memory_trust - bridge_confidence).abs()
    } else {
        0.0
    };
    stats.clean_bridge_score = score;
    OracleBridgeTarget {
        kind: OracleTargetKind::CleanBridgeScore,
        radius: label.radius,
        candidate_k: 0,
        score,
        stats,
        provenance,
    }
}

fn blinx_oracle_target_provenance(label: &BlinxOraclePositionLabel) -> OracleTargetProvenance {
    OracleTargetProvenance {
        source_family: "blinx_oracle_position_export".to_string(),
        source_path: Some(label.path.clone()),
        source_radius: Some(label.radius),
        source_position: Some(label.position),
        source_size: Some(label.size),
        leave_one_out: true,
        contamination_adjusted: true,
        future_blind_at_runtime: true,
        offline_only: true,
    }
}

fn blinx_oracle_target_stats(label: &BlinxOraclePositionLabel) -> OracleTargetStats {
    let bidirectional_fraction = normalized_support_score(label.bidi_inclusive_support_size);
    let left_only_fraction = normalized_support_score(label.left_leaveout_support_size);
    let memory_trust = left_only_fraction;
    let bridge_confidence = normalized_support_score(label.bidi_leaveout_support_size);
    let self_inclusion_uplift = if label.self_inclusion_support_changed {
        (bidirectional_fraction - bridge_confidence).abs()
    } else {
        0.0
    };
    let future_context_uplift = if label.future_context_support_changed {
        (bridge_confidence - left_only_fraction).abs()
    } else {
        0.0
    };
    let clean_bridge_score = memory_trust.min(bridge_confidence)
        - 0.25 * f64::from(label.self_inclusion_support_changed as u8)
        - 0.25 * f64::from(label.future_context_support_changed as u8);
    OracleTargetStats {
        bidirectional_fraction,
        left_only_fraction,
        self_inclusion_uplift,
        future_context_uplift,
        clean_bridge_score,
        bidirectional_support_size: label.bidi_inclusive_support_size,
        left_only_support_size: label.left_leaveout_support_size,
        support_gap: label.bidi_leaveout_support_size as isize
            - label.left_leaveout_support_size as isize,
        candidate_leq_2: label.bidi_leaveout_candidate4 && label.bidi_leaveout_support_size <= 2,
        candidate_leq_4: label.bidi_leaveout_candidate4,
        candidate_leq_8: label.bidi_leaveout_support_size <= 8
            && label.bidi_leaveout_support_size > 0,
        memory_trust,
        bridge_confidence,
    }
}

fn candidate_set_matches(label: &BlinxOraclePositionLabel, candidate_k: usize) -> bool {
    match candidate_k {
        2 => label.bidi_leaveout_support_size > 0 && label.bidi_leaveout_support_size <= 2,
        4 => label.bidi_leaveout_support_size > 0 && label.bidi_leaveout_support_size <= 4,
        8 => label.bidi_leaveout_support_size > 0 && label.bidi_leaveout_support_size <= 8,
        _ => false,
    }
}

fn normalized_support_score(support_size: usize) -> f64 {
    if support_size == 0 {
        0.0
    } else {
        1.0 / support_size as f64
    }
}

pub fn summarize_radii(corpus: &OracleAttackCorpus) -> Vec<RadiusSummary> {
    let mut rows = Vec::new();
    for &radius in &corpus.radii {
        let mut matched = Vec::new();
        for file in &corpus.files {
            if let Some(row) = file.radii.iter().find(|row| row.radius == radius) {
                matched.push(row);
            }
        }
        if matched.is_empty() {
            continue;
        }
        let mean_bidi_leaveout_candidate4 = matched
            .iter()
            .map(|row| row.bidi_leaveout_candidate4_fraction)
            .sum::<f64>()
            / matched.len() as f64;
        let mean_left_leaveout_candidate4 = matched
            .iter()
            .map(|row| row.left_leaveout_candidate4_fraction)
            .sum::<f64>()
            / matched.len() as f64;
        let mean_self_inclusion_uplift = matched
            .iter()
            .map(|row| row.self_inclusion_candidate4_uplift)
            .sum::<f64>()
            / matched.len() as f64;
        let mean_future_context_uplift = matched
            .iter()
            .map(|row| row.future_context_candidate4_uplift)
            .sum::<f64>()
            / matched.len() as f64;
        let mean_naive_net_removed_bytes = matched
            .iter()
            .map(|row| row.naive_net_removed_bytes as f64)
            .sum::<f64>()
            / matched.len() as f64;
        let causalizable_share =
            mean_left_leaveout_candidate4 / mean_bidi_leaveout_candidate4.max(f64::EPSILON);
        let clean_bridge_score = mean_left_leaveout_candidate4 - mean_self_inclusion_uplift;
        rows.push(RadiusSummary {
            radius,
            file_count: matched.len(),
            mean_bidi_leaveout_candidate4,
            mean_left_leaveout_candidate4,
            mean_self_inclusion_uplift,
            mean_future_context_uplift,
            mean_naive_net_removed_bytes,
            causalizable_share,
            clean_bridge_score,
        });
    }
    rows.sort_by_key(|row| row.radius);
    rows
}

pub fn bridgeable_rows(corpus: &OracleAttackCorpus) -> Vec<BridgeableRow> {
    let mut rows = Vec::new();
    for file in &corpus.files {
        for row in &file.radii {
            rows.push(BridgeableRow {
                path: file.path.clone(),
                size: file.size,
                radius: row.radius,
                positions: row.positions,
                bidi_leaveout_candidate4_fraction: row.bidi_leaveout_candidate4_fraction,
                left_leaveout_candidate4_fraction: row.left_leaveout_candidate4_fraction,
                self_inclusion_candidate4_uplift: row.self_inclusion_candidate4_uplift,
                future_context_candidate4_uplift: row.future_context_candidate4_uplift,
                naive_net_removed_bytes: row.naive_net_removed_bytes,
                clean_bridge_score: row.left_leaveout_candidate4_fraction
                    - row.self_inclusion_candidate4_uplift,
            });
        }
    }
    rows
}

pub fn bridge_target_from_radius_summary(row: &RadiusSummary) -> OracleBridgeTarget {
    clean_candidate4_target(
        row.radius,
        row.mean_left_leaveout_candidate4,
        row.mean_bidi_leaveout_candidate4,
        row.mean_self_inclusion_uplift,
        row.mean_future_context_uplift,
        "blinx_oracle_attack",
        None,
    )
}

pub fn bridge_target_from_bridgeable_row(row: &BridgeableRow) -> OracleBridgeTarget {
    clean_candidate4_target(
        row.radius,
        row.left_leaveout_candidate4_fraction,
        row.bidi_leaveout_candidate4_fraction,
        row.self_inclusion_candidate4_uplift,
        row.future_context_candidate4_uplift,
        "blinx_oracle_attack",
        Some(row.path.clone()),
    )
}

pub fn render_oracle_clean_summary(corpus: &OracleAttackCorpus, top_n: usize) -> String {
    let radii = summarize_radii(corpus);
    let mut rows = bridgeable_rows(corpus);
    rows.sort_by(|a, b| {
        b.clean_bridge_score
            .partial_cmp(&a.clean_bridge_score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| b.positions.cmp(&a.positions))
    });

    let mut contaminated = rows.clone();
    contaminated.sort_by(|a, b| {
        b.self_inclusion_candidate4_uplift
            .partial_cmp(&a.self_inclusion_candidate4_uplift)
            .unwrap_or(Ordering::Equal)
    });

    let best_radius = radii.iter().max_by(|a, b| {
        a.clean_bridge_score
            .partial_cmp(&b.clean_bridge_score)
            .unwrap_or(Ordering::Equal)
    });

    let mut out = String::new();
    out.push_str("chronohorn_oracle_clean_summary\n");
    out.push_str(&format!("file_count: {}\n", corpus.file_count));
    out.push_str(&format!("total_bytes: {}\n", corpus.total_bytes));
    if let Some(best) = best_radius {
        out.push_str(&format!(
            "recommended_radius: {} (clean_bridge_score={:.6} causalizable_share={:.6})\n",
            best.radius, best.clean_bridge_score, best.causalizable_share
        ));
    }
    out.push_str("radius_summaries:\n");
    for row in &radii {
        out.push_str(&format!(
            "  radius={} files={} bidi_leaveout_c4={:.6} left_leaveout_c4={:.6} self_uplift={:.6} future_uplift={:.6} causalizable_share={:.6} clean_bridge_score={:.6} naive_net_removed_bytes={:.2}\n",
            row.radius,
            row.file_count,
            row.mean_bidi_leaveout_candidate4,
            row.mean_left_leaveout_candidate4,
            row.mean_self_inclusion_uplift,
            row.mean_future_context_uplift,
            row.causalizable_share,
            row.clean_bridge_score,
            row.mean_naive_net_removed_bytes
        ));
    }
    out.push_str("top_bridgeable_rows:\n");
    for row in rows.iter().take(top_n) {
        out.push_str(&format!(
            "  radius={} clean_bridge_score={:.6} left_leaveout_c4={:.6} self_uplift={:.6} future_uplift={:.6} naive_net_removed_bytes={} positions={} size={} path={}\n",
            row.radius,
            row.clean_bridge_score,
            row.left_leaveout_candidate4_fraction,
            row.self_inclusion_candidate4_uplift,
            row.future_context_candidate4_uplift,
            row.naive_net_removed_bytes,
            row.positions,
            row.size,
            row.path
        ));
    }
    out.push_str("top_contaminated_rows:\n");
    for row in contaminated.iter().take(top_n) {
        out.push_str(&format!(
            "  radius={} self_uplift={:.6} bidi_leaveout_c4={:.6} left_leaveout_c4={:.6} future_uplift={:.6} path={}\n",
            row.radius,
            row.self_inclusion_candidate4_uplift,
            row.bidi_leaveout_candidate4_fraction,
            row.left_leaveout_candidate4_fraction,
            row.future_context_candidate4_uplift,
            row.path
        ));
    }
    out
}
