use serde::Deserialize;
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

pub fn load_oracle_attack(path: &str) -> Result<OracleAttackCorpus, String> {
    let raw = fs::read_to_string(path).map_err(|err| format!("read {path}: {err}"))?;
    serde_json::from_str(&raw).map_err(|err| format!("parse oracle attack {path}: {err}"))
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
