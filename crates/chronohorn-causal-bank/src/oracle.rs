use std::collections::HashMap;
use std::path::Path;

use chronohorn_core::bridge::{
    OracleBridgeTarget, OracleTargetKind, OracleTargetProvenance, OracleTargetStats,
};
use chronohorn_core::data::{take_train_tokens, take_val_tokens};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenOraclePositionLabel {
    pub root: String,
    pub size: usize,
    pub radius: usize,
    pub position: usize,
    pub center: usize,
    pub left_context: Vec<usize>,
    pub right_context: Vec<usize>,
    pub left_leaveout_support_size: usize,
    pub bidi_leaveout_support_size: usize,
    pub bidi_leaveout_candidates: Vec<usize>,
    pub bidi_leaveout_candidate_counts: Vec<usize>,
    pub support_gap: isize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenOracleTargetRow {
    pub label: TokenOraclePositionLabel,
    pub targets: Vec<OracleBridgeTarget>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenOracleTargetDataset {
    pub schema: String,
    pub source_family: String,
    pub rows: Vec<TokenOracleTargetRow>,
}

pub fn load_token_oracle_position_labels(
    root: &Path,
    token_budget: usize,
    radius: usize,
    stride: usize,
) -> Result<Vec<TokenOraclePositionLabel>, String> {
    let tokens = take_train_tokens(root, token_budget)?;
    build_token_oracle_position_labels(&root.display().to_string(), &tokens, radius, stride)
}

pub fn load_token_oracle_val_position_labels(
    root: &Path,
    token_budget: usize,
    radius: usize,
    stride: usize,
) -> Result<Vec<TokenOraclePositionLabel>, String> {
    let tokens = take_val_tokens(root, token_budget)?;
    build_token_oracle_position_labels(&root.display().to_string(), &tokens, radius, stride)
}

pub fn load_token_oracle_target_dataset(
    root: &Path,
    token_budget: usize,
    radius: usize,
    stride: usize,
) -> Result<TokenOracleTargetDataset, String> {
    let labels = load_token_oracle_position_labels(root, token_budget, radius, stride)?;
    Ok(token_oracle_target_dataset_from_labels(labels))
}

pub fn load_token_oracle_val_target_dataset(
    root: &Path,
    token_budget: usize,
    radius: usize,
    stride: usize,
) -> Result<TokenOracleTargetDataset, String> {
    let labels = load_token_oracle_val_position_labels(root, token_budget, radius, stride)?;
    Ok(token_oracle_target_dataset_from_labels(labels))
}

pub fn token_oracle_target_dataset_from_tokens(
    source: impl Into<String>,
    tokens: &[usize],
    radius: usize,
    stride: usize,
) -> Result<TokenOracleTargetDataset, String> {
    let labels = build_token_oracle_position_labels(&source.into(), tokens, radius, stride)?;
    Ok(token_oracle_target_dataset_from_labels(labels))
}

pub fn token_oracle_target_dataset_from_labels(
    labels: Vec<TokenOraclePositionLabel>,
) -> TokenOracleTargetDataset {
    let rows = labels.into_iter().map(token_oracle_target_row).collect();
    TokenOracleTargetDataset {
        schema: "chronohorn.token_context_oracle.position_targets".to_string(),
        source_family: "token_context_oracle".to_string(),
        rows,
    }
}

pub fn token_oracle_target_row(label: TokenOraclePositionLabel) -> TokenOracleTargetRow {
    let targets = token_oracle_targets_for_label(&label);
    TokenOracleTargetRow { label, targets }
}

pub fn token_oracle_targets_for_label(label: &TokenOraclePositionLabel) -> Vec<OracleBridgeTarget> {
    vec![
        candidate_set_target_from_label(label, 2, OracleTargetKind::CandidateSetLeq2),
        candidate_set_target_from_label(label, 4, OracleTargetKind::CandidateSetLeq4),
        candidate_set_target_from_label(label, 8, OracleTargetKind::CandidateSetLeq8),
        memory_trust_target_from_label(label),
        bridge_confidence_target_from_label(label),
        clean_bridge_score_target_from_label(label),
    ]
}

pub fn render_token_oracle_summary(dataset: &TokenOracleTargetDataset, top_n: usize) -> String {
    let mut rows = dataset.rows.iter().collect::<Vec<_>>();
    rows.sort_by(|a, b| {
        clean_bridge_score_target_from_row(b)
            .partial_cmp(&clean_bridge_score_target_from_row(a))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.label.position.cmp(&b.label.position))
    });

    let mut out = String::new();
    out.push_str("chronohorn_token_context_oracle\n");
    out.push_str(&format!("schema: {}\n", dataset.schema));
    out.push_str(&format!("source_family: {}\n", dataset.source_family));
    out.push_str(&format!("row_count: {}\n", dataset.rows.len()));
    if let Some(first) = dataset.rows.first() {
        out.push_str(&format!("root: {}\n", first.label.root));
        out.push_str(&format!("radius: {}\n", first.label.radius));
        out.push_str(&format!("source_size: {}\n", first.label.size));
    }
    out.push_str("top_clean_bridge_rows:\n");
    for row in rows.into_iter().take(top_n) {
        out.push_str(&format!(
            "  pos={} center={} left_leaveout={} bidi_leaveout={} support_gap={} clean_bridge_score={:.6}\n",
            row.label.position,
            row.label.center,
            row.label.left_leaveout_support_size,
            row.label.bidi_leaveout_support_size,
            row.label.support_gap,
            clean_bridge_score_target_from_row(row)
        ));
    }
    out
}

pub fn token_oracle_target_values(
    dataset: &TokenOracleTargetDataset,
    kind: OracleTargetKind,
) -> Vec<(usize, f64)> {
    dataset
        .rows
        .iter()
        .filter_map(|row| {
            row.targets
                .iter()
                .find(|target| target.kind == kind)
                .map(|target| (row.label.position, target.score))
        })
        .collect()
}

pub fn token_oracle_teacher_candidates(
    dataset: &TokenOracleTargetDataset,
) -> Vec<(usize, Vec<usize>)> {
    dataset
        .rows
        .iter()
        .map(|row| {
            (
                row.label.position,
                row.label.bidi_leaveout_candidates.clone(),
            )
        })
        .collect()
}

pub fn token_oracle_teacher_candidate_counts(
    dataset: &TokenOracleTargetDataset,
) -> Vec<(usize, Vec<usize>)> {
    dataset
        .rows
        .iter()
        .map(|row| {
            (
                row.label.position,
                row.label.bidi_leaveout_candidate_counts.clone(),
            )
        })
        .collect()
}

pub fn token_oracle_teacher_candidate_pairs(
    dataset: &TokenOracleTargetDataset,
) -> Vec<(usize, Vec<(usize, usize)>)> {
    dataset
        .rows
        .iter()
        .map(|row| {
            let pairs = row
                .label
                .bidi_leaveout_candidates
                .iter()
                .copied()
                .zip(row.label.bidi_leaveout_candidate_counts.iter().copied())
                .collect::<Vec<_>>();
            (row.label.position, pairs)
        })
        .collect()
}

fn build_token_oracle_position_labels(
    source: &str,
    tokens: &[usize],
    radius: usize,
    stride: usize,
) -> Result<Vec<TokenOraclePositionLabel>, String> {
    if radius == 0 {
        return Err("radius must be positive".to_string());
    }
    if stride == 0 {
        return Err("stride must be positive".to_string());
    }
    if tokens.len() <= radius * 2 {
        return Err("need more tokens than 2 * radius".to_string());
    }

    let (left_counts, bidi_counts) = context_support_counts(tokens, radius);
    let mut labels = Vec::new();
    let last = tokens.len() - radius;
    for position in radius..last {
        if (position - radius) % stride != 0 {
            continue;
        }

        let left_context = tokens[position - radius..position].to_vec();
        let right_context = tokens[position + 1..=position + radius].to_vec();
        let bidi_context = tokens[position - radius..=position + radius].to_vec();
        let left_leaveout_candidates =
            leaveout_candidates(left_counts.get(&left_context), tokens[position]);
        let bidi_leaveout_candidates =
            leaveout_candidates(bidi_counts.get(&bidi_context), tokens[position]);
        let bidi_leaveout_candidate_counts = bidi_leaveout_candidates
            .iter()
            .map(|candidate| candidate.1)
            .collect::<Vec<_>>();
        let bidi_leaveout_candidates = bidi_leaveout_candidates
            .iter()
            .map(|candidate| candidate.0)
            .collect::<Vec<_>>();
        let left_leaveout_support_size = left_leaveout_candidates.len();
        let bidi_leaveout_support_size = bidi_leaveout_candidates.len();
        labels.push(TokenOraclePositionLabel {
            root: source.to_string(),
            size: tokens.len(),
            radius,
            position,
            center: tokens[position],
            left_context,
            right_context,
            left_leaveout_support_size,
            bidi_leaveout_support_size,
            bidi_leaveout_candidates,
            bidi_leaveout_candidate_counts,
            support_gap: bidi_leaveout_support_size as isize - left_leaveout_support_size as isize,
        });
    }

    if labels.is_empty() {
        return Err("no token oracle labels were produced".to_string());
    }
    Ok(labels)
}

fn context_support_counts(
    tokens: &[usize],
    radius: usize,
) -> (
    HashMap<Vec<usize>, HashMap<usize, usize>>,
    HashMap<Vec<usize>, HashMap<usize, usize>>,
) {
    let mut left_counts: HashMap<Vec<usize>, HashMap<usize, usize>> = HashMap::new();
    let mut bidi_counts: HashMap<Vec<usize>, HashMap<usize, usize>> = HashMap::new();
    for position in radius..(tokens.len() - radius) {
        let left_context = tokens[position - radius..position].to_vec();
        let bidi_context = tokens[position - radius..=position + radius].to_vec();
        *left_counts
            .entry(left_context)
            .or_default()
            .entry(tokens[position])
            .or_insert(0) += 1;
        *bidi_counts
            .entry(bidi_context)
            .or_default()
            .entry(tokens[position])
            .or_insert(0) += 1;
    }
    (left_counts, bidi_counts)
}

fn leaveout_candidates(
    counter: Option<&HashMap<usize, usize>>,
    center: usize,
) -> Vec<(usize, usize)> {
    let Some(counter) = counter else {
        return Vec::new();
    };
    let mut candidates = counter
        .iter()
        .filter_map(|(token, count)| {
            let remaining = if *token == center {
                count.saturating_sub(1)
            } else {
                *count
            };
            (remaining > 0).then_some((*token, remaining))
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|(token_a, count_a), (token_b, count_b)| {
        count_b.cmp(count_a).then_with(|| token_a.cmp(token_b))
    });
    candidates
}

fn candidate_set_target_from_label(
    label: &TokenOraclePositionLabel,
    candidate_k: usize,
    kind: OracleTargetKind,
) -> OracleBridgeTarget {
    let provenance = token_oracle_target_provenance(label);
    let stats = token_oracle_target_stats(label);
    let score = if label.bidi_leaveout_support_size < candidate_k {
        1.0
    } else {
        0.0
    };
    OracleBridgeTarget {
        kind,
        radius: label.radius,
        candidate_k,
        score,
        stats,
        provenance,
    }
}

fn memory_trust_target_from_label(label: &TokenOraclePositionLabel) -> OracleBridgeTarget {
    let provenance = token_oracle_target_provenance(label);
    let mut stats = token_oracle_target_stats(label);
    let score = smoothed_support_score(label.left_leaveout_support_size);
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

fn bridge_confidence_target_from_label(label: &TokenOraclePositionLabel) -> OracleBridgeTarget {
    let provenance = token_oracle_target_provenance(label);
    let mut stats = token_oracle_target_stats(label);
    let score = smoothed_support_score(label.bidi_leaveout_support_size);
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

fn clean_bridge_score_target_from_label(label: &TokenOraclePositionLabel) -> OracleBridgeTarget {
    let provenance = token_oracle_target_provenance(label);
    let mut stats = token_oracle_target_stats(label);
    let memory_trust = smoothed_support_score(label.left_leaveout_support_size);
    let bridge_confidence = smoothed_support_score(label.bidi_leaveout_support_size);
    let max_support = label
        .left_leaveout_support_size
        .max(label.bidi_leaveout_support_size)
        .max(1) as f64;
    let gap_fraction = label
        .left_leaveout_support_size
        .abs_diff(label.bidi_leaveout_support_size) as f64
        / max_support;
    let score = memory_trust.min(bridge_confidence) - 0.25 * gap_fraction;
    stats.left_only_fraction = memory_trust;
    stats.bidirectional_fraction = bridge_confidence;
    stats.self_inclusion_uplift = (bridge_confidence - memory_trust).abs();
    stats.future_context_uplift = (memory_trust - bridge_confidence).abs();
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

fn token_oracle_target_provenance(label: &TokenOraclePositionLabel) -> OracleTargetProvenance {
    OracleTargetProvenance {
        source_family: "token_context_oracle".to_string(),
        source_path: Some(label.root.clone()),
        source_radius: Some(label.radius),
        source_position: Some(label.position),
        source_size: Some(label.size),
        leave_one_out: true,
        contamination_adjusted: true,
        future_blind_at_runtime: true,
        offline_only: true,
    }
}

fn token_oracle_target_stats(label: &TokenOraclePositionLabel) -> OracleTargetStats {
    let bidirectional_fraction = smoothed_support_score(label.bidi_leaveout_support_size);
    let left_only_fraction = smoothed_support_score(label.left_leaveout_support_size);
    let self_inclusion_uplift = (bidirectional_fraction - left_only_fraction).abs();
    let future_context_uplift = (left_only_fraction - bidirectional_fraction).abs();
    let clean_bridge_score = left_only_fraction.min(bidirectional_fraction)
        - 0.25
            * label
                .left_leaveout_support_size
                .abs_diff(label.bidi_leaveout_support_size) as f64
            / label
                .left_leaveout_support_size
                .max(label.bidi_leaveout_support_size)
                .max(1) as f64;
    OracleTargetStats {
        bidirectional_fraction,
        left_only_fraction,
        self_inclusion_uplift,
        future_context_uplift,
        clean_bridge_score,
        bidirectional_support_size: label.bidi_leaveout_support_size,
        left_only_support_size: label.left_leaveout_support_size,
        support_gap: label.support_gap,
        candidate_leq_2: label.bidi_leaveout_support_size < 2,
        candidate_leq_4: label.bidi_leaveout_support_size < 4,
        candidate_leq_8: label.bidi_leaveout_support_size < 8,
        memory_trust: left_only_fraction,
        bridge_confidence: bidirectional_fraction,
    }
}

fn clean_bridge_score_target_from_row(row: &TokenOracleTargetRow) -> f64 {
    row.targets
        .iter()
        .find(|target| matches!(target.kind, OracleTargetKind::CleanBridgeScore))
        .map(|target| target.score)
        .unwrap_or(0.0)
}

fn smoothed_support_score(support_size: usize) -> f64 {
    1.0 / (1.0 + support_size as f64)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;
    use chronohorn_core::data::{
        HEADER_BYTES, HEADER_INTS, PARAMETER_GOLF_MAGIC, PARAMETER_GOLF_VERSION,
    };

    #[test]
    fn builds_token_oracle_dataset_from_shards() {
        let root = scratch_root("token_oracle_smoke");
        fs::create_dir_all(&root).unwrap();
        write_shard(
            &root.join("fineweb_train_000000.bin"),
            &[10, 11, 12, 10, 11, 12, 10, 11, 12, 10, 11, 12],
        )
        .unwrap();

        let labels = load_token_oracle_position_labels(&root, 12, 1, 1).unwrap();
        assert!(!labels.is_empty());
        assert_eq!(labels[0].radius, 1);
        assert_eq!(labels[0].left_context.len(), 1);
        assert_eq!(labels[0].right_context.len(), 1);

        let dataset = token_oracle_target_dataset_from_labels(labels);
        assert_eq!(dataset.rows[0].targets.len(), 6);
        let summary = render_token_oracle_summary(&dataset, 2);
        assert!(summary.contains("chronohorn_token_context_oracle"));
        assert!(summary.contains("top_clean_bridge_rows"));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn builds_token_oracle_dataset_from_val_shards() {
        let root = scratch_root("token_oracle_val_smoke");
        fs::create_dir_all(&root).unwrap();
        write_shard(
            &root.join("fineweb_train_000000.bin"),
            &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        )
        .unwrap();
        write_shard(
            &root.join("fineweb_val_000000.bin"),
            &[20, 21, 22, 20, 21, 22, 20, 21, 22, 20, 21, 22],
        )
        .unwrap();

        let labels = load_token_oracle_val_position_labels(&root, 12, 1, 1).unwrap();
        assert!(!labels.is_empty());
        assert_eq!(labels[0].root, root.display().to_string());
        assert_eq!(labels[0].center, 21);
        assert_eq!(labels[0].left_context, vec![20]);
        assert_eq!(labels[0].right_context, vec![22]);
        assert_eq!(labels[0].bidi_leaveout_candidates, vec![21]);
        assert_eq!(labels[0].bidi_leaveout_candidate_counts, vec![3]);

        let dataset = load_token_oracle_val_target_dataset(&root, 12, 1, 1).unwrap();
        assert_eq!(
            dataset.schema,
            "chronohorn.token_context_oracle.position_targets"
        );
        assert_eq!(dataset.source_family, "token_context_oracle");
        assert_eq!(dataset.rows.len(), labels.len());
        assert_eq!(
            token_oracle_target_values(&dataset, OracleTargetKind::CandidateSetLeq4),
            labels
                .iter()
                .map(|label| (
                    label.position,
                    if label.bidi_leaveout_support_size < 4 {
                        1.0
                    } else {
                        0.0
                    }
                ))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            token_oracle_teacher_candidate_pairs(&dataset)[0],
            (labels[0].position, vec![(21, 3)])
        );
        assert_eq!(
            token_oracle_teacher_candidate_counts(&dataset),
            dataset
                .rows
                .iter()
                .map(|row| (
                    row.label.position,
                    row.label.bidi_leaveout_candidate_counts.clone()
                ))
                .collect::<Vec<_>>()
        );

        let summary = render_token_oracle_summary(&dataset, 2);
        assert!(summary.contains("chronohorn_token_context_oracle"));
        assert!(summary.contains("top_clean_bridge_rows"));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn token_oracle_target_values_preserve_row_order() {
        let root = scratch_root("token_oracle_values_smoke");
        fs::create_dir_all(&root).unwrap();
        write_shard(
            &root.join("fineweb_val_000000.bin"),
            &[4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6],
        )
        .unwrap();

        let dataset = load_token_oracle_val_target_dataset(&root, 12, 1, 1).unwrap();
        let values = token_oracle_target_values(&dataset, OracleTargetKind::CleanBridgeScore);
        let expected = dataset
            .rows
            .iter()
            .map(|row| {
                let score = row
                    .targets
                    .iter()
                    .find(|target| target.kind == OracleTargetKind::CleanBridgeScore)
                    .map(|target| target.score)
                    .unwrap();
                (row.label.position, score)
            })
            .collect::<Vec<_>>();
        assert_eq!(values, expected);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn token_oracle_ranked_teacher_pairs_preserve_row_order() {
        let root = scratch_root("token_oracle_ranked_teacher_smoke");
        fs::create_dir_all(&root).unwrap();
        write_shard(
            &root.join("fineweb_val_000000.bin"),
            &[10, 11, 12, 10, 11, 12, 10, 11, 12, 10, 11, 12],
        )
        .unwrap();

        let dataset = load_token_oracle_val_target_dataset(&root, 12, 1, 1).unwrap();
        let pairs = token_oracle_teacher_candidate_pairs(&dataset);
        let expected = dataset
            .rows
            .iter()
            .map(|row| {
                (
                    row.label.position,
                    row.label
                        .bidi_leaveout_candidates
                        .iter()
                        .copied()
                        .zip(row.label.bidi_leaveout_candidate_counts.iter().copied())
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(pairs, expected);
        assert_eq!(pairs[0], (1, vec![(11, 3)]));
        assert_eq!(pairs[1], (2, vec![(12, 2)]));

        let _ = fs::remove_dir_all(&root);
    }

    fn scratch_root(prefix: &str) -> std::path::PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}_{stamp}_{}", std::process::id()))
    }

    fn write_shard(path: &Path, tokens: &[usize]) -> Result<(), String> {
        let mut blob = Vec::with_capacity(HEADER_BYTES + tokens.len() * 2);
        let mut header = [0i32; HEADER_INTS];
        header[0] = PARAMETER_GOLF_MAGIC;
        header[1] = PARAMETER_GOLF_VERSION;
        header[2] = tokens.len() as i32;
        for value in header {
            blob.extend_from_slice(&value.to_le_bytes());
        }
        for &token in tokens {
            blob.extend_from_slice(&(token as u16).to_le_bytes());
        }
        fs::write(path, blob).map_err(|err| format!("write {}: {err}", path.display()))
    }
}
