use std::cmp::Ordering;
use std::path::Path;

use serde::Serialize;

use crate::data::{ResolvedDataRoot, resolve_data_root};
use crate::token_copy_bridge::train_token_copy_bridge_from_data_root;
use crate::token_match_bridge::train_token_match_bridge_from_data_root;
use crate::token_matchcopy_bridge::train_token_matchcopy_bridge_from_data_root;
use crate::token_matchskip_bridge::train_token_matchskip_bridge_from_data_root;
use crate::token_matchskipcopy_bridge::train_token_matchskipcopy_bridge_from_data_root;
use crate::token_skip_bridge::train_token_skip_bridge_from_data_root;

#[derive(Debug, Clone, Serialize)]
pub struct TokenExperimentMatrixConfig {
    pub train_token_budget: usize,
    pub trigram_buckets: usize,
    pub skip_buckets: usize,
    pub val_token_budget: usize,
    pub match_depth: usize,
    pub copy_window: usize,
    pub candidate_k: usize,
    pub train_stride: usize,
    pub copy_decay_bp: usize,
    pub alpha_bigram: f64,
    pub alpha_trigram: f64,
    pub alpha_skip: f64,
}

impl Default for TokenExperimentMatrixConfig {
    fn default() -> Self {
        Self {
            train_token_budget: 1_000_000,
            trigram_buckets: 8_192,
            skip_buckets: 8_192,
            val_token_budget: 32_768,
            match_depth: 8,
            copy_window: 256,
            candidate_k: 4,
            train_stride: 4,
            copy_decay_bp: 980,
            alpha_bigram: 4.0,
            alpha_trigram: 2.0,
            alpha_skip: 2.0,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct MatrixMetric {
    pub key: String,
    pub value: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct MatrixRow {
    pub family: String,
    pub label: String,
    pub base_bpt: f64,
    pub primary_bpt: f64,
    pub oracle_bpt: f64,
    pub lift_vs_base: f64,
    pub oracle_headroom: f64,
    pub selected_runtime_gate: String,
    pub selected_runtime_lambda: Option<f64>,
    pub hit_rate: Option<f64>,
    pub better_rate: Option<f64>,
    pub extra_metrics: Vec<MatrixMetric>,
}

impl MatrixRow {
    fn new(
        family: impl Into<String>,
        label: impl Into<String>,
        base_bpt: f64,
        primary_bpt: f64,
        oracle_bpt: f64,
        selected_runtime_gate: impl Into<String>,
        selected_runtime_lambda: Option<f64>,
        hit_rate: Option<f64>,
        better_rate: Option<f64>,
        extra_metrics: Vec<MatrixMetric>,
    ) -> Self {
        Self {
            family: family.into(),
            label: label.into(),
            base_bpt,
            primary_bpt,
            oracle_bpt,
            lift_vs_base: base_bpt - primary_bpt,
            oracle_headroom: primary_bpt - oracle_bpt,
            selected_runtime_gate: selected_runtime_gate.into(),
            selected_runtime_lambda,
            hit_rate,
            better_rate,
            extra_metrics,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct TokenExperimentMatrixReport {
    pub data_root_spec: String,
    pub data_root_resolution: ResolvedDataRoot,
    pub config: TokenExperimentMatrixConfig,
    pub rows: Vec<MatrixRow>,
}

impl TokenExperimentMatrixReport {
    pub fn best_by_lift(&self) -> Option<&MatrixRow> {
        self.rows.iter().max_by(|a, b| {
            a.lift_vs_base
                .partial_cmp(&b.lift_vs_base)
                .unwrap_or(Ordering::Equal)
        })
    }

    pub fn render_compact(&self) -> String {
        let mut rows = self.rows.clone();
        rows.sort_by(|a, b| {
            b.lift_vs_base
                .partial_cmp(&a.lift_vs_base)
                .unwrap_or(Ordering::Equal)
        });

        let mut out = String::new();
        out.push_str("chronohorn_token_experiment_matrix\n");
        out.push_str(&format!("data_root_spec: {}\n", self.data_root_spec));
        out.push_str(&format!(
            "resolved_alias: {}\n",
            self.data_root_resolution
                .alias
                .clone()
                .unwrap_or_else(|| "<direct>".to_string())
        ));
        out.push_str(&format!(
            "claim_tier: {}\n",
            self.data_root_resolution.report.claim_tier
        ));
        out.push_str(&format!("rows: {}\n", rows.len()));
        if let Some(best) = self.best_by_lift() {
            out.push_str(&format!(
                "best_by_lift: {} lift {:.4} base {:.4} primary {:.4} oracle {:.4} gate {} lambda {}\n",
                best.family,
                best.lift_vs_base,
                best.base_bpt,
                best.primary_bpt,
                best.oracle_bpt,
                best.selected_runtime_gate,
                best.selected_runtime_lambda
                    .map(|v| format!("{v:.3}"))
                    .unwrap_or_else(|| "-".to_string())
            ));
        }
        for row in rows {
            out.push_str(&format!(
                "{:<16} {:<28} base {:>8.4} primary {:>8.4} oracle {:>8.4} lift {:>8.4} headroom {:>8.4} gate {:<10} lambda {:>6}\n",
                row.family,
                row.label,
                row.base_bpt,
                row.primary_bpt,
                row.oracle_bpt,
                row.lift_vs_base,
                row.oracle_headroom,
                row.selected_runtime_gate,
                row.selected_runtime_lambda
                    .map(|v| format!("{v:.3}"))
                    .unwrap_or_else(|| "-".to_string())
            ));
            if !row.extra_metrics.is_empty() {
                let extras = row
                    .extra_metrics
                    .iter()
                    .map(|metric| format!("{}={:.4}", metric.key, metric.value))
                    .collect::<Vec<_>>()
                    .join(" ");
                out.push_str(&format!("  extras: {extras}\n"));
            }
        }
        out
    }
}

pub fn run_token_experiment_matrix_from_data_root(
    data_root_spec: &str,
    config: TokenExperimentMatrixConfig,
) -> Result<TokenExperimentMatrixReport, String> {
    let resolved = resolve_data_root(Some(data_root_spec))?;
    if resolved.report.claim_tier == "blocked" {
        return Err(format!(
            "data root {data_root_spec} is blocked: {}",
            resolved
                .report
                .blocked_reason
                .clone()
                .unwrap_or_else(|| "unknown".to_string())
        ));
    }
    let root = Path::new(&resolved.selected_path);
    let mut rows = Vec::with_capacity(6);

    let match_report = train_token_match_bridge_from_data_root(
        root,
        config.train_token_budget,
        config.trigram_buckets,
        config.val_token_budget,
        config.match_depth,
        config.candidate_k,
        config.train_stride,
        config.alpha_bigram,
        config.alpha_trigram,
    )?;
    rows.push(row_from_match(match_report.report()));

    let skip_report = train_token_skip_bridge_from_data_root(
        root,
        config.train_token_budget,
        config.trigram_buckets,
        config.skip_buckets,
        config.val_token_budget,
        config.alpha_bigram,
        config.alpha_trigram,
        config.alpha_skip,
        config.train_stride,
        config.candidate_k,
    )?;
    rows.push(row_from_skip(skip_report.report()));

    let copy_report = train_token_copy_bridge_from_data_root(
        root,
        config.train_token_budget,
        config.trigram_buckets,
        config.val_token_budget,
        config.copy_window,
        config.candidate_k,
        config.train_stride,
        config.alpha_bigram,
        config.alpha_trigram,
        config.copy_decay_bp as f64 / 1000.0,
    )?;
    rows.push(row_from_copy(copy_report.report()));

    let matchskip_report = train_token_matchskip_bridge_from_data_root(
        root,
        config.train_token_budget,
        config.trigram_buckets,
        config.skip_buckets,
        config.val_token_budget,
        config.match_depth,
        config.candidate_k,
        config.train_stride,
        config.alpha_bigram,
        config.alpha_trigram,
        config.alpha_skip,
    )?;
    rows.push(row_from_matchskip(matchskip_report.report()));

    let matchcopy_report = train_token_matchcopy_bridge_from_data_root(
        root,
        config.train_token_budget,
        config.trigram_buckets,
        config.val_token_budget,
        config.match_depth,
        config.copy_window,
        config.candidate_k,
        config.train_stride,
        config.alpha_bigram,
        config.alpha_trigram,
        config.copy_decay_bp as f64 / 1000.0,
    )?;
    rows.push(row_from_matchcopy(matchcopy_report.report()));

    let matchskipcopy_report = train_token_matchskipcopy_bridge_from_data_root(
        root,
        config.train_token_budget,
        config.trigram_buckets,
        config.skip_buckets,
        config.val_token_budget,
        config.match_depth,
        config.copy_window,
        config.candidate_k,
        config.train_stride,
        config.alpha_bigram,
        config.alpha_trigram,
        config.alpha_skip,
        config.copy_decay_bp as f64 / 1000.0,
    )?;
    rows.push(row_from_matchskipcopy(matchskipcopy_report.report()));

    rows.sort_by(|a, b| {
        b.lift_vs_base
            .partial_cmp(&a.lift_vs_base)
            .unwrap_or(Ordering::Equal)
    });

    Ok(TokenExperimentMatrixReport {
        data_root_spec: data_root_spec.to_string(),
        data_root_resolution: resolved,
        config,
        rows,
    })
}

fn row_from_match(report: &crate::token_match_bridge::TokenMatchBridgeReport) -> MatrixRow {
    let primary_bpt = match report.selected_runtime_gate.as_str() {
        "heuristic" => report.eval_bpt_heuristic,
        "direct" => report.eval_bpt_direct,
        _ => report.eval_bpt_heuristic.min(report.eval_bpt_direct),
    };
    MatrixRow::new(
        "match",
        format!(
            "train {} depth {} k {}",
            report.train_token_budget, report.match_depth, report.candidate_k
        ),
        report.eval_bpt_base,
        primary_bpt,
        report.eval_bpt_oracle,
        report.selected_runtime_gate.clone(),
        Some(report.selected_runtime_lambda),
        Some(report.eval_match_hit_rate),
        Some(report.eval_match_better_rate),
        vec![
            MatrixMetric {
                key: "eval_match_full".to_string(),
                value: report.eval_bpt_match_full,
            },
            MatrixMetric {
                key: "eval_match_topk".to_string(),
                value: report.eval_bpt_match_topk,
            },
        ],
    )
}

fn row_from_skip(report: &crate::token_skip_bridge::TokenSkipBridgeReport) -> MatrixRow {
    let primary_bpt = match report.selected_runtime_gate.as_str() {
        "heuristic" => report.eval_bpt_heuristic,
        "direct" => report.eval_bpt_direct,
        _ => report.eval_bpt_heuristic.min(report.eval_bpt_direct),
    };
    MatrixRow::new(
        "skip",
        format!(
            "train {} skip {}:{} k {}",
            report.train_token_budget, report.skip_far, report.skip_near, report.candidate_k
        ),
        report.eval_bpt_base,
        primary_bpt,
        report.eval_bpt_oracle,
        report.selected_runtime_gate.clone(),
        None,
        Some(report.eval_skip_candidate_hit_rate),
        Some(report.eval_skip_better_rate),
        vec![
            MatrixMetric {
                key: "eval_skip".to_string(),
                value: report.eval_bpt_skip,
            },
            MatrixMetric {
                key: "eval_oracle_gap".to_string(),
                value: report.eval_oracle_gap,
            },
            MatrixMetric {
                key: "eval_overlap".to_string(),
                value: report.eval_candidate_overlap_rate,
            },
        ],
    )
}

fn row_from_copy(report: &crate::token_copy_bridge::TokenCopyBridgeReport) -> MatrixRow {
    let primary_bpt = match report.selected_runtime_gate.as_str() {
        "heuristic" => report.eval_bpt_heuristic,
        "direct" => report.eval_bpt_direct,
        _ => report.eval_bpt_heuristic.min(report.eval_bpt_direct),
    };
    MatrixRow::new(
        "copy",
        format!(
            "train {} window {} decay {:.3} k {}",
            report.train_token_budget, report.copy_window, report.copy_decay, report.candidate_k
        ),
        report.eval_bpt_base,
        primary_bpt,
        report.eval_bpt_oracle,
        report.selected_runtime_gate.clone(),
        Some(report.selected_runtime_lambda),
        Some(report.eval_copy_hit_rate),
        Some(report.eval_copy_better_rate),
        vec![
            MatrixMetric {
                key: "eval_copy_full".to_string(),
                value: report.eval_bpt_copy_full,
            },
            MatrixMetric {
                key: "eval_copy_topk".to_string(),
                value: report.eval_bpt_copy_topk,
            },
        ],
    )
}

fn row_from_matchskip(
    report: &crate::token_matchskip_bridge::TokenMatchSkipBridgeReport,
) -> MatrixRow {
    let primary_bpt = match report.selected_runtime_gate.as_str() {
        "heuristic" => report.eval_bpt_heuristic,
        "direct" => report.eval_bpt_direct,
        _ => report.eval_bpt_heuristic.min(report.eval_bpt_direct),
    };
    MatrixRow::new(
        "match+skip",
        format!(
            "train {} depth {} trigram {} skip {} k {}",
            report.train_token_budget,
            report.match_depth,
            report.trigram_buckets,
            report.skip_buckets,
            report.candidate_k
        ),
        report.eval_bpt_match.min(report.eval_bpt_skip),
        primary_bpt,
        report.eval_bpt_oracle,
        report.selected_runtime_gate.clone(),
        Some(report.selected_runtime_lambda),
        Some(report.eval_match_better_rate),
        Some(report.eval_skip_better_rate),
        vec![
            MatrixMetric {
                key: "eval_match".to_string(),
                value: report.eval_bpt_match,
            },
            MatrixMetric {
                key: "eval_skip".to_string(),
                value: report.eval_bpt_skip,
            },
        ],
    )
}

fn row_from_matchcopy(
    report: &crate::token_matchcopy_bridge::TokenMatchCopyBridgeReport,
) -> MatrixRow {
    let primary_bpt = match report.selected_runtime_gate.as_str() {
        "heuristic" => report.eval_bpt_heuristic,
        "direct" => report.eval_bpt_direct,
        _ => report.eval_bpt_heuristic.min(report.eval_bpt_direct),
    };
    MatrixRow::new(
        "match+copy",
        format!(
            "train {} depth {} window {} k {}",
            report.train_token_budget, report.match_depth, report.copy_window, report.candidate_k
        ),
        report.eval_bpt_match.min(report.eval_bpt_copy),
        primary_bpt,
        report.eval_bpt_oracle,
        report.selected_runtime_gate.clone(),
        Some(report.selected_runtime_lambda),
        Some(report.eval_match_better_rate),
        Some(report.eval_copy_better_rate),
        vec![
            MatrixMetric {
                key: "eval_match".to_string(),
                value: report.eval_bpt_match,
            },
            MatrixMetric {
                key: "eval_copy".to_string(),
                value: report.eval_bpt_copy,
            },
        ],
    )
}

fn row_from_matchskipcopy(
    report: &crate::token_matchskipcopy_bridge::TokenMatchSkipCopyBridgeReport,
) -> MatrixRow {
    let primary_bpt = match report.selected_runtime_gate.as_str() {
        "heuristic" => report.eval_bpt_heuristic,
        "direct" => report.eval_bpt_direct,
        _ => report.eval_bpt_heuristic.min(report.eval_bpt_direct),
    };
    MatrixRow::new(
        "match+skip+copy",
        format!(
            "train {} depth {} window {} trigram {} skip {} k {}",
            report.train_token_budget,
            report.match_depth,
            report.copy_window,
            report.trigram_buckets,
            report.skip_buckets,
            report.candidate_k
        ),
        report
            .eval_bpt_match
            .min(report.eval_bpt_skip)
            .min(report.eval_bpt_copy),
        primary_bpt,
        report.eval_bpt_oracle,
        report.selected_runtime_gate.clone(),
        Some(report.selected_runtime_lambda),
        Some(report.eval_match_better_rate),
        Some(report.eval_skip_better_rate),
        vec![
            MatrixMetric {
                key: "eval_match".to_string(),
                value: report.eval_bpt_match,
            },
            MatrixMetric {
                key: "eval_skip".to_string(),
                value: report.eval_bpt_skip,
            },
            MatrixMetric {
                key: "eval_copy".to_string(),
                value: report.eval_bpt_copy,
            },
            MatrixMetric {
                key: "eval_consensus".to_string(),
                value: report.eval_top1_consensus_rate,
            },
        ],
    )
}
