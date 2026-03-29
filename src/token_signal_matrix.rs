use std::cmp::Ordering;
use std::fmt::Write;

use crate::token_bridge::TokenBridgeReport;
use crate::token_column_bridge::TokenColumnBridgeReport;
use crate::token_copy_bridge::TokenCopyBridgeReport;
use crate::token_decay_bridge::TokenDecayBridgeReport;
use crate::token_match_bridge::TokenMatchBridgeReport;
use crate::token_skip_bridge::TokenSkipBridgeReport;
use crate::token_skipcopy_bridge::TokenSkipCopyBridgeReport;
use crate::token_word_bridge::TokenWordBridgeReport;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SignalFamily {
    TokenBridge,
    TokenCopy,
    TokenSkip,
    TokenSkipCopy,
    TokenMatch,
    TokenDecay,
    TokenWord,
    TokenColumn,
}

impl SignalFamily {
    pub fn as_str(self) -> &'static str {
        match self {
            SignalFamily::TokenBridge => "token_bridge",
            SignalFamily::TokenCopy => "token_copy_bridge",
            SignalFamily::TokenSkip => "token_skip_bridge",
            SignalFamily::TokenSkipCopy => "token_skipcopy_bridge",
            SignalFamily::TokenMatch => "token_match_bridge",
            SignalFamily::TokenDecay => "token_decay_bridge",
            SignalFamily::TokenWord => "token_word_bridge",
            SignalFamily::TokenColumn => "token_column_bridge",
        }
    }
}

#[derive(Debug, Clone)]
pub struct MatrixMetric {
    pub key: String,
    pub value: f64,
}

#[derive(Debug, Clone)]
pub struct TokenSignalMatrixRow {
    pub family: SignalFamily,
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

impl TokenSignalMatrixRow {
    pub fn from_family(
        family: SignalFamily,
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
        let label = label.into();
        Self {
            family,
            label,
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

#[derive(Debug, Clone, Default)]
pub struct TokenSignalMatrix {
    pub rows: Vec<TokenSignalMatrixRow>,
}

impl TokenSignalMatrix {
    pub fn new() -> Self {
        Self { rows: Vec::new() }
    }

    pub fn from_rows(rows: Vec<TokenSignalMatrixRow>) -> Self {
        Self { rows }
    }

    pub fn push(&mut self, row: TokenSignalMatrixRow) {
        self.rows.push(row);
    }

    pub fn best_by_lift(&self) -> Option<&TokenSignalMatrixRow> {
        self.rows.iter().max_by(|a, b| {
            a.lift_vs_base
                .partial_cmp(&b.lift_vs_base)
                .unwrap_or(Ordering::Equal)
        })
    }

    pub fn best_by_oracle_headroom(&self) -> Option<&TokenSignalMatrixRow> {
        self.rows.iter().min_by(|a, b| {
            a.oracle_headroom
                .partial_cmp(&b.oracle_headroom)
                .unwrap_or(Ordering::Equal)
        })
    }

    pub fn sort_by_lift_desc(&mut self) {
        self.rows.sort_by(|a, b| {
            b.lift_vs_base
                .partial_cmp(&a.lift_vs_base)
                .unwrap_or(Ordering::Equal)
        });
    }

    pub fn render(&self) -> String {
        let mut out = String::new();
        let mut rows = self.rows.clone();
        rows.sort_by(|a, b| {
            b.lift_vs_base
                .partial_cmp(&a.lift_vs_base)
                .unwrap_or(Ordering::Equal)
        });

        let _ = writeln!(out, "chronohorn_token_signal_matrix");
        let _ = writeln!(out, "rows: {}", rows.len());
        for row in rows {
            let lambda = row
                .selected_runtime_lambda
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "-".to_string());
            let hit_rate = row
                .hit_rate
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "-".to_string());
            let better_rate = row
                .better_rate
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "-".to_string());
            let _ = writeln!(
                out,
                "{:<20} {:<24} base {:>8.4} primary {:>8.4} oracle {:>8.4} lift {:>8.4} headroom {:>8.4} gate {:<10} lambda {:>6} hit {:>6} better {:>6}",
                row.family.as_str(),
                row.label,
                row.base_bpt,
                row.primary_bpt,
                row.oracle_bpt,
                row.lift_vs_base,
                row.oracle_headroom,
                row.selected_runtime_gate,
                lambda,
                hit_rate,
                better_rate
            );
            if !row.extra_metrics.is_empty() {
                let extras = row
                    .extra_metrics
                    .iter()
                    .map(|metric| format!("{}={:.4}", metric.key, metric.value))
                    .collect::<Vec<_>>()
                    .join(" ");
                let _ = writeln!(out, "  extras: {extras}");
            }
        }
        out
    }
}

impl From<&TokenBridgeReport> for TokenSignalMatrixRow {
    fn from(report: &TokenBridgeReport) -> Self {
        let primary_bpt = match report.selected_runtime_gate.as_str() {
            "heuristic" => report.eval_bpt_heuristic,
            "direct" => report.eval_bpt_direct,
            "mlp" => report.eval_bpt_mlp,
            "bucket" => report.eval_bpt_bucket,
            _ => report
                .eval_bpt_heuristic
                .min(report.eval_bpt_direct)
                .min(report.eval_bpt_mlp)
                .min(report.eval_bpt_bucket),
        };
        Self::from_family(
            SignalFamily::TokenBridge,
            format!(
                "train {} trigram {} stride {} k {}",
                report.train_token_budget,
                report.trigram_buckets,
                report.train_stride,
                report.candidate_k
            ),
            report.eval_bpt_base,
            primary_bpt,
            report.eval_bpt_oracle,
            report.selected_runtime_gate.clone(),
            None,
            Some(report.eval_topk_hit_rate),
            Some(report.eval_topk_better_rate),
            vec![
                MatrixMetric {
                    key: "eval_heuristic".to_string(),
                    value: report.eval_bpt_heuristic,
                },
                MatrixMetric {
                    key: "eval_direct".to_string(),
                    value: report.eval_bpt_direct,
                },
                MatrixMetric {
                    key: "eval_mlp".to_string(),
                    value: report.eval_bpt_mlp,
                },
                MatrixMetric {
                    key: "eval_bucket".to_string(),
                    value: report.eval_bpt_bucket,
                },
            ],
        )
    }
}

impl From<&TokenCopyBridgeReport> for TokenSignalMatrixRow {
    fn from(report: &TokenCopyBridgeReport) -> Self {
        let primary_bpt = match report.selected_runtime_gate.as_str() {
            "heuristic" => report.eval_bpt_heuristic,
            "direct" => report.eval_bpt_direct,
            _ => report.eval_bpt_heuristic.min(report.eval_bpt_direct),
        };
        Self::from_family(
            SignalFamily::TokenCopy,
            format!(
                "train {} window {} decay {:.3} k {}",
                report.train_token_budget,
                report.copy_window,
                report.copy_decay,
                report.candidate_k
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
}

impl From<&TokenSkipBridgeReport> for TokenSignalMatrixRow {
    fn from(report: &TokenSkipBridgeReport) -> Self {
        let primary_bpt = match report.selected_runtime_gate.as_str() {
            "heuristic" => report.eval_bpt_heuristic,
            "direct" => report.eval_bpt_direct,
            _ => report.eval_bpt_heuristic.min(report.eval_bpt_direct),
        };
        let selected_runtime_lambda = match report.selected_runtime_gate.as_str() {
            "heuristic" => Some(report.tuned_heuristic_lambda),
            "direct" => Some(report.tuned_direct_lambda),
            _ => None,
        };
        Self::from_family(
            SignalFamily::TokenSkip,
            format!(
                "train {} skip {}:{} k {}",
                report.train_token_budget, report.skip_far, report.skip_near, report.candidate_k
            ),
            report.eval_bpt_base,
            primary_bpt,
            report.eval_bpt_oracle,
            report.selected_runtime_gate.clone(),
            selected_runtime_lambda,
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
}

impl From<&TokenSkipCopyBridgeReport> for TokenSignalMatrixRow {
    fn from(report: &TokenSkipCopyBridgeReport) -> Self {
        let primary_bpt = match report.selected_runtime_gate.as_str() {
            "heuristic" => report.eval_bpt_heuristic,
            "direct" => report.eval_bpt_direct,
            _ => report.eval_bpt_heuristic.min(report.eval_bpt_direct),
        };
        Self::from_family(
            SignalFamily::TokenSkipCopy,
            format!(
                "train {} window {} decay {:.3} k {}",
                report.train_token_budget,
                report.copy_window,
                report.copy_decay,
                report.candidate_k
            ),
            report.eval_bpt_skip,
            primary_bpt,
            report.eval_bpt_oracle,
            report.selected_runtime_gate.clone(),
            Some(report.selected_runtime_lambda),
            Some(report.eval_copy_better_rate),
            Some(report.eval_skip_better_rate),
            vec![
                MatrixMetric {
                    key: "eval_skip".to_string(),
                    value: report.eval_bpt_skip,
                },
                MatrixMetric {
                    key: "eval_copy".to_string(),
                    value: report.eval_bpt_copy,
                },
                MatrixMetric {
                    key: "eval_agreement".to_string(),
                    value: report.eval_agreement_rate,
                },
            ],
        )
    }
}

impl From<&TokenMatchBridgeReport> for TokenSignalMatrixRow {
    fn from(report: &TokenMatchBridgeReport) -> Self {
        let primary_bpt = match report.selected_runtime_gate.as_str() {
            "heuristic" => report.eval_bpt_heuristic,
            "direct" => report.eval_bpt_direct,
            _ => report.eval_bpt_heuristic.min(report.eval_bpt_direct),
        };
        Self::from_family(
            SignalFamily::TokenMatch,
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
}

impl From<&TokenDecayBridgeReport> for TokenSignalMatrixRow {
    fn from(report: &TokenDecayBridgeReport) -> Self {
        let primary_bpt = match report.selected_runtime_gate.as_str() {
            "heuristic" => report.eval_bpt_heuristic,
            "direct" => report.eval_bpt_direct,
            _ => report.eval_bpt_heuristic.min(report.eval_bpt_direct),
        };
        Self::from_family(
            SignalFamily::TokenDecay,
            format!(
                "train {} decay {:.3} k {}",
                report.train_token_budget, report.decay_factor, report.candidate_k
            ),
            report.eval_bpt_base,
            primary_bpt,
            report.eval_bpt_oracle,
            report.selected_runtime_gate.clone(),
            Some(report.selected_runtime_lambda),
            Some(report.eval_decay_hit_rate),
            Some(report.eval_decay_better_rate),
            vec![
                MatrixMetric {
                    key: "eval_decay_full".to_string(),
                    value: report.eval_bpt_decay_full,
                },
                MatrixMetric {
                    key: "eval_decay_topk".to_string(),
                    value: report.eval_bpt_decay_topk,
                },
            ],
        )
    }
}

impl From<&TokenWordBridgeReport> for TokenSignalMatrixRow {
    fn from(report: &TokenWordBridgeReport) -> Self {
        let primary_bpt = match report.selected_runtime_gate.as_str() {
            "heuristic" => report.eval_bpt_heuristic,
            "direct" => report.eval_bpt_direct,
            _ => report.eval_bpt_heuristic.min(report.eval_bpt_direct),
        };
        Self::from_family(
            SignalFamily::TokenWord,
            format!(
                "train {} window {} suffix {} markers {}",
                report.train_token_budget,
                report.word_window,
                report.suffix_len,
                report.boundary_markers
            ),
            report.eval_bpt_base,
            primary_bpt,
            report.eval_bpt_oracle,
            report.selected_runtime_gate.clone(),
            Some(report.selected_runtime_lambda),
            Some(report.eval_word_hit_rate),
            Some(report.eval_word_better_rate),
            vec![
                MatrixMetric {
                    key: "eval_word_full".to_string(),
                    value: report.eval_bpt_word_full,
                },
                MatrixMetric {
                    key: "eval_word_topk".to_string(),
                    value: report.eval_bpt_word_topk,
                },
            ],
        )
    }
}

impl From<&TokenColumnBridgeReport> for TokenSignalMatrixRow {
    fn from(report: &TokenColumnBridgeReport) -> Self {
        let primary_bpt = match report.selected_runtime_gate.as_str() {
            "heuristic" => report.eval_bpt_heuristic,
            "direct" => report.eval_bpt_direct,
            _ => report.eval_bpt_heuristic.min(report.eval_bpt_direct),
        };
        Self::from_family(
            SignalFamily::TokenColumn,
            format!(
                "train {} period {} slot {} k {}",
                report.train_token_budget,
                report.slot_period,
                report.slot_buckets,
                report.candidate_k
            ),
            report.eval_bpt_base,
            primary_bpt,
            report.eval_bpt_oracle,
            report.selected_runtime_gate.clone(),
            Some(report.selected_runtime_lambda),
            Some(report.eval_column_hit_rate),
            Some(report.eval_column_better_rate),
            vec![
                MatrixMetric {
                    key: "eval_column_full".to_string(),
                    value: report.eval_bpt_column_full,
                },
                MatrixMetric {
                    key: "eval_column_topk".to_string(),
                    value: report.eval_bpt_column_topk,
                },
            ],
        )
    }
}
