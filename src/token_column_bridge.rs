use std::cmp::Ordering;
use std::path::Path;

use crate::data::{take_train_tokens, take_val_tokens};
use crate::packed_memory::{PackedTables, build_packed_tables};
use crate::protocol::{Runner, SampleOutputs};

const VOCAB_SIZE: usize = 1024;
const FEATURE_DIM: usize = 9;
const UNIFORM_FLOOR: f64 = 1e-12;
const COLUMN_HASH_MUL_A: u64 = 1_002_583_641;
const COLUMN_HASH_MUL_B: u64 = 2_100_523_641;

#[derive(Debug, Clone)]
pub struct TokenColumnBridgeReport {
    pub train_token_budget: usize,
    pub trigram_buckets: usize,
    pub slot_buckets: usize,
    pub slot_period: usize,
    pub val_token_budget: usize,
    pub train_stride: usize,
    pub candidate_k: usize,
    pub train_records: usize,
    pub tune_records: usize,
    pub eval_records: usize,
    pub tuned_heuristic_lambda: f64,
    pub tuned_direct_lambda: f64,
    pub selected_runtime_gate: String,
    pub selected_runtime_lambda: f64,
    pub tune_bpt_base: f64,
    pub tune_bpt_column_full: f64,
    pub tune_bpt_column_topk: f64,
    pub tune_bpt_heuristic: f64,
    pub tune_bpt_direct: f64,
    pub tune_bpt_oracle: f64,
    pub eval_bpt_base: f64,
    pub eval_bpt_column_full: f64,
    pub eval_bpt_column_topk: f64,
    pub eval_bpt_heuristic: f64,
    pub eval_bpt_direct: f64,
    pub eval_bpt_oracle: f64,
    pub eval_column_hit_rate: f64,
    pub eval_column_better_rate: f64,
}

#[derive(Debug, Clone)]
pub struct TrainedTokenColumnBridge {
    report: TokenColumnBridgeReport,
    runner: TokenColumnBridgeRunner,
    eval_tokens: Vec<usize>,
}

impl TrainedTokenColumnBridge {
    pub fn report(&self) -> &TokenColumnBridgeReport {
        &self.report
    }

    pub fn runner(&self) -> &TokenColumnBridgeRunner {
        &self.runner
    }

    pub fn eval_tokens(&self) -> &[usize] {
        &self.eval_tokens
    }
}

#[derive(Debug, Clone)]
pub struct TokenColumnBridgeRunner {
    base_tables: PackedTables,
    slot_tables: SlotTables,
    alpha_bigram: f64,
    alpha_trigram: f64,
    alpha_slot: f64,
    standardizer: Standardizer,
    gate: GateChoice,
    lambda: f64,
    candidate_k: usize,
    slot_period: usize,
    stream_prev2: Option<usize>,
    stream_prev1: Option<usize>,
    stream_pos: usize,
}

#[derive(Debug, Clone)]
struct SlotTables {
    vocab_size: usize,
    counts: Vec<f64>,
    totals: Vec<f64>,
    buckets: usize,
    slot_period: usize,
}

#[derive(Debug, Clone)]
struct Standardizer {
    mean: [f64; FEATURE_DIM],
    std: [f64; FEATURE_DIM],
}

#[derive(Debug, Clone)]
struct LogisticModel {
    weights: [f64; FEATURE_DIM],
    bias: f64,
}

#[derive(Debug, Clone)]
enum GateChoice {
    Heuristic,
    Direct(LogisticModel),
}

#[derive(Debug, Clone)]
struct RawColumnRecord {
    features: [f64; FEATURE_DIM],
    base_gold_prob: f64,
    column_full_gold_prob: f64,
    column_topk_gold_prob: f64,
    heuristic_gate: f64,
}

#[derive(Debug, Clone)]
struct ColumnRecord {
    features: [f64; FEATURE_DIM],
    base_gold_prob: f64,
    column_full_gold_prob: f64,
    column_topk_gold_prob: f64,
    heuristic_gate: f64,
}

#[derive(Debug, Clone, Copy)]
struct DistributionStats {
    top1_prob: f64,
    top2_prob: f64,
    entropy_norm: f64,
    top1_token: usize,
}

#[derive(Debug, Clone)]
struct ColumnSummary {
    full_dist: Vec<f64>,
    topk_dist: Vec<f64>,
    top1_prob: f64,
    topk_mass: f64,
    entropy_norm: f64,
    support: usize,
    top1_token: usize,
    slot_phase: f64,
    slot_index: usize,
}

pub fn train_token_column_bridge_from_data_root(
    root: &Path,
    train_token_budget: usize,
    trigram_buckets: usize,
    slot_buckets: usize,
    slot_period: usize,
    val_token_budget: usize,
    candidate_k: usize,
    train_stride: usize,
    alpha_bigram: f64,
    alpha_trigram: f64,
    alpha_slot: f64,
) -> Result<TrainedTokenColumnBridge, String> {
    if train_stride == 0 {
        return Err("train_stride must be positive".to_string());
    }
    if candidate_k == 0 {
        return Err("candidate_k must be positive".to_string());
    }
    if slot_period == 0 {
        return Err("slot_period must be positive".to_string());
    }

    let train_tokens = take_train_tokens(root, train_token_budget)?;
    let val_tokens = take_val_tokens(root, val_token_budget)?;
    if val_tokens.len() < 8 {
        return Err("need at least 8 validation tokens".to_string());
    }
    let split = (val_tokens.len() / 2).max(1);
    let tune_tokens = val_tokens[..split].to_vec();
    let eval_tokens = val_tokens[split..].to_vec();
    if eval_tokens.is_empty() {
        return Err("validation split left no eval tokens".to_string());
    }

    let base_tables = build_packed_tables(&train_tokens, VOCAB_SIZE, trigram_buckets)?;
    let slot_tables = build_slot_tables(&train_tokens, VOCAB_SIZE, slot_buckets, slot_period)?;
    let train_raw = collect_raw_records(
        &train_tokens,
        &base_tables,
        &slot_tables,
        alpha_bigram,
        alpha_trigram,
        alpha_slot,
        true,
        train_stride,
        candidate_k,
    );
    if train_raw.is_empty() {
        return Err("no token column bridge training records collected".to_string());
    }

    let standardizer = fit_standardizer_from_raw(&train_raw);
    let train_records = standardize_records(&train_raw, &standardizer);
    let direct_model = train_logistic_model(&train_records, 72, 0.12, 1e-4);

    let tune_raw = collect_raw_records(
        &tune_tokens,
        &base_tables,
        &slot_tables,
        alpha_bigram,
        alpha_trigram,
        alpha_slot,
        false,
        1,
        candidate_k,
    );
    let eval_raw = collect_raw_records(
        &eval_tokens,
        &base_tables,
        &slot_tables,
        alpha_bigram,
        alpha_trigram,
        alpha_slot,
        false,
        1,
        candidate_k,
    );
    if tune_raw.is_empty() || eval_raw.is_empty() {
        return Err("no token column bridge tune/eval records collected".to_string());
    }

    let tune_records = standardize_records(&tune_raw, &standardizer);
    let eval_records = standardize_records(&eval_raw, &standardizer);

    let tuned_heuristic_lambda = tune_lambda(&tune_records, |record| record.heuristic_gate);
    let tuned_direct_lambda = tune_lambda(&tune_records, |record| {
        predict_probability(&direct_model, &record.features)
    });

    let tune_bpt_base = mean_bits_base(&tune_records);
    let tune_bpt_column_full = mean_bits_column_full(&tune_records);
    let tune_bpt_column_topk = mean_bits_column_topk(&tune_records);
    let tune_bpt_heuristic = mean_bits_with_gate(&tune_records, tuned_heuristic_lambda, |record| {
        record.heuristic_gate
    });
    let tune_bpt_direct = mean_bits_with_gate(&tune_records, tuned_direct_lambda, |record| {
        predict_probability(&direct_model, &record.features)
    });
    let tune_bpt_oracle = oracle_bits_per_token(&tune_records);

    let eval_bpt_base = mean_bits_base(&eval_records);
    let eval_bpt_column_full = mean_bits_column_full(&eval_records);
    let eval_bpt_column_topk = mean_bits_column_topk(&eval_records);
    let eval_bpt_heuristic = mean_bits_with_gate(&eval_records, tuned_heuristic_lambda, |record| {
        record.heuristic_gate
    });
    let eval_bpt_direct = mean_bits_with_gate(&eval_records, tuned_direct_lambda, |record| {
        predict_probability(&direct_model, &record.features)
    });
    let eval_bpt_oracle = oracle_bits_per_token(&eval_records);

    let direct_selected = tune_bpt_direct < tune_bpt_heuristic;
    let (selected_runtime_gate, gate, selected_runtime_lambda) = if direct_selected {
        (
            "direct".to_string(),
            GateChoice::Direct(direct_model.clone()),
            tuned_direct_lambda,
        )
    } else {
        (
            "heuristic".to_string(),
            GateChoice::Heuristic,
            tuned_heuristic_lambda,
        )
    };

    let report = TokenColumnBridgeReport {
        train_token_budget,
        trigram_buckets,
        slot_buckets,
        slot_period,
        val_token_budget: val_tokens.len(),
        train_stride,
        candidate_k,
        train_records: train_records.len(),
        tune_records: tune_records.len(),
        eval_records: eval_records.len(),
        tuned_heuristic_lambda,
        tuned_direct_lambda,
        selected_runtime_gate,
        selected_runtime_lambda,
        tune_bpt_base,
        tune_bpt_column_full,
        tune_bpt_column_topk,
        tune_bpt_heuristic,
        tune_bpt_direct,
        tune_bpt_oracle,
        eval_bpt_base,
        eval_bpt_column_full,
        eval_bpt_column_topk,
        eval_bpt_heuristic,
        eval_bpt_direct,
        eval_bpt_oracle,
        eval_column_hit_rate: column_hit_rate(&eval_records),
        eval_column_better_rate: column_better_rate(&eval_records),
    };

    let runner = TokenColumnBridgeRunner {
        base_tables,
        slot_tables,
        alpha_bigram,
        alpha_trigram,
        alpha_slot,
        standardizer,
        gate,
        lambda: selected_runtime_lambda,
        candidate_k,
        slot_period,
        stream_prev2: None,
        stream_prev1: None,
        stream_pos: 0,
    };

    Ok(TrainedTokenColumnBridge {
        report,
        runner,
        eval_tokens,
    })
}

pub fn run_token_column_bridge_from_data_root(
    root: &Path,
    train_token_budget: usize,
    trigram_buckets: usize,
    slot_buckets: usize,
    slot_period: usize,
    val_token_budget: usize,
    candidate_k: usize,
    train_stride: usize,
    alpha_bigram: f64,
    alpha_trigram: f64,
    alpha_slot: f64,
) -> Result<TokenColumnBridgeReport, String> {
    Ok(train_token_column_bridge_from_data_root(
        root,
        train_token_budget,
        trigram_buckets,
        slot_buckets,
        slot_period,
        val_token_budget,
        candidate_k,
        train_stride,
        alpha_bigram,
        alpha_trigram,
        alpha_slot,
    )?
    .report)
}

pub fn render_token_column_bridge_report(report: &TokenColumnBridgeReport) -> String {
    let mut out = String::new();
    out.push_str("chronohorn_token_column_bridge\n");
    out.push_str(&format!(
        "train_token_budget: {}\n",
        report.train_token_budget
    ));
    out.push_str(&format!("trigram_buckets: {}\n", report.trigram_buckets));
    out.push_str(&format!("slot_buckets: {}\n", report.slot_buckets));
    out.push_str(&format!("slot_period: {}\n", report.slot_period));
    out.push_str(&format!("val_token_budget: {}\n", report.val_token_budget));
    out.push_str(&format!("train_stride: {}\n", report.train_stride));
    out.push_str(&format!("candidate_k: {}\n", report.candidate_k));
    out.push_str(&format!("train_records: {}\n", report.train_records));
    out.push_str(&format!("tune_records: {}\n", report.tune_records));
    out.push_str(&format!("eval_records: {}\n", report.eval_records));
    out.push_str(&format!(
        "tuned_heuristic_lambda: {:.3}\n",
        report.tuned_heuristic_lambda
    ));
    out.push_str(&format!(
        "tuned_direct_lambda: {:.3}\n",
        report.tuned_direct_lambda
    ));
    out.push_str(&format!(
        "selected_runtime_gate: {}\n",
        report.selected_runtime_gate
    ));
    out.push_str(&format!(
        "selected_runtime_lambda: {:.3}\n",
        report.selected_runtime_lambda
    ));
    out.push_str(&format!("tune_bpt_base: {:.6}\n", report.tune_bpt_base));
    out.push_str(&format!(
        "tune_bpt_column_full: {:.6}\n",
        report.tune_bpt_column_full
    ));
    out.push_str(&format!(
        "tune_bpt_column_topk: {:.6}\n",
        report.tune_bpt_column_topk
    ));
    out.push_str(&format!(
        "tune_bpt_heuristic: {:.6}\n",
        report.tune_bpt_heuristic
    ));
    out.push_str(&format!("tune_bpt_direct: {:.6}\n", report.tune_bpt_direct));
    out.push_str(&format!("tune_bpt_oracle: {:.6}\n", report.tune_bpt_oracle));
    out.push_str(&format!("eval_bpt_base: {:.6}\n", report.eval_bpt_base));
    out.push_str(&format!(
        "eval_bpt_column_full: {:.6}\n",
        report.eval_bpt_column_full
    ));
    out.push_str(&format!(
        "eval_bpt_column_topk: {:.6}\n",
        report.eval_bpt_column_topk
    ));
    out.push_str(&format!(
        "eval_bpt_heuristic: {:.6}\n",
        report.eval_bpt_heuristic
    ));
    out.push_str(&format!("eval_bpt_direct: {:.6}\n", report.eval_bpt_direct));
    out.push_str(&format!("eval_bpt_oracle: {:.6}\n", report.eval_bpt_oracle));
    out.push_str(&format!(
        "eval_column_hit_rate: {:.6}\n",
        report.eval_column_hit_rate
    ));
    out.push_str(&format!(
        "eval_column_better_rate: {:.6}\n",
        report.eval_column_better_rate
    ));
    out
}

impl Runner for TokenColumnBridgeRunner {
    fn name(&self) -> &'static str {
        "TokenColumnBridgeRunner"
    }

    fn vocab_size(&self) -> usize {
        self.base_tables.vocab_size
    }

    fn score_chunk(
        &self,
        tokens: &[usize],
        sample_positions: &[usize],
    ) -> Result<SampleOutputs, String> {
        let mut sample_predictions = Vec::with_capacity(sample_positions.len());
        let mut sample_gold_logprobs = Vec::with_capacity(sample_positions.len());
        for &pos in sample_positions {
            if pos >= tokens.len() {
                return Err(format!("sample position {pos} out of bounds"));
            }
            let base_prev2 = self.context_back(tokens, pos, 2);
            let base_prev1 = self.context_back(tokens, pos, 1);
            let base = score_base_distribution(
                &self.base_tables,
                self.alpha_bigram,
                self.alpha_trigram,
                base_prev2,
                base_prev1,
            );
            let slot = (self.stream_pos + pos) % self.slot_period;
            let slot_summary = build_slot_summary(
                slot,
                base_prev2,
                base_prev1,
                &self.slot_tables,
                self.alpha_slot,
                self.candidate_k,
                &base,
            );
            let base_stats = distribution_stats(&base);
            let features = standardize_features(
                extract_features_from_summary(&base_stats, &slot_summary, self.stream_pos + pos),
                &self.standardizer,
            );
            let raw_gate = match &self.gate {
                GateChoice::Heuristic => heuristic_gate(&slot_summary),
                GateChoice::Direct(model) => predict_probability(model, &features),
            };
            let gate = (self.lambda * raw_gate).clamp(0.0, 1.0);
            let mixed = mix_with_candidate_set(&base, &slot_summary.topk_dist, gate);
            let gold = mixed
                .get(tokens[pos])
                .copied()
                .unwrap_or(f64::MIN_POSITIVE)
                .max(f64::MIN_POSITIVE)
                .ln();
            sample_gold_logprobs.push(gold);
            sample_predictions.push(mixed);
        }
        Ok(SampleOutputs {
            sample_predictions,
            sample_gold_logprobs: Some(sample_gold_logprobs),
        })
    }

    fn adapt_chunk(&mut self, tokens: &[usize]) -> Result<(), String> {
        for &tok in tokens {
            self.stream_prev2 = self.stream_prev1;
            self.stream_prev1 = Some(tok);
            self.stream_pos += 1;
        }
        Ok(())
    }
}

impl TokenColumnBridgeRunner {
    fn context_back(&self, tokens: &[usize], pos: usize, back: usize) -> Option<usize> {
        match back {
            0 => None,
            1 => {
                if pos >= 1 {
                    Some(tokens[pos - 1])
                } else {
                    self.stream_prev1
                }
            }
            2 => {
                if pos >= 2 {
                    Some(tokens[pos - 2])
                } else if pos == 1 {
                    self.stream_prev1
                } else {
                    self.stream_prev2
                }
            }
            _ => None,
        }
    }
}

fn build_slot_tables(
    tokens: &[usize],
    vocab_size: usize,
    slot_buckets: usize,
    slot_period: usize,
) -> Result<SlotTables, String> {
    if slot_buckets == 0 {
        return Err("slot_buckets must be positive".to_string());
    }
    let mut counts = vec![0.0; slot_buckets * vocab_size];
    let mut totals = vec![0.0; slot_buckets];
    let mut prev2 = None;
    let mut prev1 = None;
    for (pos, &gold) in tokens.iter().enumerate() {
        let bucket = slot_bucket(slot_period, slot_buckets, pos, prev2, prev1);
        counts[bucket * vocab_size + gold] += 1.0;
        totals[bucket] += 1.0;
        prev2 = prev1;
        prev1 = Some(gold);
    }
    Ok(SlotTables {
        vocab_size,
        counts,
        totals,
        buckets: slot_buckets,
        slot_period,
    })
}

fn collect_raw_records(
    tokens: &[usize],
    base_tables: &PackedTables,
    slot_tables: &SlotTables,
    alpha_bigram: f64,
    alpha_trigram: f64,
    alpha_slot: f64,
    train: bool,
    stride: usize,
    candidate_k: usize,
) -> Vec<RawColumnRecord> {
    let mut rows = Vec::new();
    let mut prev2 = None;
    let mut prev1 = None;
    for (pos, &gold) in tokens.iter().enumerate() {
        if !train || pos % stride == 0 {
            let base =
                score_base_distribution(base_tables, alpha_bigram, alpha_trigram, prev2, prev1);
            let slot = slot_context_distribution(pos, prev2, prev1, slot_tables, alpha_slot, &base);
            let base_stats = distribution_stats(&base);
            let slot_stats = distribution_stats(&slot.full_dist);
            let base_gold_prob = base
                .get(gold)
                .copied()
                .unwrap_or(f64::MIN_POSITIVE)
                .max(f64::MIN_POSITIVE);
            let column_full_gold_prob = slot
                .full_dist
                .get(gold)
                .copied()
                .unwrap_or(f64::MIN_POSITIVE)
                .max(f64::MIN_POSITIVE);
            let column_topk_gold_prob = slot
                .topk_dist
                .get(gold)
                .copied()
                .unwrap_or(f64::MIN_POSITIVE)
                .max(f64::MIN_POSITIVE);
            rows.push(RawColumnRecord {
                features: extract_features_from_summary(&base_stats, &slot, pos),
                base_gold_prob,
                column_full_gold_prob,
                column_topk_gold_prob,
                heuristic_gate: heuristic_gate(&slot),
            });
        }
        prev2 = prev1;
        prev1 = Some(gold);
    }
    rows
}

fn slot_context_distribution(
    pos: usize,
    prev2: Option<usize>,
    prev1: Option<usize>,
    slot_tables: &SlotTables,
    alpha_slot: f64,
    base: &[f64],
) -> ColumnSummary {
    let slot = pos % slot_tables.slot_period;
    let bucket = slot_bucket(
        slot_tables.slot_period,
        slot_tables.buckets,
        pos,
        prev2,
        prev1,
    );
    let row_start = bucket * slot_tables.vocab_size;
    let total = slot_tables.totals[bucket];
    let full_dist = if total <= 0.0 {
        base.to_vec()
    } else {
        let denom = (total + alpha_slot).max(1e-8);
        let mut out = vec![0.0; slot_tables.vocab_size];
        for tok in 0..slot_tables.vocab_size {
            out[tok] = (slot_tables.counts[row_start + tok] + alpha_slot * base[tok]) / denom;
        }
        normalize(&mut out);
        out
    };
    build_column_summary(
        full_dist,
        slot,
        slot_tables.slot_period,
        slot_tables.vocab_size,
        4,
    )
}

fn build_column_summary(
    full_dist: Vec<f64>,
    slot_index: usize,
    slot_period: usize,
    vocab_size: usize,
    candidate_k: usize,
) -> ColumnSummary {
    let (topk_dist, top1_prob, topk_mass, entropy_norm, support, top1_token) =
        summarize_distribution(&full_dist, candidate_k);
    ColumnSummary {
        full_dist,
        topk_dist,
        top1_prob,
        topk_mass,
        entropy_norm,
        support,
        top1_token,
        slot_phase: slot_index as f64 / slot_period.max(1) as f64,
        slot_index,
    }
}

fn build_slot_summary(
    slot: usize,
    prev2: Option<usize>,
    prev1: Option<usize>,
    slot_tables: &SlotTables,
    alpha_slot: f64,
    candidate_k: usize,
    base: &[f64],
) -> ColumnSummary {
    let bucket = slot_bucket(
        slot_tables.slot_period,
        slot_tables.buckets,
        slot,
        prev2,
        prev1,
    );
    let row_start = bucket * slot_tables.vocab_size;
    let total = slot_tables.totals[bucket];
    let full_dist = if total <= 0.0 {
        base.to_vec()
    } else {
        let denom = (total + alpha_slot).max(1e-8);
        let mut out = vec![0.0; slot_tables.vocab_size];
        for tok in 0..slot_tables.vocab_size {
            out[tok] = (slot_tables.counts[row_start + tok] + alpha_slot * base[tok]) / denom;
        }
        normalize(&mut out);
        out
    };
    build_column_summary(
        full_dist,
        slot,
        slot_tables.slot_period,
        slot_tables.vocab_size,
        candidate_k,
    )
}

fn fit_standardizer_from_raw(records: &[RawColumnRecord]) -> Standardizer {
    let mut mean = [0.0; FEATURE_DIM];
    for record in records {
        for (index, value) in record.features.iter().enumerate() {
            mean[index] += *value;
        }
    }
    for value in &mut mean {
        *value /= records.len() as f64;
    }
    let mut var = [0.0; FEATURE_DIM];
    for record in records {
        for (index, value) in record.features.iter().enumerate() {
            let diff = *value - mean[index];
            var[index] += diff * diff;
        }
    }
    let mut std = [1.0; FEATURE_DIM];
    for (index, value) in var.iter().enumerate() {
        std[index] = (value / records.len() as f64).sqrt().max(1e-6);
    }
    Standardizer { mean, std }
}

fn standardize_records(
    records: &[RawColumnRecord],
    standardizer: &Standardizer,
) -> Vec<ColumnRecord> {
    records
        .iter()
        .map(|record| ColumnRecord {
            features: standardize_features(record.features, standardizer),
            base_gold_prob: record.base_gold_prob,
            column_full_gold_prob: record.column_full_gold_prob,
            column_topk_gold_prob: record.column_topk_gold_prob,
            heuristic_gate: record.heuristic_gate,
        })
        .collect()
}

fn standardize_features(
    features: [f64; FEATURE_DIM],
    standardizer: &Standardizer,
) -> [f64; FEATURE_DIM] {
    let mut out = [0.0; FEATURE_DIM];
    for index in 0..FEATURE_DIM {
        out[index] = (features[index] - standardizer.mean[index]) / standardizer.std[index];
    }
    out
}

fn extract_features_from_stats(
    base: &DistributionStats,
    slot: &ColumnSummary,
    pos: usize,
) -> [f64; FEATURE_DIM] {
    [
        base.top1_prob,
        base.entropy_norm,
        slot.top1_prob,
        0.0_f64.max(slot.top1_prob - slot.topk_mass),
        slot.entropy_norm,
        slot.support as f64 / VOCAB_SIZE as f64,
        pos as f64 / 32_768.0,
        slot.slot_phase,
        if base.top1_token == slot.top1_token {
            1.0
        } else {
            0.0
        },
    ]
}

fn extract_features_from_summary(
    base: &DistributionStats,
    slot: &ColumnSummary,
    pos: usize,
) -> [f64; FEATURE_DIM] {
    extract_features_from_stats(base, slot, pos)
}

fn heuristic_gate(summary: &ColumnSummary) -> f64 {
    (summary.topk_mass * (1.0 - summary.entropy_norm)).clamp(0.0, 1.0)
}

fn tune_lambda<F>(records: &[ColumnRecord], gate_fn: F) -> f64
where
    F: Fn(&ColumnRecord) -> f64,
{
    let mut best_lambda = 0.0;
    let mut best_bits = f64::INFINITY;
    for step in 0..=20 {
        let lambda = step as f64 / 20.0;
        let bits = mean_bits_with_gate(records, lambda, &gate_fn);
        if bits < best_bits {
            best_bits = bits;
            best_lambda = lambda;
        }
    }
    best_lambda
}

fn mean_bits_base(records: &[ColumnRecord]) -> f64 {
    records
        .iter()
        .map(|record| -record.base_gold_prob.max(UNIFORM_FLOOR).log2())
        .sum::<f64>()
        / records.len() as f64
}

fn mean_bits_column_full(records: &[ColumnRecord]) -> f64 {
    records
        .iter()
        .map(|record| -record.column_full_gold_prob.max(UNIFORM_FLOOR).log2())
        .sum::<f64>()
        / records.len() as f64
}

fn mean_bits_column_topk(records: &[ColumnRecord]) -> f64 {
    records
        .iter()
        .map(|record| -record.column_topk_gold_prob.max(UNIFORM_FLOOR).log2())
        .sum::<f64>()
        / records.len() as f64
}

fn mean_bits_with_gate<F>(records: &[ColumnRecord], lambda: f64, gate_fn: F) -> f64
where
    F: Fn(&ColumnRecord) -> f64,
{
    records
        .iter()
        .map(|record| {
            let gate = (lambda * gate_fn(record)).clamp(0.0, 1.0);
            let mixed =
                mix_probabilities(record.base_gold_prob, record.column_topk_gold_prob, gate);
            -mixed.max(UNIFORM_FLOOR).log2()
        })
        .sum::<f64>()
        / records.len() as f64
}

fn oracle_bits_per_token(records: &[ColumnRecord]) -> f64 {
    records
        .iter()
        .map(|record| {
            let best = record
                .base_gold_prob
                .max(record.column_topk_gold_prob)
                .max(UNIFORM_FLOOR);
            -best.log2()
        })
        .sum::<f64>()
        / records.len() as f64
}

fn column_hit_rate(records: &[ColumnRecord]) -> f64 {
    records
        .iter()
        .filter(|record| record.column_topk_gold_prob > 0.0)
        .count() as f64
        / records.len() as f64
}

fn column_better_rate(records: &[ColumnRecord]) -> f64 {
    records
        .iter()
        .filter(|record| record.column_topk_gold_prob > record.base_gold_prob)
        .count() as f64
        / records.len() as f64
}

fn predict_probability(model: &LogisticModel, features: &[f64; FEATURE_DIM]) -> f64 {
    let mut score = model.bias;
    for (weight, feature) in model.weights.iter().zip(features.iter()) {
        score += weight * feature;
    }
    sigmoid(score)
}

fn train_logistic_model(
    records: &[ColumnRecord],
    epochs: usize,
    lr: f64,
    l2: f64,
) -> LogisticModel {
    let mut model = LogisticModel {
        weights: [0.0; FEATURE_DIM],
        bias: 0.0,
    };
    for epoch in 0..epochs {
        let rate = lr / (1.0 + epoch as f64 * 0.05);
        for record in records {
            let target = if record.column_topk_gold_prob > record.base_gold_prob {
                1.0
            } else {
                0.0
            };
            let pred = predict_probability(&model, &record.features);
            let err = pred - target;
            model.bias -= rate * err;
            for index in 0..FEATURE_DIM {
                model.weights[index] -=
                    rate * (err * record.features[index] + l2 * model.weights[index]);
            }
        }
    }
    model
}

fn score_base_distribution(
    tables: &PackedTables,
    alpha_bigram: f64,
    alpha_trigram: f64,
    prev2: Option<usize>,
    prev1: Option<usize>,
) -> Vec<f64> {
    let vocab = tables.vocab_size;
    if let Some(p1) = prev1 {
        let row_start = p1 * vocab;
        let total = tables.bigram_totals[p1];
        let denom = (total + alpha_bigram).max(1e-8);
        let mut out = vec![0.0; vocab];
        for tok in 0..vocab {
            out[tok] = (tables.bigram_counts[row_start + tok]
                + alpha_bigram * tables.unigram_probs[tok])
                / denom;
        }
        if let (Some(p2), Some(p1)) = (prev2, prev1) {
            let bucket = ((p2 as u64 * COLUMN_HASH_MUL_A + p1 as u64 * COLUMN_HASH_MUL_B)
                % tables.trigram_buckets as u64) as usize;
            let row_start = bucket * vocab;
            let total = tables.trigram_totals[bucket];
            let denom = (total + alpha_trigram).max(1e-8);
            let mut out2 = vec![0.0; vocab];
            for tok in 0..vocab {
                out2[tok] =
                    (tables.trigram_counts[row_start + tok] + alpha_trigram * out[tok]) / denom;
            }
            normalize(&mut out2);
            out2
        } else {
            normalize(&mut out);
            out
        }
    } else {
        tables.unigram_probs.clone()
    }
}

fn summarize_distribution(
    distribution: &[f64],
    candidate_k: usize,
) -> (Vec<f64>, f64, f64, f64, usize, usize) {
    let mut pairs: Vec<(usize, f64)> = distribution.iter().copied().enumerate().collect();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    let support = candidate_k.min(pairs.len());
    let mut topk_dist = vec![0.0; distribution.len()];
    let mut topk_mass = 0.0;
    for &(tok, prob) in pairs.iter().take(support) {
        topk_dist[tok] = prob;
        topk_mass += prob;
    }
    if topk_mass > 0.0 {
        for value in &mut topk_dist {
            *value /= topk_mass;
        }
    }
    let entropy = distribution
        .iter()
        .filter(|value| **value > 0.0)
        .map(|value| -value * value.ln())
        .sum::<f64>();
    let entropy_norm = if distribution.len() > 1 {
        entropy / (distribution.len() as f64).ln().max(f64::MIN_POSITIVE)
    } else {
        0.0
    };
    let top1 = pairs.first().copied().unwrap_or((0, 0.0));
    let top2 = pairs.get(1).copied().unwrap_or((0, 0.0));
    (topk_dist, top1.1, topk_mass, entropy_norm, support, top1.0)
}

fn distribution_stats(distribution: &[f64]) -> DistributionStats {
    let mut top1_prob = 0.0;
    let mut top2_prob = 0.0;
    let mut top1_token = 0usize;
    for (index, &value) in distribution.iter().enumerate() {
        if value >= top1_prob {
            top2_prob = top1_prob;
            top1_prob = value;
            top1_token = index;
        } else if value > top2_prob {
            top2_prob = value;
        }
    }
    let entropy = distribution
        .iter()
        .filter(|value| **value > 0.0)
        .map(|value| -value * value.ln())
        .sum::<f64>();
    let entropy_norm = if distribution.len() > 1 {
        entropy / (distribution.len() as f64).ln().max(f64::MIN_POSITIVE)
    } else {
        0.0
    };
    DistributionStats {
        top1_prob,
        top2_prob,
        entropy_norm,
        top1_token,
    }
}

fn slot_bucket(
    slot_period: usize,
    slot_buckets: usize,
    pos: usize,
    prev2: Option<usize>,
    prev1: Option<usize>,
) -> usize {
    let slot = pos % slot_period.max(1);
    let mut hash: u64 = slot as u64 + 0x9E37_79B9_7F4A_7C15;
    if let Some(p2) = prev2 {
        hash ^= p2 as u64 + COLUMN_HASH_MUL_A + (hash << 6) + (hash >> 2);
    }
    if let Some(p1) = prev1 {
        hash ^= p1 as u64 + COLUMN_HASH_MUL_B + (hash << 6) + (hash >> 2);
    }
    (hash % slot_buckets as u64) as usize
}

fn push_recent(history: &mut std::collections::VecDeque<usize>, tok: usize, limit: usize) {
    history.push_back(tok);
    while history.len() > limit {
        history.pop_front();
    }
}

fn normalize(values: &mut [f64]) {
    let total: f64 = values.iter().sum();
    let denom = total.max(f64::EPSILON);
    for value in values {
        *value /= denom;
    }
}

fn mix_probabilities(base: f64, expert: f64, gate: f64) -> f64 {
    (1.0 - gate) * base + gate * expert
}

fn mix_with_candidate_set(base: &[f64], candidate: &[f64], gate: f64) -> Vec<f64> {
    let mut out = vec![0.0; base.len()];
    for index in 0..base.len() {
        out[index] = mix_probabilities(base[index], candidate[index], gate);
    }
    normalize(&mut out);
    out
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        let z = (-value).exp();
        1.0 / (1.0 + z)
    } else {
        let z = value.exp();
        z / (1.0 + z)
    }
}
