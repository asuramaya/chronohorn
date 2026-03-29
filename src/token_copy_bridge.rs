use std::collections::{HashMap, VecDeque};
use std::path::Path;

use crate::data::{take_train_tokens, take_val_tokens};
use crate::packed_memory::{PackedTables, build_packed_tables};
use crate::protocol::{Runner, SampleOutputs};

const FEATURE_DIM: usize = 9;
const TRIGRAM_HASH_MUL_A: u64 = 1_315_423_911;
const TRIGRAM_HASH_MUL_B: u64 = 2_654_435_761;
const COPY_SMOOTHING: f64 = 0.02;
const UNIFORM_FLOOR: f64 = 1e-12;

#[derive(Debug, Clone)]
pub struct TokenCopyBridgeReport {
    pub train_token_budget: usize,
    pub trigram_buckets: usize,
    pub val_token_budget: usize,
    pub copy_window: usize,
    pub candidate_k: usize,
    pub copy_decay: f64,
    pub train_stride: usize,
    pub train_records: usize,
    pub tune_records: usize,
    pub eval_records: usize,
    pub tuned_heuristic_lambda: f64,
    pub tuned_direct_lambda: f64,
    pub selected_runtime_gate: String,
    pub selected_runtime_lambda: f64,
    pub tune_bpt_base: f64,
    pub tune_bpt_copy_full: f64,
    pub tune_bpt_copy_topk: f64,
    pub tune_bpt_heuristic: f64,
    pub tune_bpt_direct: f64,
    pub tune_bpt_oracle: f64,
    pub eval_bpt_base: f64,
    pub eval_bpt_copy_full: f64,
    pub eval_bpt_copy_topk: f64,
    pub eval_bpt_heuristic: f64,
    pub eval_bpt_direct: f64,
    pub eval_bpt_oracle: f64,
    pub eval_copy_hit_rate: f64,
    pub eval_copy_better_rate: f64,
}

#[derive(Debug, Clone)]
pub struct TrainedTokenCopyBridge {
    report: TokenCopyBridgeReport,
    runner: TokenCopyBridgeRunner,
    eval_tokens: Vec<usize>,
}

impl TrainedTokenCopyBridge {
    pub fn report(&self) -> &TokenCopyBridgeReport {
        &self.report
    }

    pub fn runner(&self) -> &TokenCopyBridgeRunner {
        &self.runner
    }

    pub fn eval_tokens(&self) -> &[usize] {
        &self.eval_tokens
    }
}

#[derive(Debug, Clone)]
pub struct TokenCopyBridgeRunner {
    tables: PackedTables,
    alpha_bigram: f64,
    alpha_trigram: f64,
    copy_window: usize,
    copy_decay: f64,
    candidate_k: usize,
    standardizer: Standardizer,
    gate: GateChoice,
    lambda: f64,
    stream_prev2: Option<usize>,
    stream_prev1: Option<usize>,
    recent_history: VecDeque<usize>,
}

impl TokenCopyBridgeRunner {
    pub fn tables(&self) -> &PackedTables {
        &self.tables
    }

    pub fn copy_window(&self) -> usize {
        self.copy_window
    }

    pub fn candidate_k(&self) -> usize {
        self.candidate_k
    }
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
struct RawCopyRecord {
    features: [f64; FEATURE_DIM],
    base_gold_prob: f64,
    copy_full_gold_prob: f64,
    copy_topk_gold_prob: f64,
    heuristic_gate: f64,
}

#[derive(Debug, Clone)]
struct CopyRecord {
    features: [f64; FEATURE_DIM],
    base_gold_prob: f64,
    copy_full_gold_prob: f64,
    copy_topk_gold_prob: f64,
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
struct CopySummary {
    full_dist: Vec<f64>,
    topk_dist: Vec<f64>,
    top1_prob: f64,
    topk_mass: f64,
    entropy_norm: f64,
    support: usize,
    top1_token: usize,
}

pub fn train_token_copy_bridge_from_data_root(
    root: &Path,
    train_token_budget: usize,
    trigram_buckets: usize,
    val_token_budget: usize,
    copy_window: usize,
    candidate_k: usize,
    train_stride: usize,
    alpha_bigram: f64,
    alpha_trigram: f64,
    copy_decay: f64,
) -> Result<TrainedTokenCopyBridge, String> {
    if train_stride == 0 {
        return Err("train_stride must be positive".to_string());
    }
    if copy_window == 0 {
        return Err("copy_window must be positive".to_string());
    }
    if candidate_k == 0 {
        return Err("candidate_k must be positive".to_string());
    }
    if !(0.0 < copy_decay && copy_decay <= 1.0) {
        return Err("copy_decay must be in (0, 1]".to_string());
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

    let tables = build_packed_tables(&train_tokens, 1024, trigram_buckets)?;
    let train_raw = collect_raw_records(
        &train_tokens,
        &tables,
        alpha_bigram,
        alpha_trigram,
        copy_window,
        copy_decay,
        candidate_k,
        train_stride,
    );
    if train_raw.is_empty() {
        return Err("no token copy bridge training records collected".to_string());
    }
    let standardizer = fit_standardizer_from_raw(&train_raw);
    let train_records = standardize_records(&train_raw, &standardizer);
    let direct_model = train_direct_gate(&train_records, 80, 0.1, 1e-4);

    let tune_raw = collect_raw_records(
        &tune_tokens,
        &tables,
        alpha_bigram,
        alpha_trigram,
        copy_window,
        copy_decay,
        candidate_k,
        1,
    );
    let eval_raw = collect_raw_records(
        &eval_tokens,
        &tables,
        alpha_bigram,
        alpha_trigram,
        copy_window,
        copy_decay,
        candidate_k,
        1,
    );
    let tune_records = standardize_records(&tune_raw, &standardizer);
    let eval_records = standardize_records(&eval_raw, &standardizer);

    let tuned_heuristic_lambda = tune_lambda(&tune_records, |record| record.heuristic_gate);
    let tuned_direct_lambda = tune_lambda(&tune_records, |record| {
        predict_probability(&direct_model, &record.features)
    });

    let tune_bpt_base = mean_bits_per_token_base(&tune_records);
    let tune_bpt_copy_full = mean_bits_per_token_copy_full(&tune_records);
    let tune_bpt_copy_topk = mean_bits_per_token_copy_topk(&tune_records);
    let tune_bpt_heuristic =
        mean_bits_per_token_with_heuristic(&tune_records, tuned_heuristic_lambda);
    let tune_bpt_direct =
        mean_bits_per_token_with_direct(&tune_records, tuned_direct_lambda, &direct_model);
    let tune_bpt_oracle = oracle_bits_per_token(&tune_records);

    let eval_bpt_base = mean_bits_per_token_base(&eval_records);
    let eval_bpt_copy_full = mean_bits_per_token_copy_full(&eval_records);
    let eval_bpt_copy_topk = mean_bits_per_token_copy_topk(&eval_records);
    let eval_bpt_heuristic =
        mean_bits_per_token_with_heuristic(&eval_records, tuned_heuristic_lambda);
    let eval_bpt_direct =
        mean_bits_per_token_with_direct(&eval_records, tuned_direct_lambda, &direct_model);
    let eval_bpt_oracle = oracle_bits_per_token(&eval_records);

    let (selected_runtime_gate, gate_choice, selected_runtime_lambda) =
        if tune_bpt_direct < tune_bpt_heuristic {
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

    let report = TokenCopyBridgeReport {
        train_token_budget,
        trigram_buckets,
        val_token_budget: val_tokens.len(),
        copy_window,
        candidate_k,
        copy_decay,
        train_stride,
        train_records: train_records.len(),
        tune_records: tune_records.len(),
        eval_records: eval_records.len(),
        tuned_heuristic_lambda,
        tuned_direct_lambda,
        selected_runtime_gate,
        selected_runtime_lambda,
        tune_bpt_base,
        tune_bpt_copy_full,
        tune_bpt_copy_topk,
        tune_bpt_heuristic,
        tune_bpt_direct,
        tune_bpt_oracle,
        eval_bpt_base,
        eval_bpt_copy_full,
        eval_bpt_copy_topk,
        eval_bpt_heuristic,
        eval_bpt_direct,
        eval_bpt_oracle,
        eval_copy_hit_rate: copy_hit_rate(&eval_records),
        eval_copy_better_rate: copy_better_rate(&eval_records),
    };

    let runner = TokenCopyBridgeRunner {
        tables,
        alpha_bigram,
        alpha_trigram,
        copy_window,
        copy_decay,
        candidate_k,
        standardizer,
        gate: gate_choice,
        lambda: selected_runtime_lambda,
        stream_prev2: None,
        stream_prev1: None,
        recent_history: VecDeque::new(),
    };

    Ok(TrainedTokenCopyBridge {
        report,
        runner,
        eval_tokens,
    })
}

pub fn run_token_copy_bridge_from_data_root(
    root: &Path,
    train_token_budget: usize,
    trigram_buckets: usize,
    val_token_budget: usize,
    copy_window: usize,
    candidate_k: usize,
    train_stride: usize,
    alpha_bigram: f64,
    alpha_trigram: f64,
    copy_decay: f64,
) -> Result<TokenCopyBridgeReport, String> {
    Ok(train_token_copy_bridge_from_data_root(
        root,
        train_token_budget,
        trigram_buckets,
        val_token_budget,
        copy_window,
        candidate_k,
        train_stride,
        alpha_bigram,
        alpha_trigram,
        copy_decay,
    )?
    .report)
}

pub fn render_token_copy_bridge_report(report: &TokenCopyBridgeReport) -> String {
    let mut out = String::new();
    out.push_str("chronohorn_token_copy_bridge\n");
    out.push_str(&format!(
        "train_token_budget: {}\n",
        report.train_token_budget
    ));
    out.push_str(&format!("trigram_buckets: {}\n", report.trigram_buckets));
    out.push_str(&format!("val_token_budget: {}\n", report.val_token_budget));
    out.push_str(&format!("copy_window: {}\n", report.copy_window));
    out.push_str(&format!("candidate_k: {}\n", report.candidate_k));
    out.push_str(&format!("copy_decay: {:.6}\n", report.copy_decay));
    out.push_str(&format!("train_stride: {}\n", report.train_stride));
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
        "tune_bpt_copy_full: {:.6}\n",
        report.tune_bpt_copy_full
    ));
    out.push_str(&format!(
        "tune_bpt_copy_topk: {:.6}\n",
        report.tune_bpt_copy_topk
    ));
    out.push_str(&format!(
        "tune_bpt_heuristic: {:.6}\n",
        report.tune_bpt_heuristic
    ));
    out.push_str(&format!("tune_bpt_direct: {:.6}\n", report.tune_bpt_direct));
    out.push_str(&format!("tune_bpt_oracle: {:.6}\n", report.tune_bpt_oracle));
    out.push_str(&format!("eval_bpt_base: {:.6}\n", report.eval_bpt_base));
    out.push_str(&format!(
        "eval_bpt_copy_full: {:.6}\n",
        report.eval_bpt_copy_full
    ));
    out.push_str(&format!(
        "eval_bpt_copy_topk: {:.6}\n",
        report.eval_bpt_copy_topk
    ));
    out.push_str(&format!(
        "eval_bpt_heuristic: {:.6}\n",
        report.eval_bpt_heuristic
    ));
    out.push_str(&format!("eval_bpt_direct: {:.6}\n", report.eval_bpt_direct));
    out.push_str(&format!("eval_bpt_oracle: {:.6}\n", report.eval_bpt_oracle));
    out.push_str(&format!(
        "eval_copy_hit_rate: {:.6}\n",
        report.eval_copy_hit_rate
    ));
    out.push_str(&format!(
        "eval_copy_better_rate: {:.6}\n",
        report.eval_copy_better_rate
    ));
    out
}

impl Runner for TokenCopyBridgeRunner {
    fn name(&self) -> &'static str {
        "TokenCopyBridgeRunner"
    }

    fn vocab_size(&self) -> usize {
        self.tables.vocab_size
    }

    fn score_chunk(
        &self,
        tokens: &[usize],
        sample_positions: &[usize],
    ) -> Result<SampleOutputs, String> {
        if sample_positions.is_empty() {
            return Ok(SampleOutputs {
                sample_predictions: Vec::new(),
                sample_gold_logprobs: Some(Vec::new()),
            });
        }

        let mut sample_lookup = HashMap::new();
        for (index, &pos) in sample_positions.iter().enumerate() {
            if pos >= tokens.len() {
                return Err(format!("sample position {pos} out of bounds"));
            }
            sample_lookup.insert(pos, index);
        }

        let mut sample_predictions = vec![Vec::new(); sample_positions.len()];
        let mut sample_gold_logprobs = vec![0.0; sample_positions.len()];
        let mut prev2 = self.stream_prev2;
        let mut prev1 = self.stream_prev1;
        let mut history = self.recent_history.clone();

        for (pos, &tok) in tokens.iter().enumerate() {
            let base = score_distribution(
                &self.tables,
                self.alpha_bigram,
                self.alpha_trigram,
                prev2,
                prev1,
            );
            let copy = build_recent_copy_summary(
                &history,
                self.copy_window,
                self.copy_decay,
                &self.tables,
                self.candidate_k,
            );
            let base_stats = distribution_stats(&base);
            let features =
                standardize_features(extract_features(&base_stats, &copy), &self.standardizer);
            let raw_gate = match &self.gate {
                GateChoice::Heuristic => heuristic_gate_from_summary(&copy),
                GateChoice::Direct(model) => predict_probability(model, &features),
            };
            let gate = (self.lambda * raw_gate).clamp(0.0, 1.0);
            let mixed = mix_with_topk(&base, &copy.topk_dist, gate);

            if let Some(&sample_index) = sample_lookup.get(&pos) {
                let gold = mixed
                    .get(tok)
                    .copied()
                    .unwrap_or(f64::MIN_POSITIVE)
                    .max(f64::MIN_POSITIVE)
                    .ln();
                sample_gold_logprobs[sample_index] = gold;
                sample_predictions[sample_index] = mixed;
            }

            push_recent(&mut history, tok, self.copy_window);
            prev2 = prev1;
            prev1 = Some(tok);
        }

        Ok(SampleOutputs {
            sample_predictions,
            sample_gold_logprobs: Some(sample_gold_logprobs),
        })
    }

    fn adapt_chunk(&mut self, tokens: &[usize]) -> Result<(), String> {
        for &tok in tokens {
            push_recent(&mut self.recent_history, tok, self.copy_window);
            self.stream_prev2 = self.stream_prev1;
            self.stream_prev1 = Some(tok);
        }
        Ok(())
    }
}

fn collect_raw_records(
    tokens: &[usize],
    tables: &PackedTables,
    alpha_bigram: f64,
    alpha_trigram: f64,
    copy_window: usize,
    copy_decay: f64,
    candidate_k: usize,
    stride: usize,
) -> Vec<RawCopyRecord> {
    let mut rows = Vec::new();
    let mut prev2 = None;
    let mut prev1 = None;
    let mut history = VecDeque::new();

    for (pos, &gold) in tokens.iter().enumerate() {
        if pos % stride == 0 {
            let base = score_distribution(tables, alpha_bigram, alpha_trigram, prev2, prev1);
            let copy =
                build_recent_copy_summary(&history, copy_window, copy_decay, tables, candidate_k);
            let base_stats = distribution_stats(&base);
            let features = extract_features(&base_stats, &copy);
            let base_gold_prob = base
                .get(gold)
                .copied()
                .unwrap_or(f64::MIN_POSITIVE)
                .max(f64::MIN_POSITIVE);
            let copy_full_gold_prob = copy
                .full_dist
                .get(gold)
                .copied()
                .unwrap_or(f64::MIN_POSITIVE)
                .max(f64::MIN_POSITIVE);
            let copy_topk_gold_prob = copy
                .topk_dist
                .get(gold)
                .copied()
                .unwrap_or(f64::MIN_POSITIVE)
                .max(f64::MIN_POSITIVE);
            rows.push(RawCopyRecord {
                features,
                base_gold_prob,
                copy_full_gold_prob,
                copy_topk_gold_prob,
                heuristic_gate: heuristic_gate_from_summary(&copy),
            });
        }
        push_recent(&mut history, gold, copy_window);
        prev2 = prev1;
        prev1 = Some(gold);
    }

    rows
}

fn fit_standardizer_from_raw(records: &[RawCopyRecord]) -> Standardizer {
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

fn standardize_records(records: &[RawCopyRecord], standardizer: &Standardizer) -> Vec<CopyRecord> {
    records
        .iter()
        .map(|record| CopyRecord {
            features: standardize_features(record.features, standardizer),
            base_gold_prob: record.base_gold_prob,
            copy_full_gold_prob: record.copy_full_gold_prob,
            copy_topk_gold_prob: record.copy_topk_gold_prob,
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

fn train_direct_gate(records: &[CopyRecord], epochs: usize, lr: f64, l2: f64) -> LogisticModel {
    let mut model = LogisticModel {
        weights: [0.0; FEATURE_DIM],
        bias: 0.0,
    };
    let ln2 = std::f64::consts::LN_2;
    for _ in 0..epochs {
        let mut grad_w = [0.0; FEATURE_DIM];
        let mut grad_b = 0.0;
        for record in records {
            let gate = predict_probability(&model, &record.features).clamp(1e-6, 1.0 - 1e-6);
            let mixed = ((1.0 - gate) * record.base_gold_prob + gate * record.copy_topk_gold_prob)
                .max(1e-12);
            let dloss_dgate = (record.base_gold_prob - record.copy_topk_gold_prob) / (mixed * ln2);
            let dloss_dz = dloss_dgate * gate * (1.0 - gate);
            for (index, value) in record.features.iter().enumerate() {
                grad_w[index] += dloss_dz * *value;
            }
            grad_b += dloss_dz;
        }
        let inv_n = 1.0 / records.len() as f64;
        for index in 0..FEATURE_DIM {
            model.weights[index] -= lr * (grad_w[index] * inv_n + l2 * model.weights[index]);
        }
        model.bias -= lr * grad_b * inv_n;
    }
    model
}

fn predict_probability(model: &LogisticModel, features: &[f64; FEATURE_DIM]) -> f64 {
    let mut z = model.bias;
    for (weight, value) in model.weights.iter().zip(features.iter()) {
        z += weight * value;
    }
    1.0 / (1.0 + (-z).exp())
}

fn tune_lambda(records: &[CopyRecord], gate_fn: impl Fn(&CopyRecord) -> f64) -> f64 {
    let mut best_lambda = 0.0;
    let mut best_bpt = f64::INFINITY;
    for step in 0..=12 {
        let lambda = step as f64 / 10.0;
        let bpt = mean_bits_per_token_with_gate(records, lambda, &gate_fn);
        if bpt < best_bpt {
            best_bpt = bpt;
            best_lambda = lambda;
        }
    }
    best_lambda
}

fn mean_bits_per_token_base(records: &[CopyRecord]) -> f64 {
    records
        .iter()
        .map(|record| -record.base_gold_prob.max(1e-12).log2())
        .sum::<f64>()
        / records.len() as f64
}

fn mean_bits_per_token_copy_full(records: &[CopyRecord]) -> f64 {
    records
        .iter()
        .map(|record| -record.copy_full_gold_prob.max(1e-12).log2())
        .sum::<f64>()
        / records.len() as f64
}

fn mean_bits_per_token_copy_topk(records: &[CopyRecord]) -> f64 {
    records
        .iter()
        .map(|record| -record.copy_topk_gold_prob.max(1e-12).log2())
        .sum::<f64>()
        / records.len() as f64
}

fn mean_bits_per_token_with_heuristic(records: &[CopyRecord], lambda: f64) -> f64 {
    mean_bits_per_token_with_gate(records, lambda, heuristic_gate)
}

fn mean_bits_per_token_with_direct(
    records: &[CopyRecord],
    lambda: f64,
    model: &LogisticModel,
) -> f64 {
    mean_bits_per_token_with_gate(records, lambda, |record| {
        predict_probability(model, &record.features)
    })
}

fn mean_bits_per_token_with_gate(
    records: &[CopyRecord],
    lambda: f64,
    gate_fn: impl Fn(&CopyRecord) -> f64,
) -> f64 {
    let mut bits = 0.0;
    for record in records {
        let gate = (lambda * gate_fn(record)).clamp(0.0, 1.0);
        let mixed =
            (1.0 - gate) * record.base_gold_prob + gate * record.copy_topk_gold_prob.max(1e-12);
        bits += -mixed.max(1e-12).log2();
    }
    bits / records.len() as f64
}

fn oracle_bits_per_token(records: &[CopyRecord]) -> f64 {
    records
        .iter()
        .map(|record| {
            let best = record
                .base_gold_prob
                .max(record.copy_topk_gold_prob)
                .max(1e-12);
            -best.log2()
        })
        .sum::<f64>()
        / records.len() as f64
}

fn copy_hit_rate(records: &[CopyRecord]) -> f64 {
    records
        .iter()
        .filter(|record| record.copy_topk_gold_prob > 0.0)
        .count() as f64
        / records.len() as f64
}

fn copy_better_rate(records: &[CopyRecord]) -> f64 {
    records
        .iter()
        .filter(|record| record.copy_topk_gold_prob > record.base_gold_prob)
        .count() as f64
        / records.len() as f64
}

fn heuristic_gate(record: &CopyRecord) -> f64 {
    record.heuristic_gate.clamp(0.0, 1.0)
}

fn heuristic_gate_from_summary(copy: &CopySummary) -> f64 {
    copy.topk_mass.clamp(0.0, 1.0)
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

fn build_recent_copy_summary(
    history: &VecDeque<usize>,
    recent_window: usize,
    decay: f64,
    tables: &PackedTables,
    candidate_k: usize,
) -> CopySummary {
    let vocab = tables.vocab_size;
    let mut weighted = vec![0.0; vocab];
    let mut weight = 1.0;
    for &tok in history.iter().rev().take(recent_window) {
        weighted[tok] += weight;
        weight *= decay;
    }

    let support = weighted.iter().filter(|value| **value > 0.0).count();
    let base_mass = weighted.iter().sum::<f64>();
    let smoothing_mass = base_mass.max(1.0) * COPY_SMOOTHING;
    let mut full_dist = weighted;
    for (tok, value) in full_dist.iter_mut().enumerate() {
        *value += smoothing_mass * tables.unigram_probs[tok] + UNIFORM_FLOOR;
    }
    normalize(&mut full_dist);

    let mut indexed = full_dist
        .iter()
        .enumerate()
        .map(|(index, &value)| (index, value))
        .collect::<Vec<_>>();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let topk = indexed
        .iter()
        .take(candidate_k.min(vocab.max(1)))
        .copied()
        .collect::<Vec<_>>();
    let topk_mass = topk.iter().map(|(_, p)| *p).sum::<f64>();
    let mut topk_dist = vec![0.0; vocab];
    if topk_mass > 0.0 {
        for (index, prob) in topk {
            topk_dist[index] = prob / topk_mass;
        }
    }

    let entropy = full_dist
        .iter()
        .filter(|value| **value > 0.0)
        .map(|value| -value * value.ln())
        .sum::<f64>();
    let entropy_norm = if vocab > 1 {
        entropy / (vocab as f64).ln().max(f64::MIN_POSITIVE)
    } else {
        0.0
    };

    CopySummary {
        full_dist,
        topk_dist,
        top1_prob: indexed.first().map(|(_, p)| *p).unwrap_or(0.0),
        topk_mass,
        entropy_norm,
        support,
        top1_token: indexed.first().map(|(i, _)| *i).unwrap_or(0),
    }
}

fn extract_features(base: &DistributionStats, copy: &CopySummary) -> [f64; FEATURE_DIM] {
    let agreement = if base.top1_token == copy.top1_token {
        1.0
    } else {
        0.0
    };
    [
        base.top1_prob,
        base.entropy_norm,
        base.top1_prob - base.top2_prob,
        copy.top1_prob,
        copy.topk_mass,
        copy.entropy_norm,
        (1.0 + copy.support as f64).ln(),
        agreement,
        copy.top1_prob - base.top1_prob,
    ]
}

fn score_distribution(
    tables: &PackedTables,
    alpha_bigram: f64,
    alpha_trigram: f64,
    prev2: Option<usize>,
    prev1: Option<usize>,
) -> Vec<f64> {
    let vocab = tables.vocab_size;
    let p_uni = &tables.unigram_probs;
    let p_bigram = if let Some(p1) = prev1 {
        let row_start = p1 * vocab;
        let total = tables.bigram_totals[p1];
        let denom = (total + alpha_bigram).max(1e-8);
        let mut out = vec![0.0; vocab];
        for tok in 0..vocab {
            out[tok] = (tables.bigram_counts[row_start + tok] + alpha_bigram * p_uni[tok]) / denom;
        }
        out
    } else {
        p_uni.clone()
    };
    if let (Some(p2), Some(p1)) = (prev2, prev1) {
        let bucket = trigram_bucket(tables.trigram_buckets, p2, p1);
        let row_start = bucket * vocab;
        let total = tables.trigram_totals[bucket];
        let denom = (total + alpha_trigram).max(1e-8);
        let mut out = vec![0.0; vocab];
        for tok in 0..vocab {
            out[tok] =
                (tables.trigram_counts[row_start + tok] + alpha_trigram * p_bigram[tok]) / denom;
        }
        normalize(&mut out);
        out
    } else {
        let mut out = p_bigram;
        normalize(&mut out);
        out
    }
}

fn trigram_bucket(trigram_buckets: usize, prev2: usize, prev1: usize) -> usize {
    ((prev2 as u64 * TRIGRAM_HASH_MUL_A + prev1 as u64 * TRIGRAM_HASH_MUL_B)
        % trigram_buckets as u64) as usize
}

fn mix_with_topk(base: &[f64], topk: &[f64], gate: f64) -> Vec<f64> {
    if gate <= 0.0 {
        return base.to_vec();
    }
    let mut out = Vec::with_capacity(base.len());
    for (base_prob, topk_prob) in base.iter().zip(topk.iter()) {
        out.push((1.0 - gate) * *base_prob + gate * *topk_prob);
    }
    out
}

fn push_recent(history: &mut VecDeque<usize>, token: usize, recent_window: usize) {
    history.push_back(token);
    while history.len() > recent_window {
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
