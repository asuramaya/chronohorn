use std::path::Path;

use crate::data::take_val_tokens;
use crate::protocol::{Runner, SampleOutputs};
use crate::token_match_bridge::{TokenMatchBridgeRunner, train_token_match_bridge_from_data_root};
use crate::token_skip_bridge::{TokenSkipBridgeRunner, train_token_skip_bridge_from_data_root};

const FEATURE_DIM: usize = 7;

#[derive(Debug, Clone)]
pub struct TokenMatchSkipBridgeReport {
    pub train_token_budget: usize,
    pub trigram_buckets: usize,
    pub skip_buckets: usize,
    pub val_token_budget: usize,
    pub match_depth: usize,
    pub candidate_k: usize,
    pub train_stride: usize,
    pub tune_records: usize,
    pub eval_records: usize,
    pub tuned_heuristic_lambda: f64,
    pub tuned_direct_lambda: f64,
    pub selected_runtime_gate: String,
    pub selected_runtime_lambda: f64,
    pub tune_bpt_match: f64,
    pub tune_bpt_skip: f64,
    pub tune_bpt_heuristic: f64,
    pub tune_bpt_direct: f64,
    pub tune_bpt_oracle: f64,
    pub eval_bpt_match: f64,
    pub eval_bpt_skip: f64,
    pub eval_bpt_heuristic: f64,
    pub eval_bpt_direct: f64,
    pub eval_bpt_oracle: f64,
    pub eval_match_better_rate: f64,
    pub eval_skip_better_rate: f64,
    pub eval_top1_agreement_rate: f64,
}

#[derive(Debug, Clone)]
pub struct TrainedTokenMatchSkipBridge {
    report: TokenMatchSkipBridgeReport,
    runner: TokenMatchSkipBridgeRunner,
    eval_tokens: Vec<usize>,
}

impl TrainedTokenMatchSkipBridge {
    pub fn report(&self) -> &TokenMatchSkipBridgeReport {
        &self.report
    }

    pub fn runner(&self) -> &TokenMatchSkipBridgeRunner {
        &self.runner
    }

    pub fn eval_tokens(&self) -> &[usize] {
        &self.eval_tokens
    }
}

#[derive(Debug, Clone)]
pub struct TokenMatchSkipBridgeRunner {
    match_runner: TokenMatchBridgeRunner,
    skip_runner: TokenSkipBridgeRunner,
    standardizer: Standardizer,
    gate: GateChoice,
    lambda: f64,
    candidate_k: usize,
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
struct RawRecord {
    features: [f64; FEATURE_DIM],
    match_gold_prob: f64,
    skip_gold_prob: f64,
    heuristic_gate: f64,
    top1_agreement: f64,
}

#[derive(Debug, Clone)]
struct Record {
    features: [f64; FEATURE_DIM],
    match_gold_prob: f64,
    skip_gold_prob: f64,
    heuristic_gate: f64,
    top1_agreement: f64,
}

#[derive(Debug, Clone, Copy)]
struct DistStats {
    top1_prob: f64,
    top2_prob: f64,
    top1_token: usize,
    entropy_norm: f64,
    topk_mass: f64,
}

pub fn train_token_matchskip_bridge_from_data_root(
    root: &Path,
    train_token_budget: usize,
    trigram_buckets: usize,
    skip_buckets: usize,
    val_token_budget: usize,
    match_depth: usize,
    candidate_k: usize,
    train_stride: usize,
    alpha_bigram: f64,
    alpha_trigram: f64,
    alpha_skip: f64,
) -> Result<TrainedTokenMatchSkipBridge, String> {
    if train_stride == 0 {
        return Err("train_stride must be positive".to_string());
    }
    if match_depth == 0 {
        return Err("match_depth must be positive".to_string());
    }
    if candidate_k == 0 {
        return Err("candidate_k must be positive".to_string());
    }

    let trained_match = train_token_match_bridge_from_data_root(
        root,
        train_token_budget,
        trigram_buckets,
        val_token_budget,
        match_depth,
        candidate_k,
        train_stride,
        alpha_bigram,
        alpha_trigram,
    )?;
    let trained_skip = train_token_skip_bridge_from_data_root(
        root,
        train_token_budget,
        trigram_buckets,
        skip_buckets,
        val_token_budget,
        alpha_bigram,
        alpha_trigram,
        alpha_skip,
        train_stride,
        candidate_k,
    )?;

    if trained_match.runner().vocab_size() != trained_skip.runner().vocab_size() {
        return Err("match/skip vocab size mismatch".to_string());
    }

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

    let tune_raw = collect_raw_records(
        trained_match.runner(),
        trained_skip.runner(),
        &tune_tokens,
        candidate_k,
    )?;
    let eval_raw = collect_raw_records(
        trained_match.runner(),
        trained_skip.runner(),
        &eval_tokens,
        candidate_k,
    )?;
    if tune_raw.is_empty() || eval_raw.is_empty() {
        return Err("no matchskip records collected".to_string());
    }

    let standardizer = fit_standardizer_from_raw(&tune_raw);
    let tune_records = standardize_records(&tune_raw, &standardizer);
    let eval_records = standardize_records(&eval_raw, &standardizer);
    let direct_model = train_direct_gate(&tune_records, 80, 0.1, 1e-4);

    let tuned_heuristic_lambda = tune_lambda(&tune_records, |record| record.heuristic_gate);
    let tuned_direct_lambda = tune_lambda(&tune_records, |record| {
        predict_probability(&direct_model, &record.features)
    });

    let tune_bpt_match = mean_bits_for_records(&tune_records, None, Branch::Match);
    let tune_bpt_skip = mean_bits_for_records(&tune_records, None, Branch::Skip);
    let tune_bpt_heuristic = mean_bits_for_records(
        &tune_records,
        Some((tuned_heuristic_lambda, GateKind::Heuristic, None)),
        Branch::Combined,
    );
    let tune_bpt_direct = mean_bits_for_records(
        &tune_records,
        Some((tuned_direct_lambda, GateKind::Direct, Some(&direct_model))),
        Branch::Combined,
    );
    let tune_bpt_oracle = oracle_bits_per_token(&tune_records);

    let eval_bpt_match = mean_bits_for_records(&eval_records, None, Branch::Match);
    let eval_bpt_skip = mean_bits_for_records(&eval_records, None, Branch::Skip);
    let eval_bpt_heuristic = mean_bits_for_records(
        &eval_records,
        Some((tuned_heuristic_lambda, GateKind::Heuristic, None)),
        Branch::Combined,
    );
    let eval_bpt_direct = mean_bits_for_records(
        &eval_records,
        Some((tuned_direct_lambda, GateKind::Direct, Some(&direct_model))),
        Branch::Combined,
    );
    let eval_bpt_oracle = oracle_bits_per_token(&eval_records);

    let (selected_runtime_gate, gate, selected_runtime_lambda) =
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

    let report = TokenMatchSkipBridgeReport {
        train_token_budget,
        trigram_buckets,
        skip_buckets,
        val_token_budget,
        match_depth,
        candidate_k,
        train_stride,
        tune_records: tune_records.len(),
        eval_records: eval_records.len(),
        tuned_heuristic_lambda,
        tuned_direct_lambda,
        selected_runtime_gate,
        selected_runtime_lambda,
        tune_bpt_match,
        tune_bpt_skip,
        tune_bpt_heuristic,
        tune_bpt_direct,
        tune_bpt_oracle,
        eval_bpt_match,
        eval_bpt_skip,
        eval_bpt_heuristic,
        eval_bpt_direct,
        eval_bpt_oracle,
        eval_match_better_rate: branch_better_rate(&eval_records, Branch::Match),
        eval_skip_better_rate: branch_better_rate(&eval_records, Branch::Skip),
        eval_top1_agreement_rate: agreement_rate(&eval_records),
    };

    let runner = TokenMatchSkipBridgeRunner {
        match_runner: trained_match.runner().clone(),
        skip_runner: trained_skip.runner().clone(),
        standardizer,
        gate,
        lambda: selected_runtime_lambda,
        candidate_k,
    };

    Ok(TrainedTokenMatchSkipBridge {
        report,
        runner,
        eval_tokens,
    })
}

pub fn run_token_matchskip_bridge_from_data_root(
    root: &Path,
    train_token_budget: usize,
    trigram_buckets: usize,
    skip_buckets: usize,
    val_token_budget: usize,
    match_depth: usize,
    candidate_k: usize,
    train_stride: usize,
    alpha_bigram: f64,
    alpha_trigram: f64,
    alpha_skip: f64,
) -> Result<TokenMatchSkipBridgeReport, String> {
    Ok(train_token_matchskip_bridge_from_data_root(
        root,
        train_token_budget,
        trigram_buckets,
        skip_buckets,
        val_token_budget,
        match_depth,
        candidate_k,
        train_stride,
        alpha_bigram,
        alpha_trigram,
        alpha_skip,
    )?
    .report)
}

pub fn render_token_matchskip_bridge_report(report: &TokenMatchSkipBridgeReport) -> String {
    let mut out = String::new();
    out.push_str("chronohorn_token_matchskip_bridge\n");
    out.push_str(&format!(
        "train_token_budget: {}\n",
        report.train_token_budget
    ));
    out.push_str(&format!("trigram_buckets: {}\n", report.trigram_buckets));
    out.push_str(&format!("skip_buckets: {}\n", report.skip_buckets));
    out.push_str(&format!("val_token_budget: {}\n", report.val_token_budget));
    out.push_str(&format!("match_depth: {}\n", report.match_depth));
    out.push_str(&format!("candidate_k: {}\n", report.candidate_k));
    out.push_str(&format!("train_stride: {}\n", report.train_stride));
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
    out.push_str(&format!("tune_bpt_match: {:.6}\n", report.tune_bpt_match));
    out.push_str(&format!("tune_bpt_skip: {:.6}\n", report.tune_bpt_skip));
    out.push_str(&format!(
        "tune_bpt_heuristic: {:.6}\n",
        report.tune_bpt_heuristic
    ));
    out.push_str(&format!("tune_bpt_direct: {:.6}\n", report.tune_bpt_direct));
    out.push_str(&format!("tune_bpt_oracle: {:.6}\n", report.tune_bpt_oracle));
    out.push_str(&format!("eval_bpt_match: {:.6}\n", report.eval_bpt_match));
    out.push_str(&format!("eval_bpt_skip: {:.6}\n", report.eval_bpt_skip));
    out.push_str(&format!(
        "eval_bpt_heuristic: {:.6}\n",
        report.eval_bpt_heuristic
    ));
    out.push_str(&format!("eval_bpt_direct: {:.6}\n", report.eval_bpt_direct));
    out.push_str(&format!("eval_bpt_oracle: {:.6}\n", report.eval_bpt_oracle));
    out.push_str(&format!(
        "eval_match_better_rate: {:.6}\n",
        report.eval_match_better_rate
    ));
    out.push_str(&format!(
        "eval_skip_better_rate: {:.6}\n",
        report.eval_skip_better_rate
    ));
    out.push_str(&format!(
        "eval_top1_agreement_rate: {:.6}\n",
        report.eval_top1_agreement_rate
    ));
    out
}

impl Runner for TokenMatchSkipBridgeRunner {
    fn name(&self) -> &'static str {
        "TokenMatchSkipBridgeRunner"
    }

    fn vocab_size(&self) -> usize {
        self.match_runner.vocab_size()
    }

    fn score_chunk(
        &self,
        tokens: &[usize],
        sample_positions: &[usize],
    ) -> Result<SampleOutputs, String> {
        let match_outputs = self.match_runner.score_chunk(tokens, sample_positions)?;
        let skip_outputs = self.skip_runner.score_chunk(tokens, sample_positions)?;
        if match_outputs.sample_predictions.len() != skip_outputs.sample_predictions.len() {
            return Err("match/skip sample prediction length mismatch".to_string());
        }

        let mut predictions = Vec::with_capacity(match_outputs.sample_predictions.len());
        let mut golds = Vec::with_capacity(match_outputs.sample_predictions.len());
        for (index, (match_dist, skip_dist)) in match_outputs
            .sample_predictions
            .iter()
            .zip(skip_outputs.sample_predictions.iter())
            .enumerate()
        {
            let match_stats = summarize_distribution(match_dist, self.candidate_k);
            let skip_stats = summarize_distribution(skip_dist, self.candidate_k);
            let features = standardize_features(
                extract_features(&match_stats, &skip_stats),
                &self.standardizer,
            );
            let raw_gate = match &self.gate {
                GateChoice::Heuristic => heuristic_gate_from_stats(&match_stats, &skip_stats),
                GateChoice::Direct(model) => predict_probability(model, &features),
            };
            let gate = (self.lambda * raw_gate).clamp(0.0, 1.0);
            let mixed = mix_pair(skip_dist, match_dist, gate);
            let gold = mixed
                .get(tokens[sample_positions[index]])
                .copied()
                .unwrap_or(f64::MIN_POSITIVE)
                .max(f64::MIN_POSITIVE)
                .ln();
            golds.push(gold);
            predictions.push(mixed);
        }

        Ok(SampleOutputs {
            sample_predictions: predictions,
            sample_gold_logprobs: Some(golds),
        })
    }

    fn adapt_chunk(&mut self, tokens: &[usize]) -> Result<(), String> {
        self.match_runner.adapt_chunk(tokens)?;
        self.skip_runner.adapt_chunk(tokens)?;
        Ok(())
    }
}

fn collect_raw_records(
    match_runner: &TokenMatchBridgeRunner,
    skip_runner: &TokenSkipBridgeRunner,
    tokens: &[usize],
    candidate_k: usize,
) -> Result<Vec<RawRecord>, String> {
    let sample_positions: Vec<usize> = (0..tokens.len()).collect();
    let match_outputs = match_runner.score_chunk(tokens, &sample_positions)?;
    let skip_outputs = skip_runner.score_chunk(tokens, &sample_positions)?;
    let match_golds = match_outputs
        .sample_gold_logprobs
        .ok_or_else(|| "match runner did not return gold logprobs".to_string())?;
    let skip_golds = skip_outputs
        .sample_gold_logprobs
        .ok_or_else(|| "skip runner did not return gold logprobs".to_string())?;

    let mut rows = Vec::with_capacity(tokens.len());
    for pos in 0..tokens.len() {
        let match_dist = &match_outputs.sample_predictions[pos];
        let skip_dist = &skip_outputs.sample_predictions[pos];
        let match_stats = summarize_distribution(match_dist, candidate_k);
        let skip_stats = summarize_distribution(skip_dist, candidate_k);
        rows.push(RawRecord {
            features: extract_features(&match_stats, &skip_stats),
            match_gold_prob: match_golds[pos].exp().max(1e-12),
            skip_gold_prob: skip_golds[pos].exp().max(1e-12),
            heuristic_gate: heuristic_gate_from_stats(&match_stats, &skip_stats),
            top1_agreement: if match_stats.top1_token == skip_stats.top1_token {
                1.0
            } else {
                0.0
            },
        });
    }
    Ok(rows)
}

fn fit_standardizer_from_raw(records: &[RawRecord]) -> Standardizer {
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

fn standardize_records(records: &[RawRecord], standardizer: &Standardizer) -> Vec<Record> {
    records
        .iter()
        .map(|record| Record {
            features: standardize_features(record.features, standardizer),
            match_gold_prob: record.match_gold_prob,
            skip_gold_prob: record.skip_gold_prob,
            heuristic_gate: record.heuristic_gate,
            top1_agreement: record.top1_agreement,
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

fn summarize_distribution(distribution: &[f64], candidate_k: usize) -> DistStats {
    let k = candidate_k.max(1);
    let mut top1_prob = 0.0;
    let mut top2_prob = 0.0;
    let mut top1_token = 0usize;
    let mut topk = vec![0.0; k];
    for (index, &value) in distribution.iter().enumerate() {
        if value >= top1_prob {
            top2_prob = top1_prob;
            top1_prob = value;
            top1_token = index;
        } else if value > top2_prob {
            top2_prob = value;
        }
        update_topk_dynamic(&mut topk, value);
    }
    let support = topk.iter().sum::<f64>();
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
    DistStats {
        top1_prob,
        top2_prob,
        top1_token,
        entropy_norm,
        topk_mass: support,
    }
}

fn update_topk_dynamic(topk: &mut [f64], value: f64) {
    if topk.is_empty() || value <= topk[topk.len() - 1] {
        return;
    }
    let mut insert_at = topk.len() - 1;
    while insert_at > 0 && value > topk[insert_at - 1] {
        insert_at -= 1;
    }
    for index in (insert_at + 1..topk.len()).rev() {
        topk[index] = topk[index - 1];
    }
    topk[insert_at] = value;
}

fn extract_features(match_stats: &DistStats, skip_stats: &DistStats) -> [f64; FEATURE_DIM] {
    [
        match_stats.top1_prob,
        match_stats.top1_prob - match_stats.top2_prob,
        match_stats.topk_mass,
        skip_stats.top1_prob,
        skip_stats.top1_prob - skip_stats.top2_prob,
        skip_stats.topk_mass,
        if match_stats.top1_token == skip_stats.top1_token {
            1.0
        } else {
            0.0
        } - (match_stats.entropy_norm - skip_stats.entropy_norm),
    ]
}

fn heuristic_gate_from_stats(match_stats: &DistStats, skip_stats: &DistStats) -> f64 {
    let support_adv = (match_stats.top1_prob + match_stats.topk_mass)
        - (skip_stats.top1_prob + skip_stats.topk_mass);
    let entropy_adv = skip_stats.entropy_norm - match_stats.entropy_norm;
    let agreement = if match_stats.top1_token == skip_stats.top1_token {
        1.0
    } else {
        0.0
    };
    (0.5 + 0.5 * (support_adv + 0.5 * entropy_adv + 0.25 * agreement)).clamp(0.0, 1.0)
}

fn train_direct_gate(records: &[Record], epochs: usize, lr: f64, l2: f64) -> LogisticModel {
    let mut model = LogisticModel {
        weights: [0.0; FEATURE_DIM],
        bias: 0.0,
    };
    let ln2 = std::f64::consts::LN_2;
    for _ in 0..epochs {
        let mut grad_w = [0.0; FEATURE_DIM];
        let mut grad_b = 0.0;
        for record in records {
            let prediction = predict_probability(&model, &record.features).clamp(1e-6, 1.0 - 1e-6);
            let mixed = ((1.0 - prediction) * record.skip_gold_prob
                + prediction * record.match_gold_prob)
                .max(1e-12);
            let dloss_dgate = (record.skip_gold_prob - record.match_gold_prob) / (mixed * ln2);
            let dloss_dz = dloss_dgate * prediction * (1.0 - prediction);
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

#[derive(Clone, Copy)]
enum GateKind {
    Heuristic,
    Direct,
}

#[derive(Clone, Copy)]
enum Branch {
    Match,
    Skip,
    Combined,
}

fn tune_lambda<F>(records: &[Record], mut gate_fn: F) -> f64
where
    F: FnMut(&Record) -> f64,
{
    let mut best_lambda = 0.0;
    let mut best_bits = mean_bits_with_gate(records, 0.0, &mut gate_fn);
    for step in 1..=12 {
        let lambda = step as f64 / 10.0;
        let bits = mean_bits_with_gate(records, lambda, &mut gate_fn);
        if bits < best_bits {
            best_bits = bits;
            best_lambda = lambda;
        }
    }
    best_lambda
}

fn mean_bits_for_records(
    records: &[Record],
    gate: Option<(f64, GateKind, Option<&LogisticModel>)>,
    branch: Branch,
) -> f64 {
    records
        .iter()
        .map(|record| {
            let prob = match branch {
                Branch::Match => record.match_gold_prob,
                Branch::Skip => record.skip_gold_prob,
                Branch::Combined => {
                    let (lambda, gate_kind, model) = gate.expect("combined branch requires gate");
                    let raw = match gate_kind {
                        GateKind::Heuristic => record.heuristic_gate,
                        GateKind::Direct => predict_probability(
                            model.expect("missing direct model"),
                            &record.features,
                        ),
                    };
                    let g = (lambda * raw).clamp(0.0, 1.0);
                    ((1.0 - g) * record.skip_gold_prob + g * record.match_gold_prob).max(1e-12)
                }
            };
            -prob.max(1e-12).log2()
        })
        .sum::<f64>()
        / records.len() as f64
}

fn mean_bits_with_gate<F>(records: &[Record], lambda: f64, gate_fn: &mut F) -> f64
where
    F: FnMut(&Record) -> f64,
{
    records
        .iter()
        .map(|record| {
            let gate = (lambda * gate_fn(record)).clamp(0.0, 1.0);
            let prob =
                ((1.0 - gate) * record.skip_gold_prob + gate * record.match_gold_prob).max(1e-12);
            -prob.log2()
        })
        .sum::<f64>()
        / records.len() as f64
}

fn oracle_bits_per_token(records: &[Record]) -> f64 {
    records
        .iter()
        .map(|record| {
            -record
                .match_gold_prob
                .max(record.skip_gold_prob)
                .max(1e-12)
                .log2()
        })
        .sum::<f64>()
        / records.len() as f64
}

fn branch_better_rate(records: &[Record], branch: Branch) -> f64 {
    records
        .iter()
        .filter(|record| match branch {
            Branch::Match => record.match_gold_prob > record.skip_gold_prob,
            Branch::Skip => record.skip_gold_prob > record.match_gold_prob,
            Branch::Combined => false,
        })
        .count() as f64
        / records.len() as f64
}

fn agreement_rate(records: &[Record]) -> f64 {
    records
        .iter()
        .filter(|record| record.top1_agreement > 0.5)
        .count() as f64
        / records.len() as f64
}

fn mix_pair(skip: &[f64], match_dist: &[f64], gate: f64) -> Vec<f64> {
    let mut mixed = Vec::with_capacity(skip.len());
    let mut total = 0.0;
    for (&skip_p, &match_p) in skip.iter().zip(match_dist.iter()) {
        let value = (1.0 - gate) * skip_p + gate * match_p;
        mixed.push(value);
        total += value;
    }
    if total > 0.0 {
        for value in &mut mixed {
            *value /= total;
        }
    }
    mixed
}
