use std::path::Path;

use crate::data::take_val_tokens;
use crate::protocol::{Runner, SampleOutputs};
use crate::token_copy_bridge::{TokenCopyBridgeRunner, train_token_copy_bridge_from_data_root};
use crate::token_skip_bridge::{TokenSkipBridgeRunner, train_token_skip_bridge_from_data_root};

const FEATURE_DIM: usize = 7;

#[derive(Debug, Clone)]
pub struct TokenSkipCopyBridgeReport {
    pub train_token_budget: usize,
    pub trigram_buckets: usize,
    pub skip_buckets: usize,
    pub val_token_budget: usize,
    pub copy_window: usize,
    pub candidate_k: usize,
    pub copy_decay: f64,
    pub train_stride: usize,
    pub tune_records: usize,
    pub eval_records: usize,
    pub tuned_heuristic_lambda: f64,
    pub tuned_direct_lambda: f64,
    pub selected_runtime_gate: String,
    pub selected_runtime_lambda: f64,
    pub tune_bpt_skip: f64,
    pub tune_bpt_copy: f64,
    pub tune_bpt_heuristic: f64,
    pub tune_bpt_direct: f64,
    pub tune_bpt_oracle: f64,
    pub eval_bpt_skip: f64,
    pub eval_bpt_copy: f64,
    pub eval_bpt_heuristic: f64,
    pub eval_bpt_direct: f64,
    pub eval_bpt_oracle: f64,
    pub eval_skip_better_rate: f64,
    pub eval_copy_better_rate: f64,
    pub eval_agreement_rate: f64,
}

#[derive(Debug, Clone)]
pub struct TrainedTokenSkipCopyBridge {
    report: TokenSkipCopyBridgeReport,
    runner: TokenSkipCopyBridgeRunner,
    eval_tokens: Vec<usize>,
}

impl TrainedTokenSkipCopyBridge {
    pub fn report(&self) -> &TokenSkipCopyBridgeReport {
        &self.report
    }

    pub fn runner(&self) -> &TokenSkipCopyBridgeRunner {
        &self.runner
    }

    pub fn eval_tokens(&self) -> &[usize] {
        &self.eval_tokens
    }
}

#[derive(Debug, Clone)]
pub struct TokenSkipCopyBridgeRunner {
    skip_runner: TokenSkipBridgeRunner,
    copy_runner: TokenCopyBridgeRunner,
    standardizer: Standardizer,
    gate: GateChoice,
    lambda: f64,
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
    skip_gold_prob: f64,
    copy_gold_prob: f64,
    heuristic_gate: f64,
}

#[derive(Debug, Clone)]
struct Record {
    features: [f64; FEATURE_DIM],
    skip_gold_prob: f64,
    copy_gold_prob: f64,
    heuristic_gate: f64,
}

#[derive(Debug, Clone, Copy)]
struct DistStats {
    top1_prob: f64,
    top2_prob: f64,
    top1_token: usize,
    entropy_norm: f64,
    topk_mass: f64,
}

pub fn train_token_skipcopy_bridge_from_data_root(
    root: &Path,
    train_token_budget: usize,
    trigram_buckets: usize,
    skip_buckets: usize,
    val_token_budget: usize,
    copy_window: usize,
    candidate_k: usize,
    train_stride: usize,
    alpha_bigram: f64,
    alpha_trigram: f64,
    alpha_skip: f64,
    copy_decay: f64,
) -> Result<TrainedTokenSkipCopyBridge, String> {
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
    let trained_copy = train_token_copy_bridge_from_data_root(
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
    )?;
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

    let tune_raw = collect_raw_records(trained_skip.runner(), trained_copy.runner(), &tune_tokens)?;
    let eval_raw = collect_raw_records(trained_skip.runner(), trained_copy.runner(), &eval_tokens)?;
    if tune_raw.is_empty() || eval_raw.is_empty() {
        return Err("no skipcopy records collected".to_string());
    }
    let standardizer = fit_standardizer_from_raw(&tune_raw);
    let tune_records = standardize_records(&tune_raw, &standardizer);
    let eval_records = standardize_records(&eval_raw, &standardizer);
    let direct_model = train_direct_gate(&tune_records, 80, 0.1, 1e-4);

    let tuned_heuristic_lambda = tune_lambda(&tune_records, |record| record.heuristic_gate);
    let tuned_direct_lambda = tune_lambda(&tune_records, |record| {
        predict_probability(&direct_model, &record.features)
    });

    let tune_bpt_skip = mean_bits_for_records(&tune_records, None, Branch::Skip);
    let tune_bpt_copy = mean_bits_for_records(&tune_records, None, Branch::Copy);
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

    let eval_bpt_skip = mean_bits_for_records(&eval_records, None, Branch::Skip);
    let eval_bpt_copy = mean_bits_for_records(&eval_records, None, Branch::Copy);
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

    let report = TokenSkipCopyBridgeReport {
        train_token_budget,
        trigram_buckets,
        skip_buckets,
        val_token_budget,
        copy_window,
        candidate_k,
        copy_decay,
        train_stride,
        tune_records: tune_records.len(),
        eval_records: eval_records.len(),
        tuned_heuristic_lambda,
        tuned_direct_lambda,
        selected_runtime_gate,
        selected_runtime_lambda,
        tune_bpt_skip,
        tune_bpt_copy,
        tune_bpt_heuristic,
        tune_bpt_direct,
        tune_bpt_oracle,
        eval_bpt_skip,
        eval_bpt_copy,
        eval_bpt_heuristic,
        eval_bpt_direct,
        eval_bpt_oracle,
        eval_skip_better_rate: branch_better_rate(&eval_records, Branch::Skip),
        eval_copy_better_rate: branch_better_rate(&eval_records, Branch::Copy),
        eval_agreement_rate: agreement_rate(&eval_records),
    };

    let runner = TokenSkipCopyBridgeRunner {
        skip_runner: trained_skip.runner().clone(),
        copy_runner: trained_copy.runner().clone(),
        standardizer,
        gate,
        lambda: selected_runtime_lambda,
    };

    Ok(TrainedTokenSkipCopyBridge {
        report,
        runner,
        eval_tokens,
    })
}

pub fn run_token_skipcopy_bridge_from_data_root(
    root: &Path,
    train_token_budget: usize,
    trigram_buckets: usize,
    skip_buckets: usize,
    val_token_budget: usize,
    copy_window: usize,
    candidate_k: usize,
    train_stride: usize,
    alpha_bigram: f64,
    alpha_trigram: f64,
    alpha_skip: f64,
    copy_decay: f64,
) -> Result<TokenSkipCopyBridgeReport, String> {
    Ok(train_token_skipcopy_bridge_from_data_root(
        root,
        train_token_budget,
        trigram_buckets,
        skip_buckets,
        val_token_budget,
        copy_window,
        candidate_k,
        train_stride,
        alpha_bigram,
        alpha_trigram,
        alpha_skip,
        copy_decay,
    )?
    .report)
}

pub fn render_token_skipcopy_bridge_report(report: &TokenSkipCopyBridgeReport) -> String {
    let mut out = String::new();
    out.push_str("chronohorn_token_skipcopy_bridge\n");
    out.push_str(&format!(
        "train_token_budget: {}\n",
        report.train_token_budget
    ));
    out.push_str(&format!("trigram_buckets: {}\n", report.trigram_buckets));
    out.push_str(&format!("skip_buckets: {}\n", report.skip_buckets));
    out.push_str(&format!("val_token_budget: {}\n", report.val_token_budget));
    out.push_str(&format!("copy_window: {}\n", report.copy_window));
    out.push_str(&format!("candidate_k: {}\n", report.candidate_k));
    out.push_str(&format!("copy_decay: {:.6}\n", report.copy_decay));
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
    out.push_str(&format!("tune_bpt_skip: {:.6}\n", report.tune_bpt_skip));
    out.push_str(&format!("tune_bpt_copy: {:.6}\n", report.tune_bpt_copy));
    out.push_str(&format!(
        "tune_bpt_heuristic: {:.6}\n",
        report.tune_bpt_heuristic
    ));
    out.push_str(&format!("tune_bpt_direct: {:.6}\n", report.tune_bpt_direct));
    out.push_str(&format!("tune_bpt_oracle: {:.6}\n", report.tune_bpt_oracle));
    out.push_str(&format!("eval_bpt_skip: {:.6}\n", report.eval_bpt_skip));
    out.push_str(&format!("eval_bpt_copy: {:.6}\n", report.eval_bpt_copy));
    out.push_str(&format!(
        "eval_bpt_heuristic: {:.6}\n",
        report.eval_bpt_heuristic
    ));
    out.push_str(&format!("eval_bpt_direct: {:.6}\n", report.eval_bpt_direct));
    out.push_str(&format!("eval_bpt_oracle: {:.6}\n", report.eval_bpt_oracle));
    out.push_str(&format!(
        "eval_skip_better_rate: {:.6}\n",
        report.eval_skip_better_rate
    ));
    out.push_str(&format!(
        "eval_copy_better_rate: {:.6}\n",
        report.eval_copy_better_rate
    ));
    out.push_str(&format!(
        "eval_agreement_rate: {:.6}\n",
        report.eval_agreement_rate
    ));
    out
}

impl Runner for TokenSkipCopyBridgeRunner {
    fn name(&self) -> &'static str {
        "TokenSkipCopyBridgeRunner"
    }

    fn vocab_size(&self) -> usize {
        self.skip_runner.vocab_size()
    }

    fn score_chunk(
        &self,
        tokens: &[usize],
        sample_positions: &[usize],
    ) -> Result<SampleOutputs, String> {
        let skip_outputs = self.skip_runner.score_chunk(tokens, sample_positions)?;
        let copy_outputs = self.copy_runner.score_chunk(tokens, sample_positions)?;
        if skip_outputs.sample_predictions.len() != copy_outputs.sample_predictions.len() {
            return Err("skip/copy sample prediction length mismatch".to_string());
        }
        let mut predictions = Vec::with_capacity(skip_outputs.sample_predictions.len());
        let mut golds = Vec::with_capacity(skip_outputs.sample_predictions.len());
        for (index, (skip_dist, copy_dist)) in skip_outputs
            .sample_predictions
            .iter()
            .zip(copy_outputs.sample_predictions.iter())
            .enumerate()
        {
            let features = standardize_features(
                extract_features(
                    &summarize_distribution(skip_dist),
                    &summarize_distribution(copy_dist),
                ),
                &self.standardizer,
            );
            let raw_gate = match &self.gate {
                GateChoice::Heuristic => heuristic_gate_from_stats(
                    &summarize_distribution(skip_dist),
                    &summarize_distribution(copy_dist),
                ),
                GateChoice::Direct(model) => predict_probability(model, &features),
            };
            let gate = (self.lambda * raw_gate).clamp(0.0, 1.0);
            let mixed = mix_pair(skip_dist, copy_dist, gate);
            predictions.push(mixed.clone());
            let gold = mixed
                .get(tokens[sample_positions[index]])
                .copied()
                .unwrap_or(f64::MIN_POSITIVE)
                .max(f64::MIN_POSITIVE)
                .ln();
            golds.push(gold);
        }
        Ok(SampleOutputs {
            sample_predictions: predictions,
            sample_gold_logprobs: Some(golds),
        })
    }

    fn adapt_chunk(&mut self, tokens: &[usize]) -> Result<(), String> {
        self.skip_runner.adapt_chunk(tokens)?;
        self.copy_runner.adapt_chunk(tokens)?;
        Ok(())
    }
}

fn collect_raw_records(
    skip_runner: &TokenSkipBridgeRunner,
    copy_runner: &TokenCopyBridgeRunner,
    tokens: &[usize],
) -> Result<Vec<RawRecord>, String> {
    let sample_positions: Vec<usize> = (0..tokens.len()).collect();
    let skip_outputs = skip_runner.score_chunk(tokens, &sample_positions)?;
    let copy_outputs = copy_runner.score_chunk(tokens, &sample_positions)?;
    let skip_golds = skip_outputs
        .sample_gold_logprobs
        .ok_or_else(|| "skip runner did not return gold logprobs".to_string())?;
    let copy_golds = copy_outputs
        .sample_gold_logprobs
        .ok_or_else(|| "copy runner did not return gold logprobs".to_string())?;
    let mut rows = Vec::with_capacity(tokens.len());
    for pos in 0..tokens.len() {
        let skip_dist = &skip_outputs.sample_predictions[pos];
        let copy_dist = &copy_outputs.sample_predictions[pos];
        let skip_stats = summarize_distribution(skip_dist);
        let copy_stats = summarize_distribution(copy_dist);
        rows.push(RawRecord {
            features: extract_features(&skip_stats, &copy_stats),
            skip_gold_prob: skip_golds[pos].exp().max(1e-12),
            copy_gold_prob: copy_golds[pos].exp().max(1e-12),
            heuristic_gate: heuristic_gate_from_stats(&skip_stats, &copy_stats),
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
            skip_gold_prob: record.skip_gold_prob,
            copy_gold_prob: record.copy_gold_prob,
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

fn summarize_distribution(distribution: &[f64]) -> DistStats {
    let mut top1_prob = 0.0;
    let mut top2_prob = 0.0;
    let mut top1_token = 0usize;
    let mut topk = Vec::with_capacity(4);
    for (index, &value) in distribution.iter().enumerate() {
        if value >= top1_prob {
            top2_prob = top1_prob;
            top1_prob = value;
            top1_token = index;
        } else if value > top2_prob {
            top2_prob = value;
        }
        topk.push(value);
    }
    topk.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let topk_mass = topk.into_iter().take(4).sum::<f64>();
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
        topk_mass,
    }
}

fn extract_features(skip: &DistStats, copy: &DistStats) -> [f64; FEATURE_DIM] {
    [
        skip.top1_prob,
        skip.top1_prob - skip.top2_prob,
        skip.topk_mass,
        copy.top1_prob,
        copy.top1_prob - copy.top2_prob,
        copy.topk_mass,
        if skip.top1_token == copy.top1_token {
            1.0
        } else {
            0.0
        } - (skip.entropy_norm - copy.entropy_norm),
    ]
}

fn heuristic_gate_from_stats(skip: &DistStats, copy: &DistStats) -> f64 {
    let copy_adv = (copy.top1_prob + copy.topk_mass) - (skip.top1_prob + skip.topk_mass);
    (0.5 + 0.5 * copy_adv).clamp(0.0, 1.0)
}

fn train_direct_gate(records: &[Record], epochs: usize, lr: f64, l2: f64) -> LogisticModel {
    let mut model = LogisticModel {
        weights: [0.0; FEATURE_DIM],
        bias: 0.0,
    };
    for _ in 0..epochs {
        for record in records {
            let target = if record.copy_gold_prob > record.skip_gold_prob {
                1.0
            } else {
                0.0
            };
            let prediction = predict_probability(&model, &record.features);
            let error = prediction - target;
            for index in 0..FEATURE_DIM {
                model.weights[index] -=
                    lr * (error * record.features[index] + l2 * model.weights[index]);
            }
            model.bias -= lr * error;
        }
    }
    model
}

fn predict_probability(model: &LogisticModel, features: &[f64; FEATURE_DIM]) -> f64 {
    let mut value = model.bias;
    for index in 0..FEATURE_DIM {
        value += model.weights[index] * features[index];
    }
    1.0 / (1.0 + (-value).exp())
}

#[derive(Clone, Copy)]
enum GateKind {
    Heuristic,
    Direct,
}

#[derive(Clone, Copy)]
enum Branch {
    Skip,
    Copy,
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
                Branch::Skip => record.skip_gold_prob,
                Branch::Copy => record.copy_gold_prob,
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
                    ((1.0 - g) * record.skip_gold_prob + g * record.copy_gold_prob).max(1e-12)
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
                ((1.0 - gate) * record.skip_gold_prob + gate * record.copy_gold_prob).max(1e-12);
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
                .skip_gold_prob
                .max(record.copy_gold_prob)
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
            Branch::Skip => record.skip_gold_prob > record.copy_gold_prob,
            Branch::Copy => record.copy_gold_prob > record.skip_gold_prob,
            Branch::Combined => false,
        })
        .count() as f64
        / records.len() as f64
}

fn agreement_rate(records: &[Record]) -> f64 {
    records
        .iter()
        .filter(|record| record.heuristic_gate <= 1e-9 || record.heuristic_gate >= 1.0 - 1e-9)
        .count() as f64
        / records.len() as f64
}

fn mix_pair(skip: &[f64], copy: &[f64], gate: f64) -> Vec<f64> {
    let mut mixed = Vec::with_capacity(skip.len());
    let mut total = 0.0;
    for (&s, &c) in skip.iter().zip(copy.iter()) {
        let value = (1.0 - gate) * s + gate * c;
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
