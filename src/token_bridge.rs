use std::path::Path;

use crate::data::{take_train_tokens, take_val_tokens};
use crate::packed_memory::{PackedTables, build_packed_tables};
use crate::protocol::{Runner, SampleOutputs};

const FEATURE_DIM: usize = 6;
const TRIGRAM_HASH_MUL_A: u64 = 1_315_423_911;
const TRIGRAM_HASH_MUL_B: u64 = 2_654_435_761;

#[derive(Debug, Clone)]
pub struct TokenBridgeReport {
    pub train_token_budget: usize,
    pub trigram_buckets: usize,
    pub val_token_budget: usize,
    pub train_stride: usize,
    pub candidate_k: usize,
    pub train_records: usize,
    pub tune_tokens: usize,
    pub eval_tokens: usize,
    pub tuned_heuristic_lambda: f64,
    pub tuned_direct_lambda: f64,
    pub tune_bpt_base: f64,
    pub tune_bpt_heuristic: f64,
    pub tune_bpt_direct: f64,
    pub tune_bpt_oracle: f64,
    pub eval_bpt_base: f64,
    pub eval_bpt_heuristic: f64,
    pub eval_bpt_direct: f64,
    pub eval_bpt_oracle: f64,
    pub eval_topk_hit_rate: f64,
    pub eval_topk_better_rate: f64,
}

#[derive(Debug, Clone)]
pub struct TrainedTokenBridge {
    report: TokenBridgeReport,
    runner: DirectGatedPackedMemoryRunner,
    eval_tokens: Vec<usize>,
}

impl TrainedTokenBridge {
    pub fn report(&self) -> &TokenBridgeReport {
        &self.report
    }

    pub fn runner(&self) -> &DirectGatedPackedMemoryRunner {
        &self.runner
    }

    pub fn eval_tokens(&self) -> &[usize] {
        &self.eval_tokens
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
struct RawTokenRecord {
    features: [f64; FEATURE_DIM],
    base_gold_prob: f64,
    top4_gold_prob: f64,
    heuristic_gate: f64,
}

#[derive(Debug, Clone)]
struct TokenRecord {
    features: [f64; FEATURE_DIM],
    base_gold_prob: f64,
    top4_gold_prob: f64,
    heuristic_gate: f64,
}

#[derive(Debug, Clone)]
pub struct DirectGatedPackedMemoryRunner {
    tables: PackedTables,
    alpha_bigram: f64,
    alpha_trigram: f64,
    standardizer: Standardizer,
    model: LogisticModel,
    lambda: f64,
    candidate_k: usize,
    stream_prev2: Option<usize>,
    stream_prev1: Option<usize>,
}

pub fn train_token_bridge_from_data_root(
    root: &Path,
    train_token_budget: usize,
    trigram_buckets: usize,
    val_token_budget: usize,
    alpha_bigram: f64,
    alpha_trigram: f64,
    train_stride: usize,
    candidate_k: usize,
) -> Result<TrainedTokenBridge, String> {
    if train_stride == 0 {
        return Err("train_stride must be positive".to_string());
    }
    if candidate_k == 0 {
        return Err("candidate_k must be positive".to_string());
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
        true,
        train_stride,
        candidate_k,
    );
    if train_raw.is_empty() {
        return Err("no token bridge training records collected".to_string());
    }
    let standardizer = fit_standardizer_from_raw(&train_raw);
    let train_records = standardize_records(&train_raw, &standardizer);
    let direct_model = train_compression_gate(&train_records, 60, 0.15, 1e-4);

    let tune_raw = collect_raw_records(
        &tune_tokens,
        &tables,
        alpha_bigram,
        alpha_trigram,
        false,
        1,
        candidate_k,
    );
    let eval_raw = collect_raw_records(
        &eval_tokens,
        &tables,
        alpha_bigram,
        alpha_trigram,
        false,
        1,
        candidate_k,
    );
    let tune_records = standardize_records(&tune_raw, &standardizer);
    let eval_records = standardize_records(&eval_raw, &standardizer);

    let tuned_heuristic_lambda = tune_lambda(&tune_records, |record| record.heuristic_gate);
    let tuned_direct_lambda = tune_lambda(&tune_records, |record| {
        predict_probability(&direct_model, &record.features)
    });

    let report = TokenBridgeReport {
        train_token_budget,
        trigram_buckets,
        val_token_budget: val_tokens.len(),
        train_stride,
        candidate_k,
        train_records: train_records.len(),
        tune_tokens: tune_records.len(),
        eval_tokens: eval_records.len(),
        tuned_heuristic_lambda,
        tuned_direct_lambda,
        tune_bpt_base: mean_bits_per_token(&tune_records, None),
        tune_bpt_heuristic: mean_bits_per_token(
            &tune_records,
            Some((tuned_heuristic_lambda, GateKind::Heuristic, &direct_model)),
        ),
        tune_bpt_direct: mean_bits_per_token(
            &tune_records,
            Some((tuned_direct_lambda, GateKind::Direct, &direct_model)),
        ),
        tune_bpt_oracle: oracle_bits_per_token(&tune_records),
        eval_bpt_base: mean_bits_per_token(&eval_records, None),
        eval_bpt_heuristic: mean_bits_per_token(
            &eval_records,
            Some((tuned_heuristic_lambda, GateKind::Heuristic, &direct_model)),
        ),
        eval_bpt_direct: mean_bits_per_token(
            &eval_records,
            Some((tuned_direct_lambda, GateKind::Direct, &direct_model)),
        ),
        eval_bpt_oracle: oracle_bits_per_token(&eval_records),
        eval_topk_hit_rate: topk_hit_rate(&eval_records),
        eval_topk_better_rate: topk_better_rate(&eval_records),
    };

    let runner = DirectGatedPackedMemoryRunner {
        tables,
        alpha_bigram,
        alpha_trigram,
        standardizer,
        model: direct_model,
        lambda: tuned_direct_lambda,
        candidate_k,
        stream_prev2: None,
        stream_prev1: None,
    };

    Ok(TrainedTokenBridge {
        report,
        runner,
        eval_tokens,
    })
}

pub fn run_token_bridge_from_data_root(
    root: &Path,
    train_token_budget: usize,
    trigram_buckets: usize,
    val_token_budget: usize,
    alpha_bigram: f64,
    alpha_trigram: f64,
    train_stride: usize,
    candidate_k: usize,
) -> Result<TokenBridgeReport, String> {
    Ok(train_token_bridge_from_data_root(
        root,
        train_token_budget,
        trigram_buckets,
        val_token_budget,
        alpha_bigram,
        alpha_trigram,
        train_stride,
        candidate_k,
    )?
    .report)
}

pub fn render_token_bridge_report(report: &TokenBridgeReport) -> String {
    let mut out = String::new();
    out.push_str("chronohorn_token_bridge\n");
    out.push_str(&format!(
        "train_token_budget: {}\n",
        report.train_token_budget
    ));
    out.push_str(&format!("trigram_buckets: {}\n", report.trigram_buckets));
    out.push_str(&format!("val_token_budget: {}\n", report.val_token_budget));
    out.push_str(&format!("train_stride: {}\n", report.train_stride));
    out.push_str(&format!("candidate_k: {}\n", report.candidate_k));
    out.push_str(&format!("train_records: {}\n", report.train_records));
    out.push_str(&format!("tune_tokens: {}\n", report.tune_tokens));
    out.push_str(&format!("eval_tokens: {}\n", report.eval_tokens));
    out.push_str(&format!(
        "tuned_heuristic_lambda: {:.3}\n",
        report.tuned_heuristic_lambda
    ));
    out.push_str(&format!(
        "tuned_direct_lambda: {:.3}\n",
        report.tuned_direct_lambda
    ));
    out.push_str(&format!("tune_bpt_base: {:.6}\n", report.tune_bpt_base));
    out.push_str(&format!(
        "tune_bpt_heuristic: {:.6}\n",
        report.tune_bpt_heuristic
    ));
    out.push_str(&format!("tune_bpt_direct: {:.6}\n", report.tune_bpt_direct));
    out.push_str(&format!("tune_bpt_oracle: {:.6}\n", report.tune_bpt_oracle));
    out.push_str(&format!("eval_bpt_base: {:.6}\n", report.eval_bpt_base));
    out.push_str(&format!(
        "eval_bpt_heuristic: {:.6}\n",
        report.eval_bpt_heuristic
    ));
    out.push_str(&format!("eval_bpt_direct: {:.6}\n", report.eval_bpt_direct));
    out.push_str(&format!("eval_bpt_oracle: {:.6}\n", report.eval_bpt_oracle));
    out.push_str(&format!(
        "eval_topk_hit_rate: {:.6}\n",
        report.eval_topk_hit_rate
    ));
    out.push_str(&format!(
        "eval_topk_better_rate: {:.6}\n",
        report.eval_topk_better_rate
    ));
    out
}

impl Runner for DirectGatedPackedMemoryRunner {
    fn name(&self) -> &'static str {
        "DirectGatedPackedMemoryRunner"
    }

    fn vocab_size(&self) -> usize {
        self.tables.vocab_size
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
            let (prev2, prev1) = self.context_for_position(tokens, pos);
            let base = score_distribution(
                &self.tables,
                self.alpha_bigram,
                self.alpha_trigram,
                prev2,
                prev1,
                None,
            );
            let features = standardize_features(
                extract_features(&base, prev1, prev2, &self.tables),
                &self.standardizer,
            );
            let gate = (self.lambda * predict_probability(&self.model, &features)).clamp(0.0, 1.0);
            let mixed = mix_with_topk(&base, gate, self.candidate_k);
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
        }
        Ok(())
    }
}

impl DirectGatedPackedMemoryRunner {
    fn context_for_position(&self, tokens: &[usize], pos: usize) -> (Option<usize>, Option<usize>) {
        match pos {
            0 => (self.stream_prev2, self.stream_prev1),
            1 => (self.stream_prev1, Some(tokens[0])),
            _ => (Some(tokens[pos - 2]), Some(tokens[pos - 1])),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Adjustment {
    gold: usize,
    subtract_unigram: bool,
    subtract_bigram: bool,
    subtract_trigram: bool,
}

#[derive(Debug, Clone, Copy)]
enum GateKind {
    Heuristic,
    Direct,
}

fn collect_raw_records(
    tokens: &[usize],
    tables: &PackedTables,
    alpha_bigram: f64,
    alpha_trigram: f64,
    exclude_current: bool,
    stride: usize,
    candidate_k: usize,
) -> Vec<RawTokenRecord> {
    let mut rows = Vec::new();
    for pos in 0..tokens.len() {
        if pos % stride != 0 {
            continue;
        }
        let prev2 = if pos >= 2 {
            Some(tokens[pos - 2])
        } else {
            None
        };
        let prev1 = if pos >= 1 {
            Some(tokens[pos - 1])
        } else {
            None
        };
        let gold = tokens[pos];
        let adjustment = if exclude_current {
            Some(Adjustment {
                gold,
                subtract_unigram: pos >= 1,
                subtract_bigram: pos >= 1,
                subtract_trigram: pos >= 2,
            })
        } else {
            None
        };
        let base = score_distribution(
            tables,
            alpha_bigram,
            alpha_trigram,
            prev2,
            prev1,
            adjustment,
        );
        let features = extract_features(&base, prev1, prev2, tables);
        let (_, topk_mass, _, _, topk_gold_prob) = topk_summary(&base, gold, candidate_k);
        let base_gold_prob = base
            .get(gold)
            .copied()
            .unwrap_or(f64::MIN_POSITIVE)
            .max(f64::MIN_POSITIVE);
        rows.push(RawTokenRecord {
            features,
            base_gold_prob,
            top4_gold_prob: topk_gold_prob,
            heuristic_gate: topk_mass,
        });
    }
    rows
}

fn score_distribution(
    tables: &PackedTables,
    alpha_bigram: f64,
    alpha_trigram: f64,
    prev2: Option<usize>,
    prev1: Option<usize>,
    adjustment: Option<Adjustment>,
) -> Vec<f64> {
    let vocab = tables.vocab_size;
    let unigram_total = (tables.token_budget.saturating_sub(1) as f64
        - if adjustment.is_some_and(|adj| adj.subtract_unigram) {
            1.0
        } else {
            0.0
        })
    .max(1.0);
    let mut p_uni = vec![0.0; vocab];
    for tok in 0..vocab {
        let mut count = tables.unigram_probs[tok] * (tables.token_budget.saturating_sub(1) as f64);
        if let Some(adj) = adjustment {
            if adj.subtract_unigram && tok == adj.gold {
                count = (count - 1.0).max(0.0);
            }
        }
        p_uni[tok] = count / unigram_total;
    }

    let p_bigram = if let Some(p1) = prev1 {
        let row_start = p1 * vocab;
        let mut total = tables.bigram_totals[p1];
        if let Some(adj) = adjustment {
            if adj.subtract_bigram {
                total = (total - 1.0).max(0.0);
            }
        }
        let denom = (total + alpha_bigram).max(1e-8);
        let mut out = vec![0.0; vocab];
        for tok in 0..vocab {
            let mut count = tables.bigram_counts[row_start + tok];
            if let Some(adj) = adjustment {
                if adj.subtract_bigram && tok == adj.gold {
                    count = (count - 1.0).max(0.0);
                }
            }
            out[tok] = (count + alpha_bigram * p_uni[tok]) / denom;
        }
        out
    } else {
        p_uni.clone()
    };

    if let (Some(p2), Some(p1)) = (prev2, prev1) {
        let bucket = trigram_bucket(tables.trigram_buckets, p2, p1);
        let row_start = bucket * vocab;
        let mut total = tables.trigram_totals[bucket];
        if let Some(adj) = adjustment {
            if adj.subtract_trigram {
                total = (total - 1.0).max(0.0);
            }
        }
        let denom = (total + alpha_trigram).max(1e-8);
        let mut out = vec![0.0; vocab];
        for tok in 0..vocab {
            let mut count = tables.trigram_counts[row_start + tok];
            if let Some(adj) = adjustment {
                if adj.subtract_trigram && tok == adj.gold {
                    count = (count - 1.0).max(0.0);
                }
            }
            out[tok] = (count + alpha_trigram * p_bigram[tok]) / denom;
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

fn extract_features(
    distribution: &[f64],
    prev1: Option<usize>,
    prev2: Option<usize>,
    tables: &PackedTables,
) -> [f64; FEATURE_DIM] {
    let bigram_total = prev1.map(|p1| tables.bigram_totals[p1]).unwrap_or(0.0);
    let trigram_total = match (prev2, prev1) {
        (Some(p2), Some(p1)) => {
            let bucket = trigram_bucket(tables.trigram_buckets, p2, p1);
            tables.trigram_totals[bucket]
        }
        _ => 0.0,
    };
    let (top1_prob, top4_mass, normalized_entropy, margin, _) = topk_summary(distribution, 0, 4);
    [
        (1.0 + bigram_total).ln(),
        (1.0 + trigram_total).ln(),
        top1_prob,
        top4_mass,
        normalized_entropy,
        margin,
    ]
}

fn topk_summary(
    distribution: &[f64],
    gold: usize,
    candidate_k: usize,
) -> (f64, f64, f64, f64, f64) {
    let mut indexed = distribution
        .iter()
        .enumerate()
        .map(|(index, &value)| (index, value))
        .collect::<Vec<_>>();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top1_prob = indexed.first().map(|(_, p)| *p).unwrap_or(0.0);
    let top2_prob = indexed.get(1).map(|(_, p)| *p).unwrap_or(0.0);
    let top4 = indexed
        .iter()
        .take(candidate_k)
        .copied()
        .collect::<Vec<_>>();
    let top4_mass = top4.iter().map(|(_, p)| *p).sum::<f64>();
    let top4_gold_prob = if top4_mass > 0.0 {
        top4.iter()
            .find(|(index, _)| *index == gold)
            .map(|(_, value)| *value / top4_mass)
            .unwrap_or(0.0)
    } else {
        0.0
    };
    let entropy = distribution
        .iter()
        .filter(|value| **value > 0.0)
        .map(|value| -value * value.ln())
        .sum::<f64>();
    let normalized_entropy = if distribution.len() > 1 {
        entropy / (distribution.len() as f64).ln().max(f64::MIN_POSITIVE)
    } else {
        0.0
    };
    (
        top1_prob,
        top4_mass,
        normalized_entropy,
        top1_prob - top2_prob,
        top4_gold_prob,
    )
}

fn mix_with_topk(base: &[f64], gate: f64, candidate_k: usize) -> Vec<f64> {
    if gate <= 0.0 {
        return base.to_vec();
    }
    let mut indexed = base
        .iter()
        .enumerate()
        .map(|(index, &value)| (index, value))
        .collect::<Vec<_>>();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top4 = indexed
        .iter()
        .take(candidate_k)
        .copied()
        .collect::<Vec<_>>();
    let top4_mass = top4.iter().map(|(_, p)| *p).sum::<f64>().max(1e-12);
    let mut out = Vec::with_capacity(base.len());
    for (index, &base_prob) in base.iter().enumerate() {
        let top4_prob = top4
            .iter()
            .find(|(tok, _)| *tok == index)
            .map(|(_, value)| *value / top4_mass)
            .unwrap_or(0.0);
        out.push((1.0 - gate) * base_prob + gate * top4_prob);
    }
    out
}

fn fit_standardizer_from_raw(records: &[RawTokenRecord]) -> Standardizer {
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
    records: &[RawTokenRecord],
    standardizer: &Standardizer,
) -> Vec<TokenRecord> {
    records
        .iter()
        .map(|record| TokenRecord {
            features: standardize_features(record.features, standardizer),
            base_gold_prob: record.base_gold_prob,
            top4_gold_prob: record.top4_gold_prob,
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

fn train_compression_gate(
    records: &[TokenRecord],
    epochs: usize,
    lr: f64,
    l2: f64,
) -> LogisticModel {
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
            let mixed =
                ((1.0 - gate) * record.base_gold_prob + gate * record.top4_gold_prob).max(1e-12);
            let dloss_dgate = (record.base_gold_prob - record.top4_gold_prob) / (mixed * ln2);
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

fn tune_lambda(records: &[TokenRecord], gate_fn: impl Fn(&TokenRecord) -> f64) -> f64 {
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

fn mean_bits_per_token(
    records: &[TokenRecord],
    gate: Option<(f64, GateKind, &LogisticModel)>,
) -> f64 {
    match gate {
        None => {
            records
                .iter()
                .map(|record| -record.base_gold_prob.max(1e-12).log2())
                .sum::<f64>()
                / records.len() as f64
        }
        Some((lambda, GateKind::Heuristic, _)) => {
            mean_bits_per_token_with_gate(records, lambda, |record| record.heuristic_gate)
        }
        Some((lambda, GateKind::Direct, model)) => {
            mean_bits_per_token_with_gate(records, lambda, |record| {
                predict_probability(model, &record.features)
            })
        }
    }
}

fn mean_bits_per_token_with_gate(
    records: &[TokenRecord],
    lambda: f64,
    gate_fn: impl Fn(&TokenRecord) -> f64,
) -> f64 {
    let mut bits = 0.0;
    for record in records {
        let gate = (lambda * gate_fn(record)).clamp(0.0, 1.0);
        let mixed = (1.0 - gate) * record.base_gold_prob + gate * record.top4_gold_prob.max(1e-12);
        bits += -mixed.max(1e-12).log2();
    }
    bits / records.len() as f64
}

fn oracle_bits_per_token(records: &[TokenRecord]) -> f64 {
    records
        .iter()
        .map(|record| {
            let best = record.base_gold_prob.max(record.top4_gold_prob).max(1e-12);
            -best.log2()
        })
        .sum::<f64>()
        / records.len() as f64
}

fn topk_hit_rate(records: &[TokenRecord]) -> f64 {
    records
        .iter()
        .filter(|record| record.top4_gold_prob > 0.0)
        .count() as f64
        / records.len() as f64
}

fn topk_better_rate(records: &[TokenRecord]) -> f64 {
    records
        .iter()
        .filter(|record| record.top4_gold_prob > record.base_gold_prob)
        .count() as f64
        / records.len() as f64
}

fn normalize(values: &mut [f64]) {
    let total: f64 = values.iter().sum();
    let denom = total.max(f64::EPSILON);
    for value in values {
        *value /= denom;
    }
}
