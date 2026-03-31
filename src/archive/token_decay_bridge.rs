use std::path::Path;

use chronohorn_core::data::{take_train_tokens, take_val_tokens};
use chronohorn_core::protocol::{Runner, SampleOutputs};

use super::packed_memory::{PackedTables, build_packed_tables};

const FEATURE_DIM: usize = 9;
const UNIFORM_FLOOR: f64 = 1e-12;

#[derive(Debug, Clone)]
pub struct TokenDecayBridgeReport {
    pub train_token_budget: usize,
    pub trigram_buckets: usize,
    pub val_token_budget: usize,
    pub train_stride: usize,
    pub candidate_k: usize,
    pub decay_factor: f64,
    pub train_records: usize,
    pub tune_records: usize,
    pub eval_records: usize,
    pub tuned_heuristic_lambda: f64,
    pub tuned_direct_lambda: f64,
    pub selected_runtime_gate: String,
    pub selected_runtime_lambda: f64,
    pub tune_bpt_base: f64,
    pub tune_bpt_decay_full: f64,
    pub tune_bpt_decay_topk: f64,
    pub tune_bpt_heuristic: f64,
    pub tune_bpt_direct: f64,
    pub tune_bpt_oracle: f64,
    pub eval_bpt_base: f64,
    pub eval_bpt_decay_full: f64,
    pub eval_bpt_decay_topk: f64,
    pub eval_bpt_heuristic: f64,
    pub eval_bpt_direct: f64,
    pub eval_bpt_oracle: f64,
    pub eval_decay_hit_rate: f64,
    pub eval_decay_better_rate: f64,
}

#[derive(Debug, Clone)]
pub struct TrainedTokenDecayBridge {
    report: TokenDecayBridgeReport,
    runner: TokenDecayBridgeRunner,
    eval_tokens: Vec<usize>,
}

impl TrainedTokenDecayBridge {
    pub fn report(&self) -> &TokenDecayBridgeReport {
        &self.report
    }

    pub fn runner(&self) -> &TokenDecayBridgeRunner {
        &self.runner
    }

    pub fn eval_tokens(&self) -> &[usize] {
        &self.eval_tokens
    }
}

#[derive(Debug, Clone)]
pub struct TokenDecayBridgeRunner {
    base_tables: PackedTables,
    decay_tables: DecayTables,
    alpha_bigram: f64,
    alpha_trigram: f64,
    standardizer: Standardizer,
    gate: GateChoice,
    lambda: f64,
    candidate_k: usize,
    stream_prev2: Option<usize>,
    stream_prev1: Option<usize>,
}

#[derive(Debug, Clone)]
struct DecayTables {
    vocab_size: usize,
    unigram_probs: Vec<f64>,
    bigram_counts: Vec<f64>,
    bigram_totals: Vec<f64>,
    trigram_counts: Vec<f64>,
    trigram_totals: Vec<f64>,
    trigram_buckets: usize,
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
struct RawDecayRecord {
    features: [f64; FEATURE_DIM],
    base_gold_prob: f64,
    decay_full_gold_prob: f64,
    decay_topk_gold_prob: f64,
    heuristic_gate: f64,
}

#[derive(Debug, Clone)]
struct DecayRecord {
    features: [f64; FEATURE_DIM],
    base_gold_prob: f64,
    decay_full_gold_prob: f64,
    decay_topk_gold_prob: f64,
    heuristic_gate: f64,
}

#[derive(Debug, Clone, Copy)]
struct DistributionStats {
    top1_prob: f64,
    top2_prob: f64,
    topk_mass: f64,
    entropy_norm: f64,
    top1_token: usize,
    support: usize,
}

#[derive(Debug, Clone)]
struct DistributionSummary {
    stats: DistributionStats,
    topk_dist: Vec<f64>,
}

#[derive(Debug, Clone)]
struct DecaySummary {
    full_dist: Vec<f64>,
    topk_dist: Vec<f64>,
    top1_prob: f64,
    topk_mass: f64,
    entropy_norm: f64,
    support: usize,
    top1_token: usize,
}

pub fn train_token_decay_bridge_from_data_root(
    root: &Path,
    train_token_budget: usize,
    trigram_buckets: usize,
    val_token_budget: usize,
    decay_factor: f64,
    candidate_k: usize,
    train_stride: usize,
    alpha_bigram: f64,
    alpha_trigram: f64,
) -> Result<TrainedTokenDecayBridge, String> {
    if train_stride == 0 {
        return Err("train_stride must be positive".to_string());
    }
    if candidate_k == 0 {
        return Err("candidate_k must be positive".to_string());
    }
    if !(0.0 < decay_factor && decay_factor <= 1.0) {
        return Err("decay_factor must be in (0, 1]".to_string());
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

    let base_tables = build_packed_tables(&train_tokens, 1024, trigram_buckets)?;
    let decay_tables = build_decay_tables(&train_tokens, 1024, trigram_buckets, decay_factor)?;

    let train_raw = collect_raw_records(
        &train_tokens,
        &base_tables,
        &decay_tables,
        alpha_bigram,
        alpha_trigram,
        candidate_k,
        train_stride,
    );
    if train_raw.is_empty() {
        return Err("no token decay bridge training records collected".to_string());
    }
    let standardizer = fit_standardizer_from_raw(&train_raw);
    let train_records = standardize_records(&train_raw, &standardizer);
    let direct_model = train_direct_gate(&train_records, 80, 0.1, 1e-4);

    let tune_raw = collect_raw_records(
        &tune_tokens,
        &base_tables,
        &decay_tables,
        alpha_bigram,
        alpha_trigram,
        candidate_k,
        1,
    );
    let eval_raw = collect_raw_records(
        &eval_tokens,
        &base_tables,
        &decay_tables,
        alpha_bigram,
        alpha_trigram,
        candidate_k,
        1,
    );
    if tune_raw.is_empty() {
        return Err("no token decay bridge tune records collected".to_string());
    }
    if eval_raw.is_empty() {
        return Err("no token decay bridge eval records collected".to_string());
    }

    let tune_records = standardize_records(&tune_raw, &standardizer);
    let eval_records = standardize_records(&eval_raw, &standardizer);

    let tuned_heuristic_lambda = tune_lambda(&tune_records, |record| record.heuristic_gate);
    let tuned_direct_lambda = tune_lambda(&tune_records, |record| {
        predict_probability(&direct_model, &record.features)
    });

    let tune_bpt_base = mean_bits_per_token_base(&tune_records);
    let tune_bpt_decay_full = mean_bits_per_token_decay_full(&tune_records);
    let tune_bpt_decay_topk = mean_bits_per_token_decay_topk(&tune_records);
    let tune_bpt_heuristic =
        mean_bits_per_token_with_heuristic(&tune_records, tuned_heuristic_lambda);
    let tune_bpt_direct =
        mean_bits_per_token_with_direct(&tune_records, tuned_direct_lambda, &direct_model);
    let tune_bpt_oracle = oracle_bits_per_token(&tune_records);

    let eval_bpt_base = mean_bits_per_token_base(&eval_records);
    let eval_bpt_decay_full = mean_bits_per_token_decay_full(&eval_records);
    let eval_bpt_decay_topk = mean_bits_per_token_decay_topk(&eval_records);
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

    let report = TokenDecayBridgeReport {
        train_token_budget,
        trigram_buckets,
        val_token_budget: val_tokens.len(),
        train_stride,
        candidate_k,
        decay_factor,
        train_records: train_records.len(),
        tune_records: tune_records.len(),
        eval_records: eval_records.len(),
        tuned_heuristic_lambda,
        tuned_direct_lambda,
        selected_runtime_gate,
        selected_runtime_lambda,
        tune_bpt_base,
        tune_bpt_decay_full,
        tune_bpt_decay_topk,
        tune_bpt_heuristic,
        tune_bpt_direct,
        tune_bpt_oracle,
        eval_bpt_base,
        eval_bpt_decay_full,
        eval_bpt_decay_topk,
        eval_bpt_heuristic,
        eval_bpt_direct,
        eval_bpt_oracle,
        eval_decay_hit_rate: decay_hit_rate(&eval_records),
        eval_decay_better_rate: decay_better_rate(&eval_records),
    };

    let runner = TokenDecayBridgeRunner {
        base_tables,
        decay_tables,
        alpha_bigram,
        alpha_trigram,
        standardizer,
        gate: gate_choice,
        lambda: selected_runtime_lambda,
        candidate_k,
        stream_prev2: None,
        stream_prev1: None,
    };

    Ok(TrainedTokenDecayBridge {
        report,
        runner,
        eval_tokens,
    })
}

pub fn run_token_decay_bridge_from_data_root(
    root: &Path,
    train_token_budget: usize,
    trigram_buckets: usize,
    val_token_budget: usize,
    decay_factor: f64,
    candidate_k: usize,
    train_stride: usize,
    alpha_bigram: f64,
    alpha_trigram: f64,
) -> Result<TokenDecayBridgeReport, String> {
    Ok(train_token_decay_bridge_from_data_root(
        root,
        train_token_budget,
        trigram_buckets,
        val_token_budget,
        decay_factor,
        candidate_k,
        train_stride,
        alpha_bigram,
        alpha_trigram,
    )?
    .report)
}

pub fn render_token_decay_bridge_report(report: &TokenDecayBridgeReport) -> String {
    let mut out = String::new();
    out.push_str("chronohorn_token_decay_bridge\n");
    out.push_str(&format!(
        "train_token_budget: {}\n",
        report.train_token_budget
    ));
    out.push_str(&format!("trigram_buckets: {}\n", report.trigram_buckets));
    out.push_str(&format!("val_token_budget: {}\n", report.val_token_budget));
    out.push_str(&format!("train_stride: {}\n", report.train_stride));
    out.push_str(&format!("candidate_k: {}\n", report.candidate_k));
    out.push_str(&format!("decay_factor: {:.6}\n", report.decay_factor));
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
        "tune_bpt_decay_full: {:.6}\n",
        report.tune_bpt_decay_full
    ));
    out.push_str(&format!(
        "tune_bpt_decay_topk: {:.6}\n",
        report.tune_bpt_decay_topk
    ));
    out.push_str(&format!(
        "tune_bpt_heuristic: {:.6}\n",
        report.tune_bpt_heuristic
    ));
    out.push_str(&format!("tune_bpt_direct: {:.6}\n", report.tune_bpt_direct));
    out.push_str(&format!("tune_bpt_oracle: {:.6}\n", report.tune_bpt_oracle));
    out.push_str(&format!("eval_bpt_base: {:.6}\n", report.eval_bpt_base));
    out.push_str(&format!(
        "eval_bpt_decay_full: {:.6}\n",
        report.eval_bpt_decay_full
    ));
    out.push_str(&format!(
        "eval_bpt_decay_topk: {:.6}\n",
        report.eval_bpt_decay_topk
    ));
    out.push_str(&format!(
        "eval_bpt_heuristic: {:.6}\n",
        report.eval_bpt_heuristic
    ));
    out.push_str(&format!("eval_bpt_direct: {:.6}\n", report.eval_bpt_direct));
    out.push_str(&format!("eval_bpt_oracle: {:.6}\n", report.eval_bpt_oracle));
    out.push_str(&format!(
        "eval_decay_hit_rate: {:.6}\n",
        report.eval_decay_hit_rate
    ));
    out.push_str(&format!(
        "eval_decay_better_rate: {:.6}\n",
        report.eval_decay_better_rate
    ));
    out
}

impl Runner for TokenDecayBridgeRunner {
    fn name(&self) -> &'static str {
        "TokenDecayBridgeRunner"
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
            let decay = score_decay_distribution(
                &self.decay_tables,
                self.alpha_bigram,
                self.alpha_trigram,
                base_prev2,
                base_prev1,
            );
            let base_stats = distribution_stats(&base);
            let decay_stats = distribution_stats(&decay);
            let features = standardize_features(
                extract_features(&base_stats, &decay_stats),
                &self.standardizer,
            );
            let raw_gate = match &self.gate {
                GateChoice::Heuristic => heuristic_gate_from_summary(&decay_stats),
                GateChoice::Direct(model) => predict_probability(model, &features),
            };
            let gate = (self.lambda * raw_gate).clamp(0.0, 1.0);
            let decay_topk = distribution_topk_dist(&decay, self.candidate_k);
            let mixed = mix_with_topk(&base, &decay_topk, gate);
            let tok = tokens[pos];
            let gold = mixed
                .get(tok)
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

impl TokenDecayBridgeRunner {
    fn context_back(&self, tokens: &[usize], pos: usize, back: usize) -> Option<usize> {
        if pos >= back {
            Some(tokens[pos - back])
        } else {
            match back {
                1 => self.stream_prev1,
                2 => self.stream_prev2,
                _ => None,
            }
        }
    }
}

fn collect_raw_records(
    tokens: &[usize],
    base_tables: &PackedTables,
    decay_tables: &DecayTables,
    alpha_bigram: f64,
    alpha_trigram: f64,
    candidate_k: usize,
    stride: usize,
) -> Vec<RawDecayRecord> {
    let mut rows = Vec::new();
    let mut prev2 = None;
    let mut prev1 = None;
    for (pos, &gold) in tokens.iter().enumerate() {
        if pos % stride != 0 {
            continue;
        }
        let base = score_base_distribution(base_tables, alpha_bigram, alpha_trigram, prev2, prev1);
        let decay =
            score_decay_distribution(decay_tables, alpha_bigram, alpha_trigram, prev2, prev1);
        let base_stats = distribution_stats(&base);
        let decay_stats = distribution_stats(&decay);
        let decay_topk = distribution_topk_dist(&decay, candidate_k);
        let features = extract_features(&base_stats, &decay_stats);
        let base_gold_prob = base
            .get(gold)
            .copied()
            .unwrap_or(f64::MIN_POSITIVE)
            .max(f64::MIN_POSITIVE);
        let decay_full_gold_prob = decay
            .get(gold)
            .copied()
            .unwrap_or(f64::MIN_POSITIVE)
            .max(f64::MIN_POSITIVE);
        let decay_topk_gold_prob = decay_topk
            .get(gold)
            .copied()
            .unwrap_or(f64::MIN_POSITIVE)
            .max(f64::MIN_POSITIVE);
        rows.push(RawDecayRecord {
            features,
            base_gold_prob,
            decay_full_gold_prob,
            decay_topk_gold_prob,
            heuristic_gate: heuristic_gate_from_summary(&decay_stats),
        });
        prev2 = prev1;
        prev1 = Some(gold);
    }
    rows
}

fn build_decay_tables(
    tokens: &[usize],
    vocab_size: usize,
    trigram_buckets: usize,
    decay_factor: f64,
) -> Result<DecayTables, String> {
    if trigram_buckets == 0 {
        return Err("trigram_buckets must be positive".to_string());
    }
    if tokens.len() < 4 {
        return Err("need at least 4 tokens".to_string());
    }

    let mut unigram_counts = vec![0.0; vocab_size];
    let mut bigram_counts = vec![0.0; vocab_size * vocab_size];
    let mut trigram_counts = vec![0.0; trigram_buckets * vocab_size];
    let last = tokens.len() - 1;

    for pos in 1..tokens.len() {
        let prev1 = tokens[pos - 1];
        let next = tokens[pos];
        if prev1 >= vocab_size || next >= vocab_size {
            return Err(format!("token out of vocab: {prev1} -> {next}"));
        }
        let age = (last - pos) as f64;
        let weight = decay_factor.powf(age);
        unigram_counts[next] += weight;
        bigram_counts[prev1 * vocab_size + next] += weight;
    }

    for pos in 2..tokens.len() {
        let prev2 = tokens[pos - 2];
        let prev1 = tokens[pos - 1];
        let next = tokens[pos];
        if prev2 >= vocab_size || prev1 >= vocab_size || next >= vocab_size {
            return Err(format!(
                "trigram token out of vocab: {prev2},{prev1}->{next}"
            ));
        }
        let age = (last - pos) as f64;
        let weight = decay_factor.powf(age);
        let bucket = trigram_bucket(trigram_buckets, prev2, prev1);
        trigram_counts[bucket * vocab_size + next] += weight;
    }

    let unigram_sum: f64 = unigram_counts.iter().sum();
    let mut unigram_probs = unigram_counts;
    for value in &mut unigram_probs {
        *value /= unigram_sum.max(1.0);
    }

    let mut bigram_totals = vec![0.0; vocab_size];
    for prev1 in 0..vocab_size {
        let row = &bigram_counts[prev1 * vocab_size..(prev1 + 1) * vocab_size];
        bigram_totals[prev1] = row.iter().sum();
    }

    let mut trigram_totals = vec![0.0; trigram_buckets];
    for bucket in 0..trigram_buckets {
        let row = &trigram_counts[bucket * vocab_size..(bucket + 1) * vocab_size];
        trigram_totals[bucket] = row.iter().sum();
    }

    Ok(DecayTables {
        vocab_size,
        unigram_probs,
        bigram_counts,
        bigram_totals,
        trigram_counts,
        trigram_totals,
        trigram_buckets,
    })
}

fn score_base_distribution(
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

fn score_decay_distribution(
    tables: &DecayTables,
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

fn extract_features(base: &DistributionStats, decay: &DistributionStats) -> [f64; FEATURE_DIM] {
    let agreement = if base.top1_token == decay.top1_token {
        1.0
    } else {
        0.0
    };
    [
        base.top1_prob,
        base.entropy_norm,
        base.top1_prob - base.top2_prob,
        decay.top1_prob,
        decay.topk_mass,
        decay.entropy_norm,
        (1.0_f64 + decay.support as f64).ln(),
        agreement,
        decay.top1_prob - base.top1_prob,
    ]
}

fn summarize_distribution(distribution: &[f64], candidate_k: usize) -> DistributionSummary {
    let mut indexed = distribution
        .iter()
        .enumerate()
        .map(|(index, &value)| (index, value))
        .collect::<Vec<_>>();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let k = candidate_k.min(indexed.len().max(1));
    let support = indexed.iter().take(k).copied().collect::<Vec<_>>();
    let topk_mass = support.iter().map(|(_, value)| *value).sum::<f64>();
    let mut topk_dist = vec![0.0; distribution.len()];
    for &(token, prob) in &support {
        topk_dist[token] = prob;
    }
    if topk_mass > 0.0 {
        for value in &mut topk_dist {
            *value /= topk_mass;
        }
    }
    let top1_prob = indexed.first().map(|(_, p)| *p).unwrap_or(0.0);
    let top2_prob = indexed.get(1).map(|(_, p)| *p).unwrap_or(0.0);
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
    DistributionSummary {
        stats: DistributionStats {
            top1_prob,
            top2_prob,
            topk_mass,
            entropy_norm,
            top1_token: indexed.first().map(|(i, _)| *i).unwrap_or(0),
            support: support.len(),
        },
        topk_dist,
    }
}

fn distribution_stats(distribution: &[f64]) -> DistributionStats {
    summarize_distribution(distribution, 4).stats
}

fn distribution_topk_dist(distribution: &[f64], candidate_k: usize) -> Vec<f64> {
    summarize_distribution(distribution, candidate_k).topk_dist
}

fn tune_lambda(records: &[DecayRecord], gate_fn: impl Fn(&DecayRecord) -> f64) -> f64 {
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

fn mean_bits_per_token_base(records: &[DecayRecord]) -> f64 {
    records
        .iter()
        .map(|record| -record.base_gold_prob.max(1e-12).log2())
        .sum::<f64>()
        / records.len() as f64
}

fn mean_bits_per_token_decay_full(records: &[DecayRecord]) -> f64 {
    records
        .iter()
        .map(|record| -record.decay_full_gold_prob.max(1e-12).log2())
        .sum::<f64>()
        / records.len() as f64
}

fn mean_bits_per_token_decay_topk(records: &[DecayRecord]) -> f64 {
    records
        .iter()
        .map(|record| -record.decay_topk_gold_prob.max(1e-12).log2())
        .sum::<f64>()
        / records.len() as f64
}

fn mean_bits_per_token_with_heuristic(records: &[DecayRecord], lambda: f64) -> f64 {
    mean_bits_per_token_with_gate(records, lambda, heuristic_gate)
}

fn mean_bits_per_token_with_direct(
    records: &[DecayRecord],
    lambda: f64,
    model: &LogisticModel,
) -> f64 {
    mean_bits_per_token_with_gate(records, lambda, |record| {
        predict_probability(model, &record.features)
    })
}

fn mean_bits_per_token_with_gate(
    records: &[DecayRecord],
    lambda: f64,
    gate_fn: impl Fn(&DecayRecord) -> f64,
) -> f64 {
    let mut bits = 0.0;
    for record in records {
        let gate = (lambda * gate_fn(record)).clamp(0.0, 1.0);
        let mixed =
            (1.0 - gate) * record.base_gold_prob + gate * record.decay_topk_gold_prob.max(1e-12);
        bits += -mixed.max(1e-12).log2();
    }
    bits / records.len() as f64
}

fn oracle_bits_per_token(records: &[DecayRecord]) -> f64 {
    records
        .iter()
        .map(|record| {
            let best = record
                .base_gold_prob
                .max(record.decay_topk_gold_prob)
                .max(1e-12);
            -best.log2()
        })
        .sum::<f64>()
        / records.len() as f64
}

fn decay_hit_rate(records: &[DecayRecord]) -> f64 {
    records
        .iter()
        .filter(|record| record.decay_topk_gold_prob > 0.0)
        .count() as f64
        / records.len() as f64
}

fn decay_better_rate(records: &[DecayRecord]) -> f64 {
    records
        .iter()
        .filter(|record| record.decay_topk_gold_prob > record.base_gold_prob)
        .count() as f64
        / records.len() as f64
}

fn heuristic_gate(record: &DecayRecord) -> f64 {
    record.heuristic_gate.clamp(0.0, 1.0)
}

fn heuristic_gate_from_summary(summary: &DistributionStats) -> f64 {
    summary.top1_prob.clamp(0.0, 1.0)
}

fn fit_standardizer_from_raw(records: &[RawDecayRecord]) -> Standardizer {
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
    records: &[RawDecayRecord],
    standardizer: &Standardizer,
) -> Vec<DecayRecord> {
    records
        .iter()
        .map(|record| DecayRecord {
            features: standardize_features(record.features, standardizer),
            base_gold_prob: record.base_gold_prob,
            decay_full_gold_prob: record.decay_full_gold_prob,
            decay_topk_gold_prob: record.decay_topk_gold_prob,
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

fn train_direct_gate(records: &[DecayRecord], epochs: usize, lr: f64, l2: f64) -> LogisticModel {
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
            let mixed = ((1.0 - gate) * record.base_gold_prob + gate * record.decay_topk_gold_prob)
                .max(1e-12);
            let dloss_dgate = (record.base_gold_prob - record.decay_topk_gold_prob) / (mixed * ln2);
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

fn mix_with_topk(base: &[f64], topk: &[f64], gate: f64) -> Vec<f64> {
    if gate <= 0.0 {
        return base.to_vec();
    }
    let mut out = vec![0.0; base.len()];
    for index in 0..base.len() {
        out[index] = (1.0 - gate) * base[index] + gate * topk[index];
    }
    normalize(&mut out);
    out
}

fn normalize(values: &mut [f64]) {
    let sum = values.iter().sum::<f64>().max(1e-12);
    for value in values {
        *value /= sum;
    }
}

fn trigram_bucket(trigram_buckets: usize, prev2: usize, prev1: usize) -> usize {
    ((prev2 as u64 * 1_315_423_911 + prev1 as u64 * 2_654_435_761) % trigram_buckets as u64)
        as usize
}
