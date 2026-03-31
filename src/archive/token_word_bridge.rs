use std::cmp::Ordering;
use std::collections::VecDeque;
use std::path::Path;

use chronohorn_core::data::{take_train_tokens, take_val_tokens};
use chronohorn_core::protocol::{Runner, SampleOutputs};

use super::packed_memory::{PackedTables, build_packed_tables};

const VOCAB_SIZE: usize = 1024;
const FEATURE_DIM: usize = 9;
const DEFAULT_WORD_WINDOW: usize = 24;
const DEFAULT_SUFFIX_LEN: usize = 4;
const DEFAULT_BOUNDARY_MARKERS: usize = 64;
const WORD_HASH_MUL_A: u64 = 1_160_169_841;
const WORD_HASH_MUL_B: u64 = 2_654_435_761;
const UNIFORM_FLOOR: f64 = 1e-12;

#[derive(Debug, Clone)]
pub struct TokenWordBridgeReport {
    pub train_token_budget: usize,
    pub trigram_buckets: usize,
    pub word_buckets: usize,
    pub val_token_budget: usize,
    pub word_window: usize,
    pub suffix_len: usize,
    pub boundary_markers: usize,
    pub candidate_k: usize,
    pub train_stride: usize,
    pub train_records: usize,
    pub tune_records: usize,
    pub eval_records: usize,
    pub tuned_heuristic_lambda: f64,
    pub tuned_direct_lambda: f64,
    pub selected_runtime_gate: String,
    pub selected_runtime_lambda: f64,
    pub tune_bpt_base: f64,
    pub tune_bpt_word_full: f64,
    pub tune_bpt_word_topk: f64,
    pub tune_bpt_heuristic: f64,
    pub tune_bpt_direct: f64,
    pub tune_bpt_oracle: f64,
    pub eval_bpt_base: f64,
    pub eval_bpt_word_full: f64,
    pub eval_bpt_word_topk: f64,
    pub eval_bpt_heuristic: f64,
    pub eval_bpt_direct: f64,
    pub eval_bpt_oracle: f64,
    pub eval_word_hit_rate: f64,
    pub eval_word_better_rate: f64,
}

#[derive(Debug, Clone)]
pub struct TrainedTokenWordBridge {
    report: TokenWordBridgeReport,
    runner: TokenWordBridgeRunner,
    eval_tokens: Vec<usize>,
}

impl TrainedTokenWordBridge {
    pub fn report(&self) -> &TokenWordBridgeReport {
        &self.report
    }

    pub fn runner(&self) -> &TokenWordBridgeRunner {
        &self.runner
    }

    pub fn eval_tokens(&self) -> &[usize] {
        &self.eval_tokens
    }
}

#[derive(Debug, Clone)]
pub struct TokenWordBridgeRunner {
    tables: WordTables,
    alpha_bigram: f64,
    alpha_trigram: f64,
    alpha_word: f64,
    standardizer: Standardizer,
    gate: GateChoice,
    lambda: f64,
    candidate_k: usize,
    word_window: usize,
    suffix_len: usize,
    stream_prev2: Option<usize>,
    stream_prev1: Option<usize>,
    recent_history: VecDeque<usize>,
}

#[derive(Debug, Clone)]
struct WordTables {
    base_tables: PackedTables,
    boundary_mask: Vec<bool>,
    word_counts: Vec<f64>,
    word_totals: Vec<f64>,
    word_buckets: usize,
    word_window: usize,
    suffix_len: usize,
    boundary_markers: usize,
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
struct RawWordRecord {
    features: [f64; FEATURE_DIM],
    base_gold_prob: f64,
    word_full_gold_prob: f64,
    word_topk_gold_prob: f64,
    heuristic_gate: f64,
}

#[derive(Debug, Clone)]
struct WordRecord {
    features: [f64; FEATURE_DIM],
    base_gold_prob: f64,
    word_full_gold_prob: f64,
    word_topk_gold_prob: f64,
    heuristic_gate: f64,
}

#[derive(Debug, Clone)]
struct DistributionStats {
    top1_prob: f64,
    top2_prob: f64,
    entropy_norm: f64,
    top1_token: usize,
}

#[derive(Debug, Clone)]
struct WordSummary {
    full_dist: Vec<f64>,
    topk_dist: Vec<f64>,
    top1_prob: f64,
    topk_mass: f64,
    entropy_norm: f64,
    support: usize,
    top1_token: usize,
    boundary_hit: f64,
    suffix_len: usize,
}

pub fn train_token_word_bridge_from_data_root(
    root: &Path,
    train_token_budget: usize,
    trigram_buckets: usize,
    word_buckets: usize,
    val_token_budget: usize,
    word_window: usize,
    suffix_len: usize,
    boundary_markers: usize,
    candidate_k: usize,
    train_stride: usize,
    alpha_bigram: f64,
    alpha_trigram: f64,
    alpha_word: f64,
) -> Result<TrainedTokenWordBridge, String> {
    if train_stride == 0 {
        return Err("train_stride must be positive".to_string());
    }
    if candidate_k == 0 {
        return Err("candidate_k must be positive".to_string());
    }
    if word_window == 0 {
        return Err("word_window must be positive".to_string());
    }
    if suffix_len == 0 {
        return Err("suffix_len must be positive".to_string());
    }
    if boundary_markers == 0 {
        return Err("boundary_markers must be positive".to_string());
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

    let tables = build_word_tables(
        &train_tokens,
        trigram_buckets,
        word_buckets,
        word_window,
        suffix_len,
        boundary_markers,
    )?;

    let train_raw = collect_raw_records(
        &train_tokens,
        &tables,
        alpha_bigram,
        alpha_trigram,
        alpha_word,
        candidate_k,
        train_stride,
    );
    if train_raw.is_empty() {
        return Err("no token word bridge training records collected".to_string());
    }
    let standardizer = fit_standardizer_from_raw(&train_raw);
    let train_records = standardize_records(&train_raw, &standardizer);
    let direct_model = train_logistic_model(&train_records, 72, 0.12, 1e-4);

    let tune_raw = collect_raw_records(
        &tune_tokens,
        &tables,
        alpha_bigram,
        alpha_trigram,
        alpha_word,
        candidate_k,
        1,
    );
    let eval_raw = collect_raw_records(
        &eval_tokens,
        &tables,
        alpha_bigram,
        alpha_trigram,
        alpha_word,
        candidate_k,
        1,
    );
    if tune_raw.is_empty() || eval_raw.is_empty() {
        return Err("no token word bridge tune/eval records collected".to_string());
    }

    let tune_records = standardize_records(&tune_raw, &standardizer);
    let eval_records = standardize_records(&eval_raw, &standardizer);

    let tuned_heuristic_lambda = tune_lambda(&tune_records, |record| record.heuristic_gate);
    let tuned_direct_lambda = tune_lambda(&tune_records, |record| {
        predict_probability(&direct_model, &record.features)
    });

    let tune_bpt_base = mean_bits_base(&tune_records);
    let tune_bpt_word_full = mean_bits_word_full(&tune_records);
    let tune_bpt_word_topk = mean_bits_word_topk(&tune_records);
    let tune_bpt_heuristic = mean_bits_with_gate(&tune_records, tuned_heuristic_lambda, |record| {
        record.heuristic_gate
    });
    let tune_bpt_direct = mean_bits_with_gate(&tune_records, tuned_direct_lambda, |record| {
        predict_probability(&direct_model, &record.features)
    });
    let tune_bpt_oracle = oracle_bits_per_token(&tune_records);

    let eval_bpt_base = mean_bits_base(&eval_records);
    let eval_bpt_word_full = mean_bits_word_full(&eval_records);
    let eval_bpt_word_topk = mean_bits_word_topk(&eval_records);
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

    let report = TokenWordBridgeReport {
        train_token_budget,
        trigram_buckets,
        word_buckets,
        val_token_budget: val_tokens.len(),
        word_window,
        suffix_len,
        boundary_markers,
        candidate_k,
        train_stride,
        train_records: train_records.len(),
        tune_records: tune_records.len(),
        eval_records: eval_records.len(),
        tuned_heuristic_lambda,
        tuned_direct_lambda,
        selected_runtime_gate,
        selected_runtime_lambda,
        tune_bpt_base,
        tune_bpt_word_full,
        tune_bpt_word_topk,
        tune_bpt_heuristic,
        tune_bpt_direct,
        tune_bpt_oracle,
        eval_bpt_base,
        eval_bpt_word_full,
        eval_bpt_word_topk,
        eval_bpt_heuristic,
        eval_bpt_direct,
        eval_bpt_oracle,
        eval_word_hit_rate: word_hit_rate(&eval_records),
        eval_word_better_rate: word_better_rate(&eval_records),
    };

    let runner = TokenWordBridgeRunner {
        tables,
        alpha_bigram,
        alpha_trigram,
        alpha_word,
        standardizer,
        gate,
        lambda: selected_runtime_lambda,
        candidate_k,
        word_window,
        suffix_len,
        stream_prev2: None,
        stream_prev1: None,
        recent_history: VecDeque::new(),
    };

    Ok(TrainedTokenWordBridge {
        report,
        runner,
        eval_tokens,
    })
}

pub fn run_token_word_bridge_from_data_root(
    root: &Path,
    train_token_budget: usize,
    trigram_buckets: usize,
    word_buckets: usize,
    val_token_budget: usize,
    word_window: usize,
    suffix_len: usize,
    boundary_markers: usize,
    candidate_k: usize,
    train_stride: usize,
    alpha_bigram: f64,
    alpha_trigram: f64,
    alpha_word: f64,
) -> Result<TokenWordBridgeReport, String> {
    Ok(train_token_word_bridge_from_data_root(
        root,
        train_token_budget,
        trigram_buckets,
        word_buckets,
        val_token_budget,
        word_window,
        suffix_len,
        boundary_markers,
        candidate_k,
        train_stride,
        alpha_bigram,
        alpha_trigram,
        alpha_word,
    )?
    .report)
}

pub fn render_token_word_bridge_report(report: &TokenWordBridgeReport) -> String {
    let mut out = String::new();
    out.push_str("chronohorn_token_word_bridge\n");
    out.push_str(&format!(
        "train_token_budget: {}\n",
        report.train_token_budget
    ));
    out.push_str(&format!("trigram_buckets: {}\n", report.trigram_buckets));
    out.push_str(&format!("word_buckets: {}\n", report.word_buckets));
    out.push_str(&format!("val_token_budget: {}\n", report.val_token_budget));
    out.push_str(&format!("word_window: {}\n", report.word_window));
    out.push_str(&format!("suffix_len: {}\n", report.suffix_len));
    out.push_str(&format!("boundary_markers: {}\n", report.boundary_markers));
    out.push_str(&format!("candidate_k: {}\n", report.candidate_k));
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
        "tune_bpt_word_full: {:.6}\n",
        report.tune_bpt_word_full
    ));
    out.push_str(&format!(
        "tune_bpt_word_topk: {:.6}\n",
        report.tune_bpt_word_topk
    ));
    out.push_str(&format!(
        "tune_bpt_heuristic: {:.6}\n",
        report.tune_bpt_heuristic
    ));
    out.push_str(&format!("tune_bpt_direct: {:.6}\n", report.tune_bpt_direct));
    out.push_str(&format!("tune_bpt_oracle: {:.6}\n", report.tune_bpt_oracle));
    out.push_str(&format!("eval_bpt_base: {:.6}\n", report.eval_bpt_base));
    out.push_str(&format!(
        "eval_bpt_word_full: {:.6}\n",
        report.eval_bpt_word_full
    ));
    out.push_str(&format!(
        "eval_bpt_word_topk: {:.6}\n",
        report.eval_bpt_word_topk
    ));
    out.push_str(&format!(
        "eval_bpt_heuristic: {:.6}\n",
        report.eval_bpt_heuristic
    ));
    out.push_str(&format!("eval_bpt_direct: {:.6}\n", report.eval_bpt_direct));
    out.push_str(&format!("eval_bpt_oracle: {:.6}\n", report.eval_bpt_oracle));
    out.push_str(&format!(
        "eval_word_hit_rate: {:.6}\n",
        report.eval_word_hit_rate
    ));
    out.push_str(&format!(
        "eval_word_better_rate: {:.6}\n",
        report.eval_word_better_rate
    ));
    out
}

impl Runner for TokenWordBridgeRunner {
    fn name(&self) -> &'static str {
        "TokenWordBridgeRunner"
    }

    fn vocab_size(&self) -> usize {
        self.tables.base_tables.vocab_size
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

        let mut sample_predictions = vec![Vec::new(); sample_positions.len()];
        let mut sample_gold_logprobs = vec![0.0; sample_positions.len()];
        let mut lookup = std::collections::HashMap::new();
        for (index, &pos) in sample_positions.iter().enumerate() {
            if pos >= tokens.len() {
                return Err(format!("sample position {pos} out of bounds"));
            }
            lookup.insert(pos, index);
        }

        let mut prev2 = self.stream_prev2;
        let mut prev1 = self.stream_prev1;
        let mut history = self.recent_history.clone();
        for (pos, &tok) in tokens.iter().enumerate() {
            let base = score_base_distribution(
                &self.tables.base_tables,
                self.alpha_bigram,
                self.alpha_trigram,
                prev2,
                prev1,
            );
            let summary = build_word_summary(
                &history,
                &self.tables,
                self.alpha_bigram,
                self.alpha_trigram,
                self.alpha_word,
                self.candidate_k,
            );
            let base_stats = distribution_stats(&base);
            let features =
                standardize_features(extract_features(&base_stats, &summary), &self.standardizer);
            let raw_gate = match &self.gate {
                GateChoice::Heuristic => heuristic_gate(&summary),
                GateChoice::Direct(model) => predict_probability(model, &features),
            };
            let gate = (self.lambda * raw_gate).clamp(0.0, 1.0);
            let mixed = mix_with_candidate_set(&base, &summary.topk_dist, gate);

            if let Some(&sample_index) = lookup.get(&pos) {
                let gold = mixed
                    .get(tok)
                    .copied()
                    .unwrap_or(f64::MIN_POSITIVE)
                    .max(f64::MIN_POSITIVE)
                    .ln();
                sample_gold_logprobs[sample_index] = gold;
                sample_predictions[sample_index] = mixed;
            }

            push_recent(&mut history, tok, self.word_window);
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
            push_recent(&mut self.recent_history, tok, self.word_window);
            self.stream_prev2 = self.stream_prev1;
            self.stream_prev1 = Some(tok);
        }
        Ok(())
    }
}

fn build_word_tables(
    tokens: &[usize],
    trigram_buckets: usize,
    word_buckets: usize,
    word_window: usize,
    suffix_len: usize,
    boundary_markers: usize,
) -> Result<WordTables, String> {
    let base_tables = build_packed_tables(tokens, VOCAB_SIZE, trigram_buckets)?;
    let boundary_mask = select_boundary_markers(tokens, VOCAB_SIZE, boundary_markers)?;
    let mut word_counts = vec![0.0; word_buckets * VOCAB_SIZE];
    let mut word_totals = vec![0.0; word_buckets];
    let mut history = VecDeque::new();

    for &gold in tokens {
        let (suffix, _) = word_context_suffix(&history, &boundary_mask, word_window, suffix_len);
        let bucket = word_bucket(&suffix, word_buckets);
        word_counts[bucket * VOCAB_SIZE + gold] += 1.0;
        word_totals[bucket] += 1.0;
        push_recent(&mut history, gold, word_window.max(suffix_len));
    }

    Ok(WordTables {
        base_tables,
        boundary_mask,
        word_counts,
        word_totals,
        word_buckets,
        word_window,
        suffix_len,
        boundary_markers,
    })
}

fn collect_raw_records(
    tokens: &[usize],
    tables: &WordTables,
    alpha_bigram: f64,
    alpha_trigram: f64,
    alpha_word: f64,
    candidate_k: usize,
    stride: usize,
) -> Vec<RawWordRecord> {
    let mut rows = Vec::new();
    let mut prev2 = None;
    let mut prev1 = None;
    let mut history = VecDeque::new();

    for (pos, &gold) in tokens.iter().enumerate() {
        if pos % stride == 0 {
            let base = score_base_distribution(
                &tables.base_tables,
                alpha_bigram,
                alpha_trigram,
                prev2,
                prev1,
            );
            let summary = build_word_summary_from_state(
                &history,
                tables,
                alpha_bigram,
                alpha_trigram,
                alpha_word,
                candidate_k,
                &base,
            );
            let base_gold_prob = base
                .get(gold)
                .copied()
                .unwrap_or(f64::MIN_POSITIVE)
                .max(f64::MIN_POSITIVE);
            let word_full_gold_prob = summary
                .full_dist
                .get(gold)
                .copied()
                .unwrap_or(f64::MIN_POSITIVE)
                .max(f64::MIN_POSITIVE);
            let word_topk_gold_prob = summary
                .topk_dist
                .get(gold)
                .copied()
                .unwrap_or(f64::MIN_POSITIVE)
                .max(f64::MIN_POSITIVE);
            let base_stats = distribution_stats(&base);
            rows.push(RawWordRecord {
                features: extract_features(&base_stats, &summary),
                base_gold_prob,
                word_full_gold_prob,
                word_topk_gold_prob,
                heuristic_gate: heuristic_gate(&summary),
            });
        }
        push_recent(
            &mut history,
            gold,
            tables.word_window.max(tables.suffix_len),
        );
        prev2 = prev1;
        prev1 = Some(gold);
    }

    rows
}

fn fit_standardizer_from_raw(records: &[RawWordRecord]) -> Standardizer {
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

fn standardize_records(records: &[RawWordRecord], standardizer: &Standardizer) -> Vec<WordRecord> {
    records
        .iter()
        .map(|record| WordRecord {
            features: standardize_features(record.features, standardizer),
            base_gold_prob: record.base_gold_prob,
            word_full_gold_prob: record.word_full_gold_prob,
            word_topk_gold_prob: record.word_topk_gold_prob,
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

fn extract_features(base: &DistributionStats, summary: &WordSummary) -> [f64; FEATURE_DIM] {
    [
        base.top1_prob,
        base.entropy_norm,
        summary.top1_prob,
        summary.topk_mass,
        summary.entropy_norm,
        summary.support as f64 / VOCAB_SIZE as f64,
        summary.boundary_hit,
        summary.suffix_len as f64 / DEFAULT_WORD_WINDOW as f64,
        if base.top1_token == summary.top1_token {
            1.0
        } else {
            0.0
        },
    ]
}

fn heuristic_gate(summary: &WordSummary) -> f64 {
    let boundary_bonus = if summary.boundary_hit > 0.0 { 1.0 } else { 0.6 };
    (summary.topk_mass * (1.0 - summary.entropy_norm) * boundary_bonus).clamp(0.0, 1.0)
}

fn build_word_summary(
    history: &VecDeque<usize>,
    tables: &WordTables,
    alpha_bigram: f64,
    alpha_trigram: f64,
    alpha_word: f64,
    candidate_k: usize,
) -> WordSummary {
    let base = score_word_distribution(history, tables, alpha_bigram, alpha_trigram, alpha_word);
    build_word_summary_from_base(history, tables, candidate_k, &base)
}

fn build_word_summary_from_state(
    history: &VecDeque<usize>,
    tables: &WordTables,
    alpha_bigram: f64,
    alpha_trigram: f64,
    alpha_word: f64,
    candidate_k: usize,
    base: &[f64],
) -> WordSummary {
    let word = score_word_distribution(history, tables, alpha_bigram, alpha_trigram, alpha_word);
    build_word_summary_from_base_with_word(history, tables, candidate_k, base, &word)
}

fn build_word_summary_from_base(
    history: &VecDeque<usize>,
    tables: &WordTables,
    candidate_k: usize,
    full: &[f64],
) -> WordSummary {
    build_word_summary_core(history, tables, candidate_k, full.to_vec())
}

fn build_word_summary_from_base_with_word(
    history: &VecDeque<usize>,
    tables: &WordTables,
    candidate_k: usize,
    _base: &[f64],
    word: &[f64],
) -> WordSummary {
    build_word_summary_core(history, tables, candidate_k, word.to_vec())
}

fn build_word_summary_core(
    history: &VecDeque<usize>,
    tables: &WordTables,
    candidate_k: usize,
    full_dist: Vec<f64>,
) -> WordSummary {
    let (suffix, boundary_hit) = word_context_suffix(
        history,
        &tables.boundary_mask,
        tables.word_window,
        tables.suffix_len,
    );
    let (topk_dist, top1_prob, topk_mass, entropy_norm, support, top1_token) =
        summarize_distribution(&full_dist, candidate_k);
    WordSummary {
        full_dist,
        topk_dist,
        top1_prob,
        topk_mass,
        entropy_norm,
        support,
        top1_token,
        boundary_hit: if boundary_hit { 1.0 } else { 0.0 },
        suffix_len: suffix.len(),
    }
}

fn score_word_distribution(
    history: &VecDeque<usize>,
    tables: &WordTables,
    alpha_bigram: f64,
    alpha_trigram: f64,
    alpha_word: f64,
) -> Vec<f64> {
    let base = &tables.base_tables;
    let base_dist = score_base_distribution(base, alpha_bigram, alpha_trigram, None, None);
    let (suffix, _) = word_context_suffix(
        history,
        &tables.boundary_mask,
        tables.word_window,
        tables.suffix_len,
    );
    let bucket = word_bucket(&suffix, tables.word_buckets);
    let row_start = bucket * VOCAB_SIZE;
    let total = tables.word_totals[bucket];
    if total <= 0.0 {
        return base_dist;
    }
    let denom = (total + alpha_word).max(1e-8);
    let mut out = vec![0.0; VOCAB_SIZE];
    for tok in 0..VOCAB_SIZE {
        out[tok] = (tables.word_counts[row_start + tok] + alpha_word * base_dist[tok]) / denom;
    }
    normalize(&mut out);
    out
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
            let bucket = ((p2 as u64 * WORD_HASH_MUL_A + p1 as u64 * WORD_HASH_MUL_B)
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

fn select_boundary_markers(
    tokens: &[usize],
    vocab_size: usize,
    boundary_markers: usize,
) -> Result<Vec<bool>, String> {
    let mut counts = vec![0usize; vocab_size];
    for &tok in tokens {
        if tok >= vocab_size {
            return Err(format!("token out of vocab: {tok}"));
        }
        counts[tok] += 1;
    }
    let mut order: Vec<usize> = (0..vocab_size).collect();
    order.sort_by(|&a, &b| counts[b].cmp(&counts[a]).then_with(|| a.cmp(&b)));
    let mut mask = vec![false; vocab_size];
    for &tok in order.iter().take(boundary_markers.min(vocab_size)) {
        mask[tok] = true;
    }
    Ok(mask)
}

fn word_context_suffix(
    history: &VecDeque<usize>,
    boundary_mask: &[bool],
    word_window: usize,
    suffix_len: usize,
) -> (Vec<usize>, bool) {
    if history.is_empty() {
        return (Vec::new(), false);
    }
    let window = word_window.min(history.len());
    let mut start = history.len() - window;
    let mut boundary_hit = false;
    for idx in (start..history.len()).rev() {
        let tok = history[idx];
        if boundary_mask.get(tok).copied().unwrap_or(false) {
            start = idx + 1;
            boundary_hit = true;
            break;
        }
    }
    let mut suffix: Vec<usize> = history.iter().skip(start).copied().collect();
    if suffix.len() > suffix_len {
        suffix = suffix[suffix.len() - suffix_len..].to_vec();
    }
    (suffix, boundary_hit)
}

fn word_bucket(suffix: &[usize], word_buckets: usize) -> usize {
    let mut hash: u64 = 0x9E37_79B9_7F4A_7C15;
    for &tok in suffix {
        let mixed = (tok as u64)
            .wrapping_add(WORD_HASH_MUL_A)
            .wrapping_add(hash << 6)
            .wrapping_add(hash >> 2);
        hash ^= mixed;
        hash = hash.wrapping_mul(WORD_HASH_MUL_B);
    }
    (hash % word_buckets as u64) as usize
}

fn push_recent(history: &mut VecDeque<usize>, tok: usize, limit: usize) {
    history.push_back(tok);
    while history.len() > limit {
        history.pop_front();
    }
}

fn tune_lambda<F>(records: &[WordRecord], gate_fn: F) -> f64
where
    F: Fn(&WordRecord) -> f64,
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

fn mean_bits_base(records: &[WordRecord]) -> f64 {
    records
        .iter()
        .map(|record| -record.base_gold_prob.max(UNIFORM_FLOOR).log2())
        .sum::<f64>()
        / records.len() as f64
}

fn mean_bits_word_full(records: &[WordRecord]) -> f64 {
    records
        .iter()
        .map(|record| -record.word_full_gold_prob.max(UNIFORM_FLOOR).log2())
        .sum::<f64>()
        / records.len() as f64
}

fn mean_bits_word_topk(records: &[WordRecord]) -> f64 {
    records
        .iter()
        .map(|record| -record.word_topk_gold_prob.max(UNIFORM_FLOOR).log2())
        .sum::<f64>()
        / records.len() as f64
}

fn mean_bits_with_gate<F>(records: &[WordRecord], lambda: f64, gate_fn: F) -> f64
where
    F: Fn(&WordRecord) -> f64,
{
    records
        .iter()
        .map(|record| {
            let gate = (lambda * gate_fn(record)).clamp(0.0, 1.0);
            let mixed = mix_probabilities(record.base_gold_prob, record.word_topk_gold_prob, gate);
            -mixed.max(UNIFORM_FLOOR).log2()
        })
        .sum::<f64>()
        / records.len() as f64
}

fn oracle_bits_per_token(records: &[WordRecord]) -> f64 {
    records
        .iter()
        .map(|record| {
            let best = record
                .base_gold_prob
                .max(record.word_topk_gold_prob)
                .max(UNIFORM_FLOOR);
            -best.log2()
        })
        .sum::<f64>()
        / records.len() as f64
}

fn word_hit_rate(records: &[WordRecord]) -> f64 {
    records
        .iter()
        .filter(|record| record.word_topk_gold_prob > 0.0)
        .count() as f64
        / records.len() as f64
}

fn word_better_rate(records: &[WordRecord]) -> f64 {
    records
        .iter()
        .filter(|record| record.word_topk_gold_prob > record.base_gold_prob)
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

fn train_logistic_model(records: &[WordRecord], epochs: usize, lr: f64, l2: f64) -> LogisticModel {
    let mut model = LogisticModel {
        weights: [0.0; FEATURE_DIM],
        bias: 0.0,
    };
    for epoch in 0..epochs {
        let rate = lr / (1.0 + epoch as f64 * 0.05);
        for record in records {
            let target = if record.word_topk_gold_prob > record.base_gold_prob {
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

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        let z = (-value).exp();
        1.0 / (1.0 + z)
    } else {
        let z = value.exp();
        z / (1.0 + z)
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
