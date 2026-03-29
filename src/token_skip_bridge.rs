use std::cmp::Ordering;
use std::collections::HashSet;
use std::path::Path;

use crate::data::{take_train_tokens, take_val_tokens};
use crate::packed_memory::{PackedTables, build_packed_tables};
use crate::protocol::{Runner, SampleOutputs};

const VOCAB_SIZE: usize = 1024;
const FEATURE_DIM: usize = 10;
const DEFAULT_SKIP_FAR: usize = 4;
const DEFAULT_SKIP_NEAR: usize = 2;
const SKIP_HASH_MUL_A: u64 = 1_002_583_641;
const SKIP_HASH_MUL_B: u64 = 2_100_523_641;

#[derive(Debug, Clone)]
pub struct TokenSkipBridgeReport {
    pub train_token_budget: usize,
    pub val_token_budget: usize,
    pub skip_far: usize,
    pub skip_near: usize,
    pub trigram_buckets: usize,
    pub skip_buckets: usize,
    pub train_stride: usize,
    pub candidate_k: usize,
    pub train_records: usize,
    pub tune_tokens: usize,
    pub tune_records: usize,
    pub eval_tokens: usize,
    pub eval_records: usize,
    pub tuned_heuristic_lambda: f64,
    pub tuned_direct_lambda: f64,
    pub selected_runtime_gate: String,
    pub tune_bpt_base: f64,
    pub tune_bpt_skip: f64,
    pub tune_bpt_heuristic: f64,
    pub tune_bpt_direct: f64,
    pub tune_bpt_oracle: f64,
    pub tune_oracle_gap: f64,
    pub eval_bpt_base: f64,
    pub eval_bpt_skip: f64,
    pub eval_bpt_heuristic: f64,
    pub eval_bpt_direct: f64,
    pub eval_bpt_oracle: f64,
    pub eval_oracle_gap: f64,
    pub eval_skip_candidate_hit_rate: f64,
    pub eval_skip_better_rate: f64,
    pub eval_candidate_overlap_rate: f64,
}

#[derive(Debug, Clone)]
pub struct TrainedTokenSkipBridge {
    report: TokenSkipBridgeReport,
    runner: TokenSkipBridgeRunner,
    eval_tokens: Vec<usize>,
}

impl TrainedTokenSkipBridge {
    pub fn report(&self) -> &TokenSkipBridgeReport {
        &self.report
    }

    pub fn runner(&self) -> &TokenSkipBridgeRunner {
        &self.runner
    }

    pub fn eval_tokens(&self) -> &[usize] {
        &self.eval_tokens
    }
}

#[derive(Debug, Clone)]
pub struct TokenSkipBridgeRunner {
    base_tables: PackedTables,
    skip_tables: SkipTables,
    alpha_bigram: f64,
    alpha_trigram: f64,
    alpha_skip: f64,
    standardizer: Standardizer,
    direct_model: LogisticModel,
    heuristic_lambda: f64,
    direct_lambda: f64,
    runtime_gate: RuntimeGate,
    candidate_k: usize,
    skip_far: usize,
    skip_near: usize,
    history_limit: usize,
    stream_history: Vec<usize>,
}

#[derive(Debug, Clone)]
struct SkipTables {
    vocab_size: usize,
    counts: Vec<f64>,
    totals: Vec<f64>,
    buckets: usize,
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
    skip_candidate_gold_prob: f64,
    heuristic_prob: f64,
    overlap_fraction: f64,
}

#[derive(Debug, Clone)]
struct TokenRecord {
    features: [f64; FEATURE_DIM],
    base_gold_prob: f64,
    skip_candidate_gold_prob: f64,
    heuristic_prob: f64,
    overlap_fraction: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RuntimeGate {
    Heuristic,
    Direct,
}

#[derive(Debug, Clone, Copy)]
struct Adjustment {
    gold: usize,
    subtract_unigram: bool,
    subtract_bigram: bool,
    subtract_trigram: bool,
    subtract_skip: bool,
}

#[derive(Debug, Clone)]
struct TopKStats {
    top1_prob: f64,
    topk_mass: f64,
    entropy: f64,
    margin: f64,
    support: Vec<(usize, f64)>,
}

pub fn train_token_skip_bridge_from_data_root(
    root: &Path,
    train_token_budget: usize,
    trigram_buckets: usize,
    skip_buckets: usize,
    val_token_budget: usize,
    alpha_bigram: f64,
    alpha_trigram: f64,
    alpha_skip: f64,
    train_stride: usize,
    candidate_k: usize,
) -> Result<TrainedTokenSkipBridge, String> {
    if train_stride == 0 {
        return Err("train_stride must be positive".to_string());
    }
    if candidate_k == 0 {
        return Err("candidate_k must be positive".to_string());
    }

    let train_tokens = take_train_tokens(root, train_token_budget)?;
    let val_tokens = take_val_tokens(root, val_token_budget)?;
    if val_tokens.len() < DEFAULT_SKIP_FAR + 8 {
        return Err("need at least 12 validation tokens for the skip bridge".to_string());
    }

    let split = (val_tokens.len() / 2).max(1);
    let tune_tokens = val_tokens[..split].to_vec();
    let eval_tokens = val_tokens[split..].to_vec();
    if eval_tokens.is_empty() {
        return Err("validation split left no eval tokens".to_string());
    }

    let base_tables = build_packed_tables(&train_tokens, VOCAB_SIZE, trigram_buckets)?;
    let skip_tables = build_skip_tables(&train_tokens, VOCAB_SIZE, skip_buckets)?;

    let train_raw = collect_raw_records(
        &train_tokens,
        &base_tables,
        &skip_tables,
        alpha_bigram,
        alpha_trigram,
        alpha_skip,
        true,
        train_stride,
        candidate_k,
    );
    if train_raw.is_empty() {
        return Err("no token skip bridge training records collected".to_string());
    }

    let standardizer = fit_standardizer_from_raw(&train_raw);
    let train_records = standardize_records(&train_raw, &standardizer);
    let direct_model = train_direct_gate(&train_records, 60, 0.15, 1e-4);

    let tune_raw = collect_raw_records(
        &tune_tokens,
        &base_tables,
        &skip_tables,
        alpha_bigram,
        alpha_trigram,
        alpha_skip,
        false,
        1,
        candidate_k,
    );
    let eval_raw = collect_raw_records(
        &eval_tokens,
        &base_tables,
        &skip_tables,
        alpha_bigram,
        alpha_trigram,
        alpha_skip,
        false,
        1,
        candidate_k,
    );

    if tune_raw.is_empty() {
        return Err("no token skip bridge tune records collected".to_string());
    }
    if eval_raw.is_empty() {
        return Err("no token skip bridge eval records collected".to_string());
    }

    let tune_records = standardize_records(&tune_raw, &standardizer);
    let eval_records = standardize_records(&eval_raw, &standardizer);

    let tuned_heuristic_lambda = tune_lambda(&tune_records, |record| record.heuristic_prob);
    let tuned_direct_lambda = tune_lambda(&tune_records, |record| {
        predict_probability(&direct_model, &record.features)
    });

    let tune_bpt_base = mean_bits_base(&tune_records);
    let tune_bpt_skip = mean_bits_skip(&tune_records);
    let tune_bpt_heuristic = mean_bits_with_gate(&tune_records, tuned_heuristic_lambda, |record| {
        record.heuristic_prob
    });
    let tune_bpt_direct = mean_bits_with_gate(&tune_records, tuned_direct_lambda, |record| {
        predict_probability(&direct_model, &record.features)
    });
    let tune_bpt_oracle = oracle_bits_per_token(&tune_records);

    let eval_bpt_base = mean_bits_base(&eval_records);
    let eval_bpt_skip = mean_bits_skip(&eval_records);
    let eval_bpt_heuristic = mean_bits_with_gate(&eval_records, tuned_heuristic_lambda, |record| {
        record.heuristic_prob
    });
    let eval_bpt_direct = mean_bits_with_gate(&eval_records, tuned_direct_lambda, |record| {
        predict_probability(&direct_model, &record.features)
    });
    let eval_bpt_oracle = oracle_bits_per_token(&eval_records);

    let heuristic_selected = tune_bpt_heuristic <= tune_bpt_direct;
    let runtime_gate = if heuristic_selected {
        RuntimeGate::Heuristic
    } else {
        RuntimeGate::Direct
    };
    let selected_runtime_gate = if heuristic_selected {
        "heuristic"
    } else {
        "direct"
    }
    .to_string();

    let report = TokenSkipBridgeReport {
        train_token_budget,
        val_token_budget,
        skip_far: DEFAULT_SKIP_FAR,
        skip_near: DEFAULT_SKIP_NEAR,
        trigram_buckets,
        skip_buckets,
        train_stride,
        candidate_k,
        train_records: train_records.len(),
        tune_tokens: tune_tokens.len(),
        tune_records: tune_records.len(),
        eval_tokens: eval_tokens.len(),
        eval_records: eval_records.len(),
        tuned_heuristic_lambda,
        tuned_direct_lambda,
        selected_runtime_gate,
        tune_bpt_base,
        tune_bpt_skip,
        tune_bpt_heuristic,
        tune_bpt_direct,
        tune_bpt_oracle,
        tune_oracle_gap: tune_bpt_base - tune_bpt_oracle,
        eval_bpt_base,
        eval_bpt_skip,
        eval_bpt_heuristic,
        eval_bpt_direct,
        eval_bpt_oracle,
        eval_oracle_gap: eval_bpt_base - eval_bpt_oracle,
        eval_skip_candidate_hit_rate: skip_candidate_hit_rate(&eval_records),
        eval_skip_better_rate: skip_better_rate(&eval_records),
        eval_candidate_overlap_rate: candidate_overlap_rate(&eval_records),
    };

    let runner = TokenSkipBridgeRunner {
        base_tables,
        skip_tables,
        alpha_bigram,
        alpha_trigram,
        alpha_skip,
        standardizer,
        direct_model,
        heuristic_lambda: tuned_heuristic_lambda,
        direct_lambda: tuned_direct_lambda,
        runtime_gate,
        candidate_k,
        skip_far: DEFAULT_SKIP_FAR,
        skip_near: DEFAULT_SKIP_NEAR,
        history_limit: DEFAULT_SKIP_FAR.max(2),
        stream_history: Vec::new(),
    };

    Ok(TrainedTokenSkipBridge {
        report,
        runner,
        eval_tokens,
    })
}

pub fn run_token_skip_bridge_from_data_root(
    root: &Path,
    train_token_budget: usize,
    trigram_buckets: usize,
    skip_buckets: usize,
    val_token_budget: usize,
    alpha_bigram: f64,
    alpha_trigram: f64,
    alpha_skip: f64,
    train_stride: usize,
    candidate_k: usize,
) -> Result<TokenSkipBridgeReport, String> {
    Ok(train_token_skip_bridge_from_data_root(
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
    )?
    .report)
}

pub fn render_token_skip_bridge_report(report: &TokenSkipBridgeReport) -> String {
    let mut out = String::new();
    out.push_str("chronohorn_token_skip_bridge\n");
    out.push_str(&format!(
        "train_token_budget: {}\n",
        report.train_token_budget
    ));
    out.push_str(&format!("val_token_budget: {}\n", report.val_token_budget));
    out.push_str(&format!("skip_far: {}\n", report.skip_far));
    out.push_str(&format!("skip_near: {}\n", report.skip_near));
    out.push_str(&format!("trigram_buckets: {}\n", report.trigram_buckets));
    out.push_str(&format!("skip_buckets: {}\n", report.skip_buckets));
    out.push_str(&format!("train_stride: {}\n", report.train_stride));
    out.push_str(&format!("candidate_k: {}\n", report.candidate_k));
    out.push_str(&format!("train_records: {}\n", report.train_records));
    out.push_str(&format!("tune_tokens: {}\n", report.tune_tokens));
    out.push_str(&format!("tune_records: {}\n", report.tune_records));
    out.push_str(&format!("eval_tokens: {}\n", report.eval_tokens));
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
    out.push_str(&format!("tune_bpt_base: {:.6}\n", report.tune_bpt_base));
    out.push_str(&format!("tune_bpt_skip: {:.6}\n", report.tune_bpt_skip));
    out.push_str(&format!(
        "tune_bpt_heuristic: {:.6}\n",
        report.tune_bpt_heuristic
    ));
    out.push_str(&format!("tune_bpt_direct: {:.6}\n", report.tune_bpt_direct));
    out.push_str(&format!("tune_bpt_oracle: {:.6}\n", report.tune_bpt_oracle));
    out.push_str(&format!("tune_oracle_gap: {:.6}\n", report.tune_oracle_gap));
    out.push_str(&format!("eval_bpt_base: {:.6}\n", report.eval_bpt_base));
    out.push_str(&format!("eval_bpt_skip: {:.6}\n", report.eval_bpt_skip));
    out.push_str(&format!(
        "eval_bpt_heuristic: {:.6}\n",
        report.eval_bpt_heuristic
    ));
    out.push_str(&format!("eval_bpt_direct: {:.6}\n", report.eval_bpt_direct));
    out.push_str(&format!("eval_bpt_oracle: {:.6}\n", report.eval_bpt_oracle));
    out.push_str(&format!("eval_oracle_gap: {:.6}\n", report.eval_oracle_gap));
    out.push_str(&format!(
        "eval_skip_candidate_hit_rate: {:.6}\n",
        report.eval_skip_candidate_hit_rate
    ));
    out.push_str(&format!(
        "eval_skip_better_rate: {:.6}\n",
        report.eval_skip_better_rate
    ));
    out.push_str(&format!(
        "eval_candidate_overlap_rate: {:.6}\n",
        report.eval_candidate_overlap_rate
    ));
    out
}

impl Runner for TokenSkipBridgeRunner {
    fn name(&self) -> &'static str {
        "TokenSkipBridgeRunner"
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
                None,
            );
            let skip_prev_far = self.context_back(tokens, pos, self.skip_far);
            let skip_prev_near = self.context_back(tokens, pos, self.skip_near);
            if let (Some(far), Some(near)) = (skip_prev_far, skip_prev_near) {
                let skip = score_skip_distribution(
                    &self.skip_tables,
                    &self.base_tables,
                    self.alpha_skip,
                    far,
                    near,
                    None,
                );
                let base_stats = summarize_distribution(&base, self.candidate_k);
                let skip_stats = summarize_distribution(&skip, self.candidate_k);
                let base_context_total = base_prev1
                    .map(|p1| self.base_tables.bigram_totals[p1])
                    .unwrap_or(0.0);
                let skip_context_total = self.skip_tables.context_total(far, near);
                let gate_prob = match self.runtime_gate {
                    RuntimeGate::Heuristic => (self.heuristic_lambda
                        * heuristic_probability_from_stats(
                            &base_stats,
                            &skip_stats,
                            base_context_total,
                            skip_context_total,
                            self.candidate_k,
                        ))
                    .clamp(0.0, 1.0),
                    RuntimeGate::Direct => {
                        let features = standardize_features(
                            extract_features(
                                &base_stats,
                                &skip_stats,
                                base_context_total,
                                skip_context_total,
                            ),
                            &self.standardizer,
                        );
                        (self.direct_lambda * predict_probability(&self.direct_model, &features))
                            .clamp(0.0, 1.0)
                    }
                };
                let candidate =
                    candidate_distribution_from_support(skip.len(), &skip_stats.support);
                let mixed = mix_with_candidate_set(&base, &candidate, gate_prob);
                let gold = mixed
                    .get(tokens[pos])
                    .copied()
                    .unwrap_or(f64::MIN_POSITIVE)
                    .max(f64::MIN_POSITIVE)
                    .ln();
                sample_gold_logprobs.push(gold);
                sample_predictions.push(mixed);
            } else {
                let gold = base
                    .get(tokens[pos])
                    .copied()
                    .unwrap_or(f64::MIN_POSITIVE)
                    .max(f64::MIN_POSITIVE)
                    .ln();
                sample_gold_logprobs.push(gold);
                sample_predictions.push(base);
            }
        }
        Ok(SampleOutputs {
            sample_predictions,
            sample_gold_logprobs: Some(sample_gold_logprobs),
        })
    }

    fn adapt_chunk(&mut self, tokens: &[usize]) -> Result<(), String> {
        self.stream_history.extend_from_slice(tokens);
        if self.stream_history.len() > self.history_limit {
            let excess = self.stream_history.len() - self.history_limit;
            self.stream_history.drain(0..excess);
        }
        Ok(())
    }
}

impl TokenSkipBridgeRunner {
    fn context_back(&self, tokens: &[usize], pos: usize, back: usize) -> Option<usize> {
        token_back(tokens, pos, back, &self.stream_history)
    }
}

fn collect_raw_records(
    tokens: &[usize],
    base_tables: &PackedTables,
    skip_tables: &SkipTables,
    alpha_bigram: f64,
    alpha_trigram: f64,
    alpha_skip: f64,
    exclude_current: bool,
    stride: usize,
    candidate_k: usize,
) -> Vec<RawTokenRecord> {
    let mut rows = Vec::new();
    for pos in 0..tokens.len() {
        if pos % stride != 0 || pos < DEFAULT_SKIP_FAR {
            continue;
        }
        let base_prev2 = if pos >= 2 {
            Some(tokens[pos - 2])
        } else {
            None
        };
        let base_prev1 = if pos >= 1 {
            Some(tokens[pos - 1])
        } else {
            None
        };
        let skip_prev_far = Some(tokens[pos - DEFAULT_SKIP_FAR]);
        let skip_prev_near = Some(tokens[pos - DEFAULT_SKIP_NEAR]);
        let gold = tokens[pos];

        let base_adjustment = if exclude_current {
            Some(Adjustment {
                gold,
                subtract_unigram: pos >= 1,
                subtract_bigram: pos >= 1,
                subtract_trigram: pos >= 2,
                subtract_skip: false,
            })
        } else {
            None
        };
        let skip_adjustment = if exclude_current {
            Some(Adjustment {
                gold,
                subtract_unigram: pos >= 1,
                subtract_bigram: false,
                subtract_trigram: false,
                subtract_skip: true,
            })
        } else {
            None
        };

        let base = score_base_distribution(
            base_tables,
            alpha_bigram,
            alpha_trigram,
            base_prev2,
            base_prev1,
            base_adjustment,
        );
        let skip = score_skip_distribution(
            skip_tables,
            base_tables,
            alpha_skip,
            skip_prev_far.unwrap(),
            skip_prev_near.unwrap(),
            skip_adjustment,
        );

        let base_stats = summarize_distribution(&base, candidate_k);
        let skip_stats = summarize_distribution(&skip, candidate_k);
        let base_context_total = base_prev1
            .map(|p1| base_tables.bigram_totals[p1])
            .unwrap_or(0.0);
        let skip_context_total =
            skip_tables.context_total(skip_prev_far.unwrap(), skip_prev_near.unwrap());
        let base_gold_prob = base
            .get(gold)
            .copied()
            .unwrap_or(f64::MIN_POSITIVE)
            .max(f64::MIN_POSITIVE);
        let skip_candidate_gold_prob = gold_prob_from_support(&skip_stats.support, gold);
        let heuristic_prob = heuristic_probability_from_stats(
            &base_stats,
            &skip_stats,
            base_context_total,
            skip_context_total,
            candidate_k,
        );
        let overlap_fraction = support_overlap_fraction(&base_stats.support, &skip_stats.support);
        let features = extract_features(
            &base_stats,
            &skip_stats,
            base_context_total,
            skip_context_total,
        );
        rows.push(RawTokenRecord {
            features,
            base_gold_prob,
            skip_candidate_gold_prob,
            heuristic_prob,
            overlap_fraction,
        });
    }
    rows
}

fn build_skip_tables(
    tokens: &[usize],
    vocab_size: usize,
    skip_buckets: usize,
) -> Result<SkipTables, String> {
    if skip_buckets == 0 {
        return Err("skip_buckets must be positive".to_string());
    }
    if tokens.len() < DEFAULT_SKIP_FAR + 1 {
        return Err("need at least 5 tokens to build skip tables".to_string());
    }
    let mut counts = vec![0.0; skip_buckets * vocab_size];
    let mut totals = vec![0.0; skip_buckets];
    for pos in DEFAULT_SKIP_FAR..tokens.len() {
        let far = tokens[pos - DEFAULT_SKIP_FAR];
        let near = tokens[pos - DEFAULT_SKIP_NEAR];
        let next = tokens[pos];
        if far >= vocab_size || near >= vocab_size || next >= vocab_size {
            return Err(format!("token out of vocab: {far},{near}->{next}"));
        }
        let bucket = skip_bucket(skip_buckets, far, near);
        counts[bucket * vocab_size + next] += 1.0;
        totals[bucket] += 1.0;
    }
    Ok(SkipTables {
        vocab_size,
        counts,
        totals,
        buckets: skip_buckets,
    })
}

fn score_base_distribution(
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
        let bucket = skip_bucket(tables.trigram_buckets, p2, p1);
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

fn score_skip_distribution(
    skip_tables: &SkipTables,
    base_tables: &PackedTables,
    alpha_skip: f64,
    prev_far: usize,
    prev_near: usize,
    adjustment: Option<Adjustment>,
) -> Vec<f64> {
    let vocab = skip_tables.vocab_size;
    let unigram_total = (base_tables.token_budget.saturating_sub(1) as f64
        - if adjustment.is_some_and(|adj| adj.subtract_unigram) {
            1.0
        } else {
            0.0
        })
    .max(1.0);
    let mut p_uni = vec![0.0; vocab];
    for tok in 0..vocab {
        let mut count =
            base_tables.unigram_probs[tok] * (base_tables.token_budget.saturating_sub(1) as f64);
        if let Some(adj) = adjustment {
            if adj.subtract_unigram && tok == adj.gold {
                count = (count - 1.0).max(0.0);
            }
        }
        p_uni[tok] = count / unigram_total;
    }

    let bucket = skip_bucket(skip_tables.buckets, prev_far, prev_near);
    let row_start = bucket * vocab;
    let mut total = skip_tables.totals[bucket];
    if let Some(adj) = adjustment {
        if adj.subtract_skip {
            total = (total - 1.0).max(0.0);
        }
    }
    let denom = (total + alpha_skip).max(1e-8);
    let mut out = vec![0.0; vocab];
    for tok in 0..vocab {
        let mut count = skip_tables.counts[row_start + tok];
        if let Some(adj) = adjustment {
            if adj.subtract_skip && tok == adj.gold {
                count = (count - 1.0).max(0.0);
            }
        }
        out[tok] = (count + alpha_skip * p_uni[tok]) / denom;
    }
    normalize(&mut out);
    out
}

fn extract_features(
    base: &TopKStats,
    skip: &TopKStats,
    base_context_total: f64,
    skip_context_total: f64,
) -> [f64; FEATURE_DIM] {
    let overlap = support_overlap_fraction(&base.support, &skip.support);
    let context_delta = (1.0 + skip_context_total).ln() - (1.0 + base_context_total).ln();
    [
        base.top1_prob,
        base.topk_mass,
        base.entropy,
        base.margin,
        skip.top1_prob,
        skip.topk_mass,
        skip.entropy,
        skip.margin,
        overlap,
        context_delta,
    ]
}

fn summarize_distribution(distribution: &[f64], candidate_k: usize) -> TopKStats {
    let mut indexed = distribution
        .iter()
        .enumerate()
        .map(|(index, &value)| (index, value))
        .collect::<Vec<_>>();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    let k = candidate_k.min(indexed.len().max(1));
    let support = indexed.iter().take(k).copied().collect::<Vec<_>>();
    let top1_prob = indexed.first().map(|(_, p)| *p).unwrap_or(0.0);
    let top2_prob = indexed.get(1).map(|(_, p)| *p).unwrap_or(0.0);
    let topk_mass = support.iter().map(|(_, p)| *p).sum::<f64>();
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
    TopKStats {
        top1_prob,
        topk_mass,
        entropy: normalized_entropy,
        margin: top1_prob - top2_prob,
        support,
    }
}

fn candidate_distribution_from_support(len: usize, support: &[(usize, f64)]) -> Vec<f64> {
    let total = support.iter().map(|(_, p)| *p).sum::<f64>().max(1e-12);
    let mut out = vec![0.0; len];
    for (index, value) in support {
        if *index < len {
            out[*index] = *value / total;
        }
    }
    out
}

fn gold_prob_from_support(support: &[(usize, f64)], gold: usize) -> f64 {
    let total = support.iter().map(|(_, p)| *p).sum::<f64>().max(1e-12);
    support
        .iter()
        .find(|(index, _)| *index == gold)
        .map(|(_, value)| *value / total)
        .unwrap_or(0.0)
}

fn support_overlap_fraction(left: &[(usize, f64)], right: &[(usize, f64)]) -> f64 {
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }
    let right_indices = right
        .iter()
        .map(|(index, _)| *index)
        .collect::<HashSet<_>>();
    let overlap = left
        .iter()
        .filter(|(index, _)| right_indices.contains(index))
        .count();
    overlap as f64 / left.len().min(right.len()) as f64
}

fn heuristic_probability_from_stats(
    base: &TopKStats,
    skip: &TopKStats,
    base_context_total: f64,
    skip_context_total: f64,
    candidate_k: usize,
) -> f64 {
    let overlap = support_overlap_fraction(&base.support, &skip.support);
    let context_delta = (1.0 + skip_context_total).ln() - (1.0 + base_context_total).ln();
    let support_bias = (skip.topk_mass - base.topk_mass) + 0.5 * (base.entropy - skip.entropy);
    let margin_bias = skip.margin - base.margin;
    let score = 2.0 * support_bias
        + 1.25 * margin_bias
        + 0.75 * overlap
        + 0.25 * context_delta
        + 0.1 * candidate_k as f64;
    sigmoid(score)
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
            skip_candidate_gold_prob: record.skip_candidate_gold_prob,
            heuristic_prob: record.heuristic_prob,
            overlap_fraction: record.overlap_fraction,
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

fn train_direct_gate(records: &[TokenRecord], epochs: usize, lr: f64, l2: f64) -> LogisticModel {
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
            let mixed = ((1.0 - gate) * record.base_gold_prob
                + gate * record.skip_candidate_gold_prob)
                .max(1e-12);
            let dloss_dgate =
                (record.base_gold_prob - record.skip_candidate_gold_prob) / (mixed * ln2);
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
    sigmoid(z)
}

fn tune_lambda(records: &[TokenRecord], gate_fn: impl Fn(&TokenRecord) -> f64) -> f64 {
    let mut best_lambda = 0.0;
    let mut best_bpt = f64::INFINITY;
    for step in 0..=12 {
        let lambda = step as f64 / 10.0;
        let bpt = mean_bits_with_gate(records, lambda, &gate_fn);
        if bpt < best_bpt {
            best_bpt = bpt;
            best_lambda = lambda;
        }
    }
    best_lambda
}

fn mean_bits_base(records: &[TokenRecord]) -> f64 {
    records
        .iter()
        .map(|record| -record.base_gold_prob.max(1e-12).log2())
        .sum::<f64>()
        / records.len() as f64
}

fn mean_bits_skip(records: &[TokenRecord]) -> f64 {
    records
        .iter()
        .map(|record| -record.skip_candidate_gold_prob.max(1e-12).log2())
        .sum::<f64>()
        / records.len() as f64
}

fn mean_bits_with_gate(
    records: &[TokenRecord],
    lambda: f64,
    gate_fn: impl Fn(&TokenRecord) -> f64,
) -> f64 {
    let mut bits = 0.0;
    for record in records {
        let gate = (lambda * gate_fn(record)).clamp(0.0, 1.0);
        let mixed = (1.0 - gate) * record.base_gold_prob
            + gate * record.skip_candidate_gold_prob.max(1e-12);
        bits += -mixed.max(1e-12).log2();
    }
    bits / records.len() as f64
}

fn oracle_bits_per_token(records: &[TokenRecord]) -> f64 {
    records
        .iter()
        .map(|record| {
            let best = record
                .base_gold_prob
                .max(record.skip_candidate_gold_prob)
                .max(1e-12);
            -best.log2()
        })
        .sum::<f64>()
        / records.len() as f64
}

fn skip_candidate_hit_rate(records: &[TokenRecord]) -> f64 {
    records
        .iter()
        .filter(|record| record.skip_candidate_gold_prob > 0.0)
        .count() as f64
        / records.len() as f64
}

fn skip_better_rate(records: &[TokenRecord]) -> f64 {
    records
        .iter()
        .filter(|record| record.skip_candidate_gold_prob > record.base_gold_prob)
        .count() as f64
        / records.len() as f64
}

fn candidate_overlap_rate(records: &[TokenRecord]) -> f64 {
    if records.is_empty() {
        return 0.0;
    }
    let sum = records
        .iter()
        .map(|record| record.overlap_fraction)
        .sum::<f64>();
    sum / records.len() as f64
}

fn mix_with_candidate_set(base: &[f64], candidate: &[f64], gate: f64) -> Vec<f64> {
    if gate <= 0.0 {
        return base.to_vec();
    }
    let mut out = Vec::with_capacity(base.len());
    for (index, &base_prob) in base.iter().enumerate() {
        let candidate_prob = candidate.get(index).copied().unwrap_or(0.0);
        out.push((1.0 - gate) * base_prob + gate * candidate_prob);
    }
    normalize_vec(out)
}

fn normalize_vec(mut values: Vec<f64>) -> Vec<f64> {
    let total: f64 = values.iter().sum();
    let denom = total.max(f64::EPSILON);
    for value in &mut values {
        *value /= denom;
    }
    values
}

fn token_back(tokens: &[usize], pos: usize, back: usize, history: &[usize]) -> Option<usize> {
    if pos >= back {
        Some(tokens[pos - back])
    } else {
        let need = back - pos;
        if history.len() < need {
            None
        } else {
            Some(history[history.len() - need])
        }
    }
}

fn skip_bucket(skip_buckets: usize, prev_far: usize, prev_near: usize) -> usize {
    ((prev_far as u64 * SKIP_HASH_MUL_A + prev_near as u64 * SKIP_HASH_MUL_B) % skip_buckets as u64)
        as usize
}

impl SkipTables {
    fn context_total(&self, prev_far: usize, prev_near: usize) -> f64 {
        let bucket = skip_bucket(self.buckets, prev_far, prev_near);
        self.totals[bucket]
    }
}

fn normalize(values: &mut [f64]) {
    let total: f64 = values.iter().sum();
    let denom = total.max(f64::EPSILON);
    for value in values {
        *value /= denom;
    }
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}
