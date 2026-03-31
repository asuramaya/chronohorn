use std::collections::{HashMap, VecDeque};
use std::path::Path;

use chronohorn_causal_bank::checkpoint::{
    TokenConker3CheckpointRunner, load_token_conker3_checkpoint_runner_and_metadata,
};
use chronohorn_core::data::{compute_tokens_per_byte, take_val_tokens};
use chronohorn_core::protocol::{Runner, SampleOutputs};
use serde::Serialize;

const DEFAULT_BUCKETS: usize = 4_194_304;
const MIN_ORDER: usize = 2;
const MAX_ORDER: usize = 10;
const MIX_BASE: f64 = 0.05;
const MIX_SPAN: f64 = 0.25;
const ENTROPY_MIDPOINT: f64 = 4.5;
const ENTROPY_SLOPE: f64 = 1.5;
const MIN_ROW_TOTAL: u32 = 4;
const RESIDUAL_CAP: f64 = 4.0;
const DEFAULT_EVAL_CHUNK: usize = 256;

#[derive(Debug, Clone, Serialize)]
pub struct TokenConker3NgramCheckpointReport {
    pub checkpoint_path: String,
    pub summary_path: String,
    pub train_seq_len: usize,
    pub eval_tokens: usize,
    pub scoring_mode: String,
    pub ngram_min_order: usize,
    pub ngram_max_order: usize,
    pub ngram_buckets: usize,
    pub mix_base: f64,
    pub mix_span: f64,
    pub entropy_midpoint: f64,
    pub entropy_slope: f64,
    pub min_row_total: u32,
    pub residual_cap: f64,
    pub eval_bpt_neural: f64,
    pub eval_bpt_mixed: f64,
    pub eval_tokens_per_byte: Option<f64>,
    pub eval_bytes_per_token: Option<f64>,
    pub eval_bpb_neural: Option<f64>,
    pub eval_bpb_mixed: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct LoadedTokenConker3NgramCheckpoint {
    report: TokenConker3NgramCheckpointReport,
    runner: TokenConker3NgramCheckpointRunner,
    eval_tokens: Vec<usize>,
}

impl LoadedTokenConker3NgramCheckpoint {
    pub fn report(&self) -> &TokenConker3NgramCheckpointReport {
        &self.report
    }

    pub fn runner(&self) -> &TokenConker3NgramCheckpointRunner {
        &self.runner
    }

    pub fn eval_tokens(&self) -> &[usize] {
        &self.eval_tokens
    }
}

#[derive(Debug, Clone)]
pub struct TokenConker3NgramCheckpointRunner {
    base: TokenConker3CheckpointRunner,
    ngram: NgramExpertState,
}

#[derive(Debug, Clone, Copy)]
struct NgramExpertConfig {
    min_order: usize,
    max_order: usize,
    buckets: usize,
    mix_base: f64,
    mix_span: f64,
    entropy_midpoint: f64,
    entropy_slope: f64,
    min_row_total: u32,
    residual_cap: f64,
}

impl Default for NgramExpertConfig {
    fn default() -> Self {
        Self {
            min_order: MIN_ORDER,
            max_order: MAX_ORDER,
            buckets: DEFAULT_BUCKETS,
            mix_base: MIX_BASE,
            mix_span: MIX_SPAN,
            entropy_midpoint: ENTROPY_MIDPOINT,
            entropy_slope: ENTROPY_SLOPE,
            min_row_total: MIN_ROW_TOTAL,
            residual_cap: RESIDUAL_CAP,
        }
    }
}

#[derive(Debug, Clone)]
struct NgramExpertState {
    config: NgramExpertConfig,
    vocab_size: usize,
    history: VecDeque<usize>,
    tables: Vec<HashMap<u32, SparseCounter>>,
}

#[derive(Debug, Clone, Default)]
struct SparseCounter {
    total: u32,
    counts: Vec<(usize, u32)>,
}

impl SparseCounter {
    fn increment(&mut self, token: usize) {
        self.total = self.total.saturating_add(1);
        if let Some((_, count)) = self
            .counts
            .iter_mut()
            .find(|(candidate, _)| *candidate == token)
        {
            *count = count.saturating_add(1);
        } else {
            self.counts.push((token, 1));
        }
    }
}

impl NgramExpertState {
    fn new(vocab_size: usize, config: NgramExpertConfig) -> Self {
        let context_lengths = config.max_order.saturating_sub(config.min_order) + 1;
        Self {
            config,
            vocab_size,
            history: VecDeque::with_capacity(config.max_order.saturating_sub(1)),
            tables: (0..context_lengths).map(|_| HashMap::new()).collect(),
        }
    }

    fn observe_token(&mut self, token: usize) {
        if token >= self.vocab_size {
            return;
        }
        let max_context = self
            .history
            .len()
            .min(self.config.max_order.saturating_sub(1));
        for context_len in self.config.min_order.saturating_sub(1)..=max_context {
            if context_len == 0 {
                continue;
            }
            let bucket = self.context_bucket_from_history(&self.history, context_len);
            let table_index = context_len - (self.config.min_order - 1);
            self.tables[table_index]
                .entry(bucket)
                .or_default()
                .increment(token);
        }
        self.advance_history(token);
    }

    fn advance_history(&mut self, token: usize) {
        push_history(
            &mut self.history,
            self.config.max_order.saturating_sub(1),
            token,
        );
    }

    fn lookup_counter_from_history<'a>(
        &'a self,
        history: &VecDeque<usize>,
    ) -> Option<(&'a SparseCounter, usize)> {
        let max_context = history.len().min(self.config.max_order.saturating_sub(1));
        for context_len in (self.config.min_order.saturating_sub(1)..=max_context).rev() {
            if context_len == 0 {
                continue;
            }
            let bucket = self.context_bucket_from_history(history, context_len);
            let table_index = context_len - (self.config.min_order - 1);
            if let Some(row) = self.tables[table_index].get(&bucket) {
                if row.total >= self.config.min_row_total {
                    return Some((row, context_len + 1));
                }
            }
        }
        None
    }

    fn context_bucket_from_history(&self, history: &VecDeque<usize>, context_len: usize) -> u32 {
        let mut hash = 0xcbf2_9ce4_8422_2325u64 ^ (context_len as u64);
        let start = history.len().saturating_sub(context_len);
        for &token in history.iter().skip(start) {
            hash ^= token as u64 + 0x9e37_79b9;
            hash = hash.wrapping_mul(0x1000_0000_01b3);
        }
        (hash % self.config.buckets as u64) as u32
    }
}

impl Runner for TokenConker3NgramCheckpointRunner {
    fn name(&self) -> &'static str {
        "TokenConker3NgramCheckpointRunner"
    }

    fn vocab_size(&self) -> usize {
        self.base.vocab_size()
    }

    fn score_chunk(
        &self,
        tokens: &[usize],
        sample_positions: &[usize],
    ) -> Result<SampleOutputs, String> {
        let mut wanted = vec![Vec::new(); tokens.len()];
        for (sample_index, &pos) in sample_positions.iter().enumerate() {
            if pos >= tokens.len() {
                return Err(format!("sample position {pos} out of bounds"));
            }
            wanted[pos].push(sample_index);
        }

        let mut sample_predictions = vec![Vec::new(); sample_positions.len()];
        let mut sample_gold_logprobs = vec![0.0; sample_positions.len()];
        let (mut base_state, mut base_history) = self.base.snapshot_runtime();
        let mut score_history = self.ngram.history.clone();

        for (pos, &token) in tokens.iter().enumerate() {
            if !wanted[pos].is_empty() {
                let neural_logits = self
                    .base
                    .predict_combined_logits_from(&base_state, &base_history)?;
                let row = self
                    .ngram
                    .lookup_counter_from_history(&score_history)
                    .map(|(counter, _)| counter);
                let alpha = entropy_residual_weight_from_logits(&neural_logits, self.ngram.config);
                let mixed_logits = apply_ngram_residual_logits(
                    &neural_logits,
                    row,
                    alpha,
                    self.base.vocab_size(),
                    self.ngram.config.residual_cap,
                );
                let mixed = softmax_logits_f64(&mixed_logits);
                let gold = mixed
                    .get(token)
                    .copied()
                    .unwrap_or(f64::MIN_POSITIVE)
                    .max(f64::MIN_POSITIVE)
                    .ln();
                for &sample_index in &wanted[pos] {
                    sample_predictions[sample_index] = mixed.clone();
                    sample_gold_logprobs[sample_index] = gold;
                }
            }
            self.base
                .advance_runtime_state(&mut base_state, &mut base_history, token)?;
            push_history(
                &mut score_history,
                self.ngram.config.max_order.saturating_sub(1),
                token,
            );
        }

        Ok(SampleOutputs {
            sample_predictions,
            sample_gold_logprobs: Some(sample_gold_logprobs),
        })
    }

    fn adapt_chunk(&mut self, tokens: &[usize]) -> Result<(), String> {
        self.base.adapt_chunk(tokens)?;
        for &token in tokens {
            self.ngram.observe_token(token);
        }
        Ok(())
    }
}

pub fn load_token_conker3_ngram_checkpoint_from_data_root(
    checkpoint_path: &str,
    summary_path: &str,
    root: &Path,
    val_token_budget: usize,
) -> Result<LoadedTokenConker3NgramCheckpoint, String> {
    build_loaded_ngram_checkpoint(root, checkpoint_path, summary_path, val_token_budget)
}

pub fn run_token_conker3_ngram_checkpoint_from_data_root(
    checkpoint_path: &str,
    summary_path: &str,
    root: &Path,
    val_token_budget: usize,
) -> Result<TokenConker3NgramCheckpointReport, String> {
    Ok(load_token_conker3_ngram_checkpoint_from_data_root(
        checkpoint_path,
        summary_path,
        root,
        val_token_budget,
    )?
    .report)
}

pub fn run_token_conker3_ngram_checkpoint_fast_from_data_root(
    checkpoint_path: &str,
    summary_path: &str,
    root: &Path,
    val_token_budget: usize,
) -> Result<TokenConker3NgramCheckpointReport, String> {
    let config = NgramExpertConfig::default();
    let (base_runner, metadata) =
        load_token_conker3_checkpoint_runner_and_metadata(checkpoint_path, summary_path)?;
    let eval_tokens = take_val_tokens(root, val_token_budget)?;
    let vocab_size = base_runner.vocab_size();
    let runner = TokenConker3NgramCheckpointRunner {
        base: base_runner,
        ngram: NgramExpertState::new(vocab_size, config),
    };
    let (eval_bpt_neural, eval_bpt_mixed) =
        evaluate_target_only_adjusted_nll(runner, &eval_tokens, DEFAULT_EVAL_CHUNK)?;
    let byte_accounting = compute_tokens_per_byte(root, &eval_tokens)?;
    Ok(TokenConker3NgramCheckpointReport {
        checkpoint_path: metadata.checkpoint_path,
        summary_path: metadata.summary_path,
        train_seq_len: metadata.train_seq_len,
        eval_tokens: eval_tokens.len(),
        scoring_mode: "target_only_adjusted_nll".to_string(),
        ngram_min_order: config.min_order,
        ngram_max_order: config.max_order,
        ngram_buckets: config.buckets,
        mix_base: config.mix_base,
        mix_span: config.mix_span,
        entropy_midpoint: config.entropy_midpoint,
        entropy_slope: config.entropy_slope,
        min_row_total: config.min_row_total,
        residual_cap: config.residual_cap,
        eval_bpt_neural,
        eval_bpt_mixed,
        eval_tokens_per_byte: byte_accounting.as_ref().map(|row| row.tokens_per_byte),
        eval_bytes_per_token: byte_accounting.as_ref().map(|row| row.bytes_per_token),
        eval_bpb_neural: byte_accounting
            .as_ref()
            .map(|row| eval_bpt_neural * row.tokens_per_byte),
        eval_bpb_mixed: byte_accounting
            .as_ref()
            .map(|row| eval_bpt_mixed * row.tokens_per_byte),
    })
}

pub fn render_token_conker3_ngram_checkpoint_report(
    report: &TokenConker3NgramCheckpointReport,
) -> String {
    let mut out = String::new();
    out.push_str("chronohorn_token_conker3_ngram_checkpoint\n");
    out.push_str(&format!("checkpoint_path: {}\n", report.checkpoint_path));
    out.push_str(&format!("summary_path: {}\n", report.summary_path));
    out.push_str(&format!("train_seq_len: {}\n", report.train_seq_len));
    out.push_str(&format!("eval_tokens: {}\n", report.eval_tokens));
    out.push_str(&format!("scoring_mode: {}\n", report.scoring_mode));
    out.push_str(&format!(
        "ngram_orders: {}..={}\n",
        report.ngram_min_order, report.ngram_max_order
    ));
    out.push_str(&format!("ngram_buckets: {}\n", report.ngram_buckets));
    out.push_str(&format!(
        "entropy_mix: base={:.4} span={:.4} midpoint={:.4} slope={:.4}\n",
        report.mix_base, report.mix_span, report.entropy_midpoint, report.entropy_slope
    ));
    out.push_str(&format!(
        "ngram_residual: min_row_total={} residual_cap={:.4}\n",
        report.min_row_total, report.residual_cap
    ));
    out.push_str(&format!("eval_bpt_neural: {:.6}\n", report.eval_bpt_neural));
    out.push_str(&format!("eval_bpt_mixed: {:.6}\n", report.eval_bpt_mixed));
    if let Some(value) = report.eval_tokens_per_byte {
        out.push_str(&format!("eval_tokens_per_byte: {:.6}\n", value));
    }
    if let Some(value) = report.eval_bytes_per_token {
        out.push_str(&format!("eval_bytes_per_token: {:.6}\n", value));
    }
    if let Some(value) = report.eval_bpb_neural {
        out.push_str(&format!("eval_bpb_neural: {:.6}\n", value));
    }
    if let Some(value) = report.eval_bpb_mixed {
        out.push_str(&format!("eval_bpb_mixed: {:.6}\n", value));
    }
    out
}

pub fn render_token_conker3_ngram_bundle(loaded: &LoadedTokenConker3NgramCheckpoint) -> String {
    render_token_conker3_ngram_checkpoint_report(loaded.report())
}

fn build_loaded_ngram_checkpoint(
    root: &Path,
    checkpoint_path: &str,
    summary_path: &str,
    val_token_budget: usize,
) -> Result<LoadedTokenConker3NgramCheckpoint, String> {
    let config = NgramExpertConfig::default();
    let (base_runner, metadata) =
        load_token_conker3_checkpoint_runner_and_metadata(checkpoint_path, summary_path)?;
    let eval_tokens = take_val_tokens(root, val_token_budget)?;
    let vocab_size = base_runner.vocab_size();
    let runner = TokenConker3NgramCheckpointRunner {
        base: base_runner,
        ngram: NgramExpertState::new(vocab_size, config),
    };
    let (eval_bpt_neural, eval_bpt_mixed) =
        evaluate_mixed_runner(runner.clone(), &eval_tokens, DEFAULT_EVAL_CHUNK)?;
    let byte_accounting = compute_tokens_per_byte(root, &eval_tokens)?;
    let report = TokenConker3NgramCheckpointReport {
        checkpoint_path: metadata.checkpoint_path,
        summary_path: metadata.summary_path,
        train_seq_len: metadata.train_seq_len,
        eval_tokens: eval_tokens.len(),
        scoring_mode: "full_distribution_logit_residual".to_string(),
        ngram_min_order: config.min_order,
        ngram_max_order: config.max_order,
        ngram_buckets: config.buckets,
        mix_base: config.mix_base,
        mix_span: config.mix_span,
        entropy_midpoint: config.entropy_midpoint,
        entropy_slope: config.entropy_slope,
        min_row_total: config.min_row_total,
        residual_cap: config.residual_cap,
        eval_bpt_neural,
        eval_bpt_mixed,
        eval_tokens_per_byte: byte_accounting.as_ref().map(|row| row.tokens_per_byte),
        eval_bytes_per_token: byte_accounting.as_ref().map(|row| row.bytes_per_token),
        eval_bpb_neural: byte_accounting
            .as_ref()
            .map(|row| eval_bpt_neural * row.tokens_per_byte),
        eval_bpb_mixed: byte_accounting
            .as_ref()
            .map(|row| eval_bpt_mixed * row.tokens_per_byte),
    };
    Ok(LoadedTokenConker3NgramCheckpoint {
        report,
        runner,
        eval_tokens,
    })
}

fn evaluate_mixed_runner(
    mut runner: TokenConker3NgramCheckpointRunner,
    tokens: &[usize],
    chunk_size: usize,
) -> Result<(f64, f64), String> {
    if tokens.is_empty() {
        return Err("cannot evaluate empty token slice".to_string());
    }
    let mut total_neural_nats = 0.0f64;
    let mut total_mixed_nats = 0.0f64;
    let mut total = 0usize;
    for chunk in tokens.chunks(chunk_size.max(1)) {
        let (mut base_state, mut base_history) = runner.base.snapshot_runtime();
        let mut score_history = runner.ngram.history.clone();
        for &token in chunk {
            let neural_logits = runner
                .base
                .predict_combined_logits_from(&base_state, &base_history)?;
            total_neural_nats += negative_log_prob_from_logits(&neural_logits, token);
            let row = runner
                .ngram
                .lookup_counter_from_history(&score_history)
                .map(|(counter, _)| counter);
            let alpha = entropy_residual_weight_from_logits(&neural_logits, runner.ngram.config);
            let mixed_logits = apply_ngram_residual_logits(
                &neural_logits,
                row,
                alpha,
                runner.base.vocab_size(),
                runner.ngram.config.residual_cap,
            );
            total_mixed_nats += negative_log_prob_from_logits(&mixed_logits, token);
            total += 1;
            runner
                .base
                .advance_runtime_state(&mut base_state, &mut base_history, token)?;
            push_history(
                &mut score_history,
                runner.ngram.config.max_order.saturating_sub(1),
                token,
            );
        }
        runner.adapt_chunk(chunk)?;
    }
    if total == 0 {
        return Err("mixed runner scored zero tokens".to_string());
    }
    let denom = total as f64 * std::f64::consts::LN_2;
    Ok((total_neural_nats / denom, total_mixed_nats / denom))
}

fn evaluate_target_only_adjusted_nll(
    mut runner: TokenConker3NgramCheckpointRunner,
    tokens: &[usize],
    chunk_size: usize,
) -> Result<(f64, f64), String> {
    if tokens.is_empty() {
        return Err("cannot evaluate empty token slice".to_string());
    }
    let mut total_neural_nats = 0.0f64;
    let mut total_mixed_nats = 0.0f64;
    let mut total = 0usize;
    let vocab_size = runner.base.vocab_size() as f64;
    for chunk in tokens.chunks(chunk_size.max(1)) {
        let (mut base_state, mut base_history) = runner.base.snapshot_runtime();
        let mut score_history = runner.ngram.history.clone();
        for &token in chunk {
            let neural_logits = runner
                .base
                .predict_combined_logits_from(&base_state, &base_history)?;
            let neural_prob = gold_prob_from_logits(&neural_logits, token);
            total_neural_nats -= neural_prob.max(f64::MIN_POSITIVE).ln();
            let mixed_prob =
                if let Some((row, _)) = runner.ngram.lookup_counter_from_history(&score_history) {
                    let alpha =
                        entropy_residual_weight_from_logits(&neural_logits, runner.ngram.config);
                    let count = row
                        .counts
                        .iter()
                        .find(|(candidate, _)| *candidate == token)
                        .map(|(_, count)| *count as f64)
                        .unwrap_or(0.0);
                    let p_ngram = count / (count + vocab_size);
                    (alpha * p_ngram + (1.0 - alpha) * neural_prob).max(f64::MIN_POSITIVE)
                } else {
                    neural_prob.max(f64::MIN_POSITIVE)
                };
            total_mixed_nats -= mixed_prob.ln();
            total += 1;
            runner
                .base
                .advance_runtime_state(&mut base_state, &mut base_history, token)?;
            push_history(
                &mut score_history,
                runner.ngram.config.max_order.saturating_sub(1),
                token,
            );
        }
        runner.adapt_chunk(chunk)?;
    }
    if total == 0 {
        return Err("adjusted_nll scorer scored zero tokens".to_string());
    }
    let denom = total as f64 * std::f64::consts::LN_2;
    Ok((total_neural_nats / denom, total_mixed_nats / denom))
}

fn entropy_residual_weight_from_logits(logits: &[f32], config: NgramExpertConfig) -> f64 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut unnormalized = Vec::with_capacity(logits.len());
    let mut sum = 0.0f64;
    for &value in logits {
        let weight = ((value as f64) - max).exp();
        unnormalized.push(weight);
        sum += weight;
    }
    let sum = sum.max(f64::MIN_POSITIVE);
    let entropy = unnormalized
        .into_iter()
        .map(|weight| {
            let prob = weight / sum;
            if prob <= 0.0 { 0.0 } else { -prob * prob.ln() }
        })
        .sum::<f64>();
    config.mix_base
        + config.mix_span * sigmoid(config.entropy_slope * (entropy - config.entropy_midpoint))
}

fn apply_ngram_residual_logits(
    neural_logits: &[f32],
    counter: Option<&SparseCounter>,
    alpha: f64,
    vocab_size: usize,
    residual_cap: f64,
) -> Vec<f32> {
    let Some(counter) = counter else {
        return neural_logits.to_vec();
    };
    if counter.total == 0 {
        return neural_logits.to_vec();
    }
    let mut out = neural_logits.to_vec();
    let alpha = alpha.clamp(0.0, 1.0);
    let uniform_log = -((vocab_size.max(1)) as f64).ln();
    let denom = counter.total as f64;
    for &(token, count) in &counter.counts {
        if token >= out.len() {
            continue;
        }
        let log_prob = ((count as f64) / denom).max(f64::MIN_POSITIVE).ln();
        let raw = alpha * (log_prob - uniform_log);
        let bounded = residual_cap * (raw / residual_cap).tanh();
        out[token] += bounded as f32;
    }
    out
}

fn softmax_logits_f64(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut out = Vec::with_capacity(logits.len());
    let mut sum = 0.0f64;
    for &value in logits {
        let exp = ((value as f64) - max).exp();
        out.push(exp);
        sum += exp;
    }
    let sum = sum.max(f64::MIN_POSITIVE);
    for value in &mut out {
        *value /= sum;
    }
    out
}

fn negative_log_prob_from_logits(logits: &[f32], gold: usize) -> f64 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut sum = 0.0f64;
    for &value in logits {
        sum += ((value as f64) - max).exp();
    }
    let gold_logit = logits.get(gold).copied().unwrap_or(-1e9) as f64;
    -(gold_logit - max - sum.max(f64::MIN_POSITIVE).ln())
}

fn gold_prob_from_logits(logits: &[f32], gold: usize) -> f64 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut sum = 0.0f64;
    for &value in logits {
        sum += ((value as f64) - max).exp();
    }
    let gold_logit = logits.get(gold).copied().unwrap_or(-1e9) as f64;
    (gold_logit - max - sum.max(f64::MIN_POSITIVE).ln()).exp()
}

fn push_history(history: &mut VecDeque<usize>, max_len: usize, token: usize) {
    history.push_back(token);
    while history.len() > max_len {
        history.pop_front();
    }
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
