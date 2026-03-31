use std::collections::{HashMap, VecDeque};
use std::fs;
use std::path::Path;
use std::time::Instant;

use chronohorn_core::data::{
    HEADER_BYTES, PARAMETER_GOLF_MAGIC, PARAMETER_GOLF_VERSION, compute_tokens_per_byte,
    list_shards,
};
use chronohorn_core::protocol::{Runner, SampleOutputs};
use serde::Serialize;

use crate::checkpoint::{
    TokenConker3ExactCheckpointRunner, load_token_conker3_exact_checkpoint_from_data_root,
};

const BUCKET_2: usize = 16_777_216;
const BUCKET_3: usize = 16_777_216;
const BUCKET_4: usize = 4_194_304;
const BUCKET_5: usize = 4_194_304;
const BUCKET_6: usize = 4_194_304;
const BUCKET_7: usize = 4_194_304;
const MIX_BASE: f64 = 0.05;
const MIX_SPAN: f64 = 0.25;
const ENTROPY_MIDPOINT: f64 = 4.5;
const ENTROPY_SLOPE: f64 = 1.5;
const MIN_ROW_TOTAL: u32 = 4;
const RESIDUAL_CAP: f64 = 4.0;

#[derive(Debug, Clone, Serialize)]
pub struct TokenConker3ExactNgramCheckpointReport {
    pub checkpoint_path: String,
    pub summary_path: String,
    pub tokenizer_vocab_path: String,
    pub train_seq_len: usize,
    pub eval_tokens: usize,
    pub collision_control: String,
    pub table_source: String,
    pub train_tokens_built: usize,
    pub train_build_elapsed_sec: f64,
    pub ngram_orders: Vec<usize>,
    pub ngram_bucket_profile: Vec<usize>,
    pub mix_base: f64,
    pub mix_span: f64,
    pub entropy_midpoint: f64,
    pub entropy_slope: f64,
    pub min_row_total: u32,
    pub residual_cap: f64,
    pub exact_residual_cap: f64,
    pub eval_bpt_base: f64,
    pub eval_bpt_exact: f64,
    pub eval_bpt_exact_ngram: f64,
    pub eval_tokens_per_byte: Option<f64>,
    pub eval_bytes_per_token: Option<f64>,
    pub eval_bpb_base: Option<f64>,
    pub eval_bpb_exact: Option<f64>,
    pub eval_bpb_exact_ngram: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct LoadedTokenConker3ExactNgramCheckpoint {
    report: TokenConker3ExactNgramCheckpointReport,
    runner: TokenConker3ExactNgramCheckpointRunner,
    eval_tokens: Vec<usize>,
}

impl LoadedTokenConker3ExactNgramCheckpoint {
    pub fn report(&self) -> &TokenConker3ExactNgramCheckpointReport {
        &self.report
    }

    pub fn runner(&self) -> &TokenConker3ExactNgramCheckpointRunner {
        &self.runner
    }

    pub fn eval_tokens(&self) -> &[usize] {
        &self.eval_tokens
    }
}

#[derive(Debug, Clone)]
pub struct TokenConker3ExactNgramCheckpointRunner {
    exact: TokenConker3ExactCheckpointRunner,
    frozen: FrozenFingerprintedNgramTable,
    eval_history: VecDeque<usize>,
}

#[derive(Debug, Clone)]
struct FrozenFingerprintedNgramConfig {
    enabled_orders: Vec<usize>,
    bucket_counts: Vec<usize>,
    mix_base: f64,
    mix_span: f64,
    entropy_midpoint: f64,
    entropy_slope: f64,
    min_row_total: u32,
    residual_cap: f64,
}

impl Default for FrozenFingerprintedNgramConfig {
    fn default() -> Self {
        Self {
            enabled_orders: vec![2, 3, 4, 5, 6, 7],
            bucket_counts: vec![BUCKET_2, BUCKET_3, BUCKET_4, BUCKET_5, BUCKET_6, BUCKET_7],
            mix_base: MIX_BASE,
            mix_span: MIX_SPAN,
            entropy_midpoint: ENTROPY_MIDPOINT,
            entropy_slope: ENTROPY_SLOPE,
            min_row_total: MIN_ROW_TOTAL,
            residual_cap: RESIDUAL_CAP,
        }
    }
}

impl FrozenFingerprintedNgramConfig {
    fn max_order(&self) -> usize {
        self.enabled_orders.last().copied().unwrap_or(1)
    }

    fn bucket_count_for_order(&self, order: usize) -> usize {
        self.enabled_orders
            .iter()
            .position(|&candidate| candidate == order)
            .map(|index| self.bucket_counts[index])
            .unwrap_or(1)
    }
}

#[derive(Debug, Clone)]
struct FrozenFingerprintedNgramTable {
    config: FrozenFingerprintedNgramConfig,
    build_history: VecDeque<usize>,
    tables: Vec<HashMap<u32, Vec<FingerprintedCounter>>>,
}

#[derive(Debug, Clone)]
struct FingerprintedCounter {
    fingerprint: u64,
    counter: NgramCounter,
}

#[derive(Debug, Clone, Default)]
struct NgramCounter {
    total: u32,
    counts: Vec<(usize, u32)>,
}

impl NgramCounter {
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

impl FrozenFingerprintedNgramTable {
    fn new(config: FrozenFingerprintedNgramConfig) -> Self {
        let table_count = config.enabled_orders.len();
        let history_capacity = config.max_order().saturating_sub(1);
        Self {
            config,
            build_history: VecDeque::with_capacity(history_capacity),
            tables: (0..table_count).map(|_| HashMap::new()).collect(),
        }
    }

    fn observe_training_token(&mut self, token: usize) {
        let max_context = self
            .build_history
            .len()
            .min(self.config.max_order().saturating_sub(1));
        for (table_index, &order) in self.config.enabled_orders.iter().enumerate() {
            let context_len = order.saturating_sub(1);
            if context_len > max_context {
                continue;
            }
            let (bucket, fingerprint) =
                self.context_key_from_history(&self.build_history, context_len, order);
            let rows = self.tables[table_index].entry(bucket).or_default();
            if let Some(row) = rows.iter_mut().find(|row| row.fingerprint == fingerprint) {
                row.counter.increment(token);
            } else {
                let mut counter = NgramCounter::default();
                counter.increment(token);
                rows.push(FingerprintedCounter {
                    fingerprint,
                    counter,
                });
            }
        }
        push_history(
            &mut self.build_history,
            self.config.max_order().saturating_sub(1),
            token,
        );
    }

    fn lookup_counter_from_history<'a>(
        &'a self,
        history: &VecDeque<usize>,
    ) -> Option<(&'a NgramCounter, usize)> {
        let max_context = history.len().min(self.config.max_order().saturating_sub(1));
        for (table_index, &order) in self.config.enabled_orders.iter().enumerate().rev() {
            let context_len = order.saturating_sub(1);
            if context_len > max_context {
                continue;
            }
            let (bucket, fingerprint) = self.context_key_from_history(history, context_len, order);
            if let Some(rows) = self.tables[table_index].get(&bucket) {
                if let Some(row) = rows
                    .iter()
                    .find(|row| row.fingerprint == fingerprint)
                    .map(|row| &row.counter)
                {
                    if row.total >= self.config.min_row_total {
                        return Some((row, order));
                    }
                }
            }
        }
        None
    }

    fn clear_build_history(&mut self) {
        self.build_history.clear();
    }

    fn context_key_from_history(
        &self,
        history: &VecDeque<usize>,
        context_len: usize,
        order: usize,
    ) -> (u32, u64) {
        let mut bucket_hash = 0xcbf2_9ce4_8422_2325u64 ^ (order as u64);
        let mut fingerprint = 0x9e37_79b9_7f4a_7c15u64 ^ ((order as u64) << 32);
        let start = history.len().saturating_sub(context_len);
        for &token in history.iter().skip(start) {
            let token_u64 = token as u64;
            bucket_hash ^= token_u64 + 0x9e37_79b9;
            bucket_hash = bucket_hash.wrapping_mul(0x1000_0000_01b3);
            fingerprint ^= token_u64.wrapping_add(0x517c_c1b7_2722_0a95);
            fingerprint = fingerprint.rotate_left(13);
            fingerprint = fingerprint.wrapping_mul(0x9e37_79b1_85eb_ca87);
        }
        let buckets = self.config.bucket_count_for_order(order).max(1);
        fingerprint ^= fingerprint >> 33;
        fingerprint = fingerprint.wrapping_mul(0xff51_afd7_ed55_8ccd);
        fingerprint ^= fingerprint >> 33;
        fingerprint = fingerprint.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        fingerprint ^= fingerprint >> 33;
        ((bucket_hash % buckets as u64) as u32, fingerprint)
    }
}

impl Runner for TokenConker3ExactNgramCheckpointRunner {
    fn name(&self) -> &'static str {
        "TokenConker3ExactNgramCheckpointRunner"
    }

    fn vocab_size(&self) -> usize {
        self.exact.vocab_size()
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
        let mut exact = self.exact.clone();
        let mut score_history = self.eval_history.clone();

        for (pos, &token) in tokens.iter().enumerate() {
            if !wanted[pos].is_empty() {
                let exact_logits = exact.predict_current_exact_logits()?;
                let row = self
                    .frozen
                    .lookup_counter_from_history(&score_history)
                    .map(|(counter, _)| counter);
                let alpha = entropy_residual_weight_from_logits(&exact_logits, &self.frozen.config);
                let mixed_logits = apply_ngram_residual_logits(
                    &exact_logits,
                    row,
                    alpha,
                    self.exact.vocab_size(),
                    self.frozen.config.residual_cap,
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
            exact.adapt_one(token)?;
            push_history(
                &mut score_history,
                self.frozen.config.max_order().saturating_sub(1),
                token,
            );
        }

        Ok(SampleOutputs {
            sample_predictions,
            sample_gold_logprobs: Some(sample_gold_logprobs),
        })
    }

    fn adapt_chunk(&mut self, tokens: &[usize]) -> Result<(), String> {
        for &token in tokens {
            self.exact.adapt_one(token)?;
            push_history(
                &mut self.eval_history,
                self.frozen.config.max_order().saturating_sub(1),
                token,
            );
        }
        Ok(())
    }
}

pub fn load_token_conker3_exact_ngram_checkpoint_from_data_root(
    checkpoint_path: &str,
    summary_path: &str,
    root: &Path,
    train_token_budget: usize,
    val_token_budget: usize,
) -> Result<LoadedTokenConker3ExactNgramCheckpoint, String> {
    let loaded_exact = load_token_conker3_exact_checkpoint_from_data_root(
        checkpoint_path,
        summary_path,
        root,
        val_token_budget,
    )?;
    let (frozen, train_tokens_built, train_build_elapsed_sec) = build_frozen_train_table(
        root,
        train_token_budget,
        FrozenFingerprintedNgramConfig::default(),
    )?;
    let eval_tokens = loaded_exact.eval_tokens().to_vec();
    let (eval_bpt_base, eval_bpt_exact, eval_bpt_exact_ngram) =
        evaluate_exact_ngram_sequence(loaded_exact.runner().clone(), &frozen, &eval_tokens)?;
    let byte_accounting = compute_tokens_per_byte(root, &eval_tokens)?;
    let report = TokenConker3ExactNgramCheckpointReport {
        checkpoint_path: loaded_exact.report().checkpoint_path.clone(),
        summary_path: loaded_exact.report().summary_path.clone(),
        tokenizer_vocab_path: loaded_exact.report().tokenizer_vocab_path.clone(),
        train_seq_len: loaded_exact.report().train_seq_len,
        eval_tokens: eval_tokens.len(),
        collision_control: "bucket_plus_fingerprint".to_string(),
        table_source: "prebuilt_train_frozen".to_string(),
        train_tokens_built,
        train_build_elapsed_sec,
        ngram_orders: frozen.config.enabled_orders.clone(),
        ngram_bucket_profile: frozen.config.bucket_counts.clone(),
        mix_base: frozen.config.mix_base,
        mix_span: frozen.config.mix_span,
        entropy_midpoint: frozen.config.entropy_midpoint,
        entropy_slope: frozen.config.entropy_slope,
        min_row_total: frozen.config.min_row_total,
        residual_cap: frozen.config.residual_cap,
        exact_residual_cap: loaded_exact.report().residual_cap,
        eval_bpt_base,
        eval_bpt_exact,
        eval_bpt_exact_ngram,
        eval_tokens_per_byte: byte_accounting.as_ref().map(|row| row.tokens_per_byte),
        eval_bytes_per_token: byte_accounting.as_ref().map(|row| row.bytes_per_token),
        eval_bpb_base: byte_accounting
            .as_ref()
            .map(|row| eval_bpt_base * row.tokens_per_byte),
        eval_bpb_exact: byte_accounting
            .as_ref()
            .map(|row| eval_bpt_exact * row.tokens_per_byte),
        eval_bpb_exact_ngram: byte_accounting
            .as_ref()
            .map(|row| eval_bpt_exact_ngram * row.tokens_per_byte),
    };
    let runner = TokenConker3ExactNgramCheckpointRunner {
        exact: loaded_exact.runner().clone(),
        frozen,
        eval_history: VecDeque::with_capacity(7),
    };
    Ok(LoadedTokenConker3ExactNgramCheckpoint {
        report,
        runner,
        eval_tokens,
    })
}

pub fn run_token_conker3_exact_ngram_checkpoint_from_data_root(
    checkpoint_path: &str,
    summary_path: &str,
    root: &Path,
    train_token_budget: usize,
    val_token_budget: usize,
) -> Result<TokenConker3ExactNgramCheckpointReport, String> {
    Ok(load_token_conker3_exact_ngram_checkpoint_from_data_root(
        checkpoint_path,
        summary_path,
        root,
        train_token_budget,
        val_token_budget,
    )?
    .report)
}

pub fn render_token_conker3_exact_ngram_checkpoint_report(
    report: &TokenConker3ExactNgramCheckpointReport,
) -> String {
    let mut out = String::new();
    out.push_str("chronohorn_token_conker3_exact_ngram_checkpoint\n");
    out.push_str(&format!("checkpoint_path: {}\n", report.checkpoint_path));
    out.push_str(&format!("summary_path: {}\n", report.summary_path));
    out.push_str(&format!(
        "tokenizer_vocab_path: {}\n",
        report.tokenizer_vocab_path
    ));
    out.push_str(&format!("train_seq_len: {}\n", report.train_seq_len));
    out.push_str(&format!("eval_tokens: {}\n", report.eval_tokens));
    out.push_str(&format!(
        "collision_control: {}\n",
        report.collision_control
    ));
    out.push_str(&format!("table_source: {}\n", report.table_source));
    out.push_str(&format!(
        "train_tokens_built: {}\n",
        report.train_tokens_built
    ));
    out.push_str(&format!(
        "train_build_elapsed_sec: {:.3}\n",
        report.train_build_elapsed_sec
    ));
    out.push_str(&format!(
        "ngram_orders: {}\n",
        report
            .ngram_orders
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(",")
    ));
    out.push_str(&format!(
        "ngram_bucket_profile: {}\n",
        report
            .ngram_bucket_profile
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(",")
    ));
    out.push_str(&format!(
        "entropy_mix: base={:.4} span={:.4} midpoint={:.4} slope={:.4}\n",
        report.mix_base, report.mix_span, report.entropy_midpoint, report.entropy_slope
    ));
    out.push_str(&format!(
        "residual_caps: exact={:.4} ngram={:.4}\n",
        report.exact_residual_cap, report.residual_cap
    ));
    out.push_str(&format!("eval_bpt_base: {:.6}\n", report.eval_bpt_base));
    out.push_str(&format!("eval_bpt_exact: {:.6}\n", report.eval_bpt_exact));
    out.push_str(&format!(
        "eval_bpt_exact_ngram: {:.6}\n",
        report.eval_bpt_exact_ngram
    ));
    if let Some(value) = report.eval_tokens_per_byte {
        out.push_str(&format!("eval_tokens_per_byte: {:.6}\n", value));
    }
    if let Some(value) = report.eval_bytes_per_token {
        out.push_str(&format!("eval_bytes_per_token: {:.6}\n", value));
    }
    if let Some(value) = report.eval_bpb_base {
        out.push_str(&format!("eval_bpb_base: {:.6}\n", value));
    }
    if let Some(value) = report.eval_bpb_exact {
        out.push_str(&format!("eval_bpb_exact: {:.6}\n", value));
    }
    if let Some(value) = report.eval_bpb_exact_ngram {
        out.push_str(&format!("eval_bpb_exact_ngram: {:.6}\n", value));
    }
    out
}

fn build_frozen_train_table(
    root: &Path,
    token_budget: usize,
    config: FrozenFingerprintedNgramConfig,
) -> Result<(FrozenFingerprintedNgramTable, usize, f64), String> {
    let start = Instant::now();
    let mut frozen = FrozenFingerprintedNgramTable::new(config);
    let mut tokens_built = 0usize;
    for shard in list_shards(root, "fineweb_train_")? {
        if tokens_built >= token_budget {
            break;
        }
        let left = token_budget - tokens_built;
        let scanned =
            scan_shard_tokens(&shard, left, |token| frozen.observe_training_token(token))?;
        tokens_built += scanned;
    }
    if tokens_built < 4 {
        return Err("need at least 4 train tokens to build a frozen n-gram table".to_string());
    }
    frozen.clear_build_history();
    Ok((frozen, tokens_built, start.elapsed().as_secs_f64()))
}

fn scan_shard_tokens(
    path: &Path,
    limit: usize,
    mut on_token: impl FnMut(usize),
) -> Result<usize, String> {
    let blob = fs::read(path).map_err(|err| format!("read {}: {err}", path.display()))?;
    let (payload, token_count) = if blob.len() >= HEADER_BYTES {
        let header0 = i32::from_le_bytes([blob[0], blob[1], blob[2], blob[3]]);
        let header1 = i32::from_le_bytes([blob[4], blob[5], blob[6], blob[7]]);
        let header2 = i32::from_le_bytes([blob[8], blob[9], blob[10], blob[11]]);
        if header0 == PARAMETER_GOLF_MAGIC && header1 == PARAMETER_GOLF_VERSION {
            let declared = usize::try_from(header2).map_err(|_| {
                format!(
                    "negative token count in shard header for {}",
                    path.display()
                )
            })?;
            (&blob[HEADER_BYTES..], declared.min(limit))
        } else {
            (&blob[..], (blob.len() / 2).min(limit))
        }
    } else {
        (&blob[..], (blob.len() / 2).min(limit))
    };
    for chunk in payload.chunks_exact(2).take(token_count) {
        on_token(u16::from_le_bytes([chunk[0], chunk[1]]) as usize);
    }
    Ok(token_count)
}

fn evaluate_exact_ngram_sequence(
    mut exact: TokenConker3ExactCheckpointRunner,
    frozen: &FrozenFingerprintedNgramTable,
    tokens: &[usize],
) -> Result<(f64, f64, f64), String> {
    if tokens.is_empty() {
        return Err("cannot evaluate empty token slice".to_string());
    }
    let mut base_nats = 0.0f64;
    let mut exact_nats = 0.0f64;
    let mut mixed_nats = 0.0f64;
    let mut total = 0usize;
    let mut eval_history = VecDeque::with_capacity(frozen.config.max_order().saturating_sub(1));
    for &token in tokens {
        let base_logits = exact.predict_current_base_logits()?;
        base_nats += negative_log_prob_from_logits(&base_logits, token);
        let exact_logits = exact.predict_current_exact_logits()?;
        exact_nats += negative_log_prob_from_logits(&exact_logits, token);
        let row = frozen
            .lookup_counter_from_history(&eval_history)
            .map(|(row, _)| row);
        let alpha = entropy_residual_weight_from_logits(&exact_logits, &frozen.config);
        let mixed_logits = apply_ngram_residual_logits(
            &exact_logits,
            row,
            alpha,
            exact.vocab_size(),
            frozen.config.residual_cap,
        );
        mixed_nats += negative_log_prob_from_logits(&mixed_logits, token);
        total += 1;
        exact.adapt_one(token)?;
        push_history(
            &mut eval_history,
            frozen.config.max_order().saturating_sub(1),
            token,
        );
    }
    let denom = total as f64 * std::f64::consts::LN_2;
    Ok((base_nats / denom, exact_nats / denom, mixed_nats / denom))
}

fn entropy_residual_weight_from_logits(
    logits: &[f32],
    config: &FrozenFingerprintedNgramConfig,
) -> f64 {
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
    exact_logits: &[f32],
    counter: Option<&NgramCounter>,
    alpha: f64,
    vocab_size: usize,
    residual_cap: f64,
) -> Vec<f32> {
    let Some(counter) = counter else {
        return exact_logits.to_vec();
    };
    if counter.total == 0 {
        return exact_logits.to_vec();
    }
    let mut out = exact_logits.to_vec();
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

fn negative_log_prob_from_logits(logits: &[f32], gold: usize) -> f64 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut sum = 0.0f64;
    for &value in logits {
        sum += ((value as f64) - max).exp();
    }
    let gold_logit = logits.get(gold).copied().unwrap_or(-1e9) as f64;
    -(gold_logit - max - sum.max(f64::MIN_POSITIVE).ln())
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
