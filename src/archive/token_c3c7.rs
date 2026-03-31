use std::collections::HashMap;
use std::path::Path;

use chronohorn_causal_bank::oracle::{
    token_oracle_target_dataset_from_tokens, token_oracle_teacher_candidate_pairs,
};
use chronohorn_core::data::{compute_tokens_per_byte, take_train_tokens, take_val_tokens};
use chronohorn_core::protocol::{Runner, SampleOutputs};
use serde::Serialize;

use super::packed_memory::{PackedTables, build_packed_tables};

const VOCAB_SIZE: usize = 1024;
const EXPERT_COUNT: usize = 5;
const RESIDUAL_COUNT: usize = EXPERT_COUNT - 1;
const TRAIN_EXAMPLE_CAP: usize = 4096;
const HALF_LIVES: [f64; 3] = [4.0, 8.0, 16.0];
const LOCAL_WINDOW: usize = 4;
const LOCAL_SMOOTHING: f64 = 0.5;
const ALPHA_BIGRAM: f64 = 4.0;
const ALPHA_TRIGRAM: f64 = 2.0;
const TRIGRAM_HASH_MUL_A: u64 = 1_315_423_911;
const TRIGRAM_HASH_MUL_B: u64 = 2_654_435_761;
const LOCAL_HASH_MULS: [u64; 4] = [1_315_423_911, 2_654_435_761, 1_002_583_641, 2_100_523_641];
const EPS: f64 = 1e-12;

#[derive(Debug, Clone, Serialize)]
pub struct TokenC3C7Report {
    pub train_token_budget: usize,
    pub trigram_buckets: usize,
    pub val_token_budget: usize,
    pub train_stride: usize,
    pub train_steps: usize,
    pub teacher_start_step: usize,
    pub oracle_radius: usize,
    pub oracle_stride: usize,
    pub alpha_nll: f64,
    pub beta_support: f64,
    pub gamma_rank: f64,
    pub train_records: usize,
    pub tune_records: usize,
    pub eval_records: usize,
    pub oracle_target_source: String,
    pub warm_start_coeffs: [f64; EXPERT_COUNT],
    pub distilled_coeffs: [f64; EXPERT_COUNT],
    pub selected_stage: String,
    pub tune_bpt_packed: f64,
    pub tune_bpt_local: f64,
    pub tune_bpt_bank4: f64,
    pub tune_bpt_bank8: f64,
    pub tune_bpt_bank16: f64,
    pub tune_bpt_warm_start: f64,
    pub tune_bpt_distilled: f64,
    pub tune_bpt_oracle: f64,
    pub eval_bpt_packed: f64,
    pub eval_bpt_local: f64,
    pub eval_bpt_bank4: f64,
    pub eval_bpt_bank8: f64,
    pub eval_bpt_bank16: f64,
    pub eval_bpt_warm_start: f64,
    pub eval_bpt_distilled: f64,
    pub eval_bpt_oracle: f64,
    pub eval_tokens_per_byte: Option<f64>,
    pub eval_bytes_per_token: Option<f64>,
    pub eval_bpb_packed: Option<f64>,
    pub eval_bpb_local: Option<f64>,
    pub eval_bpb_bank4: Option<f64>,
    pub eval_bpb_bank8: Option<f64>,
    pub eval_bpb_bank16: Option<f64>,
    pub eval_bpb_warm_start: Option<f64>,
    pub eval_bpb_distilled: Option<f64>,
    pub eval_bpb_oracle: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct TrainedTokenC3C7 {
    report: TokenC3C7Report,
    runner: TokenC3C7Runner,
    eval_tokens: Vec<usize>,
}

impl TrainedTokenC3C7 {
    pub fn report(&self) -> &TokenC3C7Report {
        &self.report
    }

    pub fn runner(&self) -> &TokenC3C7Runner {
        &self.runner
    }

    pub fn eval_tokens(&self) -> &[usize] {
        &self.eval_tokens
    }
}

#[derive(Debug, Clone)]
pub struct TokenC3C7Runner {
    packed_tables: PackedTables,
    local_tables: LocalWindowTables,
    coeffs: [f64; EXPERT_COUNT],
    bank_states: [DecayBankState; 3],
    stream_history: Vec<usize>,
}

#[derive(Debug, Clone)]
struct LocalWindowTables {
    contexts: HashMap<u64, SparseCounts>,
}

#[derive(Debug, Clone, Default)]
struct SparseCounts {
    total: f64,
    counts: Vec<(usize, f64)>,
}

#[derive(Debug, Clone)]
struct C3C7Example {
    gold: usize,
    packed_log_probs: Vec<f32>,
    residual_log_deltas: [Vec<f32>; RESIDUAL_COUNT],
    teacher_tokens: Vec<usize>,
    teacher_residual_expectations: [f64; RESIDUAL_COUNT],
    support_weight: f64,
}

#[derive(Debug, Clone)]
struct TrainingState {
    coeffs: [f64; EXPERT_COUNT],
    adam_m: [f64; EXPERT_COUNT],
    adam_v: [f64; EXPERT_COUNT],
    step: usize,
}

#[derive(Debug, Clone)]
struct EvalStats {
    packed: f64,
    local: f64,
    bank4: f64,
    bank8: f64,
    bank16: f64,
    warm_start: f64,
    distilled: f64,
    oracle: f64,
    records: usize,
}

#[derive(Debug, Clone)]
struct DenseTeacherRow {
    pairs: Vec<(usize, usize)>,
}

#[derive(Debug, Clone)]
struct DecayBankState {
    raw_counts: Vec<f64>,
    raw_total: f64,
    scale: f64,
    decay: f64,
}

impl DecayBankState {
    fn new(vocab_size: usize, half_life: f64) -> Self {
        let decay = (0.5f64).powf(1.0 / half_life.max(1e-6));
        Self {
            raw_counts: vec![0.0; vocab_size],
            raw_total: 0.0,
            scale: 1.0,
            decay,
        }
    }

    fn distribution(&self, fallback: &[f64]) -> Vec<f64> {
        if self.raw_total <= EPS {
            return fallback.to_vec();
        }
        self.raw_counts
            .iter()
            .map(|value| value / self.raw_total.max(EPS))
            .collect()
    }

    fn adapt(&mut self, token: usize) {
        self.scale *= self.decay;
        let increment = 1.0 / self.scale.max(EPS);
        if let Some(slot) = self.raw_counts.get_mut(token) {
            *slot += increment;
        }
        self.raw_total += increment;
    }
}

impl Runner for TokenC3C7Runner {
    fn name(&self) -> &'static str {
        "TokenC3C7Runner"
    }

    fn vocab_size(&self) -> usize {
        self.packed_tables.vocab_size
    }

    fn score_chunk(
        &self,
        tokens: &[usize],
        sample_positions: &[usize],
    ) -> Result<SampleOutputs, String> {
        let mut predictions = vec![Vec::new(); sample_positions.len()];
        let mut golds = vec![0.0; sample_positions.len()];
        let mut wanted_positions: HashMap<usize, Vec<usize>> = HashMap::new();
        for (index, &pos) in sample_positions.iter().enumerate() {
            if pos >= tokens.len() {
                return Err(format!("sample position {pos} out of bounds"));
            }
            wanted_positions.entry(pos).or_default().push(index);
        }
        let mut bank_states = self.bank_states.clone();
        let mut history = self.stream_history.clone();
        for pos in 0..tokens.len() {
            let packed = packed_distribution(
                &self.packed_tables,
                prev2(&history),
                prev1(&history),
                ALPHA_BIGRAM,
                ALPHA_TRIGRAM,
            );
            let local = local_distribution(&self.local_tables, &history, &packed);
            let bank4 = bank_states[0].distribution(&packed);
            let bank8 = bank_states[1].distribution(&packed);
            let bank16 = bank_states[2].distribution(&packed);
            let expert_dists = [packed, local, bank4, bank8, bank16];
            if let Some(indexes) = wanted_positions.get(&pos) {
                let combined = combine_experts_additive_logits(&expert_dists, &self.coeffs);
                let gold = combined
                    .get(tokens[pos])
                    .copied()
                    .unwrap_or(EPS)
                    .max(EPS)
                    .ln();
                for &index in indexes {
                    predictions[index] = combined.clone();
                    golds[index] = gold;
                }
            }
            let token = tokens[pos];
            history.push(token);
            if history.len() > LOCAL_WINDOW {
                history.remove(0);
            }
            for bank in &mut bank_states {
                bank.adapt(token);
            }
        }
        Ok(SampleOutputs {
            sample_predictions: predictions,
            sample_gold_logprobs: Some(golds),
        })
    }

    fn adapt_chunk(&mut self, tokens: &[usize]) -> Result<(), String> {
        for &token in tokens {
            self.stream_history.push(token);
            if self.stream_history.len() > LOCAL_WINDOW {
                self.stream_history.remove(0);
            }
            for bank in &mut self.bank_states {
                bank.adapt(token);
            }
        }
        Ok(())
    }
}

pub fn train_token_c3c7_from_data_root(
    root: &Path,
    train_token_budget: usize,
    trigram_buckets: usize,
    val_token_budget: usize,
    train_stride: usize,
    train_steps: usize,
    teacher_start_step: usize,
    oracle_radius: usize,
    oracle_stride: usize,
    alpha_nll: f64,
    beta_support: f64,
    gamma_rank: f64,
) -> Result<TrainedTokenC3C7, String> {
    if train_stride == 0 {
        return Err("train_stride must be positive".to_string());
    }
    if oracle_radius == 0 {
        return Err("oracle_radius must be positive".to_string());
    }
    if oracle_stride == 0 {
        return Err("oracle_stride must be positive".to_string());
    }

    let train_tokens = take_train_tokens(root, train_token_budget)?;
    let val_tokens = take_val_tokens(root, val_token_budget)?;
    if val_tokens.len() < 16 {
        return Err("need at least 16 validation tokens".to_string());
    }
    let split = (val_tokens.len() / 2).max(1);
    let tune_tokens = val_tokens[..split].to_vec();
    let eval_tokens = val_tokens[split..].to_vec();
    if eval_tokens.is_empty() {
        return Err("validation split left no eval tokens".to_string());
    }

    let packed_tables = build_packed_tables(&train_tokens, VOCAB_SIZE, trigram_buckets)?;
    let local_tables = build_local_window_tables(&train_tokens, VOCAB_SIZE)?;

    let train_teacher = build_dense_teacher_rows(&train_tokens, oracle_radius, oracle_stride)?;
    let tune_teacher = build_dense_teacher_rows(&tune_tokens, oracle_radius, 1)?;
    let eval_teacher = build_dense_teacher_rows(&eval_tokens, oracle_radius, 1)?;

    let train_examples = collect_c3c7_examples(
        &train_tokens,
        &packed_tables,
        &local_tables,
        &train_teacher,
        train_stride,
    )?;
    if train_examples.is_empty() {
        return Err("token_c3c7 collected no usable training examples".to_string());
    }

    let trained = train_c3c7_student(
        &train_examples,
        train_steps,
        teacher_start_step,
        alpha_nll,
        beta_support,
        gamma_rank,
    );

    let warm_coeffs = trained.warm_start_coeffs;
    let distilled_coeffs = trained.final_coeffs;

    let tune_stats = evaluate_tokens(
        &tune_tokens,
        &packed_tables,
        &local_tables,
        &tune_teacher,
        &warm_coeffs,
        &distilled_coeffs,
    )?;
    let eval_stats = evaluate_tokens(
        &eval_tokens,
        &packed_tables,
        &local_tables,
        &eval_teacher,
        &warm_coeffs,
        &distilled_coeffs,
    )?;
    let selected_stage = if tune_stats.distilled < tune_stats.warm_start {
        "distilled".to_string()
    } else {
        "warm_start".to_string()
    };
    let selected_coeffs = if selected_stage == "distilled" {
        distilled_coeffs
    } else {
        warm_coeffs
    };
    let eval_byte_accounting = compute_tokens_per_byte(root, &eval_tokens)?;
    let eval_tokens_per_byte = eval_byte_accounting.as_ref().map(|row| row.tokens_per_byte);
    let eval_bytes_per_token = eval_byte_accounting.as_ref().map(|row| row.bytes_per_token);

    let report = TokenC3C7Report {
        train_token_budget,
        trigram_buckets,
        val_token_budget: val_tokens.len(),
        train_stride,
        train_steps,
        teacher_start_step,
        oracle_radius,
        oracle_stride,
        alpha_nll,
        beta_support,
        gamma_rank,
        train_records: train_examples.len(),
        tune_records: tune_stats.records,
        eval_records: eval_stats.records,
        oracle_target_source: format!(
            "token_context_oracle:ranked_pairs:radius={oracle_radius}:stride={oracle_stride}"
        ),
        warm_start_coeffs: warm_coeffs,
        distilled_coeffs,
        selected_stage,
        tune_bpt_packed: tune_stats.packed,
        tune_bpt_local: tune_stats.local,
        tune_bpt_bank4: tune_stats.bank4,
        tune_bpt_bank8: tune_stats.bank8,
        tune_bpt_bank16: tune_stats.bank16,
        tune_bpt_warm_start: tune_stats.warm_start,
        tune_bpt_distilled: tune_stats.distilled,
        tune_bpt_oracle: tune_stats.oracle,
        eval_bpt_packed: eval_stats.packed,
        eval_bpt_local: eval_stats.local,
        eval_bpt_bank4: eval_stats.bank4,
        eval_bpt_bank8: eval_stats.bank8,
        eval_bpt_bank16: eval_stats.bank16,
        eval_bpt_warm_start: eval_stats.warm_start,
        eval_bpt_distilled: eval_stats.distilled,
        eval_bpt_oracle: eval_stats.oracle,
        eval_tokens_per_byte,
        eval_bytes_per_token,
        eval_bpb_packed: eval_tokens_per_byte.map(|scale| eval_stats.packed * scale),
        eval_bpb_local: eval_tokens_per_byte.map(|scale| eval_stats.local * scale),
        eval_bpb_bank4: eval_tokens_per_byte.map(|scale| eval_stats.bank4 * scale),
        eval_bpb_bank8: eval_tokens_per_byte.map(|scale| eval_stats.bank8 * scale),
        eval_bpb_bank16: eval_tokens_per_byte.map(|scale| eval_stats.bank16 * scale),
        eval_bpb_warm_start: eval_tokens_per_byte.map(|scale| eval_stats.warm_start * scale),
        eval_bpb_distilled: eval_tokens_per_byte.map(|scale| eval_stats.distilled * scale),
        eval_bpb_oracle: eval_tokens_per_byte.map(|scale| eval_stats.oracle * scale),
    };

    let runner = TokenC3C7Runner {
        packed_tables,
        local_tables,
        coeffs: selected_coeffs,
        bank_states: [
            DecayBankState::new(VOCAB_SIZE, HALF_LIVES[0]),
            DecayBankState::new(VOCAB_SIZE, HALF_LIVES[1]),
            DecayBankState::new(VOCAB_SIZE, HALF_LIVES[2]),
        ],
        stream_history: Vec::new(),
    };

    Ok(TrainedTokenC3C7 {
        report,
        runner,
        eval_tokens,
    })
}

#[derive(Debug, Clone)]
struct TrainResult {
    warm_start_coeffs: [f64; EXPERT_COUNT],
    final_coeffs: [f64; EXPERT_COUNT],
}

fn train_c3c7_student(
    records: &[C3C7Example],
    train_steps: usize,
    teacher_start_step: usize,
    alpha_nll: f64,
    beta_support: f64,
    gamma_rank: f64,
) -> TrainResult {
    let mut state = TrainingState {
        coeffs: [1.0, 0.25, 0.05, 0.05, 0.05],
        adam_m: [0.0; EXPERT_COUNT],
        adam_v: [0.0; EXPERT_COUNT],
        step: 0,
    };
    let lr = 0.02;
    let beta1 = 0.9;
    let beta2 = 0.999;
    let batch_size = 128usize.min(records.len().max(1));
    let mut warm_start_coeffs = state.coeffs;

    for step in 0..train_steps {
        let teacher_on = step >= teacher_start_step;
        let batch = batch_slice(records, step, batch_size);
        let mut grad = [0.0; EXPERT_COUNT];

        for record in batch {
            let mut logits = [0.0f64; VOCAB_SIZE];
            for tok in 0..VOCAB_SIZE {
                let packed_log = record.packed_log_probs[tok] as f64;
                let mut combined = packed_log;
                for residual in 0..RESIDUAL_COUNT {
                    combined += state.coeffs[residual + 1]
                        * record.residual_log_deltas[residual][tok] as f64;
                }
                logits[tok] = combined;
            }
            let probs = softmax_vocab(&logits);

            let mut expectation = [0.0; RESIDUAL_COUNT];
            for tok in 0..VOCAB_SIZE {
                let p = probs[tok];
                for residual in 0..RESIDUAL_COUNT {
                    expectation[residual] += p * record.residual_log_deltas[residual][tok] as f64;
                }
            }
            let gold = record.gold;

            for residual in 0..RESIDUAL_COUNT {
                let gold_delta = record.residual_log_deltas[residual][gold] as f64;
                grad[residual + 1] +=
                    alpha_nll * (expectation[residual] - gold_delta) / std::f64::consts::LN_2;
            }

            if teacher_on && record.support_weight > 0.0 && !record.teacher_tokens.is_empty() {
                let mut support_mass = 0.0;
                let mut support_expectation = [0.0; RESIDUAL_COUNT];
                for &token in &record.teacher_tokens {
                    let p = probs[token];
                    support_mass += p;
                    for residual in 0..RESIDUAL_COUNT {
                        support_expectation[residual] +=
                            p * record.residual_log_deltas[residual][token] as f64;
                    }
                }
                support_mass = support_mass.max(EPS);
                for residual in 0..RESIDUAL_COUNT {
                    let conditional_expectation = support_expectation[residual] / support_mass;
                    grad[residual + 1] += beta_support
                        * record.support_weight
                        * (expectation[residual] - conditional_expectation)
                        / std::f64::consts::LN_2;
                    grad[residual + 1] += gamma_rank
                        * record.support_weight
                        * (conditional_expectation
                            - record.teacher_residual_expectations[residual])
                        / std::f64::consts::LN_2;
                }
            }
        }

        let inv_batch = 1.0 / batch.len().max(1) as f64;
        for idx in 1..EXPERT_COUNT {
            let grad = grad[idx] * inv_batch;
            state.adam_m[idx] = beta1 * state.adam_m[idx] + (1.0 - beta1) * grad;
            state.adam_v[idx] = beta2 * state.adam_v[idx] + (1.0 - beta2) * grad * grad;
            let t = (state.step + 1) as f64;
            let m_hat = state.adam_m[idx] / (1.0 - beta1.powf(t));
            let v_hat = state.adam_v[idx] / (1.0 - beta2.powf(t));
            state.coeffs[idx] -= lr * m_hat / (v_hat.sqrt() + 1e-8);
            state.coeffs[idx] = state.coeffs[idx].clamp(0.0, 4.0);
        }
        state.step += 1;
        if step + 1 == teacher_start_step {
            warm_start_coeffs = state.coeffs;
        }
    }

    if teacher_start_step == 0 {
        warm_start_coeffs = state.coeffs;
    }

    TrainResult {
        warm_start_coeffs,
        final_coeffs: state.coeffs,
    }
}

fn batch_slice<'a>(
    records: &'a [C3C7Example],
    step: usize,
    batch_size: usize,
) -> &'a [C3C7Example] {
    if records.len() <= batch_size {
        return records;
    }
    let offset = (step * batch_size) % records.len();
    let end = (offset + batch_size).min(records.len());
    &records[offset..end]
}

fn evaluate_tokens(
    tokens: &[usize],
    packed_tables: &PackedTables,
    local_tables: &LocalWindowTables,
    teacher_rows: &[DenseTeacherRow],
    warm_coeffs: &[f64; EXPERT_COUNT],
    distilled_coeffs: &[f64; EXPERT_COUNT],
) -> Result<EvalStats, String> {
    let mut packed = 0.0;
    let mut local = 0.0;
    let mut bank4 = 0.0;
    let mut bank8 = 0.0;
    let mut bank16 = 0.0;
    let mut warm_start = 0.0;
    let mut distilled = 0.0;
    let mut oracle = 0.0;
    let mut records = 0usize;
    let mut history = Vec::new();
    let mut bank_states = [
        DecayBankState::new(VOCAB_SIZE, HALF_LIVES[0]),
        DecayBankState::new(VOCAB_SIZE, HALF_LIVES[1]),
        DecayBankState::new(VOCAB_SIZE, HALF_LIVES[2]),
    ];

    for pos in 0..tokens.len() {
        let packed_dist = packed_distribution(
            packed_tables,
            prev2(&history),
            prev1(&history),
            ALPHA_BIGRAM,
            ALPHA_TRIGRAM,
        );
        let local_dist = local_distribution(local_tables, &history, &packed_dist);
        let bank4_dist = bank_states[0].distribution(&packed_dist);
        let bank8_dist = bank_states[1].distribution(&packed_dist);
        let bank16_dist = bank_states[2].distribution(&packed_dist);
        let expert_dists = [packed_dist, local_dist, bank4_dist, bank8_dist, bank16_dist];

        if pos >= LOCAL_WINDOW && pos < teacher_rows.len() && !teacher_rows[pos].pairs.is_empty() {
            let gold = tokens[pos];
            packed += -expert_dists[0][gold].max(EPS).log2();
            local += -expert_dists[1][gold].max(EPS).log2();
            bank4 += -expert_dists[2][gold].max(EPS).log2();
            bank8 += -expert_dists[3][gold].max(EPS).log2();
            bank16 += -expert_dists[4][gold].max(EPS).log2();
            let warm_dist = combine_experts_additive_logits(&expert_dists, warm_coeffs);
            let distilled_dist = combine_experts_additive_logits(&expert_dists, distilled_coeffs);
            warm_start += -warm_dist[gold].max(EPS).log2();
            distilled += -distilled_dist[gold].max(EPS).log2();
            let total_teacher = teacher_rows[pos]
                .pairs
                .iter()
                .map(|(_, count)| *count as f64)
                .sum::<f64>()
                .max(EPS);
            let teacher_gold_prob = teacher_rows[pos]
                .pairs
                .iter()
                .find_map(|(token, count)| {
                    if *token == gold {
                        Some(*count as f64 / total_teacher)
                    } else {
                        None
                    }
                })
                .unwrap_or(EPS);
            oracle += -teacher_gold_prob.max(EPS).log2();
            records += 1;
        }

        let token = tokens[pos];
        history.push(token);
        if history.len() > LOCAL_WINDOW {
            history.remove(0);
        }
        for bank in &mut bank_states {
            bank.adapt(token);
        }
    }

    let scale = 1.0 / records.max(1) as f64;
    Ok(EvalStats {
        packed: packed * scale,
        local: local * scale,
        bank4: bank4 * scale,
        bank8: bank8 * scale,
        bank16: bank16 * scale,
        warm_start: warm_start * scale,
        distilled: distilled * scale,
        oracle: oracle * scale,
        records,
    })
}

fn collect_c3c7_examples(
    tokens: &[usize],
    packed_tables: &PackedTables,
    local_tables: &LocalWindowTables,
    teacher_rows: &[DenseTeacherRow],
    stride: usize,
) -> Result<Vec<C3C7Example>, String> {
    let mut records = Vec::new();
    let mut history = Vec::new();
    let mut bank_states = [
        DecayBankState::new(VOCAB_SIZE, HALF_LIVES[0]),
        DecayBankState::new(VOCAB_SIZE, HALF_LIVES[1]),
        DecayBankState::new(VOCAB_SIZE, HALF_LIVES[2]),
    ];
    let available = ((tokens.len().saturating_sub(LOCAL_WINDOW)) / stride.max(1)).max(1);
    let sample_every = ((available + TRAIN_EXAMPLE_CAP - 1) / TRAIN_EXAMPLE_CAP.max(1)).max(1);
    for pos in 0..tokens.len() {
        let packed = packed_distribution(
            packed_tables,
            prev2(&history),
            prev1(&history),
            ALPHA_BIGRAM,
            ALPHA_TRIGRAM,
        );
        let local = local_distribution(local_tables, &history, &packed);
        let bank4 = bank_states[0].distribution(&packed);
        let bank8 = bank_states[1].distribution(&packed);
        let bank16 = bank_states[2].distribution(&packed);
        let sample_index = pos.saturating_sub(LOCAL_WINDOW) / stride.max(1);
        if pos >= LOCAL_WINDOW
            && pos < teacher_rows.len()
            && (pos - LOCAL_WINDOW) % stride == 0
            && sample_index % sample_every == 0
        {
            let expert_dists = [packed, local, bank4, bank8, bank16];
            let gold = tokens[pos];
            let teacher_pairs = &teacher_rows[pos].pairs;
            let mut teacher_tokens = Vec::new();
            let total_teacher = teacher_pairs
                .iter()
                .map(|(_, count)| *count as f64)
                .sum::<f64>()
                .max(EPS);
            let mut teacher_residual_expectations = [0.0; RESIDUAL_COUNT];
            for &(token, count) in teacher_pairs {
                let q = count as f64 / total_teacher;
                teacher_tokens.push(token);
                let packed_log = expert_dists[0][token].max(EPS).ln();
                for residual in 0..RESIDUAL_COUNT {
                    teacher_residual_expectations[residual] +=
                        q * (expert_dists[residual + 1][token].max(EPS).ln() - packed_log);
                }
            }
            let support_size = teacher_pairs.len();
            let support_weight = if support_size <= 1 {
                0.0
            } else {
                ((support_size as f64).ln() / 4.0_f64.ln()).clamp(0.0, 1.0)
            };
            let packed_log_probs = expert_dists[0]
                .iter()
                .map(|value| value.max(EPS).ln() as f32)
                .collect::<Vec<_>>();
            let residual_log_deltas = std::array::from_fn(|residual| {
                expert_dists[residual + 1]
                    .iter()
                    .zip(expert_dists[0].iter())
                    .map(|(expert, packed)| (expert.max(EPS).ln() - packed.max(EPS).ln()) as f32)
                    .collect::<Vec<_>>()
            });
            records.push(C3C7Example {
                gold,
                packed_log_probs,
                residual_log_deltas,
                teacher_tokens,
                teacher_residual_expectations,
                support_weight,
            });
        }
        let token = tokens[pos];
        history.push(token);
        if history.len() > LOCAL_WINDOW {
            history.remove(0);
        }
        for bank in &mut bank_states {
            bank.adapt(token);
        }
    }
    Ok(records)
}

fn build_dense_teacher_rows(
    tokens: &[usize],
    oracle_radius: usize,
    oracle_stride: usize,
) -> Result<Vec<DenseTeacherRow>, String> {
    let dataset = token_oracle_target_dataset_from_tokens(
        "chronohorn_c3c7",
        tokens,
        oracle_radius,
        oracle_stride,
    )?;
    let mut dense = vec![DenseTeacherRow { pairs: Vec::new() }; tokens.len()];
    for (position, pairs) in token_oracle_teacher_candidate_pairs(&dataset) {
        if position < dense.len() {
            dense[position].pairs = pairs;
        }
    }
    Ok(dense)
}

fn build_local_window_tables(
    tokens: &[usize],
    vocab_size: usize,
) -> Result<LocalWindowTables, String> {
    if tokens.len() <= LOCAL_WINDOW {
        return Err("need more tokens than local window".to_string());
    }
    let mut contexts: HashMap<u64, SparseCounts> = HashMap::new();
    for window in tokens.windows(LOCAL_WINDOW + 1) {
        let next = window[LOCAL_WINDOW];
        if next >= vocab_size {
            return Err(format!("token out of vocab in local table: {next}"));
        }
        let key = local_context_key(&window[..LOCAL_WINDOW]);
        let entry = contexts.entry(key).or_default();
        entry.total += 1.0;
        if let Some((_, count)) = entry.counts.iter_mut().find(|(token, _)| *token == next) {
            *count += 1.0;
        } else {
            entry.counts.push((next, 1.0));
        }
    }
    let _ = vocab_size;
    Ok(LocalWindowTables { contexts })
}

fn local_context_key(context: &[usize]) -> u64 {
    let mut key = 0u64;
    for (idx, &token) in context.iter().enumerate() {
        key = key.wrapping_mul(LOCAL_HASH_MULS[idx % LOCAL_HASH_MULS.len()]);
        key ^= token as u64 + 1;
    }
    key
}

fn local_distribution(
    local_tables: &LocalWindowTables,
    history: &[usize],
    packed: &[f64],
) -> Vec<f64> {
    if history.len() < LOCAL_WINDOW {
        return packed.to_vec();
    }
    let key = local_context_key(&history[history.len() - LOCAL_WINDOW..]);
    let Some(row) = local_tables.contexts.get(&key) else {
        return packed.to_vec();
    };
    let denom = row.total + LOCAL_SMOOTHING;
    let mut out = packed
        .iter()
        .map(|value| (LOCAL_SMOOTHING * *value) / denom.max(EPS))
        .collect::<Vec<_>>();
    for &(token, count) in &row.counts {
        if token < out.len() {
            out[token] += count / denom.max(EPS);
        }
    }
    normalize(&mut out);
    out
}

fn packed_distribution(
    tables: &PackedTables,
    prev2: Option<usize>,
    prev1: Option<usize>,
    alpha_bigram: f64,
    alpha_trigram: f64,
) -> Vec<f64> {
    let vocab = tables.vocab_size;
    let p_uni = &tables.unigram_probs;
    let p_bigram = if let Some(p1) = prev1 {
        let row_start = p1 * vocab;
        let total = tables.bigram_totals[p1];
        let denom = (total + alpha_bigram).max(EPS);
        let mut out = vec![0.0; vocab];
        for tok in 0..vocab {
            out[tok] = (tables.bigram_counts[row_start + tok] + alpha_bigram * p_uni[tok]) / denom;
        }
        out
    } else {
        p_uni.clone()
    };
    if let (Some(p2), Some(p1)) = (prev2, prev1) {
        let bucket = ((p2 as u64 * TRIGRAM_HASH_MUL_A + p1 as u64 * TRIGRAM_HASH_MUL_B)
            % tables.trigram_buckets as u64) as usize;
        let row_start = bucket * vocab;
        let total = tables.trigram_totals[bucket];
        let denom = (total + alpha_trigram).max(EPS);
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

fn prev1(history: &[usize]) -> Option<usize> {
    history.last().copied()
}

fn prev2(history: &[usize]) -> Option<usize> {
    if history.len() >= 2 {
        Some(history[history.len() - 2])
    } else {
        None
    }
}

fn normalize(values: &mut [f64]) {
    let total = values.iter().sum::<f64>().max(EPS);
    for value in values {
        *value /= total;
    }
}

fn combine_experts_additive_logits(
    experts: &[Vec<f64>; EXPERT_COUNT],
    coeffs: &[f64; EXPERT_COUNT],
) -> Vec<f64> {
    let mut logits = [0.0f64; VOCAB_SIZE];
    for tok in 0..VOCAB_SIZE {
        let packed_log = experts[0][tok].max(EPS).ln();
        let mut combined = packed_log;
        for residual in 0..RESIDUAL_COUNT {
            combined +=
                coeffs[residual + 1] * (experts[residual + 1][tok].max(EPS).ln() - packed_log);
        }
        logits[tok] = combined;
    }
    softmax_vocab(&logits).to_vec()
}

fn softmax_vocab(logits: &[f64; VOCAB_SIZE]) -> [f64; VOCAB_SIZE] {
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut exps = [0.0; VOCAB_SIZE];
    let mut total = 0.0;
    for tok in 0..VOCAB_SIZE {
        exps[tok] = (logits[tok] - max).exp();
        total += exps[tok];
    }
    let denom = total.max(EPS);
    for value in &mut exps {
        *value /= denom;
    }
    exps
}

pub fn run_token_c3c7_from_data_root(
    root: &Path,
    train_token_budget: usize,
    trigram_buckets: usize,
    val_token_budget: usize,
    train_stride: usize,
    train_steps: usize,
    teacher_start_step: usize,
    oracle_radius: usize,
    oracle_stride: usize,
    alpha_nll: f64,
    beta_support: f64,
    gamma_rank: f64,
) -> Result<TokenC3C7Report, String> {
    Ok(train_token_c3c7_from_data_root(
        root,
        train_token_budget,
        trigram_buckets,
        val_token_budget,
        train_stride,
        train_steps,
        teacher_start_step,
        oracle_radius,
        oracle_stride,
        alpha_nll,
        beta_support,
        gamma_rank,
    )?
    .report
    .clone())
}

pub fn render_token_c3c7_report(report: &TokenC3C7Report) -> String {
    let mut out = String::new();
    out.push_str("chronohorn_token_c3c7\n");
    out.push_str(&format!(
        "train_token_budget: {}\n",
        report.train_token_budget
    ));
    out.push_str(&format!("trigram_buckets: {}\n", report.trigram_buckets));
    out.push_str(&format!("val_token_budget: {}\n", report.val_token_budget));
    out.push_str(&format!("train_stride: {}\n", report.train_stride));
    out.push_str(&format!("train_steps: {}\n", report.train_steps));
    out.push_str(&format!(
        "teacher_start_step: {}\n",
        report.teacher_start_step
    ));
    out.push_str(&format!("oracle_radius: {}\n", report.oracle_radius));
    out.push_str(&format!("oracle_stride: {}\n", report.oracle_stride));
    out.push_str(&format!("alpha_nll: {:.3}\n", report.alpha_nll));
    out.push_str(&format!("beta_support: {:.3}\n", report.beta_support));
    out.push_str(&format!("gamma_rank: {:.3}\n", report.gamma_rank));
    out.push_str(&format!("train_records: {}\n", report.train_records));
    out.push_str(&format!("tune_records: {}\n", report.tune_records));
    out.push_str(&format!("eval_records: {}\n", report.eval_records));
    out.push_str(&format!(
        "oracle_target_source: {}\n",
        report.oracle_target_source
    ));
    out.push_str(&format!(
        "warm_start_coeffs: {:?}\n",
        report.warm_start_coeffs
    ));
    out.push_str(&format!(
        "distilled_coeffs: {:?}\n",
        report.distilled_coeffs
    ));
    out.push_str(&format!("selected_stage: {}\n", report.selected_stage));
    out.push_str(&format!("tune_bpt_packed: {:.6}\n", report.tune_bpt_packed));
    out.push_str(&format!("tune_bpt_local: {:.6}\n", report.tune_bpt_local));
    out.push_str(&format!("tune_bpt_bank4: {:.6}\n", report.tune_bpt_bank4));
    out.push_str(&format!("tune_bpt_bank8: {:.6}\n", report.tune_bpt_bank8));
    out.push_str(&format!("tune_bpt_bank16: {:.6}\n", report.tune_bpt_bank16));
    out.push_str(&format!(
        "tune_bpt_warm_start: {:.6}\n",
        report.tune_bpt_warm_start
    ));
    out.push_str(&format!(
        "tune_bpt_distilled: {:.6}\n",
        report.tune_bpt_distilled
    ));
    out.push_str(&format!("tune_bpt_oracle: {:.6}\n", report.tune_bpt_oracle));
    out.push_str(&format!("eval_bpt_packed: {:.6}\n", report.eval_bpt_packed));
    out.push_str(&format!("eval_bpt_local: {:.6}\n", report.eval_bpt_local));
    out.push_str(&format!("eval_bpt_bank4: {:.6}\n", report.eval_bpt_bank4));
    out.push_str(&format!("eval_bpt_bank8: {:.6}\n", report.eval_bpt_bank8));
    out.push_str(&format!("eval_bpt_bank16: {:.6}\n", report.eval_bpt_bank16));
    out.push_str(&format!(
        "eval_bpt_warm_start: {:.6}\n",
        report.eval_bpt_warm_start
    ));
    out.push_str(&format!(
        "eval_bpt_distilled: {:.6}\n",
        report.eval_bpt_distilled
    ));
    out.push_str(&format!("eval_bpt_oracle: {:.6}\n", report.eval_bpt_oracle));
    if let Some(value) = report.eval_bpb_packed {
        out.push_str(&format!("eval_bpb_packed: {:.6}\n", value));
    }
    if let Some(value) = report.eval_bpb_local {
        out.push_str(&format!("eval_bpb_local: {:.6}\n", value));
    }
    if let Some(value) = report.eval_bpb_bank4 {
        out.push_str(&format!("eval_bpb_bank4: {:.6}\n", value));
    }
    if let Some(value) = report.eval_bpb_bank8 {
        out.push_str(&format!("eval_bpb_bank8: {:.6}\n", value));
    }
    if let Some(value) = report.eval_bpb_bank16 {
        out.push_str(&format!("eval_bpb_bank16: {:.6}\n", value));
    }
    if let Some(value) = report.eval_bpb_warm_start {
        out.push_str(&format!("eval_bpb_warm_start: {:.6}\n", value));
    }
    if let Some(value) = report.eval_bpb_distilled {
        out.push_str(&format!("eval_bpb_distilled: {:.6}\n", value));
    }
    if let Some(value) = report.eval_bpb_oracle {
        out.push_str(&format!("eval_bpb_oracle: {:.6}\n", value));
    }
    out
}
