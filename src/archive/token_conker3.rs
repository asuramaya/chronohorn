use std::collections::VecDeque;
use std::f32::consts::PI;
use std::path::Path;

use chronohorn_core::data::{compute_tokens_per_byte, take_train_tokens, take_val_tokens};
use chronohorn_core::protocol::{Runner, SampleOutputs};
use serde::Serialize;

const VOCAB_SIZE: usize = 1024;
const LOCAL_WINDOW: usize = 4;
const LOCAL_SCALE: f32 = 0.25;
const HALF_LIFE_MIN: f32 = 1.5;
const HALF_LIFE_MAX: f32 = 16.0;
const OSCILLATORY_FRAC: f32 = 0.875;
const OSC_PERIOD_MIN: f32 = 4.0;
const OSC_PERIOD_MAX: f32 = 64.0;
const BANK_GATE_SPAN: f32 = 0.5;
const EPS: f32 = 1e-8;

#[derive(Debug, Clone, Serialize)]
pub struct TokenConker3Report {
    pub train_token_budget: usize,
    pub val_token_budget: usize,
    pub scale: f64,
    pub train_stride: usize,
    pub epochs: usize,
    pub negatives: usize,
    pub learning_rate: f64,
    pub embedding_dim: usize,
    pub linear_modes: usize,
    pub non_osc_modes: usize,
    pub osc_pairs: usize,
    pub local_window: usize,
    pub local_scale: f64,
    pub static_non_osc_scale: f64,
    pub static_osc_scale: f64,
    pub train_samples: usize,
    pub eval_tokens: usize,
    pub eval_bpt_linear_only: f64,
    pub eval_bpt_local_only: f64,
    pub eval_bpt_combined: f64,
    pub eval_tokens_per_byte: Option<f64>,
    pub eval_bytes_per_token: Option<f64>,
    pub eval_bpb_linear_only: Option<f64>,
    pub eval_bpb_local_only: Option<f64>,
    pub eval_bpb_combined: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct TrainedTokenConker3 {
    report: TokenConker3Report,
    runner: TokenConker3Runner,
    eval_tokens: Vec<usize>,
}

impl TrainedTokenConker3 {
    pub fn report(&self) -> &TokenConker3Report {
        &self.report
    }

    pub fn runner(&self) -> &TokenConker3Runner {
        &self.runner
    }

    pub fn eval_tokens(&self) -> &[usize] {
        &self.eval_tokens
    }
}

#[derive(Debug, Clone)]
pub struct TokenConker3Runner {
    model: TokenConker3Model,
    state: Vec<f32>,
    history: VecDeque<usize>,
}

#[derive(Debug, Clone)]
struct TokenConker3Model {
    config: Conker3NativeConfig,
    linear_embeddings: Vec<f32>,
    local_embeddings: Vec<f32>,
    linear_drive_non_osc: Vec<f32>,
    linear_drive_osc: Vec<f32>,
    linear_readout: LinearReadout,
    local_readout: LinearReadout,
}

#[derive(Debug, Clone)]
struct Conker3NativeConfig {
    embedding_dim: usize,
    linear_modes: usize,
    non_osc_modes: usize,
    osc_pairs: usize,
    local_window: usize,
    local_scale: f32,
    non_osc_scale: f32,
    osc_scale: f32,
    non_osc_decays: Vec<f32>,
    osc_decays: Vec<f32>,
    osc_cos: Vec<f32>,
    osc_sin: Vec<f32>,
}

#[derive(Debug, Clone)]
struct LinearReadout {
    feature_dim: usize,
    weights: Vec<f32>,
    bias: Vec<f32>,
}

#[derive(Debug, Clone)]
struct EvalBreakdown {
    linear_bpt: f64,
    local_bpt: f64,
    combined_bpt: f64,
}

#[derive(Debug, Clone)]
struct XorShift64 {
    state: u64,
    cached_normal: Option<f32>,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        let state = if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        };
        Self {
            state,
            cached_normal: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        (bits as f32) / ((1u32 << 24) as f32)
    }

    fn normal_f32(&mut self) -> f32 {
        if let Some(value) = self.cached_normal.take() {
            return value;
        }
        let u1 = self.next_f32().max(1e-7);
        let u2 = self.next_f32().max(1e-7);
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;
        let z0 = radius * theta.cos();
        let z1 = radius * theta.sin();
        self.cached_normal = Some(z1);
        z0
    }

    fn next_usize(&mut self, upper: usize) -> usize {
        if upper <= 1 {
            return 0;
        }
        (self.next_u64() % upper as u64) as usize
    }
}

impl LinearReadout {
    fn new(vocab_size: usize, feature_dim: usize) -> Self {
        Self {
            feature_dim,
            weights: vec![0.0; vocab_size * feature_dim],
            bias: vec![0.0; vocab_size],
        }
    }

    fn row_score(&self, token: usize, features: &[f32]) -> f32 {
        let start = token * self.feature_dim;
        let row = &self.weights[start..start + self.feature_dim];
        self.bias[token] + dot(row, features)
    }

    fn full_scores(&self, features: &[f32]) -> Vec<f32> {
        let mut out = self.bias.clone();
        for (token, chunk) in self.weights.chunks_exact(self.feature_dim).enumerate() {
            out[token] += dot(chunk, features);
        }
        out
    }

    fn update_row(
        &mut self,
        token: usize,
        features: &[f32],
        gradient: f32,
        learning_rate: f32,
        weight_decay: f32,
    ) {
        let start = token * self.feature_dim;
        let row = &mut self.weights[start..start + self.feature_dim];
        for (weight, &feature) in row.iter_mut().zip(features.iter()) {
            let grad = gradient * feature + weight_decay * *weight;
            *weight -= learning_rate * grad;
        }
        self.bias[token] -= learning_rate * gradient;
    }
}

impl Runner for TokenConker3Runner {
    fn name(&self) -> &'static str {
        "TokenConker3Runner"
    }

    fn vocab_size(&self) -> usize {
        VOCAB_SIZE
    }

    fn score_chunk(
        &self,
        tokens: &[usize],
        sample_positions: &[usize],
    ) -> Result<SampleOutputs, String> {
        let mut sample_predictions = vec![Vec::new(); sample_positions.len()];
        let mut sample_gold_logprobs = vec![0.0; sample_positions.len()];
        let mut wanted = vec![Vec::new(); tokens.len()];
        for (sample_index, &pos) in sample_positions.iter().enumerate() {
            if pos >= tokens.len() {
                return Err(format!("sample position {pos} out of bounds"));
            }
            wanted[pos].push(sample_index);
        }

        let mut state = self.state.clone();
        let mut history = self.history.clone();
        for (pos, &token) in tokens.iter().enumerate() {
            if !wanted[pos].is_empty() {
                let dist = self.model.predict_distribution(&state, &history);
                let gold = dist
                    .get(token)
                    .copied()
                    .unwrap_or(f64::MIN_POSITIVE)
                    .max(f64::MIN_POSITIVE)
                    .ln();
                for &sample_index in &wanted[pos] {
                    sample_predictions[sample_index] = dist.clone();
                    sample_gold_logprobs[sample_index] = gold;
                }
            }
            self.model.advance_state(&mut state, &mut history, token);
        }

        Ok(SampleOutputs {
            sample_predictions,
            sample_gold_logprobs: Some(sample_gold_logprobs),
        })
    }

    fn adapt_chunk(&mut self, tokens: &[usize]) -> Result<(), String> {
        for &token in tokens {
            self.model
                .advance_state(&mut self.state, &mut self.history, token);
        }
        Ok(())
    }
}

pub fn train_token_conker3_from_data_root(
    root: &Path,
    train_token_budget: usize,
    val_token_budget: usize,
    scale: f64,
    train_stride: usize,
    epochs: usize,
    negatives: usize,
    learning_rate: f64,
) -> Result<TrainedTokenConker3, String> {
    if train_stride == 0 {
        return Err("train_stride must be positive".to_string());
    }
    if epochs == 0 {
        return Err("epochs must be positive".to_string());
    }
    if negatives == 0 {
        return Err("negatives must be positive".to_string());
    }

    let train_tokens = take_train_tokens(root, train_token_budget)?;
    let eval_tokens = take_val_tokens(root, val_token_budget)?;
    let mut rng = XorShift64::new(42);
    let config = build_native_config(scale as f32);
    let mut model = TokenConker3Model::new(&config, &mut rng);

    let mut train_samples = 0usize;
    for _ in 0..epochs {
        let mut state = vec![0.0; config.linear_modes];
        let mut history = VecDeque::with_capacity(config.local_window);
        for (index, &token) in train_tokens.iter().enumerate() {
            let (linear_features, local_features) = model.features_from_prefix(&state, &history);
            if index % train_stride == 0 {
                train_negative_sampling_step(
                    &mut model,
                    token,
                    &linear_features,
                    &local_features,
                    negatives,
                    learning_rate as f32,
                    &mut rng,
                );
                train_samples += 1;
            }
            model.advance_state(&mut state, &mut history, token);
        }
    }

    let eval_breakdown = evaluate_model(&model, &eval_tokens);
    let byte_accounting = compute_tokens_per_byte(root, &eval_tokens)?;
    let report = TokenConker3Report {
        train_token_budget,
        val_token_budget: eval_tokens.len(),
        scale,
        train_stride,
        epochs,
        negatives,
        learning_rate,
        embedding_dim: config.embedding_dim,
        linear_modes: config.linear_modes,
        non_osc_modes: config.non_osc_modes,
        osc_pairs: config.osc_pairs,
        local_window: config.local_window,
        local_scale: config.local_scale as f64,
        static_non_osc_scale: config.non_osc_scale as f64,
        static_osc_scale: config.osc_scale as f64,
        train_samples,
        eval_tokens: eval_tokens.len(),
        eval_bpt_linear_only: eval_breakdown.linear_bpt,
        eval_bpt_local_only: eval_breakdown.local_bpt,
        eval_bpt_combined: eval_breakdown.combined_bpt,
        eval_tokens_per_byte: byte_accounting.as_ref().map(|row| row.tokens_per_byte),
        eval_bytes_per_token: byte_accounting.as_ref().map(|row| row.bytes_per_token),
        eval_bpb_linear_only: byte_accounting
            .as_ref()
            .map(|row| eval_breakdown.linear_bpt * row.tokens_per_byte),
        eval_bpb_local_only: byte_accounting
            .as_ref()
            .map(|row| eval_breakdown.local_bpt * row.tokens_per_byte),
        eval_bpb_combined: byte_accounting
            .as_ref()
            .map(|row| eval_breakdown.combined_bpt * row.tokens_per_byte),
    };
    let runner = TokenConker3Runner {
        model,
        state: vec![0.0; config.linear_modes],
        history: VecDeque::with_capacity(config.local_window),
    };
    Ok(TrainedTokenConker3 {
        report,
        runner,
        eval_tokens,
    })
}

pub fn run_token_conker3_from_data_root(
    root: &Path,
    train_token_budget: usize,
    val_token_budget: usize,
    scale: f64,
    train_stride: usize,
    epochs: usize,
    negatives: usize,
    learning_rate: f64,
) -> Result<TokenConker3Report, String> {
    Ok(train_token_conker3_from_data_root(
        root,
        train_token_budget,
        val_token_budget,
        scale,
        train_stride,
        epochs,
        negatives,
        learning_rate,
    )?
    .report)
}

pub fn render_token_conker3_report(report: &TokenConker3Report) -> String {
    let mut out = String::new();
    out.push_str("chronohorn_token_conker3\n");
    out.push_str(&format!(
        "train_token_budget: {}\n",
        report.train_token_budget
    ));
    out.push_str(&format!("val_token_budget: {}\n", report.val_token_budget));
    out.push_str(&format!("scale: {:.3}\n", report.scale));
    out.push_str(&format!("train_stride: {}\n", report.train_stride));
    out.push_str(&format!("epochs: {}\n", report.epochs));
    out.push_str(&format!("negatives: {}\n", report.negatives));
    out.push_str(&format!("learning_rate: {:.6}\n", report.learning_rate));
    out.push_str(&format!("embedding_dim: {}\n", report.embedding_dim));
    out.push_str(&format!("linear_modes: {}\n", report.linear_modes));
    out.push_str(&format!("non_osc_modes: {}\n", report.non_osc_modes));
    out.push_str(&format!("osc_pairs: {}\n", report.osc_pairs));
    out.push_str(&format!("local_window: {}\n", report.local_window));
    out.push_str(&format!("local_scale: {:.4}\n", report.local_scale));
    out.push_str(&format!(
        "static_bank_scales: non_osc={:.4} osc={:.4}\n",
        report.static_non_osc_scale, report.static_osc_scale
    ));
    out.push_str(&format!("train_samples: {}\n", report.train_samples));
    out.push_str(&format!("eval_tokens: {}\n", report.eval_tokens));
    out.push_str(&format!(
        "eval_bpt_linear_only: {:.6}\n",
        report.eval_bpt_linear_only
    ));
    out.push_str(&format!(
        "eval_bpt_local_only: {:.6}\n",
        report.eval_bpt_local_only
    ));
    out.push_str(&format!(
        "eval_bpt_combined: {:.6}\n",
        report.eval_bpt_combined
    ));
    if let Some(value) = report.eval_tokens_per_byte {
        out.push_str(&format!("eval_tokens_per_byte: {:.6}\n", value));
    }
    if let Some(value) = report.eval_bytes_per_token {
        out.push_str(&format!("eval_bytes_per_token: {:.6}\n", value));
    }
    if let Some(value) = report.eval_bpb_linear_only {
        out.push_str(&format!("eval_bpb_linear_only: {:.6}\n", value));
    }
    if let Some(value) = report.eval_bpb_local_only {
        out.push_str(&format!("eval_bpb_local_only: {:.6}\n", value));
    }
    if let Some(value) = report.eval_bpb_combined {
        out.push_str(&format!("eval_bpb_combined: {:.6}\n", value));
    }
    out
}

impl TokenConker3Model {
    fn new(config: &Conker3NativeConfig, rng: &mut XorShift64) -> Self {
        let linear_embeddings = random_table(VOCAB_SIZE, config.embedding_dim, rng, 1.0);
        let local_embeddings = random_table(VOCAB_SIZE, config.embedding_dim, rng, 1.0);
        let (linear_drive_non_osc, linear_drive_osc) =
            build_drive_tables(config, &linear_embeddings, rng);
        let linear_feature_dim = config.linear_modes + config.embedding_dim;
        let local_feature_dim = config.local_window * config.embedding_dim;
        Self {
            config: config.clone(),
            linear_embeddings,
            local_embeddings,
            linear_drive_non_osc,
            linear_drive_osc,
            linear_readout: LinearReadout::new(VOCAB_SIZE, linear_feature_dim),
            local_readout: LinearReadout::new(VOCAB_SIZE, local_feature_dim),
        }
    }

    fn linear_embedding(&self, token: Option<usize>) -> &[f32] {
        static ZERO: [f32; 0] = [];
        match token {
            Some(index) if index < VOCAB_SIZE => {
                let start = index * self.config.embedding_dim;
                &self.linear_embeddings[start..start + self.config.embedding_dim]
            }
            _ => &ZERO,
        }
    }

    fn local_embedding(&self, token: Option<usize>) -> &[f32] {
        static ZERO: [f32; 0] = [];
        match token {
            Some(index) if index < VOCAB_SIZE => {
                let start = index * self.config.embedding_dim;
                &self.local_embeddings[start..start + self.config.embedding_dim]
            }
            _ => &ZERO,
        }
    }

    fn features_from_prefix(
        &self,
        state: &[f32],
        history: &VecDeque<usize>,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut linear_features =
            Vec::with_capacity(self.config.linear_modes + self.config.embedding_dim);
        for (index, &value) in state.iter().enumerate() {
            let scaled = if index < self.config.non_osc_modes {
                value * self.config.non_osc_scale
            } else {
                value * self.config.osc_scale
            };
            linear_features.push(scaled);
        }
        if let Some(&token) = history.front() {
            let embed = self.linear_embedding(Some(token));
            linear_features.extend_from_slice(embed);
        } else {
            linear_features.resize(linear_features.len() + self.config.embedding_dim, 0.0);
        }

        let mut local_features =
            Vec::with_capacity(self.config.local_window * self.config.embedding_dim);
        for offset in 0..self.config.local_window {
            let embed = history
                .get(offset)
                .map(|&token| self.local_embedding(Some(token)))
                .unwrap_or(&[]);
            if embed.is_empty() {
                local_features.resize(local_features.len() + self.config.embedding_dim, 0.0);
            } else {
                local_features.extend_from_slice(embed);
            }
        }
        (linear_features, local_features)
    }

    fn predict_distribution(&self, state: &[f32], history: &VecDeque<usize>) -> Vec<f64> {
        let (linear_features, local_features) = self.features_from_prefix(state, history);
        let mut logits = self.linear_readout.full_scores(&linear_features);
        let local_logits = self.local_readout.full_scores(&local_features);
        for (logit, local) in logits.iter_mut().zip(local_logits.iter()) {
            *logit += self.config.local_scale * *local;
        }
        softmax_f64(&logits)
    }

    fn advance_state(&self, state: &mut [f32], history: &mut VecDeque<usize>, token: usize) {
        if token >= VOCAB_SIZE {
            return;
        }
        for mode in 0..self.config.non_osc_modes {
            let drive = self.linear_drive_non_osc[token * self.config.non_osc_modes + mode];
            state[mode] = self.config.non_osc_decays[mode] * state[mode] + drive;
        }
        for pair in 0..self.config.osc_pairs {
            let drive = self.linear_drive_osc[token * self.config.osc_pairs + pair];
            let decay = self.config.osc_decays[pair];
            let cos = self.config.osc_cos[pair];
            let sin = self.config.osc_sin[pair];
            let index = self.config.non_osc_modes + 2 * pair;
            let prev_cos = state[index];
            let prev_sin = state[index + 1];
            state[index] = decay * (cos * prev_cos - sin * prev_sin) + drive;
            state[index + 1] = decay * (sin * prev_cos + cos * prev_sin);
        }
        if history.len() == self.config.local_window {
            history.pop_back();
        }
        history.push_front(token);
    }
}

fn build_native_config(scale: f32) -> Conker3NativeConfig {
    let scale = scale.max(0.125);
    let embedding_dim = ((32.0 * scale).round() as usize).max(4);
    let linear_modes = ((256.0 * scale).round() as usize).max(16);
    let osc_pairs =
        (((linear_modes as f32) * OSCILLATORY_FRAC).floor() as usize / 2).min(linear_modes / 2);
    let non_osc_modes = linear_modes.saturating_sub(2 * osc_pairs);
    let non_osc_decays = logspace_decays(HALF_LIFE_MIN, HALF_LIFE_MAX, non_osc_modes);
    let osc_half_lives = logspace_values(HALF_LIFE_MIN.max(2.0), HALF_LIFE_MAX, osc_pairs);
    let osc_periods = logspace_values(OSC_PERIOD_MIN, OSC_PERIOD_MAX, osc_pairs);
    let osc_decays = osc_half_lives
        .iter()
        .map(|&half_life| (0.5f32).powf(1.0 / half_life.max(1e-6)))
        .collect::<Vec<_>>();
    let osc_cos = osc_periods
        .iter()
        .map(|&period| (2.0 * PI / period.max(1e-6)).cos())
        .collect::<Vec<_>>();
    let osc_sin = osc_periods
        .iter()
        .map(|&period| (2.0 * PI / period.max(1e-6)).sin())
        .collect::<Vec<_>>();
    let gate_value = 1.0 + BANK_GATE_SPAN * 0.0f32.tanh();
    Conker3NativeConfig {
        embedding_dim,
        linear_modes,
        non_osc_modes,
        osc_pairs,
        local_window: LOCAL_WINDOW,
        local_scale: LOCAL_SCALE,
        non_osc_scale: gate_value,
        osc_scale: gate_value,
        non_osc_decays,
        osc_decays,
        osc_cos,
        osc_sin,
    }
}

fn logspace_values(start: f32, end: f32, count: usize) -> Vec<f32> {
    if count == 0 {
        return Vec::new();
    }
    if count == 1 {
        return vec![start];
    }
    let start_ln = start.ln();
    let end_ln = end.ln();
    (0..count)
        .map(|index| {
            let alpha = index as f32 / (count - 1) as f32;
            ((1.0 - alpha) * start_ln + alpha * end_ln).exp()
        })
        .collect()
}

fn logspace_decays(start: f32, end: f32, count: usize) -> Vec<f32> {
    logspace_values(start, end, count)
        .into_iter()
        .map(|half_life| (0.5f32).powf(1.0 / half_life.max(1e-6)))
        .collect()
}

fn random_table(rows: usize, cols: usize, rng: &mut XorShift64, scale: f32) -> Vec<f32> {
    let std = scale / (cols as f32).sqrt().max(1.0);
    (0..rows * cols).map(|_| rng.normal_f32() * std).collect()
}

fn build_drive_tables(
    config: &Conker3NativeConfig,
    linear_embeddings: &[f32],
    rng: &mut XorShift64,
) -> (Vec<f32>, Vec<f32>) {
    let non_osc_proj = random_table(config.embedding_dim, config.non_osc_modes, rng, 1.0);
    let osc_proj_shared = random_table(config.embedding_dim, config.osc_pairs, rng, 1.0);
    let mut non_osc = vec![0.0; VOCAB_SIZE * config.non_osc_modes];
    let mut osc = vec![0.0; VOCAB_SIZE * config.osc_pairs];
    for token in 0..VOCAB_SIZE {
        let embed =
            &linear_embeddings[token * config.embedding_dim..(token + 1) * config.embedding_dim];
        for mode in 0..config.non_osc_modes {
            let mut total = 0.0;
            for dim in 0..config.embedding_dim {
                total += embed[dim] * non_osc_proj[dim * config.non_osc_modes + mode];
            }
            non_osc[token * config.non_osc_modes + mode] = total;
        }
        for pair in 0..config.osc_pairs {
            let mut total = 0.0;
            for dim in 0..config.embedding_dim {
                total += embed[dim] * osc_proj_shared[dim * config.osc_pairs + pair];
            }
            osc[token * config.osc_pairs + pair] = total;
        }
    }
    (non_osc, osc)
}

fn train_negative_sampling_step(
    model: &mut TokenConker3Model,
    gold: usize,
    linear_features: &[f32],
    local_features: &[f32],
    negatives: usize,
    learning_rate: f32,
    rng: &mut XorShift64,
) {
    let gold_linear = model.linear_readout.row_score(gold, linear_features);
    let gold_local = model.local_readout.row_score(gold, local_features);
    let gold_score = gold_linear + model.config.local_scale * gold_local;
    let gold_grad = sigmoid(gold_score) - 1.0;
    model
        .linear_readout
        .update_row(gold, linear_features, gold_grad, learning_rate, 1e-6);
    model.local_readout.update_row(
        gold,
        local_features,
        gold_grad * model.config.local_scale,
        learning_rate,
        1e-6,
    );

    for _ in 0..negatives {
        let mut negative = rng.next_usize(VOCAB_SIZE);
        if negative == gold {
            negative = (negative + 1) % VOCAB_SIZE;
        }
        let neg_linear = model.linear_readout.row_score(negative, linear_features);
        let neg_local = model.local_readout.row_score(negative, local_features);
        let neg_score = neg_linear + model.config.local_scale * neg_local;
        let neg_grad = sigmoid(neg_score);
        model
            .linear_readout
            .update_row(negative, linear_features, neg_grad, learning_rate, 1e-6);
        model.local_readout.update_row(
            negative,
            local_features,
            neg_grad * model.config.local_scale,
            learning_rate,
            1e-6,
        );
    }
}

fn evaluate_model(model: &TokenConker3Model, tokens: &[usize]) -> EvalBreakdown {
    let mut state = vec![0.0; model.config.linear_modes];
    let mut history = VecDeque::with_capacity(model.config.local_window);
    let mut linear_nats = 0.0f64;
    let mut local_nats = 0.0f64;
    let mut combined_nats = 0.0f64;
    let mut count = 0usize;

    for &token in tokens {
        let (linear_features, local_features) = model.features_from_prefix(&state, &history);
        let linear_logits = model.linear_readout.full_scores(&linear_features);
        let local_logits = model.local_readout.full_scores(&local_features);
        let mut combined_logits = linear_logits.clone();
        for (combined, local) in combined_logits.iter_mut().zip(local_logits.iter()) {
            *combined += model.config.local_scale * *local;
        }

        linear_nats += negative_log_prob_from_logits(&linear_logits, token);
        local_nats += negative_log_prob_from_logits(&local_logits, token);
        combined_nats += negative_log_prob_from_logits(&combined_logits, token);
        count += 1;

        model.advance_state(&mut state, &mut history, token);
    }

    let denom = (count as f64).max(1.0);
    let to_bits = 1.0 / std::f64::consts::LN_2;
    EvalBreakdown {
        linear_bpt: linear_nats * to_bits / denom,
        local_bpt: local_nats * to_bits / denom,
        combined_bpt: combined_nats * to_bits / denom,
    }
}

fn negative_log_prob_from_logits(logits: &[f32], gold: usize) -> f64 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut sum = 0.0f64;
    for &value in logits {
        sum += ((value as f64) - max).exp();
    }
    let gold_logit = logits.get(gold).copied().unwrap_or(-1e9) as f64;
    -(gold_logit - max - sum.max(EPS as f64).ln())
}

fn softmax_f64(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut values = Vec::with_capacity(logits.len());
    let mut sum = 0.0f64;
    for &logit in logits {
        let value = ((logit as f64) - max).exp();
        values.push(value);
        sum += value;
    }
    let denom = sum.max(EPS as f64);
    for value in &mut values {
        *value /= denom;
    }
    let residual: f64 = 1.0 - values.iter().sum::<f64>();
    if let Some(first) = values.first_mut() {
        *first += residual;
    }
    values
}

fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter().zip(rhs.iter()).map(|(a, b)| *a * *b).sum()
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        let exp = (-value).exp();
        1.0 / (1.0 + exp)
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}
