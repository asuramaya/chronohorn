use std::collections::{BTreeMap, HashMap, VecDeque};
use std::path::Path;

use chronohorn_core::checkpoint::{F32Array, load_named_f32_arrays};
use chronohorn_core::data::{compute_tokens_per_byte, materialize_data_root, take_val_tokens};
use chronohorn_core::protocol::{Runner, SampleOutputs};
use chronohorn_runtime::{
    load_export_bundle_material, load_json_file, resolve_export_reference_path,
};
use serde::{Deserialize, Serialize};

const DEFAULT_VOCAB_SIZE: usize = 1024;
const EPS: f32 = 1e-8;
const LOGPROB_MATCH_TOLERANCE: f64 = 1e-4;

#[cfg(any(target_os = "macos", target_os = "linux"))]
const CBLAS_ROW_MAJOR: i32 = 101;
#[cfg(any(target_os = "macos", target_os = "linux"))]
const CBLAS_NO_TRANS: i32 = 111;
#[cfg(any(target_os = "macos", target_os = "linux"))]
const CBLAS_TRANS: i32 = 112;

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_sgemm(
        order: i32,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

#[cfg(target_os = "linux")]
#[link(name = "openblas")]
unsafe extern "C" {
    fn cblas_sgemm(
        order: i32,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

#[derive(Debug, Clone, Deserialize)]
struct Conker3CheckpointSummary {
    config: Conker3CheckpointSummaryConfig,
    dataset: Conker3CheckpointSummaryDataset,
    model: Conker3CheckpointSummaryModel,
}

#[derive(Debug, Clone, Deserialize)]
struct Conker3CheckpointSummaryConfig {
    train: Conker3CheckpointSummaryTrain,
}

#[derive(Debug, Clone, Deserialize)]
struct Conker3CheckpointSummaryTrain {
    seq_len: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct Conker3CheckpointSummaryDataset {
    test_tokens_per_byte: Option<f64>,
    test_bytes_per_token: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct Conker3CheckpointSummaryModel {
    linear_modes: usize,
    local_window: usize,
    share_embedding: bool,
    #[serde(default = "default_linear_readout_kind")]
    linear_readout_kind: String,
    #[serde(default = "default_linear_readout_num_experts")]
    linear_readout_num_experts: usize,
    oscillatory_frac: f64,
    oscillatory_period_min: f64,
    oscillatory_period_max: f64,
    static_bank_gate: bool,
    bank_gate_span: f64,
    embedding_dim: usize,
    linear_hidden: Vec<usize>,
    local_hidden: Vec<usize>,
    local_scale: f64,
    mix_mode: String,
    params: usize,
    variant: String,
}

fn default_linear_readout_kind() -> String {
    "mlp".to_string()
}

fn default_linear_readout_num_experts() -> usize {
    4
}

#[derive(Debug, Clone, Serialize)]
pub struct TokenConker3CheckpointReport {
    pub checkpoint_path: String,
    pub summary_path: String,
    pub train_seq_len: usize,
    pub embedding_dim: usize,
    pub linear_modes: usize,
    pub non_osc_modes: usize,
    pub osc_pairs: usize,
    pub local_window: usize,
    pub static_non_osc_scale: f64,
    pub static_osc_scale: f64,
    pub local_scale: f64,
    pub model_variant: String,
    pub parameter_count: usize,
    pub eval_tokens: usize,
    pub eval_bpt_linear_only: Option<f64>,
    pub eval_bpt_local_only: Option<f64>,
    pub eval_bpt_combined: f64,
    pub eval_tokens_per_byte: Option<f64>,
    pub eval_bytes_per_token: Option<f64>,
    pub eval_bpb_linear_only: Option<f64>,
    pub eval_bpb_local_only: Option<f64>,
    pub eval_bpb_combined: Option<f64>,
    pub summary_test_tokens_per_byte: Option<f64>,
    pub summary_test_bytes_per_token: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TokenConker3BoundaryCheckpointReport {
    pub checkpoint_path: String,
    pub summary_path: String,
    pub tokenizer_vocab_path: String,
    pub train_seq_len: usize,
    pub eval_tokens: usize,
    pub boundary_mode: String,
    pub flushed_spans: usize,
    pub mean_span_len: f64,
    pub max_span_len: usize,
    pub boundary_tokens_seen: usize,
    pub eval_bpt_base: f64,
    pub eval_bpt_boundary: f64,
    pub eval_tokens_per_byte: Option<f64>,
    pub eval_bytes_per_token: Option<f64>,
    pub eval_bpb_base: Option<f64>,
    pub eval_bpb_boundary: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct TokenConker3CheckpointMetadata {
    pub checkpoint_path: String,
    pub summary_path: String,
    pub train_seq_len: usize,
    pub embedding_dim: usize,
    pub linear_modes: usize,
    pub non_osc_modes: usize,
    pub osc_pairs: usize,
    pub local_window: usize,
    pub static_non_osc_scale: f64,
    pub static_osc_scale: f64,
    pub local_scale: f64,
    pub model_variant: String,
    pub parameter_count: usize,
    pub summary_test_tokens_per_byte: Option<f64>,
    pub summary_test_bytes_per_token: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TokenConker3ExactCheckpointReport {
    pub checkpoint_path: String,
    pub summary_path: String,
    pub tokenizer_vocab_path: String,
    pub train_seq_len: usize,
    pub eval_tokens: usize,
    pub residual_cap: f64,
    pub exact1_weight: f64,
    pub exact2_weight: f64,
    pub exact3_weight: f64,
    pub delim2_weight: f64,
    pub special2_weight: f64,
    pub number2_weight: f64,
    pub markup2_weight: f64,
    pub attr2_weight: f64,
    pub base_center_weight: f64,
    pub eval_bpt_base: f64,
    pub eval_bpt_exact: f64,
    pub eval_tokens_per_byte: Option<f64>,
    pub eval_bytes_per_token: Option<f64>,
    pub eval_bpb_base: Option<f64>,
    pub eval_bpb_exact: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct Conker3ExportNotesPayload {
    #[serde(default)]
    summary_json: Option<String>,
    #[serde(default)]
    replay_fixture: Option<Conker3ExportReplayFixture>,
}

#[derive(Debug, Clone, Deserialize)]
struct Conker3ExportReplayFixture {
    split: String,
    token_count: usize,
    source_file: Option<String>,
    source_file_index: usize,
    source_token_offset: usize,
    input_token_ids: Vec<usize>,
    target_token_ids: Vec<usize>,
    #[serde(default)]
    runner_target_count: Option<usize>,
    #[serde(default)]
    expected_first_target_position: Option<usize>,
    #[serde(default)]
    expected_first_target_token_id: Option<usize>,
    #[serde(default)]
    expected_first_gold_logprob: Option<f64>,
    #[serde(default)]
    expected_prefix_gold_logprob_sum: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TokenConker3ExportBundleProbeReport {
    pub export_root: String,
    pub manifest_path: String,
    pub notes_path: String,
    pub summary_path: String,
    pub export_model_variant_id: String,
    pub summary_model_variant: String,
    pub train_seq_len: usize,
    pub fixture_split: String,
    pub fixture_token_count: usize,
    pub fixture_source_file: Option<String>,
    pub fixture_source_file_index: usize,
    pub fixture_source_token_offset: usize,
    pub runner_target_count: usize,
    pub expected_first_target_position: usize,
    pub expected_first_target_token_id: usize,
    pub actual_first_target_token_id: usize,
    pub expected_first_gold_logprob: f64,
    pub actual_first_gold_logprob: f64,
    pub first_gold_logprob_delta: f64,
    pub expected_prefix_gold_logprob_sum: f64,
    pub actual_prefix_gold_logprob_sum: f64,
    pub prefix_gold_logprob_delta: f64,
    pub first_logprob_match_tolerance: f64,
    pub prefix_logprob_match_tolerance: f64,
    pub first_gold_logprob_match: bool,
    pub prefix_gold_logprob_match: bool,
}

#[derive(Debug, Clone)]
pub struct LoadedTokenConker3Checkpoint {
    report: TokenConker3CheckpointReport,
    runner: TokenConker3CheckpointRunner,
    eval_tokens: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct LoadedTokenConker3ExactCheckpoint {
    report: TokenConker3ExactCheckpointReport,
    runner: TokenConker3ExactCheckpointRunner,
    eval_tokens: Vec<usize>,
}

impl LoadedTokenConker3Checkpoint {
    pub fn report(&self) -> &TokenConker3CheckpointReport {
        &self.report
    }

    pub fn runner(&self) -> &TokenConker3CheckpointRunner {
        &self.runner
    }

    pub fn eval_tokens(&self) -> &[usize] {
        &self.eval_tokens
    }
}

impl LoadedTokenConker3ExactCheckpoint {
    pub fn report(&self) -> &TokenConker3ExactCheckpointReport {
        &self.report
    }

    pub fn runner(&self) -> &TokenConker3ExactCheckpointRunner {
        &self.runner
    }

    pub fn eval_tokens(&self) -> &[usize] {
        &self.eval_tokens
    }
}

#[derive(Debug, Clone)]
pub struct TokenConker3CheckpointRunner {
    model: TokenConker3CheckpointModel,
    state: Vec<f32>,
    history: VecDeque<usize>,
}

#[derive(Debug, Clone)]
pub struct TokenConker3ExactCheckpointRunner {
    model: TokenConker3CheckpointModel,
    experts: ExactExpertModel,
    state: Vec<f32>,
    history: VecDeque<usize>,
    exact_state: ExactExpertState,
}

#[derive(Debug, Clone)]
struct TokenConker3CheckpointModel {
    config: TokenConker3CheckpointConfig,
    linear_embedding: Option<Vec<f32>>,
    local_embedding: Option<Vec<f32>>,
    linear_in_proj_t: Option<Vec<f32>>,
    linear_readout: Option<LinearReadout>,
    local_readout: Option<MlpReadout>,
}

#[derive(Debug, Clone)]
struct ExactExpertModel {
    config: ExactExpertConfig,
    classes: TokenClassLuts,
}

#[derive(Debug, Clone)]
struct ExactExpertConfig {
    residual_cap: f32,
    base_center_weight: f32,
    exact1_weight: f32,
    exact1_flag_weight: f32,
    exact2_weight: f32,
    exact2_flag_weight: f32,
    exact3_weight: f32,
    exact3_flag_weight: f32,
    delim2_weight: f32,
    delim2_flag_weight: f32,
    special2_weight: f32,
    special2_flag_weight: f32,
    number2_weight: f32,
    number2_flag_weight: f32,
    markup2_weight: f32,
    markup2_flag_weight: f32,
    attr2_weight: f32,
    attr2_flag_weight: f32,
}

#[derive(Debug, Clone)]
struct TokenClassLuts {
    class_ids: Vec<u8>,
    delimiter_mask: Vec<bool>,
    number_mask: Vec<bool>,
    special_mask: Vec<bool>,
    markup_mask: Vec<bool>,
    attr_mask: Vec<bool>,
}

#[derive(Debug, Clone, Default)]
struct ExactExpertState {
    prefix: Vec<usize>,
    exact1: HashMap<usize, SparseCounter>,
    exact2: HashMap<u32, SparseCounter>,
    exact3: HashMap<u64, SparseCounter>,
    class2: HashMap<u16, SparseCounter>,
}

#[derive(Debug, Clone, Default)]
struct SparseCounter {
    entries: Vec<(usize, u32)>,
}

#[derive(Debug, Clone)]
struct TokenConker3CheckpointConfig {
    vocab_size: usize,
    train_seq_len: usize,
    embedding_dim: usize,
    linear_modes: usize,
    non_osc_modes: usize,
    osc_pairs: usize,
    local_window: usize,
    local_scale: f32,
    static_non_osc_scale: f32,
    static_osc_scale: f32,
    non_osc_decays: Vec<f32>,
    osc_decays: Vec<f32>,
    osc_cos: Vec<f32>,
    osc_sin: Vec<f32>,
}

#[derive(Debug, Clone)]
struct BoundaryAccumState {
    non_osc_sum: Vec<f32>,
    osc_drive_sum: Vec<f32>,
    span_len: usize,
}

#[derive(Debug, Clone, Copy, Default)]
struct BoundaryEvalBreakdown {
    boundary_bpt: f64,
    flushed_spans: usize,
    span_tokens: usize,
    max_span_len: usize,
    boundary_tokens_seen: usize,
}

#[derive(Debug, Clone)]
struct DenseLayer {
    input_dim: usize,
    output_dim: usize,
    weights: Vec<f32>,
    bias: Vec<f32>,
}

#[derive(Debug, Clone)]
struct MlpReadout {
    hidden: Vec<DenseLayer>,
    output: DenseLayer,
}

#[derive(Debug, Clone)]
struct RoutedSquaredReLUReadout {
    router: DenseLayer,
    experts_in: Vec<DenseLayer>,
    experts_out: Vec<DenseLayer>,
}

#[derive(Debug, Clone)]
enum LinearReadout {
    Mlp(MlpReadout),
    RoutedSquaredReLU(RoutedSquaredReLUReadout),
}

#[derive(Debug, Clone)]
struct EvalBreakdown {
    linear_bpt: Option<f64>,
    local_bpt: Option<f64>,
    combined_bpt: f64,
}

#[derive(Debug, Clone)]
struct PredictionLogits {
    linear: Option<Vec<f32>>,
    local: Option<Vec<f32>>,
    combined: Vec<f32>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct TokenConker3FeatureDims {
    pub linear: usize,
    pub local: usize,
}

impl Runner for TokenConker3CheckpointRunner {
    fn name(&self) -> &'static str {
        "TokenConker3CheckpointRunner"
    }

    fn vocab_size(&self) -> usize {
        self.model.config.vocab_size
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
                let dist = self.model.predict_distribution(&state, &history)?;
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
            self.model.advance_state(&mut state, &mut history, token)?;
        }

        Ok(SampleOutputs {
            sample_predictions,
            sample_gold_logprobs: Some(sample_gold_logprobs),
        })
    }

    fn adapt_chunk(&mut self, tokens: &[usize]) -> Result<(), String> {
        for &token in tokens {
            self.model
                .advance_state(&mut self.state, &mut self.history, token)?;
        }
        Ok(())
    }
}

impl TokenConker3CheckpointRunner {
    // Exposed for legacy archive probes that still live outside this crate.
    pub fn snapshot_runtime(&self) -> (Vec<f32>, VecDeque<usize>) {
        (self.state.clone(), self.history.clone())
    }

    pub(crate) fn feature_dims(&self) -> TokenConker3FeatureDims {
        self.model.feature_dims()
    }

    // Exposed for legacy archive probes that still live outside this crate.
    pub fn predict_combined_logits_from(
        &self,
        state: &[f32],
        history: &VecDeque<usize>,
    ) -> Result<Vec<f32>, String> {
        Ok(self.model.predict_logits(state, history)?.combined)
    }

    // Exposed for legacy archive probes that still live outside this crate.
    pub fn advance_runtime_state(
        &self,
        state: &mut [f32],
        history: &mut VecDeque<usize>,
        token: usize,
    ) -> Result<(), String> {
        self.model.advance_state(state, history, token)
    }

    pub(crate) fn append_feature_batches_from_prefix(
        &self,
        state: &[f32],
        history: &VecDeque<usize>,
        linear_batch: &mut Vec<f32>,
        local_batch: &mut Vec<f32>,
    ) -> Result<(), String> {
        self.model
            .append_feature_batches_from_prefix(state, history, linear_batch, local_batch)
    }

    pub(crate) fn predict_combined_logits_batch_from_features(
        &self,
        linear_batch: &[f32],
        local_batch: &[f32],
        batch_size: usize,
    ) -> Result<Vec<f32>, String> {
        self.model.predict_combined_logits_batch_from_features(
            linear_batch,
            local_batch,
            batch_size,
        )
    }
}

impl Runner for TokenConker3ExactCheckpointRunner {
    fn name(&self) -> &'static str {
        "TokenConker3ExactCheckpointRunner"
    }

    fn vocab_size(&self) -> usize {
        self.model.config.vocab_size
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
        let mut exact_state = self.exact_state.clone();
        for (pos, &token) in tokens.iter().enumerate() {
            if !wanted[pos].is_empty() {
                let dist = self.experts.predict_distribution(
                    &self.model,
                    &state,
                    &history,
                    &exact_state,
                )?;
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
            self.model.advance_state(&mut state, &mut history, token)?;
            self.experts.observe_token(&mut exact_state, token);
        }

        Ok(SampleOutputs {
            sample_predictions,
            sample_gold_logprobs: Some(sample_gold_logprobs),
        })
    }

    fn adapt_chunk(&mut self, tokens: &[usize]) -> Result<(), String> {
        for &token in tokens {
            self.model
                .advance_state(&mut self.state, &mut self.history, token)?;
            self.experts.observe_token(&mut self.exact_state, token);
        }
        Ok(())
    }
}

impl TokenConker3ExactCheckpointRunner {
    pub(crate) fn predict_current_base_logits(&self) -> Result<Vec<f32>, String> {
        Ok(self
            .model
            .predict_logits(&self.state, &self.history)?
            .combined)
    }

    pub(crate) fn predict_current_exact_logits(&self) -> Result<Vec<f32>, String> {
        self.experts
            .predict_logits(&self.model, &self.state, &self.history, &self.exact_state)
    }

    pub(crate) fn adapt_one(&mut self, token: usize) -> Result<(), String> {
        self.model
            .advance_state(&mut self.state, &mut self.history, token)?;
        self.experts.observe_token(&mut self.exact_state, token);
        Ok(())
    }
}

pub fn load_token_conker3_checkpoint_from_data_root(
    checkpoint_path: &str,
    summary_path: &str,
    root: &Path,
    val_token_budget: usize,
) -> Result<LoadedTokenConker3Checkpoint, String> {
    let (runner, metadata) =
        load_token_conker3_checkpoint_runner_and_metadata(checkpoint_path, summary_path)?;
    let eval_tokens = take_val_tokens(root, val_token_budget)?;
    let eval_breakdown = evaluate_model(&runner.model, &eval_tokens)?;
    let byte_accounting = compute_tokens_per_byte(root, &eval_tokens)?;
    let report = TokenConker3CheckpointReport {
        checkpoint_path: metadata.checkpoint_path,
        summary_path: metadata.summary_path,
        train_seq_len: metadata.train_seq_len,
        embedding_dim: metadata.embedding_dim,
        linear_modes: metadata.linear_modes,
        non_osc_modes: metadata.non_osc_modes,
        osc_pairs: metadata.osc_pairs,
        local_window: metadata.local_window,
        static_non_osc_scale: metadata.static_non_osc_scale,
        static_osc_scale: metadata.static_osc_scale,
        local_scale: metadata.local_scale,
        model_variant: metadata.model_variant,
        parameter_count: metadata.parameter_count,
        eval_tokens: eval_tokens.len(),
        eval_bpt_linear_only: eval_breakdown.linear_bpt,
        eval_bpt_local_only: eval_breakdown.local_bpt,
        eval_bpt_combined: eval_breakdown.combined_bpt,
        eval_tokens_per_byte: byte_accounting.as_ref().map(|row| row.tokens_per_byte),
        eval_bytes_per_token: byte_accounting.as_ref().map(|row| row.bytes_per_token),
        eval_bpb_linear_only: byte_accounting.as_ref().and_then(|row| {
            eval_breakdown
                .linear_bpt
                .map(|value| value * row.tokens_per_byte)
        }),
        eval_bpb_local_only: byte_accounting.as_ref().and_then(|row| {
            eval_breakdown
                .local_bpt
                .map(|value| value * row.tokens_per_byte)
        }),
        eval_bpb_combined: byte_accounting
            .as_ref()
            .map(|row| eval_breakdown.combined_bpt * row.tokens_per_byte),
        summary_test_tokens_per_byte: metadata.summary_test_tokens_per_byte,
        summary_test_bytes_per_token: metadata.summary_test_bytes_per_token,
    };
    Ok(LoadedTokenConker3Checkpoint {
        report,
        runner,
        eval_tokens,
    })
}

// Exposed for legacy archive probes that still live outside this crate.
pub fn load_token_conker3_checkpoint_runner_and_metadata(
    checkpoint_path: &str,
    summary_path: &str,
) -> Result<(TokenConker3CheckpointRunner, TokenConker3CheckpointMetadata), String> {
    let summary_text = std::fs::read_to_string(summary_path)
        .map_err(|err| format!("read summary {summary_path}: {err}"))?;
    let summary: Conker3CheckpointSummary = serde_json::from_str(&summary_text)
        .map_err(|err| format!("parse summary {summary_path}: {err}"))?;
    if summary.model.mix_mode != "additive" {
        return Err(format!(
            "only additive Conker-3 checkpoints are currently supported, got {}",
            summary.model.mix_mode
        ));
    }
    if summary.model.linear_readout_kind != "mlp"
        && summary.model.linear_readout_kind != "routed_sqrelu_experts"
    {
        return Err(format!(
            "unsupported Conker-3 linear_readout_kind for Rust replay: {}",
            summary.model.linear_readout_kind
        ));
    }
    let tensor_names = required_tensor_names(&summary);
    let tensor_refs = tensor_names
        .iter()
        .map(|name| name.as_str())
        .collect::<Vec<_>>();
    let arrays = load_named_f32_arrays(checkpoint_path, &tensor_refs)?;
    let config = build_checkpoint_config(&summary, &arrays)?;
    let model = TokenConker3CheckpointModel::from_summary_and_arrays(&summary, &config, &arrays)?;
    let metadata = TokenConker3CheckpointMetadata {
        checkpoint_path: checkpoint_path.to_string(),
        summary_path: summary_path.to_string(),
        train_seq_len: summary.config.train.seq_len,
        embedding_dim: config.embedding_dim,
        linear_modes: config.linear_modes,
        non_osc_modes: config.non_osc_modes,
        osc_pairs: config.osc_pairs,
        local_window: config.local_window,
        static_non_osc_scale: config.static_non_osc_scale as f64,
        static_osc_scale: config.static_osc_scale as f64,
        local_scale: config.local_scale as f64,
        model_variant: summary.model.variant.clone(),
        parameter_count: summary.model.params,
        summary_test_tokens_per_byte: summary.dataset.test_tokens_per_byte,
        summary_test_bytes_per_token: summary.dataset.test_bytes_per_token,
    };
    let runner = TokenConker3CheckpointRunner {
        model,
        state: vec![0.0; config.linear_modes],
        history: VecDeque::with_capacity(config.local_window),
    };
    Ok((runner, metadata))
}

pub fn run_token_conker3_checkpoint_from_data_root(
    checkpoint_path: &str,
    summary_path: &str,
    root: &Path,
    val_token_budget: usize,
) -> Result<TokenConker3CheckpointReport, String> {
    Ok(load_token_conker3_checkpoint_from_data_root(
        checkpoint_path,
        summary_path,
        root,
        val_token_budget,
    )?
    .report)
}

pub fn probe_token_conker3_export_bundle(
    export_root: &str,
) -> Result<TokenConker3ExportBundleProbeReport, String> {
    let export_root_path = Path::new(export_root);
    let loaded_bundle = load_export_bundle_material(export_root_path)?;
    let manifest = loaded_bundle.manifest;
    let bundle = loaded_bundle.bundle;
    let notes_ref = manifest.notes_ref.as_deref().ok_or_else(|| {
        format!(
            "export bundle {} has no notes_ref",
            bundle.export_root.display()
        )
    })?;
    let notes_path = resolve_export_reference_path(&bundle.export_root, notes_ref);
    let notes: Conker3ExportNotesPayload = load_json_file(&notes_path)?;
    let replay_fixture = notes
        .replay_fixture
        .ok_or_else(|| format!("notes {} has no replay_fixture", notes_path.display()))?;
    let summary_ref = notes.summary_json.ok_or_else(|| {
        format!(
            "notes {} has no summary_json; rerun the bridge export with a summary path",
            notes_path.display()
        )
    })?;
    let summary_path = resolve_export_reference_path(&bundle.export_root, &summary_ref);
    let summary_path_text = summary_path.display().to_string();
    let (runner, metadata) =
        load_token_conker3_checkpoint_runner_and_metadata(export_root, &summary_path_text)?;

    let runner_target_count = replay_fixture.runner_target_count.unwrap_or_else(|| {
        replay_fixture
            .input_token_ids
            .len()
            .saturating_sub(1)
            .min(replay_fixture.target_token_ids.len())
    });
    if runner_target_count == 0 {
        return Err(format!(
            "replay fixture in {} has no usable targets",
            notes_path.display()
        ));
    }
    if replay_fixture.input_token_ids.len() < runner_target_count + 1 {
        return Err(format!(
            "replay fixture input length {} is too short for runner_target_count {}",
            replay_fixture.input_token_ids.len(),
            runner_target_count
        ));
    }
    if replay_fixture.target_token_ids.len() < runner_target_count {
        return Err(format!(
            "replay fixture target length {} is too short for runner_target_count {}",
            replay_fixture.target_token_ids.len(),
            runner_target_count
        ));
    }
    let actual_first_target_token_id = replay_fixture.target_token_ids[0];
    if replay_fixture.input_token_ids[1] != actual_first_target_token_id {
        return Err(format!(
            "replay fixture shift mismatch: input_token_ids[1]={} but target_token_ids[0]={}",
            replay_fixture.input_token_ids[1], actual_first_target_token_id
        ));
    }

    let expected_first_target_position = replay_fixture.expected_first_target_position.unwrap_or(1);
    let expected_first_target_token_id = replay_fixture
        .expected_first_target_token_id
        .unwrap_or(actual_first_target_token_id);
    let expected_first_gold_logprob =
        replay_fixture.expected_first_gold_logprob.ok_or_else(|| {
            format!(
                "replay fixture in {} is missing expected_first_gold_logprob",
                notes_path.display()
            )
        })?;
    let expected_prefix_gold_logprob_sum = replay_fixture
        .expected_prefix_gold_logprob_sum
        .ok_or_else(|| {
            format!(
                "replay fixture in {} is missing expected_prefix_gold_logprob_sum",
                notes_path.display()
            )
        })?;

    let sample_positions = (1..=runner_target_count).collect::<Vec<_>>();
    let outputs = runner.score_chunk(&replay_fixture.input_token_ids, &sample_positions)?;
    let gold_logprobs = outputs.sample_gold_logprobs.ok_or_else(|| {
        format!(
            "runner {} did not return gold logprobs for replay fixture",
            runner.name()
        )
    })?;
    if gold_logprobs.len() != runner_target_count {
        return Err(format!(
            "runner returned {} gold logprobs for {} sample positions",
            gold_logprobs.len(),
            runner_target_count
        ));
    }

    let actual_first_gold_logprob = gold_logprobs[0];
    let actual_prefix_gold_logprob_sum = gold_logprobs.iter().sum::<f64>();
    let first_gold_logprob_delta = actual_first_gold_logprob - expected_first_gold_logprob;
    let prefix_gold_logprob_delta =
        actual_prefix_gold_logprob_sum - expected_prefix_gold_logprob_sum;
    let prefix_logprob_match_tolerance = LOGPROB_MATCH_TOLERANCE * runner_target_count as f64;
    let first_gold_logprob_match = first_gold_logprob_delta.abs() <= LOGPROB_MATCH_TOLERANCE;
    let prefix_gold_logprob_match =
        prefix_gold_logprob_delta.abs() <= prefix_logprob_match_tolerance;

    Ok(TokenConker3ExportBundleProbeReport {
        export_root: bundle.export_root.display().to_string(),
        manifest_path: bundle.manifest_path.display().to_string(),
        notes_path: notes_path.display().to_string(),
        summary_path: summary_path.display().to_string(),
        export_model_variant_id: manifest.model_variant_id,
        summary_model_variant: metadata.model_variant,
        train_seq_len: metadata.train_seq_len,
        fixture_split: replay_fixture.split,
        fixture_token_count: replay_fixture.token_count,
        fixture_source_file: replay_fixture.source_file,
        fixture_source_file_index: replay_fixture.source_file_index,
        fixture_source_token_offset: replay_fixture.source_token_offset,
        runner_target_count,
        expected_first_target_position,
        expected_first_target_token_id,
        actual_first_target_token_id,
        expected_first_gold_logprob,
        actual_first_gold_logprob,
        first_gold_logprob_delta,
        expected_prefix_gold_logprob_sum,
        actual_prefix_gold_logprob_sum,
        prefix_gold_logprob_delta,
        first_logprob_match_tolerance: LOGPROB_MATCH_TOLERANCE,
        prefix_logprob_match_tolerance,
        first_gold_logprob_match,
        prefix_gold_logprob_match,
    })
}

pub fn render_token_conker3_checkpoint_report(report: &TokenConker3CheckpointReport) -> String {
    let mut out = String::new();
    out.push_str("chronohorn_token_conker3_checkpoint\n");
    out.push_str(&format!("checkpoint_path: {}\n", report.checkpoint_path));
    out.push_str(&format!("summary_path: {}\n", report.summary_path));
    out.push_str(&format!("train_seq_len: {}\n", report.train_seq_len));
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
    out.push_str(&format!("model_variant: {}\n", report.model_variant));
    out.push_str(&format!("parameter_count: {}\n", report.parameter_count));
    out.push_str(&format!("eval_tokens: {}\n", report.eval_tokens));
    if let Some(value) = report.eval_bpt_linear_only {
        out.push_str(&format!("eval_bpt_linear_only: {:.6}\n", value));
    }
    if let Some(value) = report.eval_bpt_local_only {
        out.push_str(&format!("eval_bpt_local_only: {:.6}\n", value));
    }
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
    if let Some(value) = report.summary_test_tokens_per_byte {
        out.push_str(&format!("summary_test_tokens_per_byte: {:.6}\n", value));
    }
    if let Some(value) = report.summary_test_bytes_per_token {
        out.push_str(&format!("summary_test_bytes_per_token: {:.6}\n", value));
    }
    out
}

pub fn render_token_conker3_export_bundle_probe_report(
    report: &TokenConker3ExportBundleProbeReport,
) -> String {
    let mut out = String::new();
    out.push_str("chronohorn_token_conker3_export_bundle_probe\n");
    out.push_str(&format!("export_root: {}\n", report.export_root));
    out.push_str(&format!("manifest_path: {}\n", report.manifest_path));
    out.push_str(&format!("notes_path: {}\n", report.notes_path));
    out.push_str(&format!("summary_path: {}\n", report.summary_path));
    out.push_str(&format!(
        "export_model_variant_id: {}\n",
        report.export_model_variant_id
    ));
    out.push_str(&format!(
        "summary_model_variant: {}\n",
        report.summary_model_variant
    ));
    out.push_str(&format!("train_seq_len: {}\n", report.train_seq_len));
    out.push_str(&format!("fixture_split: {}\n", report.fixture_split));
    out.push_str(&format!(
        "fixture_token_count: {}\n",
        report.fixture_token_count
    ));
    if let Some(source_file) = &report.fixture_source_file {
        out.push_str(&format!("fixture_source_file: {}\n", source_file));
    }
    out.push_str(&format!(
        "fixture_source_file_index: {}\n",
        report.fixture_source_file_index
    ));
    out.push_str(&format!(
        "fixture_source_token_offset: {}\n",
        report.fixture_source_token_offset
    ));
    out.push_str(&format!(
        "runner_target_count: {}\n",
        report.runner_target_count
    ));
    out.push_str(&format!(
        "expected_first_target_position: {}\n",
        report.expected_first_target_position
    ));
    out.push_str(&format!(
        "expected_first_target_token_id: {}\n",
        report.expected_first_target_token_id
    ));
    out.push_str(&format!(
        "actual_first_target_token_id: {}\n",
        report.actual_first_target_token_id
    ));
    out.push_str(&format!(
        "expected_first_gold_logprob: {:.9}\n",
        report.expected_first_gold_logprob
    ));
    out.push_str(&format!(
        "actual_first_gold_logprob: {:.9}\n",
        report.actual_first_gold_logprob
    ));
    out.push_str(&format!(
        "first_gold_logprob_delta: {:.9}\n",
        report.first_gold_logprob_delta
    ));
    out.push_str(&format!(
        "expected_prefix_gold_logprob_sum: {:.9}\n",
        report.expected_prefix_gold_logprob_sum
    ));
    out.push_str(&format!(
        "actual_prefix_gold_logprob_sum: {:.9}\n",
        report.actual_prefix_gold_logprob_sum
    ));
    out.push_str(&format!(
        "prefix_gold_logprob_delta: {:.9}\n",
        report.prefix_gold_logprob_delta
    ));
    out.push_str(&format!(
        "first_logprob_match_tolerance: {:.9}\n",
        report.first_logprob_match_tolerance
    ));
    out.push_str(&format!(
        "prefix_logprob_match_tolerance: {:.9}\n",
        report.prefix_logprob_match_tolerance
    ));
    out.push_str(&format!(
        "first_gold_logprob_match: {}\n",
        report.first_gold_logprob_match
    ));
    out.push_str(&format!(
        "prefix_gold_logprob_match: {}\n",
        report.prefix_gold_logprob_match
    ));
    out
}

pub fn run_token_conker3_boundary_checkpoint_from_data_root(
    checkpoint_path: &str,
    summary_path: &str,
    root: &Path,
    val_token_budget: usize,
) -> Result<TokenConker3BoundaryCheckpointReport, String> {
    let (runner, metadata) =
        load_token_conker3_checkpoint_runner_and_metadata(checkpoint_path, summary_path)?;
    let eval_tokens = take_val_tokens(root, val_token_budget)?;
    let base = evaluate_model(&runner.model, &eval_tokens)?;
    let tokenizer_vocab_path = derive_tokenizer_vocab_path(root)?;
    let boundary_mask =
        load_whitespace_boundary_mask(&tokenizer_vocab_path, runner.model.config.vocab_size)?;
    let boundary =
        evaluate_model_with_whitespace_boundaries(&runner.model, &eval_tokens, &boundary_mask)?;
    let byte_accounting = compute_tokens_per_byte(root, &eval_tokens)?;
    Ok(TokenConker3BoundaryCheckpointReport {
        checkpoint_path: metadata.checkpoint_path,
        summary_path: metadata.summary_path,
        tokenizer_vocab_path: tokenizer_vocab_path.display().to_string(),
        train_seq_len: metadata.train_seq_len,
        eval_tokens: eval_tokens.len(),
        boundary_mode: "leading_space_or_whitespace_sum_accum".to_string(),
        flushed_spans: boundary.flushed_spans,
        mean_span_len: boundary.span_tokens as f64 / (boundary.flushed_spans as f64).max(1.0),
        max_span_len: boundary.max_span_len,
        boundary_tokens_seen: boundary.boundary_tokens_seen,
        eval_bpt_base: base.combined_bpt,
        eval_bpt_boundary: boundary.boundary_bpt,
        eval_tokens_per_byte: byte_accounting.as_ref().map(|row| row.tokens_per_byte),
        eval_bytes_per_token: byte_accounting.as_ref().map(|row| row.bytes_per_token),
        eval_bpb_base: byte_accounting
            .as_ref()
            .map(|row| base.combined_bpt * row.tokens_per_byte),
        eval_bpb_boundary: byte_accounting
            .as_ref()
            .map(|row| boundary.boundary_bpt * row.tokens_per_byte),
    })
}

pub fn render_token_conker3_boundary_checkpoint_report(
    report: &TokenConker3BoundaryCheckpointReport,
) -> String {
    let mut out = String::new();
    out.push_str("chronohorn_token_conker3_boundary_checkpoint\n");
    out.push_str(&format!("checkpoint_path: {}\n", report.checkpoint_path));
    out.push_str(&format!("summary_path: {}\n", report.summary_path));
    out.push_str(&format!(
        "tokenizer_vocab_path: {}\n",
        report.tokenizer_vocab_path
    ));
    out.push_str(&format!("train_seq_len: {}\n", report.train_seq_len));
    out.push_str(&format!("eval_tokens: {}\n", report.eval_tokens));
    out.push_str(&format!("boundary_mode: {}\n", report.boundary_mode));
    out.push_str(&format!("flushed_spans: {}\n", report.flushed_spans));
    out.push_str(&format!("mean_span_len: {:.6}\n", report.mean_span_len));
    out.push_str(&format!("max_span_len: {}\n", report.max_span_len));
    out.push_str(&format!(
        "boundary_tokens_seen: {}\n",
        report.boundary_tokens_seen
    ));
    out.push_str(&format!("eval_bpt_base: {:.6}\n", report.eval_bpt_base));
    out.push_str(&format!(
        "eval_bpt_boundary: {:.6}\n",
        report.eval_bpt_boundary
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
    if let Some(value) = report.eval_bpb_boundary {
        out.push_str(&format!("eval_bpb_boundary: {:.6}\n", value));
    }
    out
}

pub fn load_token_conker3_exact_checkpoint_from_data_root(
    checkpoint_path: &str,
    summary_path: &str,
    root: &Path,
    val_token_budget: usize,
) -> Result<LoadedTokenConker3ExactCheckpoint, String> {
    let base = load_token_conker3_checkpoint_from_data_root(
        checkpoint_path,
        summary_path,
        root,
        val_token_budget,
    )?;
    let tokenizer_vocab_path = derive_tokenizer_vocab_path(root)?;
    let classes =
        load_token_class_luts(&tokenizer_vocab_path, base.runner.model.config.vocab_size)?;
    let experts = ExactExpertModel {
        config: ExactExpertConfig::default(),
        classes,
    };
    let eval_bpt_exact = evaluate_exact_model(&base.runner.model, &experts, &base.eval_tokens)?;
    let byte_accounting = compute_tokens_per_byte(root, &base.eval_tokens)?;
    let report = TokenConker3ExactCheckpointReport {
        checkpoint_path: checkpoint_path.to_string(),
        summary_path: summary_path.to_string(),
        tokenizer_vocab_path: tokenizer_vocab_path.display().to_string(),
        train_seq_len: base.report.train_seq_len,
        eval_tokens: base.eval_tokens.len(),
        residual_cap: experts.config.residual_cap as f64,
        exact1_weight: experts.config.exact1_weight as f64,
        exact2_weight: experts.config.exact2_weight as f64,
        exact3_weight: experts.config.exact3_weight as f64,
        delim2_weight: experts.config.delim2_weight as f64,
        special2_weight: experts.config.special2_weight as f64,
        number2_weight: experts.config.number2_weight as f64,
        markup2_weight: experts.config.markup2_weight as f64,
        attr2_weight: experts.config.attr2_weight as f64,
        base_center_weight: experts.config.base_center_weight as f64,
        eval_bpt_base: base.report.eval_bpt_combined,
        eval_bpt_exact,
        eval_tokens_per_byte: byte_accounting.as_ref().map(|row| row.tokens_per_byte),
        eval_bytes_per_token: byte_accounting.as_ref().map(|row| row.bytes_per_token),
        eval_bpb_base: byte_accounting
            .as_ref()
            .map(|row| base.report.eval_bpt_combined * row.tokens_per_byte),
        eval_bpb_exact: byte_accounting
            .as_ref()
            .map(|row| eval_bpt_exact * row.tokens_per_byte),
    };
    let runner = TokenConker3ExactCheckpointRunner {
        model: base.runner.model.clone(),
        experts,
        state: vec![0.0; base.runner.model.config.linear_modes],
        history: VecDeque::with_capacity(base.runner.model.config.local_window),
        exact_state: ExactExpertState::default(),
    };
    Ok(LoadedTokenConker3ExactCheckpoint {
        report,
        runner,
        eval_tokens: base.eval_tokens,
    })
}

pub fn run_token_conker3_exact_checkpoint_from_data_root(
    checkpoint_path: &str,
    summary_path: &str,
    root: &Path,
    val_token_budget: usize,
) -> Result<TokenConker3ExactCheckpointReport, String> {
    Ok(load_token_conker3_exact_checkpoint_from_data_root(
        checkpoint_path,
        summary_path,
        root,
        val_token_budget,
    )?
    .report)
}

pub fn render_token_conker3_exact_checkpoint_report(
    report: &TokenConker3ExactCheckpointReport,
) -> String {
    let mut out = String::new();
    out.push_str("chronohorn_token_conker3_exact_checkpoint\n");
    out.push_str(&format!("checkpoint_path: {}\n", report.checkpoint_path));
    out.push_str(&format!("summary_path: {}\n", report.summary_path));
    out.push_str(&format!(
        "tokenizer_vocab_path: {}\n",
        report.tokenizer_vocab_path
    ));
    out.push_str(&format!("train_seq_len: {}\n", report.train_seq_len));
    out.push_str(&format!("eval_tokens: {}\n", report.eval_tokens));
    out.push_str(&format!("residual_cap: {:.4}\n", report.residual_cap));
    out.push_str(&format!(
        "weights: base={:.4} exact1={:.4} exact2={:.4} exact3={:.4} delim2={:.4} special2={:.4} number2={:.4} markup2={:.4} attr2={:.4}\n",
        report.base_center_weight,
        report.exact1_weight,
        report.exact2_weight,
        report.exact3_weight,
        report.delim2_weight,
        report.special2_weight,
        report.number2_weight,
        report.markup2_weight,
        report.attr2_weight,
    ));
    out.push_str(&format!("eval_bpt_base: {:.6}\n", report.eval_bpt_base));
    out.push_str(&format!("eval_bpt_exact: {:.6}\n", report.eval_bpt_exact));
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
    out
}

impl TokenConker3CheckpointModel {
    fn feature_dims(&self) -> TokenConker3FeatureDims {
        TokenConker3FeatureDims {
            linear: self.config.linear_modes + self.config.embedding_dim,
            local: self.config.local_window * self.config.embedding_dim,
        }
    }

    fn from_summary_and_arrays(
        summary: &Conker3CheckpointSummary,
        config: &TokenConker3CheckpointConfig,
        arrays: &BTreeMap<String, F32Array>,
    ) -> Result<Self, String> {
        let linear_embedding = if summary.model.share_embedding {
            Some(load_embedding(
                arrays,
                "shared_embedding.weight",
                config.vocab_size,
                config.embedding_dim,
            )?)
        } else if summary.model.linear_modes > 0 {
            Some(load_embedding(
                arrays,
                "linear_embedding.weight",
                config.vocab_size,
                config.embedding_dim,
            )?)
        } else {
            None
        };
        let local_embedding = if summary.model.share_embedding {
            Some(load_embedding(
                arrays,
                "shared_embedding.weight",
                config.vocab_size,
                config.embedding_dim,
            )?)
        } else {
            Some(load_embedding(
                arrays,
                "local_embedding.weight",
                config.vocab_size,
                config.embedding_dim,
            )?)
        };

        let linear_in_proj = load_matrix(
            arrays,
            "linear_in_proj",
            &[config.embedding_dim, config.linear_modes],
        )?;
        let linear_readout = Some(load_linear_readout(
            arrays,
            summary,
            config.linear_modes + config.embedding_dim,
            config.vocab_size,
        )?);
        let local_readout = Some(load_mlp(
            arrays,
            "local_readout",
            config.local_window * config.embedding_dim,
            &summary.model.local_hidden,
            config.vocab_size,
        )?);

        Ok(Self {
            config: config.clone(),
            linear_embedding,
            local_embedding,
            linear_in_proj_t: Some(transpose_row_major(
                &linear_in_proj,
                config.embedding_dim,
                config.linear_modes,
            )),
            linear_readout,
            local_readout,
        })
    }

    fn linear_embedding_row(&self, token: Option<usize>) -> &[f32] {
        static ZERO: [f32; 0] = [];
        match (&self.linear_embedding, token) {
            (Some(table), Some(index)) if index < self.config.vocab_size => {
                let start = index * self.config.embedding_dim;
                &table[start..start + self.config.embedding_dim]
            }
            _ => &ZERO,
        }
    }

    fn local_embedding_row(&self, token: Option<usize>) -> &[f32] {
        static ZERO: [f32; 0] = [];
        match (&self.local_embedding, token) {
            (Some(table), Some(index)) if index < self.config.vocab_size => {
                let start = index * self.config.embedding_dim;
                &table[start..start + self.config.embedding_dim]
            }
            _ => &ZERO,
        }
    }

    fn linear_features_from_prefix(
        &self,
        state: &[f32],
        history: &VecDeque<usize>,
    ) -> Option<Vec<f32>> {
        let mut features = Vec::with_capacity(self.config.linear_modes + self.config.embedding_dim);
        for (index, &value) in state.iter().enumerate() {
            let scaled = if index < self.config.non_osc_modes {
                value * self.config.static_non_osc_scale
            } else {
                value * self.config.static_osc_scale
            };
            features.push(scaled);
        }
        let embed = self.linear_embedding_row(history.front().copied());
        if embed.is_empty() {
            features.resize(features.len() + self.config.embedding_dim, 0.0);
        } else {
            features.extend_from_slice(embed);
        }
        Some(features)
    }

    fn append_linear_features_from_prefix(
        &self,
        state: &[f32],
        history: &VecDeque<usize>,
        out: &mut Vec<f32>,
    ) {
        out.reserve(self.config.linear_modes + self.config.embedding_dim);
        for (index, &value) in state.iter().enumerate() {
            let scaled = if index < self.config.non_osc_modes {
                value * self.config.static_non_osc_scale
            } else {
                value * self.config.static_osc_scale
            };
            out.push(scaled);
        }
        let embed = self.linear_embedding_row(history.front().copied());
        if embed.is_empty() {
            out.resize(out.len() + self.config.embedding_dim, 0.0);
        } else {
            out.extend_from_slice(embed);
        }
    }

    fn local_features_from_prefix(&self, history: &VecDeque<usize>) -> Option<Vec<f32>> {
        let mut features = Vec::with_capacity(self.config.local_window * self.config.embedding_dim);
        for offset in 0..self.config.local_window {
            let embed = history
                .get(offset)
                .map(|&token| self.local_embedding_row(Some(token)))
                .unwrap_or(&[]);
            if embed.is_empty() {
                features.resize(features.len() + self.config.embedding_dim, 0.0);
            } else {
                features.extend_from_slice(embed);
            }
        }
        Some(features)
    }

    fn append_local_features_from_prefix(&self, history: &VecDeque<usize>, out: &mut Vec<f32>) {
        out.reserve(self.config.local_window * self.config.embedding_dim);
        for offset in 0..self.config.local_window {
            let embed = history
                .get(offset)
                .map(|&token| self.local_embedding_row(Some(token)))
                .unwrap_or(&[]);
            if embed.is_empty() {
                out.resize(out.len() + self.config.embedding_dim, 0.0);
            } else {
                out.extend_from_slice(embed);
            }
        }
    }

    fn append_feature_batches_from_prefix(
        &self,
        state: &[f32],
        history: &VecDeque<usize>,
        linear_batch: &mut Vec<f32>,
        local_batch: &mut Vec<f32>,
    ) -> Result<(), String> {
        if self.linear_readout.is_some() {
            self.append_linear_features_from_prefix(state, history, linear_batch);
        }
        if self.local_readout.is_some() {
            self.append_local_features_from_prefix(history, local_batch);
        }
        Ok(())
    }

    fn predict_combined_logits_batch_from_features(
        &self,
        linear_batch: &[f32],
        local_batch: &[f32],
        batch_size: usize,
    ) -> Result<Vec<f32>, String> {
        let vocab_size = self.config.vocab_size;
        let linear_logits = match self.linear_readout.as_ref() {
            Some(readout) => Some(readout.forward_batch(
                linear_batch,
                batch_size,
                self.config.linear_modes + self.config.embedding_dim,
            )?),
            None => None,
        };
        let local_logits = match self.local_readout.as_ref() {
            Some(readout) => Some(readout.forward_batch(
                local_batch,
                batch_size,
                self.config.local_window * self.config.embedding_dim,
            )?),
            None => None,
        };
        let mut combined = vec![0.0f32; batch_size * vocab_size];
        match (&linear_logits, &local_logits) {
            (Some(linear), Some(local)) => {
                for batch_index in 0..batch_size {
                    let start = batch_index * vocab_size;
                    let end = start + vocab_size;
                    for ((out, lhs), rhs) in combined[start..end]
                        .iter_mut()
                        .zip(linear[start..end].iter())
                        .zip(local[start..end].iter())
                    {
                        *out = *lhs + self.config.local_scale * *rhs;
                    }
                }
            }
            (Some(linear), None) => combined.copy_from_slice(linear),
            (None, Some(local)) => combined.copy_from_slice(local),
            (None, None) => return Err("Conker-3 checkpoint model has no active path".to_string()),
        }
        Ok(combined)
    }

    fn predict_logits(
        &self,
        state: &[f32],
        history: &VecDeque<usize>,
    ) -> Result<PredictionLogits, String> {
        let linear_logits = match (
            self.linear_features_from_prefix(state, history),
            self.linear_readout.as_ref(),
        ) {
            (Some(features), Some(readout)) => Some(readout.forward(&features)?),
            _ => None,
        };
        let local_logits = match (
            self.local_features_from_prefix(history),
            self.local_readout.as_ref(),
        ) {
            (Some(features), Some(readout)) => Some(readout.forward(&features)?),
            _ => None,
        };

        let combined = match (&linear_logits, &local_logits) {
            (Some(linear), Some(local)) => linear
                .iter()
                .zip(local.iter())
                .map(|(lhs, rhs)| *lhs + self.config.local_scale * *rhs)
                .collect::<Vec<_>>(),
            (Some(linear), None) => linear.clone(),
            (None, Some(local)) => local.clone(),
            (None, None) => return Err("Conker-3 checkpoint model has no active path".to_string()),
        };
        Ok(PredictionLogits {
            linear: linear_logits,
            local: local_logits,
            combined,
        })
    }

    fn predict_distribution(
        &self,
        state: &[f32],
        history: &VecDeque<usize>,
    ) -> Result<Vec<f64>, String> {
        let logits = self.predict_logits(state, history)?;
        Ok(softmax_f64(&logits.combined))
    }

    fn advance_state(
        &self,
        state: &mut [f32],
        history: &mut VecDeque<usize>,
        token: usize,
    ) -> Result<(), String> {
        if token >= self.config.vocab_size {
            return Ok(());
        }
        let Some(in_proj_t) = &self.linear_in_proj_t else {
            if history.len() == self.config.local_window {
                history.pop_back();
            }
            history.push_front(token);
            return Ok(());
        };
        let embed = self.linear_embedding_row(Some(token));
        if embed.len() != self.config.embedding_dim {
            return Err("missing linear embedding row".to_string());
        }
        for mode in 0..self.config.non_osc_modes {
            let start = mode * self.config.embedding_dim;
            let drive = dot(embed, &in_proj_t[start..start + self.config.embedding_dim]);
            state[mode] = self.config.non_osc_decays[mode] * state[mode] + drive;
        }
        for pair in 0..self.config.osc_pairs {
            let mode = self.config.non_osc_modes + 2 * pair;
            let start = mode * self.config.embedding_dim;
            let drive = dot(embed, &in_proj_t[start..start + self.config.embedding_dim]);
            let decay = self.config.osc_decays[pair];
            let cos = self.config.osc_cos[pair];
            let sin = self.config.osc_sin[pair];
            let prev_cos = state[mode];
            let prev_sin = state[mode + 1];
            state[mode] = decay * (cos * prev_cos - sin * prev_sin) + drive;
            state[mode + 1] = decay * (sin * prev_cos + cos * prev_sin);
        }
        if history.len() == self.config.local_window {
            history.pop_back();
        }
        history.push_front(token);
        Ok(())
    }

    fn accumulate_boundary_token(
        &self,
        history: &mut VecDeque<usize>,
        accum: &mut BoundaryAccumState,
        token: usize,
    ) -> Result<(), String> {
        if token < self.config.vocab_size {
            let Some(in_proj_t) = &self.linear_in_proj_t else {
                if history.len() == self.config.local_window {
                    history.pop_back();
                }
                history.push_front(token);
                return Ok(());
            };
            let embed = self.linear_embedding_row(Some(token));
            if embed.len() != self.config.embedding_dim {
                return Err("missing linear embedding row".to_string());
            }
            for mode in 0..self.config.non_osc_modes {
                let start = mode * self.config.embedding_dim;
                let drive = dot(embed, &in_proj_t[start..start + self.config.embedding_dim]);
                accum.non_osc_sum[mode] += drive;
            }
            for pair in 0..self.config.osc_pairs {
                let mode = self.config.non_osc_modes + 2 * pair;
                let start = mode * self.config.embedding_dim;
                let drive = dot(embed, &in_proj_t[start..start + self.config.embedding_dim]);
                accum.osc_drive_sum[pair] += drive;
            }
            accum.span_len += 1;
        }
        if history.len() == self.config.local_window {
            history.pop_back();
        }
        history.push_front(token);
        Ok(())
    }

    fn flush_boundary_accumulator(&self, state: &mut [f32], accum: &mut BoundaryAccumState) {
        if accum.span_len == 0 {
            return;
        }
        let span = accum.span_len as i32;
        for mode in 0..self.config.non_osc_modes {
            state[mode] =
                self.config.non_osc_decays[mode].powi(span) * state[mode] + accum.non_osc_sum[mode];
            accum.non_osc_sum[mode] = 0.0;
        }
        for pair in 0..self.config.osc_pairs {
            let mode = self.config.non_osc_modes + 2 * pair;
            let decay_pow = self.config.osc_decays[pair].powi(span);
            let theta = self.config.osc_sin[pair].atan2(self.config.osc_cos[pair]);
            let angle = theta * accum.span_len as f32;
            let cos_span = angle.cos();
            let sin_span = angle.sin();
            let prev_cos = state[mode];
            let prev_sin = state[mode + 1];
            state[mode] =
                decay_pow * (cos_span * prev_cos - sin_span * prev_sin) + accum.osc_drive_sum[pair];
            state[mode + 1] = decay_pow * (sin_span * prev_cos + cos_span * prev_sin);
            accum.osc_drive_sum[pair] = 0.0;
        }
        accum.span_len = 0;
    }
}

impl BoundaryAccumState {
    fn new(model: &TokenConker3CheckpointModel) -> Self {
        Self {
            non_osc_sum: vec![0.0; model.config.non_osc_modes],
            osc_drive_sum: vec![0.0; model.config.osc_pairs],
            span_len: 0,
        }
    }
}

impl DenseLayer {
    fn forward(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        if input.len() != self.input_dim {
            return Err(format!(
                "dense layer expected input dim {}, got {}",
                self.input_dim,
                input.len()
            ));
        }
        let mut out = self.bias.clone();
        for (row_index, row) in self.weights.chunks_exact(self.input_dim).enumerate() {
            out[row_index] += dot(row, input);
        }
        Ok(out)
    }

    fn forward_batch(&self, input: &[f32], batch_size: usize) -> Result<Vec<f32>, String> {
        if input.len() != batch_size * self.input_dim {
            return Err(format!(
                "dense layer batch expected {} values, got {}",
                batch_size * self.input_dim,
                input.len()
            ));
        }
        let mut out = vec![0.0f32; batch_size * self.output_dim];
        for batch_index in 0..batch_size {
            let start = batch_index * self.output_dim;
            let end = start + self.output_dim;
            out[start..end].copy_from_slice(&self.bias);
        }
        self.accumulate_batch_matmul(input, batch_size, &mut out)?;
        Ok(out)
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    fn accumulate_batch_matmul(
        &self,
        input: &[f32],
        batch_size: usize,
        out: &mut [f32],
    ) -> Result<(), String> {
        let m = i32::try_from(batch_size)
            .map_err(|_| format!("batch_size {} exceeds i32", batch_size))?;
        let n = i32::try_from(self.output_dim)
            .map_err(|_| format!("output_dim {} exceeds i32", self.output_dim))?;
        let k = i32::try_from(self.input_dim)
            .map_err(|_| format!("input_dim {} exceeds i32", self.input_dim))?;
        unsafe {
            cblas_sgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_TRANS,
                m,
                n,
                k,
                1.0,
                input.as_ptr(),
                k,
                self.weights.as_ptr(),
                k,
                1.0,
                out.as_mut_ptr(),
                n,
            );
        }
        Ok(())
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    fn accumulate_batch_matmul(
        &self,
        input: &[f32],
        batch_size: usize,
        out: &mut [f32],
    ) -> Result<(), String> {
        for batch_index in 0..batch_size {
            let in_start = batch_index * self.input_dim;
            let in_end = in_start + self.input_dim;
            let input_row = &input[in_start..in_end];
            let out_start = batch_index * self.output_dim;
            let out_end = out_start + self.output_dim;
            let out_row = &mut out[out_start..out_end];
            for (row_index, row) in self.weights.chunks_exact(self.input_dim).enumerate() {
                out_row[row_index] += dot(row, input_row);
            }
        }
        Ok(())
    }
}

impl MlpReadout {
    fn forward(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        let mut activations = input.to_vec();
        for layer in &self.hidden {
            activations = layer
                .forward(&activations)?
                .into_iter()
                .map(gelu_approx)
                .collect();
        }
        self.output.forward(&activations)
    }

    fn forward_batch(
        &self,
        input: &[f32],
        batch_size: usize,
        input_dim: usize,
    ) -> Result<Vec<f32>, String> {
        if input.len() != batch_size * input_dim {
            return Err(format!(
                "mlp batch expected {} values, got {}",
                batch_size * input_dim,
                input.len()
            ));
        }
        let mut activations = input.to_vec();
        let mut current_dim = input_dim;
        for layer in &self.hidden {
            activations = layer.forward_batch(&activations, batch_size)?;
            for value in &mut activations {
                *value = gelu_approx(*value);
            }
            current_dim = layer.output_dim;
        }
        self.output
            .forward_batch(&activations, batch_size)
            .or_else(|_| {
                Err(format!(
                    "mlp output batch forward failed for batch_size={} current_dim={}",
                    batch_size, current_dim
                ))
            })
    }
}

impl RoutedSquaredReLUReadout {
    fn output_dim(&self) -> Result<usize, String> {
        self.experts_out
            .first()
            .map(|layer| layer.output_dim)
            .ok_or_else(|| "routed readout has no experts".to_string())
    }

    fn forward(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        let route = softmax_f32(&self.router.forward(input)?);
        if route.len() != self.experts_in.len() || route.len() != self.experts_out.len() {
            return Err(format!(
                "routed readout expert shape mismatch: route={} experts_in={} experts_out={}",
                route.len(),
                self.experts_in.len(),
                self.experts_out.len()
            ));
        }
        let output_dim = self.output_dim()?;
        let mut combined = vec![0.0f32; output_dim];
        for ((expert_in, expert_out), &weight) in self
            .experts_in
            .iter()
            .zip(self.experts_out.iter())
            .zip(route.iter())
        {
            let mut hidden = expert_in.forward(input)?;
            for value in &mut hidden {
                *value = value.max(0.0);
                *value *= *value;
            }
            let logits = expert_out.forward(&hidden)?;
            for (out, value) in combined.iter_mut().zip(logits.iter()) {
                *out += weight * *value;
            }
        }
        Ok(combined)
    }

    fn forward_batch(
        &self,
        input: &[f32],
        batch_size: usize,
        input_dim: usize,
    ) -> Result<Vec<f32>, String> {
        if input.len() != batch_size * input_dim {
            return Err(format!(
                "routed readout batch expected {} values, got {}",
                batch_size * input_dim,
                input.len()
            ));
        }
        let output_dim = self.output_dim()?;
        let mut out = Vec::with_capacity(batch_size * output_dim);
        for row in input.chunks_exact(input_dim) {
            out.extend(self.forward(row)?);
        }
        Ok(out)
    }
}

impl LinearReadout {
    fn forward(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        match self {
            Self::Mlp(readout) => readout.forward(input),
            Self::RoutedSquaredReLU(readout) => readout.forward(input),
        }
    }

    fn forward_batch(
        &self,
        input: &[f32],
        batch_size: usize,
        input_dim: usize,
    ) -> Result<Vec<f32>, String> {
        match self {
            Self::Mlp(readout) => readout.forward_batch(input, batch_size, input_dim),
            Self::RoutedSquaredReLU(readout) => readout.forward_batch(input, batch_size, input_dim),
        }
    }
}

impl Default for ExactExpertConfig {
    fn default() -> Self {
        Self {
            residual_cap: 4.0,
            base_center_weight: 1.0,
            exact1_weight: 0.20,
            exact1_flag_weight: 0.05,
            exact2_weight: 0.85,
            exact2_flag_weight: 0.15,
            exact3_weight: 1.10,
            exact3_flag_weight: 0.20,
            delim2_weight: 0.18,
            delim2_flag_weight: 0.04,
            special2_weight: 0.10,
            special2_flag_weight: 0.03,
            number2_weight: 0.14,
            number2_flag_weight: 0.03,
            markup2_weight: 0.12,
            markup2_flag_weight: 0.03,
            attr2_weight: 0.12,
            attr2_flag_weight: 0.03,
        }
    }
}

impl SparseCounter {
    fn increment(&mut self, token: usize) {
        if let Some((_, count)) = self
            .entries
            .iter_mut()
            .find(|(candidate, _)| *candidate == token)
        {
            *count = count.saturating_add(1);
        } else {
            self.entries.push((token, 1));
        }
    }

    fn iter(&self) -> impl Iterator<Item = (usize, u32)> + '_ {
        self.entries.iter().copied()
    }
}

impl ExactExpertState {
    fn observe_token(&mut self, classes: &TokenClassLuts, token: usize) {
        if let Some(&prev) = self.prefix.last() {
            self.exact1.entry(prev).or_default().increment(token);
            let prev_class = classes.class_ids.get(prev).copied().unwrap_or(0) as u16;
            let curr_class = classes.class_ids.get(token).copied().unwrap_or(0) as u16;
            let class_key = (prev_class << 8) | curr_class;
            self.class2.entry(class_key).or_default().increment(token);
        }
        if self.prefix.len() >= 2 {
            let prev = self.prefix[self.prefix.len() - 1];
            let prev2 = self.prefix[self.prefix.len() - 2];
            self.exact2
                .entry(pack_bigram(prev2, prev))
                .or_default()
                .increment(token);
        }
        if self.prefix.len() >= 3 {
            let prev = self.prefix[self.prefix.len() - 1];
            let prev2 = self.prefix[self.prefix.len() - 2];
            let prev3 = self.prefix[self.prefix.len() - 3];
            self.exact3
                .entry(pack_trigram(prev3, prev2, prev))
                .or_default()
                .increment(token);
        }
        self.prefix.push(token);
    }
}

impl ExactExpertModel {
    fn observe_token(&self, state: &mut ExactExpertState, token: usize) {
        state.observe_token(&self.classes, token);
    }

    fn predict_distribution(
        &self,
        model: &TokenConker3CheckpointModel,
        state: &[f32],
        history: &VecDeque<usize>,
        exact_state: &ExactExpertState,
    ) -> Result<Vec<f64>, String> {
        let logits = self.predict_logits(model, state, history, exact_state)?;
        Ok(softmax_f64(&logits))
    }

    fn predict_logits(
        &self,
        model: &TokenConker3CheckpointModel,
        state: &[f32],
        history: &VecDeque<usize>,
        exact_state: &ExactExpertState,
    ) -> Result<Vec<f32>, String> {
        let base = model.predict_logits(state, history)?;
        Ok(self.apply_exact_residual(&base.combined, exact_state))
    }

    fn apply_exact_residual(
        &self,
        base_logits: &[f32],
        exact_state: &ExactExpertState,
    ) -> Vec<f32> {
        let Some(&current) = exact_state.prefix.last() else {
            return base_logits.to_vec();
        };
        let mut pre = centered_log_probs(base_logits)
            .into_iter()
            .map(|value| value * self.config.base_center_weight)
            .collect::<Vec<_>>();
        let mut candidate_mask = vec![false; base_logits.len()];

        if let Some(source) = exact_state.exact1.get(&current) {
            add_source(
                &mut pre,
                &mut candidate_mask,
                source.iter(),
                self.config.exact1_weight,
                self.config.exact1_flag_weight,
                false,
            );
        }

        if exact_state.prefix.len() >= 2 {
            let prev = exact_state.prefix[exact_state.prefix.len() - 2];
            if let Some(source) = exact_state.exact2.get(&pack_bigram(prev, current)) {
                add_source(
                    &mut pre,
                    &mut candidate_mask,
                    source.iter(),
                    self.config.exact2_weight,
                    self.config.exact2_flag_weight,
                    true,
                );
                add_masked_source(
                    &mut pre,
                    &candidate_mask,
                    source.iter(),
                    &self.classes.special_mask,
                    self.config.special2_weight,
                    self.config.special2_flag_weight,
                );
                add_masked_source(
                    &mut pre,
                    &candidate_mask,
                    source.iter(),
                    &self.classes.number_mask,
                    self.config.number2_weight,
                    self.config.number2_flag_weight,
                );
                add_masked_source(
                    &mut pre,
                    &candidate_mask,
                    source.iter(),
                    &self.classes.markup_mask,
                    self.config.markup2_weight,
                    self.config.markup2_flag_weight,
                );
                add_masked_source(
                    &mut pre,
                    &candidate_mask,
                    source.iter(),
                    &self.classes.attr_mask,
                    self.config.attr2_weight,
                    self.config.attr2_flag_weight,
                );
            }
            let prev_class = self.classes.class_ids.get(prev).copied().unwrap_or(0) as u16;
            let curr_class = self.classes.class_ids.get(current).copied().unwrap_or(0) as u16;
            if let Some(source) = exact_state.class2.get(&((prev_class << 8) | curr_class)) {
                add_masked_source(
                    &mut pre,
                    &candidate_mask,
                    source.iter(),
                    &self.classes.delimiter_mask,
                    self.config.delim2_weight,
                    self.config.delim2_flag_weight,
                );
            }
        }

        if exact_state.prefix.len() >= 3 {
            let prev = exact_state.prefix[exact_state.prefix.len() - 2];
            let prev2 = exact_state.prefix[exact_state.prefix.len() - 3];
            if let Some(source) = exact_state.exact3.get(&pack_trigram(prev2, prev, current)) {
                add_source(
                    &mut pre,
                    &mut candidate_mask,
                    source.iter(),
                    self.config.exact3_weight,
                    self.config.exact3_flag_weight,
                    true,
                );
            }
        }

        let mut out = base_logits.to_vec();
        for (index, value) in out.iter_mut().enumerate() {
            if candidate_mask[index] {
                *value += self.config.residual_cap * (pre[index] / self.config.residual_cap).tanh();
            }
        }
        out
    }
}

fn add_source<I>(
    pre: &mut [f32],
    candidate_mask: &mut [bool],
    entries: I,
    weight: f32,
    flag_weight: f32,
    opens_mask: bool,
) where
    I: Iterator<Item = (usize, u32)>,
{
    for (token, count) in entries {
        if token >= pre.len() {
            continue;
        }
        pre[token] += weight * (count as f32).ln_1p() + flag_weight;
        if opens_mask {
            candidate_mask[token] = true;
        }
    }
}

fn add_masked_source<I>(
    pre: &mut [f32],
    candidate_mask: &[bool],
    entries: I,
    mask: &[bool],
    weight: f32,
    flag_weight: f32,
) where
    I: Iterator<Item = (usize, u32)>,
{
    for (token, count) in entries {
        if token >= pre.len() || token >= mask.len() || !mask[token] || !candidate_mask[token] {
            continue;
        }
        pre[token] += weight * (count as f32).ln_1p() + flag_weight;
    }
}

fn centered_log_probs(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for &value in logits {
        sum += (value - max).exp();
    }
    let log_z = max + sum.max(EPS).ln();
    let mut out = logits
        .iter()
        .map(|&value| value - log_z)
        .collect::<Vec<_>>();
    let mean = out.iter().copied().sum::<f32>() / out.len().max(1) as f32;
    for value in &mut out {
        *value -= mean;
    }
    out
}

fn pack_bigram(lhs: usize, rhs: usize) -> u32 {
    ((lhs as u32) << 16) | rhs as u32
}

fn pack_trigram(a: usize, b: usize, c: usize) -> u64 {
    ((a as u64) << 32) | ((b as u64) << 16) | c as u64
}

fn derive_tokenizer_vocab_path(root: &Path) -> Result<std::path::PathBuf, String> {
    let root = materialize_data_root(root)?;
    let vocab = root
        .parent()
        .and_then(|path| path.parent())
        .map(|path| path.join("tokenizers").join("fineweb_1024_bpe.vocab"))
        .ok_or_else(|| format!("cannot derive tokenizer path from {}", root.display()))?;
    if !vocab.is_file() {
        return Err(format!("missing tokenizer vocab at {}", vocab.display()));
    }
    Ok(vocab)
}

fn load_token_class_luts(path: &Path, vocab_size: usize) -> Result<TokenClassLuts, String> {
    let text =
        std::fs::read_to_string(path).map_err(|err| format!("read {}: {err}", path.display()))?;
    let mut class_ids = vec![0u8; vocab_size];
    let mut delimiter_mask = vec![false; vocab_size];
    let mut number_mask = vec![false; vocab_size];
    let mut special_mask = vec![false; vocab_size];
    let mut markup_mask = vec![false; vocab_size];
    let mut attr_mask = vec![false; vocab_size];
    let mut scores = vec![f32::NAN; vocab_size];
    for (token_id, line) in text.lines().enumerate() {
        if token_id >= vocab_size {
            break;
        }
        let (piece, score) = parse_vocab_line(line);
        let (class_id, is_delim, core, leading_space) = classify_vocab_piece(&piece);
        class_ids[token_id] = class_id;
        delimiter_mask[token_id] = is_delim;
        if let Some(value) = score {
            scores[token_id] = value;
        }
        number_mask[token_id] = is_number_like(&core);
        special_mask[token_id] = is_identifier_like(&core, leading_space);
        markup_mask[token_id] = is_markup_like(&core);
        attr_mask[token_id] = is_attr_like(&core);
    }
    let mut finite_scores = scores
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    finite_scores.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(std::cmp::Ordering::Equal));
    if !finite_scores.is_empty() {
        let rare_index = ((finite_scores.len() as f32) * 0.2).floor() as usize;
        let rare_threshold = finite_scores[rare_index.min(finite_scores.len() - 1)];
        for token_id in 0..vocab_size {
            if !scores[token_id].is_finite()
                || delimiter_mask[token_id]
                || number_mask[token_id]
                || special_mask[token_id]
                || markup_mask[token_id]
                || attr_mask[token_id]
            {
                continue;
            }
            if class_ids[token_id] == 1 || class_ids[token_id] == 2 {
                if scores[token_id] <= rare_threshold {
                    special_mask[token_id] = true;
                }
            }
        }
    }
    Ok(TokenClassLuts {
        class_ids,
        delimiter_mask,
        number_mask,
        special_mask,
        markup_mask,
        attr_mask,
    })
}

fn load_whitespace_boundary_mask(path: &Path, vocab_size: usize) -> Result<Vec<bool>, String> {
    let text =
        std::fs::read_to_string(path).map_err(|err| format!("read {}: {err}", path.display()))?;
    let mut boundary = vec![false; vocab_size];
    for (token_id, line) in text.lines().enumerate() {
        if token_id >= vocab_size {
            break;
        }
        let (piece, _) = parse_vocab_line(line);
        let (core, leading_space) = decode_vocab_piece(&piece);
        boundary[token_id] = leading_space || core.chars().any(char::is_whitespace);
    }
    Ok(boundary)
}

fn parse_vocab_line(line: &str) -> (String, Option<f32>) {
    let mut parts = line.split('\t');
    let piece = parts.next().unwrap_or_default().to_string();
    let score = parts.next().and_then(|value| value.parse::<f32>().ok());
    (piece, score)
}

fn classify_vocab_piece(piece: &str) -> (u8, bool, String, bool) {
    let (core, leading_space) = decode_vocab_piece(piece);
    if core.is_empty() {
        return (
            if leading_space { 4 } else { 0 },
            false,
            core,
            leading_space,
        );
    }
    if core.chars().all(char::is_whitespace) {
        return (4, false, core, leading_space);
    }
    let punct_chars = ".,;:!?()[]{}<>\"'`/\\\\|-_=+*&^%$#@~";
    if core.chars().all(|ch| punct_chars.contains(ch)) {
        return (3, true, core, leading_space);
    }
    if core.chars().any(|ch| ch.is_alphanumeric()) {
        return (
            if leading_space { 2 } else { 1 },
            false,
            core,
            leading_space,
        );
    }
    (0, false, core, leading_space)
}

fn decode_vocab_piece(piece: &str) -> (String, bool) {
    let leading_space = piece.starts_with('▁');
    let mut core = piece.strip_prefix('▁').unwrap_or(piece).to_string();
    if core.starts_with("<0x") && core.ends_with('>') && core.len() == 6 {
        if let Ok(byte) = u8::from_str_radix(&core[3..5], 16) {
            core = String::from_utf8_lossy(&[byte]).to_string();
        } else {
            core.clear();
        }
    }
    (core, leading_space)
}

fn is_number_like(core: &str) -> bool {
    !core.is_empty()
        && core.chars().any(|ch| ch.is_ascii_digit())
        && core.chars().all(|ch| "0123456789.,:+-/%".contains(ch))
}

fn is_identifier_like(core: &str, leading_space: bool) -> bool {
    if core.is_empty() {
        return false;
    }
    let has_alpha = core.chars().any(|ch| ch.is_ascii_alphabetic());
    let has_digit = core.chars().any(|ch| ch.is_ascii_digit());
    let has_ident_punct = core.chars().any(|ch| "_-/.:@#".contains(ch));
    let has_inner_upper = !leading_space && core.chars().any(|ch| ch.is_ascii_uppercase());
    (has_alpha && has_digit) || has_ident_punct || has_inner_upper
}

fn is_markup_like(core: &str) -> bool {
    if core.is_empty() {
        return false;
    }
    let lowered = core.to_ascii_lowercase();
    let terms = [
        "<", ">", "</", "/>", "html", "body", "div", "span", "meta", "link", "script", "style",
        "class", "href", "src", "alt", "id=", "rel=", "type=", "content=", "data-", "&lt", "&gt",
        "&amp",
    ];
    terms.iter().any(|term| lowered.contains(term))
        || (core.chars().any(|ch| "<>=\"'".contains(ch))
            && core.chars().any(|ch| ch.is_ascii_alphabetic()))
}

fn is_attr_like(core: &str) -> bool {
    if core.is_empty() {
        return false;
    }
    let lowered = core.to_ascii_lowercase();
    let terms = [
        "class", "href", "src", "alt", "id", "rel", "type", "name", "value", "content", "style",
        "title", "property", "charset", "data-", "aria-", "=",
    ];
    terms.iter().any(|term| lowered.contains(term))
        || (core.contains('=') && core.chars().any(|ch| ch.is_ascii_alphabetic()))
}

fn build_checkpoint_config(
    summary: &Conker3CheckpointSummary,
    arrays: &BTreeMap<String, F32Array>,
) -> Result<TokenConker3CheckpointConfig, String> {
    let linear_modes = summary.model.linear_modes;
    let osc_pairs = (((linear_modes as f64) * summary.model.oscillatory_frac).floor() as usize / 2)
        .min(linear_modes / 2);
    let non_osc_modes = linear_modes.saturating_sub(2 * osc_pairs);
    let decays_full = load_vector(arrays, "linear_decays", &[linear_modes])?;
    let periods = logspace_values(
        summary.model.oscillatory_period_min as f32,
        summary.model.oscillatory_period_max as f32,
        osc_pairs,
    );
    let osc_decays = (0..osc_pairs)
        .map(|index| decays_full[non_osc_modes + 2 * index])
        .collect::<Vec<_>>();
    let osc_cos = periods
        .iter()
        .map(|&period| (2.0 * std::f32::consts::PI / period.max(1e-6)).cos())
        .collect::<Vec<_>>();
    let osc_sin = periods
        .iter()
        .map(|&period| (2.0 * std::f32::consts::PI / period.max(1e-6)).sin())
        .collect::<Vec<_>>();
    let (static_non_osc_scale, static_osc_scale) =
        if summary.model.static_bank_gate && osc_pairs > 0 {
            let logits = load_vector(arrays, "bank_gate_logits", &[2])?;
            (
                1.0 + summary.model.bank_gate_span as f32 * logits[0].tanh(),
                1.0 + summary.model.bank_gate_span as f32 * logits[1].tanh(),
            )
        } else {
            (1.0, 1.0)
        };

    Ok(TokenConker3CheckpointConfig {
        vocab_size: DEFAULT_VOCAB_SIZE,
        train_seq_len: summary.config.train.seq_len,
        embedding_dim: summary.model.embedding_dim,
        linear_modes,
        non_osc_modes,
        osc_pairs,
        local_window: summary.model.local_window,
        local_scale: summary.model.local_scale as f32,
        static_non_osc_scale,
        static_osc_scale,
        non_osc_decays: decays_full[..non_osc_modes].to_vec(),
        osc_decays,
        osc_cos,
        osc_sin,
    })
}

fn required_tensor_names(summary: &Conker3CheckpointSummary) -> Vec<String> {
    let mut names = Vec::new();
    if summary.model.share_embedding {
        names.push("shared_embedding.weight".to_string());
    } else {
        names.push("linear_embedding.weight".to_string());
        names.push("local_embedding.weight".to_string());
    }
    names.push("linear_in_proj".to_string());
    names.push("linear_decays".to_string());
    names.extend(required_linear_readout_tensor_names(summary));
    for index in 0..summary.model.local_hidden.len() {
        names.push(format!("local_readout.layers.{index}.weight"));
        names.push(format!("local_readout.layers.{index}.bias"));
    }
    names.push("local_readout.out.weight".to_string());
    names.push("local_readout.out.bias".to_string());
    if summary.model.static_bank_gate && summary.model.oscillatory_frac > 0.0 {
        names.push("bank_gate_logits".to_string());
    }
    names
}

fn required_linear_readout_tensor_names(summary: &Conker3CheckpointSummary) -> Vec<String> {
    match summary.model.linear_readout_kind.as_str() {
        "mlp" => {
            let mut names = Vec::new();
            for index in 0..summary.model.linear_hidden.len() {
                names.push(format!("linear_readout.layers.{index}.weight"));
                names.push(format!("linear_readout.layers.{index}.bias"));
            }
            names.push("linear_readout.out.weight".to_string());
            names.push("linear_readout.out.bias".to_string());
            names
        }
        "routed_sqrelu_experts" => {
            let mut names = vec![
                "linear_readout.router.weight".to_string(),
                "linear_readout.router.bias".to_string(),
            ];
            for index in 0..summary.model.linear_readout_num_experts {
                names.push(format!("linear_readout.experts_in.{index}.weight"));
                names.push(format!("linear_readout.experts_in.{index}.bias"));
                names.push(format!("linear_readout.experts_out.{index}.weight"));
                names.push(format!("linear_readout.experts_out.{index}.bias"));
            }
            names
        }
        _ => Vec::new(),
    }
}

fn load_embedding(
    arrays: &BTreeMap<String, F32Array>,
    name: &str,
    vocab_size: usize,
    embedding_dim: usize,
) -> Result<Vec<f32>, String> {
    load_matrix(arrays, name, &[vocab_size, embedding_dim])
}

fn load_vector(
    arrays: &BTreeMap<String, F32Array>,
    name: &str,
    expected_shape: &[usize],
) -> Result<Vec<f32>, String> {
    load_matrix(arrays, name, expected_shape)
}

fn load_matrix(
    arrays: &BTreeMap<String, F32Array>,
    name: &str,
    expected_shape: &[usize],
) -> Result<Vec<f32>, String> {
    let array = arrays
        .get(name)
        .ok_or_else(|| format!("missing tensor {name}"))?;
    if array.shape != expected_shape {
        return Err(format!(
            "tensor {name} shape mismatch: expected {:?}, got {:?}",
            expected_shape, array.shape
        ));
    }
    Ok(array.values.clone())
}

fn load_dense_layer(
    arrays: &BTreeMap<String, F32Array>,
    weight_name: &str,
    bias_name: &str,
    input_dim: usize,
    output_dim: usize,
) -> Result<DenseLayer, String> {
    let weights = load_matrix(arrays, weight_name, &[output_dim, input_dim])?;
    let bias = load_matrix(arrays, bias_name, &[output_dim])?;
    Ok(DenseLayer {
        input_dim,
        output_dim,
        weights,
        bias,
    })
}

fn load_mlp(
    arrays: &BTreeMap<String, F32Array>,
    prefix: &str,
    input_dim: usize,
    hidden_dims: &[usize],
    output_dim: usize,
) -> Result<MlpReadout, String> {
    let mut hidden = Vec::new();
    let mut prev = input_dim;
    for (index, &width) in hidden_dims.iter().enumerate() {
        hidden.push(load_dense_layer(
            arrays,
            &format!("{prefix}.layers.{index}.weight"),
            &format!("{prefix}.layers.{index}.bias"),
            prev,
            width,
        )?);
        prev = width;
    }
    let output = load_dense_layer(
        arrays,
        &format!("{prefix}.out.weight"),
        &format!("{prefix}.out.bias"),
        prev,
        output_dim,
    )?;
    Ok(MlpReadout { hidden, output })
}

fn load_routed_sqrelu_readout(
    arrays: &BTreeMap<String, F32Array>,
    prefix: &str,
    input_dim: usize,
    hidden_dims: &[usize],
    output_dim: usize,
    num_experts: usize,
) -> Result<RoutedSquaredReLUReadout, String> {
    if hidden_dims.len() != 1 {
        return Err(format!(
            "routed_sqrelu_experts expects exactly one hidden width, got {:?}",
            hidden_dims
        ));
    }
    if num_experts < 2 {
        return Err(format!(
            "routed_sqrelu_experts expects at least 2 experts, got {}",
            num_experts
        ));
    }
    let hidden_dim = hidden_dims[0];
    let router = load_dense_layer(
        arrays,
        &format!("{prefix}.router.weight"),
        &format!("{prefix}.router.bias"),
        input_dim,
        num_experts,
    )?;
    let mut experts_in = Vec::with_capacity(num_experts);
    let mut experts_out = Vec::with_capacity(num_experts);
    for index in 0..num_experts {
        experts_in.push(load_dense_layer(
            arrays,
            &format!("{prefix}.experts_in.{index}.weight"),
            &format!("{prefix}.experts_in.{index}.bias"),
            input_dim,
            hidden_dim,
        )?);
        experts_out.push(load_dense_layer(
            arrays,
            &format!("{prefix}.experts_out.{index}.weight"),
            &format!("{prefix}.experts_out.{index}.bias"),
            hidden_dim,
            output_dim,
        )?);
    }
    Ok(RoutedSquaredReLUReadout {
        router,
        experts_in,
        experts_out,
    })
}

fn load_linear_readout(
    arrays: &BTreeMap<String, F32Array>,
    summary: &Conker3CheckpointSummary,
    input_dim: usize,
    output_dim: usize,
) -> Result<LinearReadout, String> {
    match summary.model.linear_readout_kind.as_str() {
        "mlp" => Ok(LinearReadout::Mlp(load_mlp(
            arrays,
            "linear_readout",
            input_dim,
            &summary.model.linear_hidden,
            output_dim,
        )?)),
        "routed_sqrelu_experts" => Ok(LinearReadout::RoutedSquaredReLU(
            load_routed_sqrelu_readout(
                arrays,
                "linear_readout",
                input_dim,
                &summary.model.linear_hidden,
                output_dim,
                summary.model.linear_readout_num_experts,
            )?,
        )),
        other => Err(format!(
            "unsupported Conker-3 linear_readout_kind in Rust replay: {}",
            other
        )),
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

fn evaluate_model(
    model: &TokenConker3CheckpointModel,
    tokens: &[usize],
) -> Result<EvalBreakdown, String> {
    let seq_len = model.config.train_seq_len;
    let usable = ((tokens.len().saturating_sub(1)) / seq_len) * seq_len;
    if usable == 0 {
        return Err(format!(
            "need at least {} validation tokens for chunked parity eval, got {}",
            seq_len + 1,
            tokens.len()
        ));
    }
    let eval_tokens = &tokens[..usable + 1];
    let mut linear_nats = 0.0f64;
    let mut local_nats = 0.0f64;
    let mut combined_nats = 0.0f64;
    let mut count = 0usize;

    for seq_start in (0..usable).step_by(seq_len) {
        let mut state = vec![0.0; model.config.linear_modes];
        let mut history = VecDeque::with_capacity(model.config.local_window);
        let x = &eval_tokens[seq_start..seq_start + seq_len];
        let y = &eval_tokens[seq_start + 1..seq_start + seq_len + 1];
        for (&input_token, &target_token) in x.iter().zip(y.iter()) {
            model.advance_state(&mut state, &mut history, input_token)?;
            let logits = model.predict_logits(&state, &history)?;
            if let Some(ref linear_logits) = logits.linear {
                linear_nats += negative_log_prob_from_logits(linear_logits, target_token);
            }
            if let Some(ref local_logits) = logits.local {
                local_nats += negative_log_prob_from_logits(local_logits, target_token);
            }
            combined_nats += negative_log_prob_from_logits(&logits.combined, target_token);
            count += 1;
        }
    }

    let denom = (count as f64).max(1.0);
    let to_bits = 1.0 / std::f64::consts::LN_2;
    Ok(EvalBreakdown {
        linear_bpt: if model.linear_readout.is_some() {
            Some(linear_nats * to_bits / denom)
        } else {
            None
        },
        local_bpt: if model.local_readout.is_some() {
            Some(local_nats * to_bits / denom)
        } else {
            None
        },
        combined_bpt: combined_nats * to_bits / denom,
    })
}

fn evaluate_model_with_whitespace_boundaries(
    model: &TokenConker3CheckpointModel,
    tokens: &[usize],
    boundary_mask: &[bool],
) -> Result<BoundaryEvalBreakdown, String> {
    let seq_len = model.config.train_seq_len;
    let usable = ((tokens.len().saturating_sub(1)) / seq_len) * seq_len;
    if usable == 0 {
        return Err(format!(
            "need at least {} validation tokens for chunked parity eval, got {}",
            seq_len + 1,
            tokens.len()
        ));
    }
    let eval_tokens = &tokens[..usable + 1];
    let mut nats = 0.0f64;
    let mut count = 0usize;
    let mut flushed_spans = 0usize;
    let mut span_tokens = 0usize;
    let mut max_span_len = 0usize;
    let mut boundary_tokens_seen = 0usize;

    for seq_start in (0..usable).step_by(seq_len) {
        let mut state = vec![0.0; model.config.linear_modes];
        let mut history = VecDeque::with_capacity(model.config.local_window);
        let mut accum = BoundaryAccumState::new(model);
        let x = &eval_tokens[seq_start..seq_start + seq_len];
        let y = &eval_tokens[seq_start + 1..seq_start + seq_len + 1];
        for (&input_token, &target_token) in x.iter().zip(y.iter()) {
            if boundary_mask.get(input_token).copied().unwrap_or(false) {
                boundary_tokens_seen += 1;
                if accum.span_len > 0 {
                    max_span_len = max_span_len.max(accum.span_len);
                    span_tokens += accum.span_len;
                    flushed_spans += 1;
                    model.flush_boundary_accumulator(&mut state, &mut accum);
                }
            }
            model.accumulate_boundary_token(&mut history, &mut accum, input_token)?;
            let logits = model.predict_logits(&state, &history)?;
            nats += negative_log_prob_from_logits(&logits.combined, target_token);
            count += 1;
        }
    }

    let denom = (count as f64).max(1.0);
    Ok(BoundaryEvalBreakdown {
        boundary_bpt: nats * (1.0 / std::f64::consts::LN_2) / denom,
        flushed_spans,
        span_tokens,
        max_span_len,
        boundary_tokens_seen,
    })
}

fn evaluate_exact_model(
    model: &TokenConker3CheckpointModel,
    experts: &ExactExpertModel,
    tokens: &[usize],
) -> Result<f64, String> {
    let seq_len = model.config.train_seq_len;
    let usable = ((tokens.len().saturating_sub(1)) / seq_len) * seq_len;
    if usable == 0 {
        return Err(format!(
            "need at least {} validation tokens for chunked parity eval, got {}",
            seq_len + 1,
            tokens.len()
        ));
    }
    let eval_tokens = &tokens[..usable + 1];
    let mut nats = 0.0f64;
    let mut count = 0usize;

    for seq_start in (0..usable).step_by(seq_len) {
        let mut state = vec![0.0; model.config.linear_modes];
        let mut history = VecDeque::with_capacity(model.config.local_window);
        let mut exact_state = ExactExpertState::default();
        let x = &eval_tokens[seq_start..seq_start + seq_len];
        let y = &eval_tokens[seq_start + 1..seq_start + seq_len + 1];
        for (&input_token, &target_token) in x.iter().zip(y.iter()) {
            model.advance_state(&mut state, &mut history, input_token)?;
            experts.observe_token(&mut exact_state, input_token);
            let logits = experts.predict_logits(model, &state, &history, &exact_state)?;
            nats += negative_log_prob_from_logits(&logits, target_token);
            count += 1;
        }
    }

    let denom = (count as f64).max(1.0);
    Ok(nats * (1.0 / std::f64::consts::LN_2) / denom)
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
    let residual = 1.0 - values.iter().sum::<f64>();
    if let Some(first) = values.first_mut() {
        *first += residual;
    }
    values
}

fn softmax_f32(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut values = Vec::with_capacity(logits.len());
    let mut sum = 0.0f32;
    for &logit in logits {
        let value = (logit - max).exp();
        values.push(value);
        sum += value;
    }
    let denom = sum.max(EPS);
    for value in &mut values {
        *value /= denom;
    }
    let residual = 1.0 - values.iter().sum::<f32>();
    if let Some(first) = values.first_mut() {
        *first += residual;
    }
    values
}

fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter().zip(rhs.iter()).map(|(a, b)| *a * *b).sum()
}

fn transpose_row_major(values: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0; values.len()];
    for row in 0..rows {
        for col in 0..cols {
            out[col * rows + row] = values[row * cols + col];
        }
    }
    out
}

fn gelu_approx(value: f32) -> f32 {
    let cubic = value * value * value;
    let inner = (2.0 / std::f32::consts::PI).sqrt() * (value + 0.044_715 * cubic);
    0.5 * value * (1.0 + inner.tanh())
}
