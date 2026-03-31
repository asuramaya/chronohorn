use std::path::Path;

use chronohorn_core::data::{compute_tokens_per_byte, take_val_tokens};
use chronohorn_core::protocol::{Runner, SampleOutputs};
use serde::Serialize;

use super::token_match_bridge::{TokenMatchBridgeRunner, train_token_match_bridge_from_data_root};
use super::token_skip_bridge::{TokenSkipBridgeRunner, train_token_skip_bridge_from_data_root};
const FEATURE_DIM: usize = 12;
const PRIVILEGED_SURFACE_DIM: usize = 6;

#[derive(Debug, Clone, Serialize)]
pub struct TokenMatchSkipBridgeReport {
    pub train_token_budget: usize,
    pub trigram_buckets: usize,
    pub skip_buckets: usize,
    pub val_token_budget: usize,
    pub match_depth: usize,
    pub candidate_k: usize,
    pub train_stride: usize,
    pub tune_records: usize,
    pub eval_records: usize,
    pub privileged_surface_dim: usize,
    pub oracle_target_source: String,
    pub tuned_heuristic_lambda: f64,
    pub tuned_direct_lambda: f64,
    pub tuned_weighted_direct_lambda: f64,
    pub tuned_distilled_direct_lambda: f64,
    pub tuned_oracle_supervised_lambda: f64,
    pub tuned_hybrid_lambda: f64,
    pub tuned_trust_head_lambda: f64,
    pub selected_runtime_gate: String,
    pub selected_runtime_lambda: f64,
    pub tune_bpt_match: f64,
    pub tune_bpt_skip: f64,
    pub tune_bpt_heuristic: f64,
    pub tune_bpt_direct: f64,
    pub tune_bpt_weighted_direct: f64,
    pub tune_bpt_distilled_direct: f64,
    pub tune_bpt_oracle_supervised: f64,
    pub tune_bpt_hybrid: f64,
    pub tune_bpt_trust_head: f64,
    pub tune_bpt_oracle: f64,
    pub eval_bpt_match: f64,
    pub eval_bpt_skip: f64,
    pub eval_bpt_heuristic: f64,
    pub eval_bpt_direct: f64,
    pub eval_bpt_weighted_direct: f64,
    pub eval_bpt_distilled_direct: f64,
    pub eval_bpt_oracle_supervised: f64,
    pub eval_bpt_hybrid: f64,
    pub eval_bpt_trust_head: f64,
    pub eval_bpt_oracle: f64,
    pub eval_tokens_per_byte: Option<f64>,
    pub eval_bytes_per_token: Option<f64>,
    pub eval_bpb_match: Option<f64>,
    pub eval_bpb_skip: Option<f64>,
    pub eval_bpb_heuristic: Option<f64>,
    pub eval_bpb_direct: Option<f64>,
    pub eval_bpb_weighted_direct: Option<f64>,
    pub eval_bpb_distilled_direct: Option<f64>,
    pub eval_bpb_oracle_supervised: Option<f64>,
    pub eval_bpb_hybrid: Option<f64>,
    pub eval_bpb_trust_head: Option<f64>,
    pub eval_bpb_oracle: Option<f64>,
    pub tune_trust_head_brier: f64,
    pub eval_trust_head_brier: f64,
    pub eval_match_better_rate: f64,
    pub eval_skip_better_rate: f64,
    pub eval_top1_agreement_rate: f64,
}

#[derive(Debug, Clone)]
pub struct TrainedTokenMatchSkipBridge {
    report: TokenMatchSkipBridgeReport,
    runner: TokenMatchSkipBridgeRunner,
    eval_tokens: Vec<usize>,
}

impl TrainedTokenMatchSkipBridge {
    pub fn report(&self) -> &TokenMatchSkipBridgeReport {
        &self.report
    }

    pub fn runner(&self) -> &TokenMatchSkipBridgeRunner {
        &self.runner
    }

    pub fn eval_tokens(&self) -> &[usize] {
        &self.eval_tokens
    }
}

#[derive(Debug, Clone)]
pub struct TokenMatchSkipBridgeRunner {
    match_runner: TokenMatchBridgeRunner,
    skip_runner: TokenSkipBridgeRunner,
    standardizer: Standardizer,
    privileged_standardizer: PrivilegedSurfaceStandardizer,
    gate: GateChoice,
    lambda: f64,
    candidate_k: usize,
}

#[derive(Debug, Clone)]
struct Standardizer {
    mean: [f64; FEATURE_DIM],
    std: [f64; FEATURE_DIM],
}

#[derive(Debug, Clone)]
struct PrivilegedSurfaceStandardizer {
    mean: [f64; PRIVILEGED_SURFACE_DIM],
    std: [f64; PRIVILEGED_SURFACE_DIM],
}

#[derive(Debug, Clone)]
struct LogisticModel {
    weights: [f64; FEATURE_DIM],
    bias: f64,
}

#[derive(Debug, Clone)]
struct TinyTrustHead {
    weights: [f64; PRIVILEGED_SURFACE_DIM],
    bias: f64,
}

#[derive(Debug, Clone)]
enum GateChoice {
    Heuristic,
    DirectNll(LogisticModel),
    OracleSupervised(LogisticModel),
    HybridOracleNll(LogisticModel),
    TrustHead(TinyTrustHead),
}

#[derive(Debug, Clone)]
struct RuntimeGateCandidate {
    label: &'static str,
    lambda: f64,
    tune_bpt: f64,
    gate: GateChoice,
}

#[derive(Debug, Clone)]
struct RawRecord {
    features: [f64; FEATURE_DIM],
    privileged_surfaces: [f64; PRIVILEGED_SURFACE_DIM],
    match_gold_prob: f64,
    skip_gold_prob: f64,
    heuristic_gate: f64,
    oracle_gate_target: f64,
    trust_target: f64,
    sample_weight: f64,
    teacher_target_probs: Vec<f64>,
    teacher_match_probs: Vec<f64>,
    teacher_skip_probs: Vec<f64>,
    teacher_support_weight: f64,
    top1_agreement: f64,
}

#[derive(Debug, Clone)]
struct Record {
    features: [f64; FEATURE_DIM],
    privileged_surfaces: [f64; PRIVILEGED_SURFACE_DIM],
    match_gold_prob: f64,
    skip_gold_prob: f64,
    heuristic_gate: f64,
    oracle_gate_target: f64,
    trust_target: f64,
    sample_weight: f64,
    teacher_target_probs: Vec<f64>,
    teacher_match_probs: Vec<f64>,
    teacher_skip_probs: Vec<f64>,
    teacher_support_weight: f64,
    top1_agreement: f64,
}

#[derive(Debug, Clone)]
struct DistStats {
    top1_prob: f64,
    top1_token: usize,
    entropy_norm: f64,
    topk_mass: f64,
    margin: f64,
    support: Vec<(usize, f64)>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TokenMatchSkipOfflineTargetStream {
    pub source: String,
    pub tune_gate_targets: Vec<f64>,
    pub eval_gate_targets: Vec<f64>,
    pub tune_trust_targets: Option<Vec<f64>>,
    pub eval_trust_targets: Option<Vec<f64>>,
    pub tune_teacher_candidate_pairs: Option<Vec<Vec<(usize, usize)>>>,
    pub eval_teacher_candidate_pairs: Option<Vec<Vec<(usize, usize)>>>,
}

pub fn train_token_matchskip_bridge_from_data_root(
    root: &Path,
    train_token_budget: usize,
    trigram_buckets: usize,
    skip_buckets: usize,
    val_token_budget: usize,
    match_depth: usize,
    candidate_k: usize,
    train_stride: usize,
    alpha_bigram: f64,
    alpha_trigram: f64,
    alpha_skip: f64,
) -> Result<TrainedTokenMatchSkipBridge, String> {
    train_token_matchskip_bridge_from_data_root_with_offline_targets(
        root,
        train_token_budget,
        trigram_buckets,
        skip_buckets,
        val_token_budget,
        match_depth,
        candidate_k,
        train_stride,
        alpha_bigram,
        alpha_trigram,
        alpha_skip,
        None,
    )
}

pub fn train_token_matchskip_bridge_from_data_root_with_offline_targets(
    root: &Path,
    train_token_budget: usize,
    trigram_buckets: usize,
    skip_buckets: usize,
    val_token_budget: usize,
    match_depth: usize,
    candidate_k: usize,
    train_stride: usize,
    alpha_bigram: f64,
    alpha_trigram: f64,
    alpha_skip: f64,
    offline_targets: Option<&TokenMatchSkipOfflineTargetStream>,
) -> Result<TrainedTokenMatchSkipBridge, String> {
    if train_stride == 0 {
        return Err("train_stride must be positive".to_string());
    }
    if match_depth == 0 {
        return Err("match_depth must be positive".to_string());
    }
    if candidate_k == 0 {
        return Err("candidate_k must be positive".to_string());
    }

    let trained_match = train_token_match_bridge_from_data_root(
        root,
        train_token_budget,
        trigram_buckets,
        val_token_budget,
        match_depth,
        candidate_k,
        train_stride,
        alpha_bigram,
        alpha_trigram,
    )?;
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

    if trained_match.runner().vocab_size() != trained_skip.runner().vocab_size() {
        return Err("match/skip vocab size mismatch".to_string());
    }

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

    let mut tune_raw = collect_raw_records(
        trained_match.runner(),
        trained_skip.runner(),
        &tune_tokens,
        candidate_k,
        offline_targets.and_then(|targets| targets.tune_teacher_candidate_pairs.as_deref()),
    )?;
    let mut eval_raw = collect_raw_records(
        trained_match.runner(),
        trained_skip.runner(),
        &eval_tokens,
        candidate_k,
        offline_targets.and_then(|targets| targets.eval_teacher_candidate_pairs.as_deref()),
    )?;
    if tune_raw.is_empty() || eval_raw.is_empty() {
        return Err("no matchskip records collected".to_string());
    }
    let oracle_target_source =
        apply_offline_target_stream(&mut tune_raw, &mut eval_raw, offline_targets)?;

    let standardizer = fit_standardizer_from_raw(&tune_raw);
    let privileged_standardizer = fit_privileged_standardizer_from_raw(&tune_raw);
    let tune_records = standardize_records(&tune_raw, &standardizer, &privileged_standardizer);
    let eval_records = standardize_records(&eval_raw, &standardizer, &privileged_standardizer);
    let direct_model = train_direct_gate(&tune_records, 80, 0.1, 1e-4);
    let weighted_direct_model = train_weighted_direct_gate(&tune_records, 80, 0.1, 1e-4);
    let distilled_direct_model =
        train_distilled_direct_gate(&tune_records, 80, 0.1, 1e-4, 0.5, 0.4, 0.1);
    let oracle_supervised_model = train_oracle_supervised_gate(&tune_records, 80, 0.08, 1e-4);
    let trust_head = train_tiny_trust_head(&tune_records, 64, 0.08, 3e-4);
    let hybrid_model = train_hybrid_gate(&tune_records, 80, 0.1, 1e-4, 0.35);

    let tuned_heuristic_lambda = tune_lambda(&tune_records, |record| record.heuristic_gate);
    let tuned_direct_lambda = tune_lambda(&tune_records, |record| {
        predict_probability(&direct_model, &record.features)
    });
    let tuned_weighted_direct_lambda = tune_lambda(&tune_records, |record| {
        predict_probability(&weighted_direct_model, &record.features)
    });
    let tuned_distilled_direct_lambda = tune_lambda(&tune_records, |record| {
        predict_probability(&distilled_direct_model, &record.features)
    });
    let tuned_oracle_supervised_lambda = tune_lambda(&tune_records, |record| {
        predict_probability(&oracle_supervised_model, &record.features)
    });
    let tuned_hybrid_lambda = tune_lambda(&tune_records, |record| {
        predict_probability(&hybrid_model, &record.features)
    });
    let tuned_trust_head_lambda = tune_lambda(&tune_records, |record| {
        predict_trust_probability(&trust_head, &record.privileged_surfaces)
    });

    let tune_bpt_match = mean_bits_for_branch(&tune_records, Branch::Match);
    let tune_bpt_skip = mean_bits_for_branch(&tune_records, Branch::Skip);
    let tune_bpt_heuristic = mean_bits_with_gate(&tune_records, tuned_heuristic_lambda, |record| {
        record.heuristic_gate
    });
    let tune_bpt_direct = mean_bits_with_gate(&tune_records, tuned_direct_lambda, |record| {
        predict_probability(&direct_model, &record.features)
    });
    let tune_bpt_weighted_direct =
        mean_bits_with_gate(&tune_records, tuned_weighted_direct_lambda, |record| {
            predict_probability(&weighted_direct_model, &record.features)
        });
    let tune_bpt_distilled_direct =
        mean_bits_with_gate(&tune_records, tuned_distilled_direct_lambda, |record| {
            predict_probability(&distilled_direct_model, &record.features)
        });
    let tune_bpt_oracle_supervised =
        mean_bits_with_gate(&tune_records, tuned_oracle_supervised_lambda, |record| {
            predict_probability(&oracle_supervised_model, &record.features)
        });
    let tune_bpt_hybrid = mean_bits_with_gate(&tune_records, tuned_hybrid_lambda, |record| {
        predict_probability(&hybrid_model, &record.features)
    });
    let tune_bpt_trust_head =
        mean_bits_with_gate(&tune_records, tuned_trust_head_lambda, |record| {
            predict_trust_probability(&trust_head, &record.privileged_surfaces)
        });
    let tune_bpt_oracle = oracle_bits_per_token(&tune_records);
    let tune_trust_head_brier = trust_brier_score(&tune_records, |record| {
        predict_trust_probability(&trust_head, &record.privileged_surfaces)
    });

    let eval_bpt_match = mean_bits_for_branch(&eval_records, Branch::Match);
    let eval_bpt_skip = mean_bits_for_branch(&eval_records, Branch::Skip);
    let eval_bpt_heuristic = mean_bits_with_gate(&eval_records, tuned_heuristic_lambda, |record| {
        record.heuristic_gate
    });
    let eval_bpt_direct = mean_bits_with_gate(&eval_records, tuned_direct_lambda, |record| {
        predict_probability(&direct_model, &record.features)
    });
    let eval_bpt_weighted_direct =
        mean_bits_with_gate(&eval_records, tuned_weighted_direct_lambda, |record| {
            predict_probability(&weighted_direct_model, &record.features)
        });
    let eval_bpt_distilled_direct =
        mean_bits_with_gate(&eval_records, tuned_distilled_direct_lambda, |record| {
            predict_probability(&distilled_direct_model, &record.features)
        });
    let eval_bpt_oracle_supervised =
        mean_bits_with_gate(&eval_records, tuned_oracle_supervised_lambda, |record| {
            predict_probability(&oracle_supervised_model, &record.features)
        });
    let eval_bpt_hybrid = mean_bits_with_gate(&eval_records, tuned_hybrid_lambda, |record| {
        predict_probability(&hybrid_model, &record.features)
    });
    let eval_bpt_trust_head =
        mean_bits_with_gate(&eval_records, tuned_trust_head_lambda, |record| {
            predict_trust_probability(&trust_head, &record.privileged_surfaces)
        });
    let eval_bpt_oracle = oracle_bits_per_token(&eval_records);
    let eval_trust_head_brier = trust_brier_score(&eval_records, |record| {
        predict_trust_probability(&trust_head, &record.privileged_surfaces)
    });
    let eval_byte_accounting = compute_tokens_per_byte(root, &eval_tokens)?;
    let eval_tokens_per_byte = eval_byte_accounting.as_ref().map(|row| row.tokens_per_byte);
    let eval_bytes_per_token = eval_byte_accounting.as_ref().map(|row| row.bytes_per_token);

    let RuntimeGateCandidate {
        label: selected_runtime_gate,
        lambda: selected_runtime_lambda,
        gate,
        ..
    } = select_runtime_gate([
        RuntimeGateCandidate {
            label: "heuristic",
            lambda: tuned_heuristic_lambda,
            tune_bpt: tune_bpt_heuristic,
            gate: GateChoice::Heuristic,
        },
        RuntimeGateCandidate {
            label: "direct_nll",
            lambda: tuned_direct_lambda,
            tune_bpt: tune_bpt_direct,
            gate: GateChoice::DirectNll(direct_model.clone()),
        },
        RuntimeGateCandidate {
            label: "weighted_direct_nll",
            lambda: tuned_weighted_direct_lambda,
            tune_bpt: tune_bpt_weighted_direct,
            gate: GateChoice::DirectNll(weighted_direct_model.clone()),
        },
        RuntimeGateCandidate {
            label: "distilled_direct_nll",
            lambda: tuned_distilled_direct_lambda,
            tune_bpt: tune_bpt_distilled_direct,
            gate: GateChoice::DirectNll(distilled_direct_model.clone()),
        },
        RuntimeGateCandidate {
            label: "oracle_supervised",
            lambda: tuned_oracle_supervised_lambda,
            tune_bpt: tune_bpt_oracle_supervised,
            gate: GateChoice::OracleSupervised(oracle_supervised_model.clone()),
        },
        RuntimeGateCandidate {
            label: "hybrid_oracle_nll",
            lambda: tuned_hybrid_lambda,
            tune_bpt: tune_bpt_hybrid,
            gate: GateChoice::HybridOracleNll(hybrid_model.clone()),
        },
        RuntimeGateCandidate {
            label: "trust_head",
            lambda: tuned_trust_head_lambda,
            tune_bpt: tune_bpt_trust_head,
            gate: GateChoice::TrustHead(trust_head.clone()),
        },
    ]);

    let report = TokenMatchSkipBridgeReport {
        train_token_budget,
        trigram_buckets,
        skip_buckets,
        val_token_budget,
        match_depth,
        candidate_k,
        train_stride,
        tune_records: tune_records.len(),
        eval_records: eval_records.len(),
        privileged_surface_dim: PRIVILEGED_SURFACE_DIM,
        oracle_target_source,
        tuned_heuristic_lambda,
        tuned_direct_lambda,
        tuned_weighted_direct_lambda,
        tuned_distilled_direct_lambda,
        tuned_oracle_supervised_lambda,
        tuned_hybrid_lambda,
        tuned_trust_head_lambda,
        selected_runtime_gate: selected_runtime_gate.to_string(),
        selected_runtime_lambda,
        tune_bpt_match,
        tune_bpt_skip,
        tune_bpt_heuristic,
        tune_bpt_direct,
        tune_bpt_weighted_direct,
        tune_bpt_distilled_direct,
        tune_bpt_oracle_supervised,
        tune_bpt_hybrid,
        tune_bpt_trust_head,
        tune_bpt_oracle,
        eval_bpt_match,
        eval_bpt_skip,
        eval_bpt_heuristic,
        eval_bpt_direct,
        eval_bpt_weighted_direct,
        eval_bpt_distilled_direct,
        eval_bpt_oracle_supervised,
        eval_bpt_hybrid,
        eval_bpt_trust_head,
        eval_bpt_oracle,
        eval_tokens_per_byte,
        eval_bytes_per_token,
        eval_bpb_match: eval_tokens_per_byte.map(|scale| eval_bpt_match * scale),
        eval_bpb_skip: eval_tokens_per_byte.map(|scale| eval_bpt_skip * scale),
        eval_bpb_heuristic: eval_tokens_per_byte.map(|scale| eval_bpt_heuristic * scale),
        eval_bpb_direct: eval_tokens_per_byte.map(|scale| eval_bpt_direct * scale),
        eval_bpb_weighted_direct: eval_tokens_per_byte
            .map(|scale| eval_bpt_weighted_direct * scale),
        eval_bpb_distilled_direct: eval_tokens_per_byte
            .map(|scale| eval_bpt_distilled_direct * scale),
        eval_bpb_oracle_supervised: eval_tokens_per_byte
            .map(|scale| eval_bpt_oracle_supervised * scale),
        eval_bpb_hybrid: eval_tokens_per_byte.map(|scale| eval_bpt_hybrid * scale),
        eval_bpb_trust_head: eval_tokens_per_byte.map(|scale| eval_bpt_trust_head * scale),
        eval_bpb_oracle: eval_tokens_per_byte.map(|scale| eval_bpt_oracle * scale),
        tune_trust_head_brier,
        eval_trust_head_brier,
        eval_match_better_rate: branch_better_rate(&eval_records, Branch::Match),
        eval_skip_better_rate: branch_better_rate(&eval_records, Branch::Skip),
        eval_top1_agreement_rate: agreement_rate(&eval_records),
    };

    let runner = TokenMatchSkipBridgeRunner {
        match_runner: trained_match.runner().clone(),
        skip_runner: trained_skip.runner().clone(),
        standardizer,
        privileged_standardizer,
        gate,
        lambda: selected_runtime_lambda,
        candidate_k,
    };

    Ok(TrainedTokenMatchSkipBridge {
        report,
        runner,
        eval_tokens,
    })
}

pub fn run_token_matchskip_bridge_from_data_root(
    root: &Path,
    train_token_budget: usize,
    trigram_buckets: usize,
    skip_buckets: usize,
    val_token_budget: usize,
    match_depth: usize,
    candidate_k: usize,
    train_stride: usize,
    alpha_bigram: f64,
    alpha_trigram: f64,
    alpha_skip: f64,
) -> Result<TokenMatchSkipBridgeReport, String> {
    Ok(train_token_matchskip_bridge_from_data_root(
        root,
        train_token_budget,
        trigram_buckets,
        skip_buckets,
        val_token_budget,
        match_depth,
        candidate_k,
        train_stride,
        alpha_bigram,
        alpha_trigram,
        alpha_skip,
    )?
    .report)
}

pub fn render_token_matchskip_bridge_report(report: &TokenMatchSkipBridgeReport) -> String {
    let mut out = String::new();
    out.push_str("chronohorn_token_matchskip_bridge\n");
    out.push_str(&format!(
        "train_token_budget: {}\n",
        report.train_token_budget
    ));
    out.push_str(&format!("trigram_buckets: {}\n", report.trigram_buckets));
    out.push_str(&format!("skip_buckets: {}\n", report.skip_buckets));
    out.push_str(&format!("val_token_budget: {}\n", report.val_token_budget));
    out.push_str(&format!("match_depth: {}\n", report.match_depth));
    out.push_str(&format!("candidate_k: {}\n", report.candidate_k));
    out.push_str(&format!("train_stride: {}\n", report.train_stride));
    out.push_str(&format!("tune_records: {}\n", report.tune_records));
    out.push_str(&format!("eval_records: {}\n", report.eval_records));
    out.push_str(&format!(
        "privileged_surface_dim: {}\n",
        report.privileged_surface_dim
    ));
    out.push_str(&format!(
        "oracle_target_source: {}\n",
        report.oracle_target_source
    ));
    out.push_str(&format!(
        "tuned_heuristic_lambda: {:.3}\n",
        report.tuned_heuristic_lambda
    ));
    out.push_str(&format!(
        "tuned_direct_lambda: {:.3}\n",
        report.tuned_direct_lambda
    ));
    out.push_str(&format!(
        "tuned_weighted_direct_lambda: {:.3}\n",
        report.tuned_weighted_direct_lambda
    ));
    out.push_str(&format!(
        "tuned_distilled_direct_lambda: {:.3}\n",
        report.tuned_distilled_direct_lambda
    ));
    out.push_str(&format!(
        "tuned_oracle_supervised_lambda: {:.3}\n",
        report.tuned_oracle_supervised_lambda
    ));
    out.push_str(&format!(
        "tuned_hybrid_lambda: {:.3}\n",
        report.tuned_hybrid_lambda
    ));
    out.push_str(&format!(
        "tuned_trust_head_lambda: {:.3}\n",
        report.tuned_trust_head_lambda
    ));
    out.push_str(&format!(
        "selected_runtime_gate: {}\n",
        report.selected_runtime_gate
    ));
    out.push_str(&format!(
        "selected_runtime_lambda: {:.3}\n",
        report.selected_runtime_lambda
    ));
    out.push_str(&format!("tune_bpt_match: {:.6}\n", report.tune_bpt_match));
    out.push_str(&format!("tune_bpt_skip: {:.6}\n", report.tune_bpt_skip));
    out.push_str(&format!(
        "tune_bpt_heuristic: {:.6}\n",
        report.tune_bpt_heuristic
    ));
    out.push_str(&format!("tune_bpt_direct: {:.6}\n", report.tune_bpt_direct));
    out.push_str(&format!(
        "tune_bpt_weighted_direct: {:.6}\n",
        report.tune_bpt_weighted_direct
    ));
    out.push_str(&format!(
        "tune_bpt_distilled_direct: {:.6}\n",
        report.tune_bpt_distilled_direct
    ));
    out.push_str(&format!(
        "tune_bpt_oracle_supervised: {:.6}\n",
        report.tune_bpt_oracle_supervised
    ));
    out.push_str(&format!("tune_bpt_hybrid: {:.6}\n", report.tune_bpt_hybrid));
    out.push_str(&format!(
        "tune_bpt_trust_head: {:.6}\n",
        report.tune_bpt_trust_head
    ));
    out.push_str(&format!("tune_bpt_oracle: {:.6}\n", report.tune_bpt_oracle));
    out.push_str(&format!("eval_bpt_match: {:.6}\n", report.eval_bpt_match));
    out.push_str(&format!("eval_bpt_skip: {:.6}\n", report.eval_bpt_skip));
    out.push_str(&format!(
        "eval_bpt_heuristic: {:.6}\n",
        report.eval_bpt_heuristic
    ));
    out.push_str(&format!("eval_bpt_direct: {:.6}\n", report.eval_bpt_direct));
    out.push_str(&format!(
        "eval_bpt_weighted_direct: {:.6}\n",
        report.eval_bpt_weighted_direct
    ));
    out.push_str(&format!(
        "eval_bpt_distilled_direct: {:.6}\n",
        report.eval_bpt_distilled_direct
    ));
    out.push_str(&format!(
        "eval_bpt_oracle_supervised: {:.6}\n",
        report.eval_bpt_oracle_supervised
    ));
    out.push_str(&format!("eval_bpt_hybrid: {:.6}\n", report.eval_bpt_hybrid));
    out.push_str(&format!(
        "eval_bpt_trust_head: {:.6}\n",
        report.eval_bpt_trust_head
    ));
    out.push_str(&format!("eval_bpt_oracle: {:.6}\n", report.eval_bpt_oracle));
    if let Some(value) = report.eval_tokens_per_byte {
        out.push_str(&format!("eval_tokens_per_byte: {:.6}\n", value));
    }
    if let Some(value) = report.eval_bytes_per_token {
        out.push_str(&format!("eval_bytes_per_token: {:.6}\n", value));
    }
    if let Some(value) = report.eval_bpb_match {
        out.push_str(&format!("eval_bpb_match: {:.6}\n", value));
    }
    if let Some(value) = report.eval_bpb_skip {
        out.push_str(&format!("eval_bpb_skip: {:.6}\n", value));
    }
    if let Some(value) = report.eval_bpb_heuristic {
        out.push_str(&format!("eval_bpb_heuristic: {:.6}\n", value));
    }
    if let Some(value) = report.eval_bpb_direct {
        out.push_str(&format!("eval_bpb_direct: {:.6}\n", value));
    }
    if let Some(value) = report.eval_bpb_weighted_direct {
        out.push_str(&format!("eval_bpb_weighted_direct: {:.6}\n", value));
    }
    if let Some(value) = report.eval_bpb_distilled_direct {
        out.push_str(&format!("eval_bpb_distilled_direct: {:.6}\n", value));
    }
    if let Some(value) = report.eval_bpb_oracle_supervised {
        out.push_str(&format!("eval_bpb_oracle_supervised: {:.6}\n", value));
    }
    if let Some(value) = report.eval_bpb_hybrid {
        out.push_str(&format!("eval_bpb_hybrid: {:.6}\n", value));
    }
    if let Some(value) = report.eval_bpb_trust_head {
        out.push_str(&format!("eval_bpb_trust_head: {:.6}\n", value));
    }
    if let Some(value) = report.eval_bpb_oracle {
        out.push_str(&format!("eval_bpb_oracle: {:.6}\n", value));
    }
    out.push_str(&format!(
        "tune_trust_head_brier: {:.6}\n",
        report.tune_trust_head_brier
    ));
    out.push_str(&format!(
        "eval_trust_head_brier: {:.6}\n",
        report.eval_trust_head_brier
    ));
    out.push_str(&format!(
        "eval_match_better_rate: {:.6}\n",
        report.eval_match_better_rate
    ));
    out.push_str(&format!(
        "eval_skip_better_rate: {:.6}\n",
        report.eval_skip_better_rate
    ));
    out.push_str(&format!(
        "eval_top1_agreement_rate: {:.6}\n",
        report.eval_top1_agreement_rate
    ));
    out
}

impl Runner for TokenMatchSkipBridgeRunner {
    fn name(&self) -> &'static str {
        "TokenMatchSkipBridgeRunner"
    }

    fn vocab_size(&self) -> usize {
        self.match_runner.vocab_size()
    }

    fn score_chunk(
        &self,
        tokens: &[usize],
        sample_positions: &[usize],
    ) -> Result<SampleOutputs, String> {
        let match_outputs = self.match_runner.score_chunk(tokens, sample_positions)?;
        let skip_outputs = self.skip_runner.score_chunk(tokens, sample_positions)?;
        if match_outputs.sample_predictions.len() != skip_outputs.sample_predictions.len() {
            return Err("match/skip sample prediction length mismatch".to_string());
        }

        let mut predictions = Vec::with_capacity(match_outputs.sample_predictions.len());
        let mut golds = Vec::with_capacity(match_outputs.sample_predictions.len());
        for (index, (match_dist, skip_dist)) in match_outputs
            .sample_predictions
            .iter()
            .zip(skip_outputs.sample_predictions.iter())
            .enumerate()
        {
            let match_stats = summarize_distribution(match_dist, self.candidate_k);
            let skip_stats = summarize_distribution(skip_dist, self.candidate_k);
            let features = standardize_features(
                extract_features(&match_stats, &skip_stats),
                &self.standardizer,
            );
            let privileged_surfaces = standardize_privileged_surfaces(
                extract_privileged_runtime_surfaces(&match_stats, &skip_stats),
                &self.privileged_standardizer,
            );
            let raw_gate = match &self.gate {
                GateChoice::Heuristic => heuristic_gate_from_stats(&match_stats, &skip_stats),
                GateChoice::DirectNll(model) => predict_probability(model, &features),
                GateChoice::OracleSupervised(model) => predict_probability(model, &features),
                GateChoice::HybridOracleNll(model) => predict_probability(model, &features),
                GateChoice::TrustHead(model) => {
                    predict_trust_probability(model, &privileged_surfaces)
                }
            };
            let gate = (self.lambda * raw_gate).clamp(0.0, 1.0);
            let mixed = mix_pair(skip_dist, match_dist, gate);
            let gold = mixed
                .get(tokens[sample_positions[index]])
                .copied()
                .unwrap_or(f64::MIN_POSITIVE)
                .max(f64::MIN_POSITIVE)
                .ln();
            golds.push(gold);
            predictions.push(mixed);
        }

        Ok(SampleOutputs {
            sample_predictions: predictions,
            sample_gold_logprobs: Some(golds),
        })
    }

    fn adapt_chunk(&mut self, tokens: &[usize]) -> Result<(), String> {
        self.match_runner.adapt_chunk(tokens)?;
        self.skip_runner.adapt_chunk(tokens)?;
        Ok(())
    }
}

fn collect_raw_records(
    match_runner: &TokenMatchBridgeRunner,
    skip_runner: &TokenSkipBridgeRunner,
    tokens: &[usize],
    candidate_k: usize,
    teacher_candidate_pairs: Option<&[Vec<(usize, usize)>]>,
) -> Result<Vec<RawRecord>, String> {
    let sample_positions: Vec<usize> = (0..tokens.len()).collect();
    let match_outputs = match_runner.score_chunk(tokens, &sample_positions)?;
    let skip_outputs = skip_runner.score_chunk(tokens, &sample_positions)?;
    let match_golds = match_outputs
        .sample_gold_logprobs
        .ok_or_else(|| "match runner did not return gold logprobs".to_string())?;
    let skip_golds = skip_outputs
        .sample_gold_logprobs
        .ok_or_else(|| "skip runner did not return gold logprobs".to_string())?;

    let mut rows = Vec::with_capacity(tokens.len());
    for pos in 0..tokens.len() {
        let match_dist = &match_outputs.sample_predictions[pos];
        let skip_dist = &skip_outputs.sample_predictions[pos];
        let match_stats = summarize_distribution(match_dist, candidate_k);
        let skip_stats = summarize_distribution(skip_dist, candidate_k);
        let surrogate_target = candidate_trust_target(&match_stats, &skip_stats, tokens[pos]);
        let (teacher_target_probs, teacher_match_probs, teacher_skip_probs, teacher_support_weight) =
            teacher_candidate_probs(
                match_dist,
                skip_dist,
                teacher_candidate_pairs.and_then(|pairs| pairs.get(pos)),
            );
        rows.push(RawRecord {
            features: extract_features(&match_stats, &skip_stats),
            privileged_surfaces: extract_privileged_runtime_surfaces(&match_stats, &skip_stats),
            match_gold_prob: match_golds[pos].exp().max(1e-12),
            skip_gold_prob: skip_golds[pos].exp().max(1e-12),
            heuristic_gate: heuristic_gate_from_stats(&match_stats, &skip_stats),
            oracle_gate_target: surrogate_target,
            trust_target: surrogate_target,
            sample_weight: 1.0,
            teacher_target_probs,
            teacher_match_probs,
            teacher_skip_probs,
            teacher_support_weight,
            top1_agreement: if match_stats.top1_token == skip_stats.top1_token {
                1.0
            } else {
                0.0
            },
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

fn fit_privileged_standardizer_from_raw(records: &[RawRecord]) -> PrivilegedSurfaceStandardizer {
    let mut mean = [0.0; PRIVILEGED_SURFACE_DIM];
    for record in records {
        for (index, value) in record.privileged_surfaces.iter().enumerate() {
            mean[index] += *value;
        }
    }
    for value in &mut mean {
        *value /= records.len() as f64;
    }

    let mut var = [0.0; PRIVILEGED_SURFACE_DIM];
    for record in records {
        for (index, value) in record.privileged_surfaces.iter().enumerate() {
            let diff = *value - mean[index];
            var[index] += diff * diff;
        }
    }
    let mut std = [1.0; PRIVILEGED_SURFACE_DIM];
    for (index, value) in var.iter().enumerate() {
        std[index] = (value / records.len() as f64).sqrt().max(1e-6);
    }
    PrivilegedSurfaceStandardizer { mean, std }
}

fn standardize_records(
    records: &[RawRecord],
    standardizer: &Standardizer,
    privileged_standardizer: &PrivilegedSurfaceStandardizer,
) -> Vec<Record> {
    records
        .iter()
        .map(|record| Record {
            features: standardize_features(record.features, standardizer),
            privileged_surfaces: standardize_privileged_surfaces(
                record.privileged_surfaces,
                privileged_standardizer,
            ),
            match_gold_prob: record.match_gold_prob,
            skip_gold_prob: record.skip_gold_prob,
            heuristic_gate: record.heuristic_gate,
            oracle_gate_target: record.oracle_gate_target,
            trust_target: record.trust_target,
            sample_weight: record.sample_weight,
            teacher_target_probs: record.teacher_target_probs.clone(),
            teacher_match_probs: record.teacher_match_probs.clone(),
            teacher_skip_probs: record.teacher_skip_probs.clone(),
            teacher_support_weight: record.teacher_support_weight,
            top1_agreement: record.top1_agreement,
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

fn standardize_privileged_surfaces(
    privileged_surfaces: [f64; PRIVILEGED_SURFACE_DIM],
    standardizer: &PrivilegedSurfaceStandardizer,
) -> [f64; PRIVILEGED_SURFACE_DIM] {
    let mut out = [0.0; PRIVILEGED_SURFACE_DIM];
    for index in 0..PRIVILEGED_SURFACE_DIM {
        out[index] =
            (privileged_surfaces[index] - standardizer.mean[index]) / standardizer.std[index];
    }
    out
}

fn apply_offline_target_stream(
    tune_raw: &mut [RawRecord],
    eval_raw: &mut [RawRecord],
    offline_targets: Option<&TokenMatchSkipOfflineTargetStream>,
) -> Result<String, String> {
    match offline_targets {
        Some(offline_targets) => {
            apply_target_slice(
                tune_raw,
                &offline_targets.tune_gate_targets,
                offline_targets.tune_trust_targets.as_deref(),
                offline_targets.tune_teacher_candidate_pairs.as_deref(),
                "tune",
            )?;
            apply_target_slice(
                eval_raw,
                &offline_targets.eval_gate_targets,
                offline_targets.eval_trust_targets.as_deref(),
                offline_targets.eval_teacher_candidate_pairs.as_deref(),
                "eval",
            )?;
            Ok(offline_targets.source.clone())
        }
        None => Ok("surrogate_candidate_trust".to_string()),
    }
}

fn apply_target_slice(
    records: &mut [RawRecord],
    gate_targets: &[f64],
    trust_targets: Option<&[f64]>,
    teacher_candidate_pairs: Option<&[Vec<(usize, usize)>]>,
    split_name: &str,
) -> Result<(), String> {
    if records.len() != gate_targets.len() {
        return Err(format!(
            "{split_name} gate target length mismatch: records {} targets {}",
            records.len(),
            gate_targets.len()
        ));
    }
    let trust_targets = trust_targets.unwrap_or(gate_targets);
    if records.len() != trust_targets.len() {
        return Err(format!(
            "{split_name} trust target length mismatch: records {} targets {}",
            records.len(),
            trust_targets.len()
        ));
    }
    if let Some(teacher_candidate_pairs) = teacher_candidate_pairs {
        if records.len() != teacher_candidate_pairs.len() {
            return Err(format!(
                "{split_name} teacher candidate length mismatch: records {} candidates {}",
                records.len(),
                teacher_candidate_pairs.len()
            ));
        }
    }

    let sample_weights = normalized_sample_weights(gate_targets);

    for (index, record) in records.iter_mut().enumerate() {
        record.oracle_gate_target =
            normalize_offline_target(gate_targets[index], split_name, "gate", index)?;
        record.trust_target =
            normalize_offline_target(trust_targets[index], split_name, "trust", index)?;
        record.sample_weight = sample_weights[index];
    }
    Ok(())
}

fn normalize_offline_target(
    value: f64,
    split_name: &str,
    target_kind: &str,
    index: usize,
) -> Result<f64, String> {
    if !value.is_finite() {
        return Err(format!(
            "{split_name} {target_kind} target at record {index} is not finite"
        ));
    }
    Ok(value.clamp(0.0, 1.0))
}

fn normalized_sample_weights(gate_targets: &[f64]) -> Vec<f64> {
    if gate_targets.is_empty() {
        return Vec::new();
    }
    let mut weights = gate_targets
        .iter()
        .map(|value| 0.5 + value.clamp(0.0, 1.0))
        .collect::<Vec<_>>();
    let mean = weights.iter().sum::<f64>() / weights.len() as f64;
    let scale = if mean > 0.0 { 1.0 / mean } else { 1.0 };
    for weight in &mut weights {
        *weight *= scale;
    }
    weights
}

fn teacher_candidate_probs(
    match_dist: &[f64],
    skip_dist: &[f64],
    teacher_candidate_pairs: Option<&Vec<(usize, usize)>>,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, f64) {
    let Some(teacher_candidate_pairs) = teacher_candidate_pairs else {
        return (Vec::new(), Vec::new(), Vec::new(), 0.0);
    };
    let total = teacher_candidate_pairs
        .iter()
        .map(|(_, count)| *count as f64)
        .sum::<f64>()
        .max(1e-12);
    let mut teacher_probs = Vec::with_capacity(teacher_candidate_pairs.len());
    let mut match_probs = Vec::with_capacity(teacher_candidate_pairs.len());
    let mut skip_probs = Vec::with_capacity(teacher_candidate_pairs.len());
    for &(token, count) in teacher_candidate_pairs {
        teacher_probs.push((count as f64) / total);
        match_probs.push(match_dist.get(token).copied().unwrap_or(0.0).max(1e-12));
        skip_probs.push(skip_dist.get(token).copied().unwrap_or(0.0).max(1e-12));
    }
    let support_size = teacher_candidate_pairs.len();
    let support_weight = if support_size <= 1 {
        0.0
    } else {
        ((support_size as f64).ln() / 4.0_f64.ln()).clamp(0.0, 1.0)
    };
    (teacher_probs, match_probs, skip_probs, support_weight)
}

fn summarize_distribution(distribution: &[f64], candidate_k: usize) -> DistStats {
    let mut indexed = distribution
        .iter()
        .enumerate()
        .map(|(index, &value)| (index, value))
        .collect::<Vec<_>>();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let support = indexed
        .iter()
        .take(candidate_k.max(1).min(indexed.len().max(1)))
        .copied()
        .collect::<Vec<_>>();
    let top1_prob = indexed.first().map(|(_, value)| *value).unwrap_or(0.0);
    let top2_prob = indexed.get(1).map(|(_, value)| *value).unwrap_or(0.0);
    let top1_token = indexed.first().map(|(index, _)| *index).unwrap_or(0);
    let topk_mass = support.iter().map(|(_, value)| *value).sum::<f64>();
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
        top1_token,
        entropy_norm,
        topk_mass,
        margin: top1_prob - top2_prob,
        support,
    }
}

fn extract_features(match_stats: &DistStats, skip_stats: &DistStats) -> [f64; FEATURE_DIM] {
    let top1_agreement = if match_stats.top1_token == skip_stats.top1_token {
        1.0
    } else {
        0.0
    };
    let support_overlap = support_overlap_fraction(&match_stats.support, &skip_stats.support);
    let shared_support_mass =
        shared_support_mass_fraction(&match_stats.support, &skip_stats.support);
    let exclusive_mass_delta =
        exclusive_support_mass_delta(&match_stats.support, &skip_stats.support);
    [
        match_stats.top1_prob,
        match_stats.margin,
        match_stats.topk_mass,
        match_stats.entropy_norm,
        skip_stats.top1_prob,
        skip_stats.margin,
        skip_stats.topk_mass,
        skip_stats.entropy_norm,
        top1_agreement,
        support_overlap,
        shared_support_mass,
        exclusive_mass_delta,
    ]
}

fn extract_privileged_runtime_surfaces(
    match_stats: &DistStats,
    skip_stats: &DistStats,
) -> [f64; PRIVILEGED_SURFACE_DIM] {
    let top1_agreement = if match_stats.top1_token == skip_stats.top1_token {
        1.0
    } else {
        0.0
    };
    let support_overlap = support_overlap_fraction(&match_stats.support, &skip_stats.support);
    [
        match_stats.top1_prob,
        match_stats.margin,
        skip_stats.top1_prob,
        skip_stats.margin,
        support_overlap,
        top1_agreement,
    ]
}

fn heuristic_gate_from_stats(match_stats: &DistStats, skip_stats: &DistStats) -> f64 {
    let top1_agreement = if match_stats.top1_token == skip_stats.top1_token {
        1.0
    } else {
        0.0
    };
    let support_overlap = support_overlap_fraction(&match_stats.support, &skip_stats.support);
    let shared_support_mass =
        shared_support_mass_fraction(&match_stats.support, &skip_stats.support);
    let exclusive_mass_delta =
        exclusive_support_mass_delta(&match_stats.support, &skip_stats.support);
    let confidence_delta =
        (match_stats.top1_prob + match_stats.margin) - (skip_stats.top1_prob + skip_stats.margin);
    let concentration_delta = (match_stats.topk_mass - skip_stats.topk_mass)
        + 0.5 * (skip_stats.entropy_norm - match_stats.entropy_norm);
    sigmoid(
        1.75 * confidence_delta
            + 1.25 * concentration_delta
            + 0.75 * support_overlap
            + 0.75 * shared_support_mass
            + 0.5 * exclusive_mass_delta
            + 0.25 * top1_agreement,
    )
}

fn candidate_trust_target(
    match_stats: &DistStats,
    skip_stats: &DistStats,
    gold_token: usize,
) -> f64 {
    let match_has = match_stats
        .support
        .iter()
        .any(|(token, _)| *token == gold_token);
    let skip_has = skip_stats
        .support
        .iter()
        .any(|(token, _)| *token == gold_token);
    match (match_has, skip_has) {
        (true, false) => 1.0,
        (false, true) => 0.0,
        _ => heuristic_gate_from_stats(match_stats, skip_stats),
    }
}

fn support_overlap_fraction(left: &[(usize, f64)], right: &[(usize, f64)]) -> f64 {
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }
    let overlap = left
        .iter()
        .filter(|(index, _)| right.iter().any(|(other, _)| other == index))
        .count();
    overlap as f64 / left.len().min(right.len()) as f64
}

fn shared_support_mass_fraction(left: &[(usize, f64)], right: &[(usize, f64)]) -> f64 {
    let left_total = left.iter().map(|(_, value)| *value).sum::<f64>();
    let right_total = right.iter().map(|(_, value)| *value).sum::<f64>();
    let denom = left_total.min(right_total).max(1e-12);
    let shared = left
        .iter()
        .filter_map(|(index, left_value)| {
            right
                .iter()
                .find(|(other, _)| other == index)
                .map(|(_, right_value)| left_value.min(*right_value))
        })
        .sum::<f64>();
    (shared / denom).clamp(0.0, 1.0)
}

fn exclusive_support_mass_delta(left: &[(usize, f64)], right: &[(usize, f64)]) -> f64 {
    let shared_left = left
        .iter()
        .filter(|(index, _)| right.iter().any(|(other, _)| other == index))
        .map(|(_, value)| *value)
        .sum::<f64>();
    let shared_right = right
        .iter()
        .filter(|(index, _)| left.iter().any(|(other, _)| other == index))
        .map(|(_, value)| *value)
        .sum::<f64>();
    (left.iter().map(|(_, value)| *value).sum::<f64>() - shared_left)
        - (right.iter().map(|(_, value)| *value).sum::<f64>() - shared_right)
}

fn sigmoid(value: f64) -> f64 {
    1.0 / (1.0 + (-value).exp())
}

fn train_direct_gate(records: &[Record], epochs: usize, lr: f64, l2: f64) -> LogisticModel {
    let mut model = LogisticModel {
        weights: [0.0; FEATURE_DIM],
        bias: 0.0,
    };
    let ln2 = std::f64::consts::LN_2;
    for _ in 0..epochs {
        let mut grad_w = [0.0; FEATURE_DIM];
        let mut grad_b = 0.0;
        for record in records {
            let prediction = predict_probability(&model, &record.features).clamp(1e-6, 1.0 - 1e-6);
            let mixed = ((1.0 - prediction) * record.skip_gold_prob
                + prediction * record.match_gold_prob)
                .max(1e-12);
            let dloss_dgate = (record.skip_gold_prob - record.match_gold_prob) / (mixed * ln2);
            let dloss_dz = dloss_dgate * prediction * (1.0 - prediction);
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

fn train_weighted_direct_gate(
    records: &[Record],
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
        let mut total_weight = 0.0;
        for record in records {
            let prediction = predict_probability(&model, &record.features).clamp(1e-6, 1.0 - 1e-6);
            let mixed = ((1.0 - prediction) * record.skip_gold_prob
                + prediction * record.match_gold_prob)
                .max(1e-12);
            let dloss_dgate = (record.skip_gold_prob - record.match_gold_prob) / (mixed * ln2);
            let dloss_dz = record.sample_weight * dloss_dgate * prediction * (1.0 - prediction);
            for (index, value) in record.features.iter().enumerate() {
                grad_w[index] += dloss_dz * *value;
            }
            grad_b += dloss_dz;
            total_weight += record.sample_weight;
        }
        let inv_n = 1.0 / total_weight.max(1e-12);
        for index in 0..FEATURE_DIM {
            model.weights[index] -= lr * (grad_w[index] * inv_n + l2 * model.weights[index]);
        }
        model.bias -= lr * grad_b * inv_n;
    }
    model
}

fn train_distilled_direct_gate(
    records: &[Record],
    epochs: usize,
    lr: f64,
    l2: f64,
    alpha_nll: f64,
    beta_support: f64,
    gamma_rank: f64,
) -> LogisticModel {
    let mut model = LogisticModel {
        weights: [0.0; FEATURE_DIM],
        bias: 0.0,
    };
    let ln2 = std::f64::consts::LN_2;
    for _ in 0..epochs {
        let mut grad_w = [0.0; FEATURE_DIM];
        let mut grad_b = 0.0;
        let mut total_weight = 0.0;
        for record in records {
            let prediction = predict_probability(&model, &record.features).clamp(1e-6, 1.0 - 1e-6);
            let mixed_gold = ((1.0 - prediction) * record.skip_gold_prob
                + prediction * record.match_gold_prob)
                .max(1e-12);
            let nll_grad = (record.skip_gold_prob - record.match_gold_prob) / (mixed_gold * ln2);
            let support_grad = support_set_gate_grad(record, prediction, ln2);
            let rank_grad = ranked_teacher_gate_grad(record, prediction, ln2);
            let dloss_dgate = record.sample_weight
                * (alpha_nll * nll_grad
                    + beta_support * record.teacher_support_weight * support_grad
                    + gamma_rank * record.teacher_support_weight * rank_grad);
            let dloss_dz = dloss_dgate * prediction * (1.0 - prediction);
            for (index, value) in record.features.iter().enumerate() {
                grad_w[index] += dloss_dz * *value;
            }
            grad_b += dloss_dz;
            total_weight += record.sample_weight;
        }
        let inv_n = 1.0 / total_weight.max(1e-12);
        for index in 0..FEATURE_DIM {
            model.weights[index] -= lr * (grad_w[index] * inv_n + l2 * model.weights[index]);
        }
        model.bias -= lr * grad_b * inv_n;
    }
    model
}

fn train_oracle_supervised_gate(
    records: &[Record],
    epochs: usize,
    lr: f64,
    l2: f64,
) -> LogisticModel {
    let mut model = LogisticModel {
        weights: [0.0; FEATURE_DIM],
        bias: 0.0,
    };
    for _ in 0..epochs {
        let mut grad_w = [0.0; FEATURE_DIM];
        let mut grad_b = 0.0;
        for record in records {
            let prediction = predict_probability(&model, &record.features);
            let dloss_dz = prediction - record.oracle_gate_target;
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

fn train_tiny_trust_head(records: &[Record], epochs: usize, lr: f64, l2: f64) -> TinyTrustHead {
    let mut model = TinyTrustHead {
        weights: [0.0; PRIVILEGED_SURFACE_DIM],
        bias: 0.0,
    };
    for _ in 0..epochs {
        let mut grad_w = [0.0; PRIVILEGED_SURFACE_DIM];
        let mut grad_b = 0.0;
        for record in records {
            let prediction = predict_trust_probability(&model, &record.privileged_surfaces);
            let dloss_dz = prediction - record.trust_target;
            for (index, value) in record.privileged_surfaces.iter().enumerate() {
                grad_w[index] += dloss_dz * *value;
            }
            grad_b += dloss_dz;
        }
        let inv_n = 1.0 / records.len() as f64;
        for index in 0..PRIVILEGED_SURFACE_DIM {
            model.weights[index] -= lr * (grad_w[index] * inv_n + l2 * model.weights[index]);
        }
        model.bias -= lr * grad_b * inv_n;
    }
    model
}

fn train_hybrid_gate(
    records: &[Record],
    epochs: usize,
    lr: f64,
    l2: f64,
    aux_weight: f64,
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
            let prediction = predict_probability(&model, &record.features).clamp(1e-6, 1.0 - 1e-6);
            let mixed = ((1.0 - prediction) * record.skip_gold_prob
                + prediction * record.match_gold_prob)
                .max(1e-12);
            let dloss_dgate = (record.skip_gold_prob - record.match_gold_prob) / (mixed * ln2);
            let dloss_dz = dloss_dgate * prediction * (1.0 - prediction)
                + aux_weight * (prediction - record.oracle_gate_target);
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

fn predict_trust_probability(
    model: &TinyTrustHead,
    privileged_surfaces: &[f64; PRIVILEGED_SURFACE_DIM],
) -> f64 {
    let mut z = model.bias;
    for (weight, value) in model.weights.iter().zip(privileged_surfaces.iter()) {
        z += weight * value;
    }
    1.0 / (1.0 + (-z).exp())
}

fn support_set_gate_grad(record: &Record, prediction: f64, ln2: f64) -> f64 {
    if record.teacher_match_probs.is_empty()
        || record.teacher_match_probs.len() != record.teacher_skip_probs.len()
    {
        return 0.0;
    }
    let mut support_mass = 0.0;
    let mut support_delta = 0.0;
    for (&match_prob, &skip_prob) in record
        .teacher_match_probs
        .iter()
        .zip(record.teacher_skip_probs.iter())
    {
        let mixed = ((1.0 - prediction) * skip_prob + prediction * match_prob).max(1e-12);
        support_mass += mixed;
        support_delta += match_prob - skip_prob;
    }
    if support_mass <= 0.0 {
        return 0.0;
    }
    -support_delta / (support_mass * ln2)
}

fn ranked_teacher_gate_grad(record: &Record, prediction: f64, ln2: f64) -> f64 {
    if record.teacher_target_probs.is_empty()
        || record.teacher_target_probs.len() != record.teacher_match_probs.len()
        || record.teacher_target_probs.len() != record.teacher_skip_probs.len()
    {
        return 0.0;
    }
    let mut support_mass = 0.0;
    let mut support_delta = 0.0;
    let mut ranked_term = 0.0;
    for ((&teacher_prob, &match_prob), &skip_prob) in record
        .teacher_target_probs
        .iter()
        .zip(record.teacher_match_probs.iter())
        .zip(record.teacher_skip_probs.iter())
    {
        let mixed = ((1.0 - prediction) * skip_prob + prediction * match_prob).max(1e-12);
        let delta = match_prob - skip_prob;
        support_mass += mixed;
        support_delta += delta;
        ranked_term += teacher_prob * (delta / mixed);
    }
    if support_mass <= 0.0 {
        return 0.0;
    }
    (support_delta / support_mass - ranked_term) / ln2
}

fn select_runtime_gate<const N: usize>(
    candidates: [RuntimeGateCandidate; N],
) -> RuntimeGateCandidate {
    let mut best_index = 0usize;
    for index in 1..N {
        let ordering = candidates[index]
            .tune_bpt
            .partial_cmp(&candidates[best_index].tune_bpt)
            .unwrap_or(std::cmp::Ordering::Greater);
        if ordering == std::cmp::Ordering::Less {
            best_index = index;
        }
    }
    candidates
        .into_iter()
        .nth(best_index)
        .expect("runtime gate candidate selection should not be empty")
}

#[derive(Clone, Copy)]
enum Branch {
    Match,
    Skip,
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

fn mean_bits_for_branch(records: &[Record], branch: Branch) -> f64 {
    records
        .iter()
        .map(|record| {
            let prob = match branch {
                Branch::Match => record.match_gold_prob,
                Branch::Skip => record.skip_gold_prob,
            };
            -prob.max(1e-12).log2()
        })
        .sum::<f64>()
        / records.len() as f64
}

fn mean_bits_with_gate<F>(records: &[Record], lambda: f64, mut gate_fn: F) -> f64
where
    F: FnMut(&Record) -> f64,
{
    records
        .iter()
        .map(|record| {
            let gate = (lambda * gate_fn(record)).clamp(0.0, 1.0);
            let prob =
                ((1.0 - gate) * record.skip_gold_prob + gate * record.match_gold_prob).max(1e-12);
            -prob.log2()
        })
        .sum::<f64>()
        / records.len() as f64
}

fn trust_brier_score<F>(records: &[Record], mut trust_fn: F) -> f64
where
    F: FnMut(&Record) -> f64,
{
    records
        .iter()
        .map(|record| {
            let diff = trust_fn(record) - record.trust_target;
            diff * diff
        })
        .sum::<f64>()
        / records.len() as f64
}

fn oracle_bits_per_token(records: &[Record]) -> f64 {
    records
        .iter()
        .map(|record| {
            -record
                .match_gold_prob
                .max(record.skip_gold_prob)
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
            Branch::Match => record.match_gold_prob > record.skip_gold_prob,
            Branch::Skip => record.skip_gold_prob > record.match_gold_prob,
        })
        .count() as f64
        / records.len() as f64
}

fn agreement_rate(records: &[Record]) -> f64 {
    records
        .iter()
        .filter(|record| record.top1_agreement > 0.5)
        .count() as f64
        / records.len() as f64
}

fn mix_pair(skip: &[f64], match_dist: &[f64], gate: f64) -> Vec<f64> {
    let mut mixed = Vec::with_capacity(skip.len());
    let mut total = 0.0;
    for (&skip_p, &match_p) in skip.iter().zip(match_dist.iter()) {
        let value = (1.0 - gate) * skip_p + gate * match_p;
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
