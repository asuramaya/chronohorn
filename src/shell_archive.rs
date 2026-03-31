use std::path::Path;

use chronohorn::archive::byte_bridge::{
    render_byte_bridge_codec_report, render_byte_bridge_report, run_byte_bridge,
    run_byte_bridge_codec,
};
use chronohorn::archive::legacy_bridge_report_schema::{
    BridgeReportBundle, BridgeReportManifest, TOKEN_SEQUENCE_SKIP_BRIDGE_ALIASES,
};
use chronohorn::archive::token_bridge::{
    render_token_bridge_report, run_token_bridge_from_data_root, train_token_bridge_from_data_root,
};
use chronohorn::archive::token_c3c7::{
    render_token_c3c7_report, run_token_c3c7_from_data_root, train_token_c3c7_from_data_root,
};
use chronohorn::archive::token_column_bridge::{
    render_token_column_bridge_report, run_token_column_bridge_from_data_root,
    train_token_column_bridge_from_data_root,
};
use chronohorn::archive::token_conker3::{
    render_token_conker3_report, run_token_conker3_from_data_root,
    train_token_conker3_from_data_root,
};
use chronohorn::archive::token_copy_bridge::{
    render_token_copy_bridge_report, run_token_copy_bridge_from_data_root,
    train_token_copy_bridge_from_data_root,
};
use chronohorn::archive::token_decay_bridge::{
    render_token_decay_bridge_report, run_token_decay_bridge_from_data_root,
    train_token_decay_bridge_from_data_root,
};
use chronohorn::archive::token_experiment_matrix::{
    TokenExperimentMatrixConfig, run_token_experiment_matrix_from_data_root,
};
use chronohorn::archive::token_match_bridge::{
    render_token_match_bridge_report, run_token_match_bridge_from_data_root,
    train_token_match_bridge_from_data_root,
};
use chronohorn::archive::token_matchcopy_bridge::{
    render_token_matchcopy_bridge_report, run_token_matchcopy_bridge_from_data_root,
    train_token_matchcopy_bridge_from_data_root,
};
use chronohorn::archive::token_matchskip_bridge::{
    TokenMatchSkipBridgeReport, TokenMatchSkipOfflineTargetStream,
    render_token_matchskip_bridge_report, run_token_matchskip_bridge_from_data_root,
    train_token_matchskip_bridge_from_data_root,
    train_token_matchskip_bridge_from_data_root_with_offline_targets,
};
use chronohorn::archive::token_matchskipcopy_bridge::{
    render_token_matchskipcopy_bridge_report, run_token_matchskipcopy_bridge_from_data_root,
    train_token_matchskipcopy_bridge_from_data_root,
};
use chronohorn::archive::token_skip_bridge::{
    render_token_skip_bridge_report, run_token_skip_bridge_from_data_root,
    train_token_skip_bridge_from_data_root,
};
use chronohorn::archive::token_skipcopy_bridge::{
    render_token_skipcopy_bridge_report, run_token_skipcopy_bridge_from_data_root,
    train_token_skipcopy_bridge_from_data_root,
};
use chronohorn::archive::token_word_bridge::{
    render_token_word_bridge_report, run_token_word_bridge_from_data_root,
    train_token_word_bridge_from_data_root,
};
use chronohorn::causal_bank::oracle::{
    load_token_oracle_val_target_dataset, token_oracle_target_dataset_from_tokens,
    token_oracle_target_values, token_oracle_teacher_candidate_pairs,
};
use chronohorn::core::audit::{LegalityReport, audit_parameter_golf};
use chronohorn::core::bridge::{OracleTargetKind, bridge_doctrine};
use chronohorn::core::data::{resolve_data_root, take_val_tokens};
use chronohorn::core::protocol::Runner;

use crate::shell_support::{indent_block, parse_f64_flag, parse_usize_flag};
use crate::shell_usage::print_archive_usage;

#[derive(Debug)]
struct MatchskipExportArgs {
    data_root: String,
    train_tokens: usize,
    trigram_buckets: usize,
    skip_buckets: usize,
    val_tokens: usize,
    match_depth: usize,
    candidate_k: usize,
    train_stride: usize,
    chunk_size: usize,
    max_chunks: usize,
}

#[derive(Debug)]
struct MatchskipExportContext {
    manifest: BridgeReportManifest,
    report: TokenMatchSkipBridgeReport,
    audit: LegalityReport,
}

fn parse_oracle_target_kind_flag(
    value: Option<String>,
    label: &str,
    default: OracleTargetKind,
) -> Result<OracleTargetKind, String> {
    match value {
        Some(raw) => {
            parse_oracle_target_kind(&raw).map_err(|err| format!("invalid {label} {raw}: {err}"))
        }
        None => Ok(default),
    }
}

fn parse_oracle_target_kind(raw: &str) -> Result<OracleTargetKind, String> {
    match raw {
        "candidate_set_leq_2" => Ok(OracleTargetKind::CandidateSetLeq2),
        "candidate_set_leq_4" => Ok(OracleTargetKind::CandidateSetLeq4),
        "candidate_set_leq_8" => Ok(OracleTargetKind::CandidateSetLeq8),
        "memory_trust" => Ok(OracleTargetKind::MemoryTrust),
        "bridge_confidence" => Ok(OracleTargetKind::BridgeConfidence),
        "clean_bridge_score" => Ok(OracleTargetKind::CleanBridgeScore),
        _ => Err(
            "expected one of candidate_set_leq_2, candidate_set_leq_4, candidate_set_leq_8, memory_trust, bridge_confidence, clean_bridge_score"
                .to_string(),
        ),
    }
}

fn oracle_target_kind_name(kind: OracleTargetKind) -> &'static str {
    match kind {
        OracleTargetKind::CandidateSetLeq2 => "candidate_set_leq_2",
        OracleTargetKind::CandidateSetLeq4 => "candidate_set_leq_4",
        OracleTargetKind::CandidateSetLeq8 => "candidate_set_leq_8",
        OracleTargetKind::MemoryTrust => "memory_trust",
        OracleTargetKind::BridgeConfidence => "bridge_confidence",
        OracleTargetKind::CleanBridgeScore => "clean_bridge_score",
    }
}

fn dense_oracle_target_vector(
    total_len: usize,
    entries: Vec<(usize, f64)>,
    target_name: &str,
) -> Result<Vec<f64>, String> {
    let mut dense = vec![0.0; total_len];
    let mut seen = vec![false; total_len];
    for (position, score) in entries {
        if position >= total_len {
            return Err(format!(
                "{target_name} target position {position} out of bounds for length {total_len}"
            ));
        }
        if seen[position] {
            return Err(format!(
                "{target_name} target position {position} was emitted more than once"
            ));
        }
        dense[position] = score;
        seen[position] = true;
    }
    Ok(dense)
}

fn dense_teacher_candidate_pair_vector(
    total_len: usize,
    entries: Vec<(usize, Vec<(usize, usize)>)>,
) -> Result<Vec<Vec<(usize, usize)>>, String> {
    let mut dense = vec![Vec::new(); total_len];
    let mut seen = vec![false; total_len];
    for (position, candidates) in entries {
        if position >= total_len {
            return Err(format!(
                "teacher candidate position {position} out of bounds for length {total_len}"
            ));
        }
        if seen[position] {
            return Err(format!(
                "teacher candidate position {position} was emitted more than once"
            ));
        }
        dense[position] = candidates;
        seen[position] = true;
    }
    Ok(dense)
}

fn build_token_matchskip_offline_targets(
    root: &Path,
    val_token_budget: usize,
    oracle_radius: usize,
    oracle_stride: usize,
    gate_target_kind: OracleTargetKind,
    trust_target_kind: OracleTargetKind,
) -> Result<TokenMatchSkipOfflineTargetStream, String> {
    let val_tokens = take_val_tokens(root, val_token_budget)?;
    if val_tokens.len() < 8 {
        return Err("need at least 8 validation tokens for token oracle supervision".to_string());
    }
    let split = (val_tokens.len() / 2).max(1);
    if split >= val_tokens.len() {
        return Err(
            "validation split left no eval tokens for token oracle supervision".to_string(),
        );
    }

    let dataset =
        load_token_oracle_val_target_dataset(root, val_token_budget, oracle_radius, oracle_stride)?;
    let gate_targets = dense_oracle_target_vector(
        val_tokens.len(),
        token_oracle_target_values(&dataset, gate_target_kind),
        oracle_target_kind_name(gate_target_kind),
    )?;
    let trust_targets = dense_oracle_target_vector(
        val_tokens.len(),
        token_oracle_target_values(&dataset, trust_target_kind),
        oracle_target_kind_name(trust_target_kind),
    )?;
    let teacher_candidate_pairs = dense_teacher_candidate_pair_vector(
        val_tokens.len(),
        token_oracle_teacher_candidate_pairs(&dataset),
    )?;

    Ok(TokenMatchSkipOfflineTargetStream {
        source: format!(
            "{}:gate={}:trust={}:radius={}:stride={}",
            dataset.source_family,
            oracle_target_kind_name(gate_target_kind),
            oracle_target_kind_name(trust_target_kind),
            oracle_radius,
            oracle_stride
        ),
        tune_gate_targets: gate_targets[..split].to_vec(),
        eval_gate_targets: gate_targets[split..].to_vec(),
        tune_trust_targets: Some(trust_targets[..split].to_vec()),
        eval_trust_targets: Some(trust_targets[split..].to_vec()),
        tune_teacher_candidate_pairs: Some(teacher_candidate_pairs[..split].to_vec()),
        eval_teacher_candidate_pairs: Some(teacher_candidate_pairs[split..].to_vec()),
    })
}

fn distribution_entropy_bits(distribution: &[f64]) -> f64 {
    distribution
        .iter()
        .filter(|value| **value > 0.0)
        .map(|value| -value * value.log2())
        .sum::<f64>()
}

fn distribution_top1_prob(distribution: &[f64]) -> f64 {
    distribution
        .iter()
        .copied()
        .fold(0.0, |best, value| best.max(value))
}

fn parse_matchskip_export_args(
    args: &mut (impl Iterator<Item = String> + ?Sized),
    command_name: &str,
) -> Result<MatchskipExportArgs, String> {
    let data_root = args
        .next()
        .ok_or_else(|| format!("{command_name} requires a data root"))?;
    let train_tokens = parse_usize_flag(args.next(), "train_tokens", 1_000_000)?;
    let trigram_buckets = parse_usize_flag(args.next(), "trigram_buckets", 2_048)?;
    let skip_buckets = parse_usize_flag(args.next(), "skip_buckets", 2_048)?;
    let val_tokens = parse_usize_flag(args.next(), "val_tokens", 32_768)?;
    let match_depth = parse_usize_flag(args.next(), "match_depth", 8)?;
    let candidate_k = parse_usize_flag(args.next(), "candidate_k", 4)?;
    let train_stride = parse_usize_flag(args.next(), "train_stride", 1)?;
    let chunk_size = parse_usize_flag(args.next(), "chunk_size", 64)?;
    let max_chunks = parse_usize_flag(args.next(), "max_chunks", 8)?;
    if args.next().is_some() {
        return Err(format!(
            "{command_name} takes <data-root> [train_tokens] [trigram_buckets] [skip_buckets] [val_tokens] [match_depth] [candidate_k] [train_stride] [chunk_size] [max_chunks]"
        ));
    }
    Ok(MatchskipExportArgs {
        data_root,
        train_tokens,
        trigram_buckets,
        skip_buckets,
        val_tokens,
        match_depth,
        candidate_k,
        train_stride,
        chunk_size,
        max_chunks,
    })
}

fn build_matchskip_export_context(
    args: MatchskipExportArgs,
    source_command: &'static str,
) -> Result<MatchskipExportContext, String> {
    let resolved = resolve_data_root(Some(&args.data_root))?;
    let doctrine = bridge_doctrine();
    let oracle_contract_declared = doctrine.oracle_contract.contains(&"offline only")
        && doctrine.oracle_contract.contains(&"never in eval path")
        && doctrine
            .audit_contract
            .contains(&"prove the bridge did not smuggle the oracle into runtime");
    let trained = train_token_matchskip_bridge_from_data_root(
        Path::new(&resolved.selected_path),
        args.train_tokens,
        args.trigram_buckets,
        args.skip_buckets,
        args.val_tokens,
        args.match_depth,
        args.candidate_k,
        args.train_stride,
        4.0,
        2.0,
        2.0,
    )?;
    let audit = audit_parameter_golf(
        trained.runner(),
        trained.eval_tokens(),
        args.chunk_size,
        args.max_chunks,
    )?;
    let report = trained.report().clone();
    let manifest = BridgeReportManifest {
        schema: "chronohorn.export.bundle",
        schema_version: "1",
        family: "token_matchskip_bridge",
        aliases: TOKEN_SEQUENCE_SKIP_BRIDGE_ALIASES,
        oracle_contract_declared,
        source_command,
        data_root_spec: args.data_root,
        data_root_resolution: resolved,
        runner_name: trained.runner().name(),
        selected_runtime_gate: report.selected_runtime_gate.clone(),
        selected_runtime_lambda: Some(report.selected_runtime_lambda),
        chunk_size: args.chunk_size,
        max_chunks: args.max_chunks,
    };
    Ok(MatchskipExportContext {
        manifest,
        report,
        audit,
    })
}

pub(crate) fn handle(
    command: &str,
    args: &mut dyn Iterator<Item = String>,
) -> Option<Result<(), String>> {
    let result = match command {
        "help-archive" => {
            if args.next().is_some() {
                return Some(Err("help-archive takes no arguments".to_string()));
            }
            print_archive_usage();
            Some(Ok(()))
        }
        "train-byte-bridge" => {
            let radius = match parse_usize_flag(args.next(), "radius", 4) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            let stride = match parse_usize_flag(args.next(), "stride", 16) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            let max_files = match parse_usize_flag(args.next(), "max_files", 80) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            if args.next().is_some() {
                return Some(Err(
                    "train-byte-bridge takes [radius] [stride] [max-files]".to_string()
                ));
            }
            Some(
                run_byte_bridge(Path::new("."), radius, stride, max_files).map(|report| {
                    print!("{}", render_byte_bridge_report(&report));
                }),
            )
        }
        "run-byte-bridge-codec" => {
            let radius = match parse_usize_flag(args.next(), "radius", 4) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            let stride = match parse_usize_flag(args.next(), "stride", 16) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            let max_files = match parse_usize_flag(args.next(), "max_files", 80) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            if args.next().is_some() {
                return Some(Err(
                    "run-byte-bridge-codec takes [radius] [stride] [max-files]".to_string(),
                ));
            }
            Some(
                run_byte_bridge_codec(Path::new("."), radius, stride, max_files).map(|report| {
                    print!("{}", render_byte_bridge_codec_report(&report));
                }),
            )
        }
        "run-token-bridge" => Some(run_token_bridge(args)),
        "audit-token-bridge" => Some(audit_token_bridge(args)),
        "run-token-copy-bridge" => Some(run_token_copy_bridge(args)),
        "audit-token-copy-bridge" => Some(audit_token_copy_bridge(args)),
        "run-token-skip-bridge" => Some(run_token_skip_bridge(args)),
        "audit-token-skip-bridge" => Some(audit_token_skip_bridge(args)),
        "run-token-skipcopy-bridge" => Some(run_token_skipcopy_bridge(args)),
        "audit-token-skipcopy-bridge" => Some(audit_token_skipcopy_bridge(args)),
        "run-token-match-bridge" => Some(run_token_match_bridge(args)),
        "audit-token-match-bridge" => Some(audit_token_match_bridge(args)),
        "run-token-matchcopy-bridge" => Some(run_token_matchcopy_bridge(args)),
        "audit-token-matchcopy-bridge" => Some(audit_token_matchcopy_bridge(args)),
        "run-token-matchskip-bridge" => Some(run_token_matchskip_bridge(args)),
        "run-token-matchskip-token-oracle-bridge" => {
            Some(run_token_matchskip_token_oracle_bridge(args))
        }
        "audit-token-matchskip-token-oracle-bridge" => {
            Some(audit_token_matchskip_token_oracle_bridge(args))
        }
        "run-token-c3c7" => Some(run_token_c3c7(args)),
        "audit-token-c3c7" => Some(audit_token_c3c7(args)),
        "run-token-conker3" => Some(run_token_conker3(args)),
        "audit-token-conker3" => Some(audit_token_conker3(args)),
        "analyze-token-matchskip-oracle-entropy" => {
            Some(analyze_token_matchskip_oracle_entropy(args))
        }
        "audit-token-matchskip-bridge" => Some(audit_token_matchskip_bridge(args)),
        "run-token-matchskip-bundle-json" => Some(run_token_matchskip_bundle_json(args)),
        "run-token-matchskip-manifest-json" => Some(run_token_matchskip_manifest_json(args)),
        "run-token-experiment-matrix" => Some(run_token_experiment_matrix(args)),
        "run-token-matchskipcopy-bridge" => Some(run_token_matchskipcopy_bridge(args)),
        "audit-token-matchskipcopy-bridge" => Some(audit_token_matchskipcopy_bridge(args)),
        "run-token-decay-bridge" => Some(run_token_decay_bridge(args)),
        "audit-token-decay-bridge" => Some(audit_token_decay_bridge(args)),
        "run-token-word-bridge" => Some(run_token_word_bridge(args)),
        "audit-token-word-bridge" => Some(audit_token_word_bridge(args)),
        "run-token-column-bridge" => Some(run_token_column_bridge(args)),
        "audit-token-column-bridge" => Some(audit_token_column_bridge(args)),
        _ => None,
    };
    result
}

fn run_token_bridge(args: &mut dyn Iterator<Item = String>) -> Result<(), String> {
    let data_root = args
        .next()
        .ok_or_else(|| "run-token-bridge requires a data root".to_string())?;
    let train_tokens = parse_usize_flag(args.next(), "train_tokens", 1_000_000)?;
    let trigram_buckets = parse_usize_flag(args.next(), "trigram_buckets", 2_048)?;
    let val_tokens = parse_usize_flag(args.next(), "val_tokens", 32_768)?;
    let train_stride = parse_usize_flag(args.next(), "train_stride", 1)?;
    let candidate_k = parse_usize_flag(args.next(), "candidate_k", 4)?;
    if args.next().is_some() {
        return Err(
            "run-token-bridge takes <data-root> [train_tokens] [trigram_buckets] [val_tokens] [train_stride] [candidate_k]"
                .to_string(),
        );
    }
    let report = run_token_bridge_from_data_root(
        Path::new(&data_root),
        train_tokens,
        trigram_buckets,
        val_tokens,
        4.0,
        2.0,
        train_stride,
        candidate_k,
    )?;
    print!("{}", render_token_bridge_report(&report));
    Ok(())
}

fn audit_token_bridge(args: &mut dyn Iterator<Item = String>) -> Result<(), String> {
    let data_root = args
        .next()
        .ok_or_else(|| "audit-token-bridge requires a data root".to_string())?;
    let train_tokens = parse_usize_flag(args.next(), "train_tokens", 1_000_000)?;
    let trigram_buckets = parse_usize_flag(args.next(), "trigram_buckets", 2_048)?;
    let val_tokens = parse_usize_flag(args.next(), "val_tokens", 32_768)?;
    let train_stride = parse_usize_flag(args.next(), "train_stride", 1)?;
    let candidate_k = parse_usize_flag(args.next(), "candidate_k", 4)?;
    let chunk_size = parse_usize_flag(args.next(), "chunk_size", 64)?;
    let max_chunks = parse_usize_flag(args.next(), "max_chunks", 8)?;
    if args.next().is_some() {
        return Err(
            "audit-token-bridge takes <data-root> [train_tokens] [trigram_buckets] [val_tokens] [train_stride] [candidate_k] [chunk_size] [max_chunks]"
                .to_string(),
        );
    }
    let trained = train_token_bridge_from_data_root(
        Path::new(&data_root),
        train_tokens,
        trigram_buckets,
        val_tokens,
        4.0,
        2.0,
        train_stride,
        candidate_k,
    )?;
    print!("{}", render_token_bridge_report(trained.report()));
    let report = audit_parameter_golf(
        trained.runner(),
        trained.eval_tokens(),
        chunk_size,
        max_chunks,
    )?;
    println!("token_bridge_audit:");
    print!(
        "{}",
        indent_block(&report.render(trained.runner().name()), "  ")
    );
    Ok(())
}

fn run_token_copy_bridge(args: &mut dyn Iterator<Item = String>) -> Result<(), String> {
    let data_root = args
        .next()
        .ok_or_else(|| "run-token-copy-bridge requires a data root".to_string())?;
    let train_tokens = parse_usize_flag(args.next(), "train_tokens", 1_000_000)?;
    let trigram_buckets = parse_usize_flag(args.next(), "trigram_buckets", 2_048)?;
    let val_tokens = parse_usize_flag(args.next(), "val_tokens", 32_768)?;
    let copy_window = parse_usize_flag(args.next(), "copy_window", 256)?;
    let candidate_k = parse_usize_flag(args.next(), "candidate_k", 4)?;
    let train_stride = parse_usize_flag(args.next(), "train_stride", 1)?;
    let copy_decay_bp = parse_usize_flag(args.next(), "copy_decay_bp", 980)?;
    if args.next().is_some() {
        return Err(
            "run-token-copy-bridge takes <data-root> [train_tokens] [trigram_buckets] [val_tokens] [copy_window] [candidate_k] [train_stride] [copy_decay_bp]"
                .to_string(),
        );
    }
    let report = run_token_copy_bridge_from_data_root(
        Path::new(&data_root),
        train_tokens,
        trigram_buckets,
        val_tokens,
        copy_window,
        candidate_k,
        train_stride,
        4.0,
        2.0,
        (copy_decay_bp as f64) / 1000.0,
    )?;
    print!("{}", render_token_copy_bridge_report(&report));
    Ok(())
}

fn audit_token_copy_bridge(args: &mut dyn Iterator<Item = String>) -> Result<(), String> {
    let data_root = args
        .next()
        .ok_or_else(|| "audit-token-copy-bridge requires a data root".to_string())?;
    let train_tokens = parse_usize_flag(args.next(), "train_tokens", 1_000_000)?;
    let trigram_buckets = parse_usize_flag(args.next(), "trigram_buckets", 2_048)?;
    let val_tokens = parse_usize_flag(args.next(), "val_tokens", 32_768)?;
    let copy_window = parse_usize_flag(args.next(), "copy_window", 256)?;
    let candidate_k = parse_usize_flag(args.next(), "candidate_k", 4)?;
    let train_stride = parse_usize_flag(args.next(), "train_stride", 1)?;
    let copy_decay_bp = parse_usize_flag(args.next(), "copy_decay_bp", 980)?;
    let chunk_size = parse_usize_flag(args.next(), "chunk_size", 64)?;
    let max_chunks = parse_usize_flag(args.next(), "max_chunks", 8)?;
    if args.next().is_some() {
        return Err(
            "audit-token-copy-bridge takes <data-root> [train_tokens] [trigram_buckets] [val_tokens] [copy_window] [candidate_k] [train_stride] [copy_decay_bp] [chunk_size] [max_chunks]"
                .to_string(),
        );
    }
    let trained = train_token_copy_bridge_from_data_root(
        Path::new(&data_root),
        train_tokens,
        trigram_buckets,
        val_tokens,
        copy_window,
        candidate_k,
        train_stride,
        4.0,
        2.0,
        (copy_decay_bp as f64) / 1000.0,
    )?;
    print!("{}", render_token_copy_bridge_report(trained.report()));
    let report = audit_parameter_golf(
        trained.runner(),
        trained.eval_tokens(),
        chunk_size,
        max_chunks,
    )?;
    println!("token_copy_bridge_audit:");
    print!(
        "{}",
        indent_block(&report.render(trained.runner().name()), "  ")
    );
    Ok(())
}

fn run_token_skip_bridge(args: &mut dyn Iterator<Item = String>) -> Result<(), String> {
    let data_root = args
        .next()
        .ok_or_else(|| "run-token-skip-bridge requires a data root".to_string())?;
    let train_tokens = parse_usize_flag(args.next(), "train_tokens", 1_000_000)?;
    let trigram_buckets = parse_usize_flag(args.next(), "trigram_buckets", 2_048)?;
    let skip_buckets = parse_usize_flag(args.next(), "skip_buckets", 2_048)?;
    let val_tokens = parse_usize_flag(args.next(), "val_tokens", 32_768)?;
    let train_stride = parse_usize_flag(args.next(), "train_stride", 1)?;
    let candidate_k = parse_usize_flag(args.next(), "candidate_k", 4)?;
    if args.next().is_some() {
        return Err(
            "run-token-skip-bridge takes <data-root> [train_tokens] [trigram_buckets] [skip_buckets] [val_tokens] [train_stride] [candidate_k]"
                .to_string(),
        );
    }
    let report = run_token_skip_bridge_from_data_root(
        Path::new(&data_root),
        train_tokens,
        trigram_buckets,
        skip_buckets,
        val_tokens,
        4.0,
        2.0,
        2.0,
        train_stride,
        candidate_k,
    )?;
    print!("{}", render_token_skip_bridge_report(&report));
    Ok(())
}

fn audit_token_skip_bridge(args: &mut dyn Iterator<Item = String>) -> Result<(), String> {
    let data_root = args
        .next()
        .ok_or_else(|| "audit-token-skip-bridge requires a data root".to_string())?;
    let train_tokens = parse_usize_flag(args.next(), "train_tokens", 1_000_000)?;
    let trigram_buckets = parse_usize_flag(args.next(), "trigram_buckets", 2_048)?;
    let skip_buckets = parse_usize_flag(args.next(), "skip_buckets", 2_048)?;
    let val_tokens = parse_usize_flag(args.next(), "val_tokens", 32_768)?;
    let train_stride = parse_usize_flag(args.next(), "train_stride", 1)?;
    let candidate_k = parse_usize_flag(args.next(), "candidate_k", 4)?;
    let chunk_size = parse_usize_flag(args.next(), "chunk_size", 64)?;
    let max_chunks = parse_usize_flag(args.next(), "max_chunks", 8)?;
    if args.next().is_some() {
        return Err(
            "audit-token-skip-bridge takes <data-root> [train_tokens] [trigram_buckets] [skip_buckets] [val_tokens] [train_stride] [candidate_k] [chunk_size] [max_chunks]"
                .to_string(),
        );
    }
    let trained = train_token_skip_bridge_from_data_root(
        Path::new(&data_root),
        train_tokens,
        trigram_buckets,
        skip_buckets,
        val_tokens,
        4.0,
        2.0,
        2.0,
        train_stride,
        candidate_k,
    )?;
    print!("{}", render_token_skip_bridge_report(trained.report()));
    let report = audit_parameter_golf(
        trained.runner(),
        trained.eval_tokens(),
        chunk_size,
        max_chunks,
    )?;
    println!("token_skip_bridge_audit:");
    print!(
        "{}",
        indent_block(&report.render(trained.runner().name()), "  ")
    );
    Ok(())
}

fn run_token_skipcopy_bridge(args: &mut dyn Iterator<Item = String>) -> Result<(), String> {
    let data_root = args
        .next()
        .ok_or_else(|| "run-token-skipcopy-bridge requires a data root".to_string())?;
    let train_tokens = parse_usize_flag(args.next(), "train_tokens", 1_000_000)?;
    let trigram_buckets = parse_usize_flag(args.next(), "trigram_buckets", 2_048)?;
    let skip_buckets = parse_usize_flag(args.next(), "skip_buckets", 2_048)?;
    let val_tokens = parse_usize_flag(args.next(), "val_tokens", 32_768)?;
    let copy_window = parse_usize_flag(args.next(), "copy_window", 256)?;
    let candidate_k = parse_usize_flag(args.next(), "candidate_k", 4)?;
    let train_stride = parse_usize_flag(args.next(), "train_stride", 1)?;
    let copy_decay_bp = parse_usize_flag(args.next(), "copy_decay_bp", 980)?;
    if args.next().is_some() {
        return Err(
            "run-token-skipcopy-bridge takes <data-root> [train_tokens] [trigram_buckets] [skip_buckets] [val_tokens] [copy_window] [candidate_k] [train_stride] [copy_decay_bp]"
                .to_string(),
        );
    }
    let report = run_token_skipcopy_bridge_from_data_root(
        Path::new(&data_root),
        train_tokens,
        trigram_buckets,
        skip_buckets,
        val_tokens,
        copy_window,
        candidate_k,
        train_stride,
        4.0,
        2.0,
        2.0,
        (copy_decay_bp as f64) / 1000.0,
    )?;
    print!("{}", render_token_skipcopy_bridge_report(&report));
    Ok(())
}

fn audit_token_skipcopy_bridge(args: &mut dyn Iterator<Item = String>) -> Result<(), String> {
    let data_root = args
        .next()
        .ok_or_else(|| "audit-token-skipcopy-bridge requires a data root".to_string())?;
    let train_tokens = parse_usize_flag(args.next(), "train_tokens", 1_000_000)?;
    let trigram_buckets = parse_usize_flag(args.next(), "trigram_buckets", 2_048)?;
    let skip_buckets = parse_usize_flag(args.next(), "skip_buckets", 2_048)?;
    let val_tokens = parse_usize_flag(args.next(), "val_tokens", 32_768)?;
    let copy_window = parse_usize_flag(args.next(), "copy_window", 256)?;
    let candidate_k = parse_usize_flag(args.next(), "candidate_k", 4)?;
    let train_stride = parse_usize_flag(args.next(), "train_stride", 1)?;
    let copy_decay_bp = parse_usize_flag(args.next(), "copy_decay_bp", 980)?;
    let chunk_size = parse_usize_flag(args.next(), "chunk_size", 64)?;
    let max_chunks = parse_usize_flag(args.next(), "max_chunks", 8)?;
    if args.next().is_some() {
        return Err(
            "audit-token-skipcopy-bridge takes <data-root> [train_tokens] [trigram_buckets] [skip_buckets] [val_tokens] [copy_window] [candidate_k] [train_stride] [copy_decay_bp] [chunk_size] [max_chunks]"
                .to_string(),
        );
    }
    let trained = train_token_skipcopy_bridge_from_data_root(
        Path::new(&data_root),
        train_tokens,
        trigram_buckets,
        skip_buckets,
        val_tokens,
        copy_window,
        candidate_k,
        train_stride,
        4.0,
        2.0,
        2.0,
        (copy_decay_bp as f64) / 1000.0,
    )?;
    print!("{}", render_token_skipcopy_bridge_report(trained.report()));
    let report = audit_parameter_golf(
        trained.runner(),
        trained.eval_tokens(),
        chunk_size,
        max_chunks,
    )?;
    println!("token_skipcopy_bridge_audit:");
    print!(
        "{}",
        indent_block(&report.render(trained.runner().name()), "  ")
    );
    Ok(())
}

fn run_token_match_bridge(args: &mut dyn Iterator<Item = String>) -> Result<(), String> {
    let data_root = args
        .next()
        .ok_or_else(|| "run-token-match-bridge requires a data root".to_string())?;
    let train_tokens = parse_usize_flag(args.next(), "train_tokens", 1_000_000)?;
    let trigram_buckets = parse_usize_flag(args.next(), "trigram_buckets", 2_048)?;
    let val_tokens = parse_usize_flag(args.next(), "val_tokens", 32_768)?;
    let match_depth = parse_usize_flag(args.next(), "match_depth", 8)?;
    let candidate_k = parse_usize_flag(args.next(), "candidate_k", 4)?;
    let train_stride = parse_usize_flag(args.next(), "train_stride", 1)?;
    if args.next().is_some() {
        return Err(
            "run-token-match-bridge takes <data-root> [train_tokens] [trigram_buckets] [val_tokens] [match_depth] [candidate_k] [train_stride]"
                .to_string(),
        );
    }
    let report = run_token_match_bridge_from_data_root(
        Path::new(&data_root),
        train_tokens,
        trigram_buckets,
        val_tokens,
        match_depth,
        candidate_k,
        train_stride,
        4.0,
        2.0,
    )?;
    print!("{}", render_token_match_bridge_report(&report));
    Ok(())
}

fn audit_token_match_bridge(args: &mut dyn Iterator<Item = String>) -> Result<(), String> {
    let data_root = args
        .next()
        .ok_or_else(|| "audit-token-match-bridge requires a data root".to_string())?;
    let train_tokens = parse_usize_flag(args.next(), "train_tokens", 1_000_000)?;
    let trigram_buckets = parse_usize_flag(args.next(), "trigram_buckets", 2_048)?;
    let val_tokens = parse_usize_flag(args.next(), "val_tokens", 32_768)?;
    let match_depth = parse_usize_flag(args.next(), "match_depth", 8)?;
    let candidate_k = parse_usize_flag(args.next(), "candidate_k", 4)?;
    let train_stride = parse_usize_flag(args.next(), "train_stride", 1)?;
    let chunk_size = parse_usize_flag(args.next(), "chunk_size", 64)?;
    let max_chunks = parse_usize_flag(args.next(), "max_chunks", 8)?;
    if args.next().is_some() {
        return Err(
            "audit-token-match-bridge takes <data-root> [train_tokens] [trigram_buckets] [val_tokens] [match_depth] [candidate_k] [train_stride] [chunk_size] [max_chunks]"
                .to_string(),
        );
    }
    let trained = train_token_match_bridge_from_data_root(
        Path::new(&data_root),
        train_tokens,
        trigram_buckets,
        val_tokens,
        match_depth,
        candidate_k,
        train_stride,
        4.0,
        2.0,
    )?;
    print!("{}", render_token_match_bridge_report(trained.report()));
    let report = audit_parameter_golf(
        trained.runner(),
        trained.eval_tokens(),
        chunk_size,
        max_chunks,
    )?;
    println!("token_match_bridge_audit:");
    print!(
        "{}",
        indent_block(&report.render(trained.runner().name()), "  ")
    );
    Ok(())
}

fn run_token_matchcopy_bridge(args: &mut dyn Iterator<Item = String>) -> Result<(), String> {
    let data_root = args
        .next()
        .ok_or_else(|| "run-token-matchcopy-bridge requires a data root".to_string())?;
    let train_tokens = parse_usize_flag(args.next(), "train_tokens", 1_000_000)?;
    let trigram_buckets = parse_usize_flag(args.next(), "trigram_buckets", 2_048)?;
    let val_tokens = parse_usize_flag(args.next(), "val_tokens", 32_768)?;
    let match_depth = parse_usize_flag(args.next(), "match_depth", 8)?;
    let copy_window = parse_usize_flag(args.next(), "copy_window", 256)?;
    let candidate_k = parse_usize_flag(args.next(), "candidate_k", 4)?;
    let train_stride = parse_usize_flag(args.next(), "train_stride", 1)?;
    let copy_decay_bp = parse_usize_flag(args.next(), "copy_decay_bp", 980)?;
    if args.next().is_some() {
        return Err(
            "run-token-matchcopy-bridge takes <data-root> [train_tokens] [trigram_buckets] [val_tokens] [match_depth] [copy_window] [candidate_k] [train_stride] [copy_decay_bp]"
                .to_string(),
        );
    }
    let report = run_token_matchcopy_bridge_from_data_root(
        Path::new(&data_root),
        train_tokens,
        trigram_buckets,
        val_tokens,
        match_depth,
        copy_window,
        candidate_k,
        train_stride,
        4.0,
        2.0,
        (copy_decay_bp as f64) / 1000.0,
    )?;
    print!("{}", render_token_matchcopy_bridge_report(&report));
    Ok(())
}

fn audit_token_matchcopy_bridge(args: &mut dyn Iterator<Item = String>) -> Result<(), String> {
    let data_root = args
        .next()
        .ok_or_else(|| "audit-token-matchcopy-bridge requires a data root".to_string())?;
    let train_tokens = parse_usize_flag(args.next(), "train_tokens", 1_000_000)?;
    let trigram_buckets = parse_usize_flag(args.next(), "trigram_buckets", 2_048)?;
    let val_tokens = parse_usize_flag(args.next(), "val_tokens", 32_768)?;
    let match_depth = parse_usize_flag(args.next(), "match_depth", 8)?;
    let copy_window = parse_usize_flag(args.next(), "copy_window", 256)?;
    let candidate_k = parse_usize_flag(args.next(), "candidate_k", 4)?;
    let train_stride = parse_usize_flag(args.next(), "train_stride", 1)?;
    let copy_decay_bp = parse_usize_flag(args.next(), "copy_decay_bp", 980)?;
    let chunk_size = parse_usize_flag(args.next(), "chunk_size", 64)?;
    let max_chunks = parse_usize_flag(args.next(), "max_chunks", 8)?;
    if args.next().is_some() {
        return Err(
            "audit-token-matchcopy-bridge takes <data-root> [train_tokens] [trigram_buckets] [val_tokens] [match_depth] [copy_window] [candidate_k] [train_stride] [copy_decay_bp] [chunk_size] [max_chunks]"
                .to_string(),
        );
    }
    let trained = train_token_matchcopy_bridge_from_data_root(
        Path::new(&data_root),
        train_tokens,
        trigram_buckets,
        val_tokens,
        match_depth,
        copy_window,
        candidate_k,
        train_stride,
        4.0,
        2.0,
        (copy_decay_bp as f64) / 1000.0,
    )?;
    print!("{}", render_token_matchcopy_bridge_report(trained.report()));
    let report = audit_parameter_golf(
        trained.runner(),
        trained.eval_tokens(),
        chunk_size,
        max_chunks,
    )?;
    println!("token_matchcopy_bridge_audit:");
    print!(
        "{}",
        indent_block(&report.render(trained.runner().name()), "  ")
    );
    Ok(())
}

fn run_token_matchskip_bridge(args: &mut dyn Iterator<Item = String>) -> Result<(), String> {
    let data_root = args
        .next()
        .ok_or_else(|| "run-token-matchskip-bridge requires a data root".to_string())?;
    let train_tokens = parse_usize_flag(args.next(), "train_tokens", 1_000_000)?;
    let trigram_buckets = parse_usize_flag(args.next(), "trigram_buckets", 2_048)?;
    let skip_buckets = parse_usize_flag(args.next(), "skip_buckets", 2_048)?;
    let val_tokens = parse_usize_flag(args.next(), "val_tokens", 32_768)?;
    let match_depth = parse_usize_flag(args.next(), "match_depth", 8)?;
    let candidate_k = parse_usize_flag(args.next(), "candidate_k", 4)?;
    let train_stride = parse_usize_flag(args.next(), "train_stride", 1)?;
    if args.next().is_some() {
        return Err(
            "run-token-matchskip-bridge takes <data-root> [train_tokens] [trigram_buckets] [skip_buckets] [val_tokens] [match_depth] [candidate_k] [train_stride]"
                .to_string(),
        );
    }
    let report = run_token_matchskip_bridge_from_data_root(
        Path::new(&data_root),
        train_tokens,
        trigram_buckets,
        skip_buckets,
        val_tokens,
        match_depth,
        candidate_k,
        train_stride,
        4.0,
        2.0,
        2.0,
    )?;
    print!("{}", render_token_matchskip_bridge_report(&report));
    Ok(())
}

fn run_token_matchskip_token_oracle_bridge(
    args: &mut dyn Iterator<Item = String>,
) -> Result<(), String> {
    let data_root = args.next().ok_or_else(|| {
        "run-token-matchskip-token-oracle-bridge requires a data root".to_string()
    })?;
    let train_tokens = parse_usize_flag(args.next(), "train_tokens", 1_000_000)?;
    let trigram_buckets = parse_usize_flag(args.next(), "trigram_buckets", 16_384)?;
    let skip_buckets = parse_usize_flag(args.next(), "skip_buckets", 16_384)?;
    let val_tokens = parse_usize_flag(args.next(), "val_tokens", 16_384)?;
    let match_depth = parse_usize_flag(args.next(), "match_depth", 12)?;
    let candidate_k = parse_usize_flag(args.next(), "candidate_k", 16)?;
    let train_stride = parse_usize_flag(args.next(), "train_stride", 4)?;
    let oracle_radius = parse_usize_flag(args.next(), "oracle_radius", 2)?;
    let oracle_stride = parse_usize_flag(args.next(), "oracle_stride", 1)?;
    let gate_target_kind = parse_oracle_target_kind_flag(
        args.next(),
        "gate_target_kind",
        OracleTargetKind::CandidateSetLeq8,
    )?;
    let trust_target_kind = parse_oracle_target_kind_flag(
        args.next(),
        "trust_target_kind",
        OracleTargetKind::MemoryTrust,
    )?;
    if args.next().is_some() {
        return Err(
            "run-token-matchskip-token-oracle-bridge takes <data-root> [train_tokens] [trigram_buckets] [skip_buckets] [val_tokens] [match_depth] [candidate_k] [train_stride] [oracle_radius] [oracle_stride] [gate_target_kind] [trust_target_kind]"
                .to_string(),
        );
    }
    let resolved = resolve_data_root(Some(&data_root))?;
    let offline_targets = build_token_matchskip_offline_targets(
        Path::new(&resolved.selected_path),
        val_tokens,
        oracle_radius,
        oracle_stride,
        gate_target_kind,
        trust_target_kind,
    )?;
    let trained = train_token_matchskip_bridge_from_data_root_with_offline_targets(
        Path::new(&resolved.selected_path),
        train_tokens,
        trigram_buckets,
        skip_buckets,
        val_tokens,
        match_depth,
        candidate_k,
        train_stride,
        4.0,
        2.0,
        2.0,
        Some(&offline_targets),
    )?;
    println!("data_root_spec: {data_root}");
    println!("data_root_selected_path: {}", resolved.selected_path);
    println!("oracle_radius: {oracle_radius}");
    println!("oracle_stride: {oracle_stride}");
    println!(
        "oracle_gate_target_kind: {}",
        oracle_target_kind_name(gate_target_kind)
    );
    println!(
        "oracle_trust_target_kind: {}",
        oracle_target_kind_name(trust_target_kind)
    );
    print!("{}", render_token_matchskip_bridge_report(trained.report()));
    Ok(())
}

fn audit_token_matchskip_token_oracle_bridge(
    args: &mut dyn Iterator<Item = String>,
) -> Result<(), String> {
    let data_root = args.next().ok_or_else(|| {
        "audit-token-matchskip-token-oracle-bridge requires a data root".to_string()
    })?;
    let train_tokens = parse_usize_flag(args.next(), "train_tokens", 1_000_000)?;
    let trigram_buckets = parse_usize_flag(args.next(), "trigram_buckets", 16_384)?;
    let skip_buckets = parse_usize_flag(args.next(), "skip_buckets", 16_384)?;
    let val_tokens = parse_usize_flag(args.next(), "val_tokens", 16_384)?;
    let match_depth = parse_usize_flag(args.next(), "match_depth", 12)?;
    let candidate_k = parse_usize_flag(args.next(), "candidate_k", 16)?;
    let train_stride = parse_usize_flag(args.next(), "train_stride", 4)?;
    let oracle_radius = parse_usize_flag(args.next(), "oracle_radius", 2)?;
    let oracle_stride = parse_usize_flag(args.next(), "oracle_stride", 1)?;
    let gate_target_kind = parse_oracle_target_kind_flag(
        args.next(),
        "gate_target_kind",
        OracleTargetKind::CandidateSetLeq8,
    )?;
    let trust_target_kind = parse_oracle_target_kind_flag(
        args.next(),
        "trust_target_kind",
        OracleTargetKind::MemoryTrust,
    )?;
    let chunk_size = parse_usize_flag(args.next(), "chunk_size", 64)?;
    let max_chunks = parse_usize_flag(args.next(), "max_chunks", 8)?;
    if args.next().is_some() {
        return Err(
            "audit-token-matchskip-token-oracle-bridge takes <data-root> [train_tokens] [trigram_buckets] [skip_buckets] [val_tokens] [match_depth] [candidate_k] [train_stride] [oracle_radius] [oracle_stride] [gate_target_kind] [trust_target_kind] [chunk_size] [max_chunks]"
                .to_string(),
        );
    }
    let resolved = resolve_data_root(Some(&data_root))?;
    let offline_targets = build_token_matchskip_offline_targets(
        Path::new(&resolved.selected_path),
        val_tokens,
        oracle_radius,
        oracle_stride,
        gate_target_kind,
        trust_target_kind,
    )?;
    let trained = train_token_matchskip_bridge_from_data_root_with_offline_targets(
        Path::new(&resolved.selected_path),
        train_tokens,
        trigram_buckets,
        skip_buckets,
        val_tokens,
        match_depth,
        candidate_k,
        train_stride,
        4.0,
        2.0,
        2.0,
        Some(&offline_targets),
    )?;
    println!("data_root_spec: {data_root}");
    println!("data_root_selected_path: {}", resolved.selected_path);
    println!("oracle_radius: {oracle_radius}");
    println!("oracle_stride: {oracle_stride}");
    println!(
        "oracle_gate_target_kind: {}",
        oracle_target_kind_name(gate_target_kind)
    );
    println!(
        "oracle_trust_target_kind: {}",
        oracle_target_kind_name(trust_target_kind)
    );
    print!("{}", render_token_matchskip_bridge_report(trained.report()));
    let report = audit_parameter_golf(
        trained.runner(),
        trained.eval_tokens(),
        chunk_size,
        max_chunks,
    )?;
    println!("token_matchskip_token_oracle_bridge_audit:");
    print!(
        "{}",
        indent_block(&report.render(trained.runner().name()), "  ")
    );
    Ok(())
}

fn run_token_c3c7(args: &mut dyn Iterator<Item = String>) -> Result<(), String> {
    let data_root = args
        .next()
        .ok_or_else(|| "run-token-c3c7 requires a data root".to_string())?;
    let train_tokens = parse_usize_flag(args.next(), "train_tokens", 1_000_000)?;
    let trigram_buckets = parse_usize_flag(args.next(), "trigram_buckets", 16_384)?;
    let val_tokens = parse_usize_flag(args.next(), "val_tokens", 16_384)?;
    let train_stride = parse_usize_flag(args.next(), "train_stride", 4)?;
    let train_steps = parse_usize_flag(args.next(), "train_steps", 2_200)?;
    let teacher_start_step = parse_usize_flag(args.next(), "teacher_start_step", 1_100)?;
    let oracle_radius = parse_usize_flag(args.next(), "oracle_radius", 2)?;
    let oracle_stride = parse_usize_flag(args.next(), "oracle_stride", 4)?;
    if args.next().is_some() {
        return Err(
            "run-token-c3c7 takes <data-root> [train_tokens] [trigram_buckets] [val_tokens] [train_stride] [train_steps] [teacher_start_step] [oracle_radius] [oracle_stride]"
                .to_string(),
        );
    }
    let report = run_token_c3c7_from_data_root(
        Path::new(&data_root),
        train_tokens,
        trigram_buckets,
        val_tokens,
        train_stride,
        train_steps,
        teacher_start_step,
        oracle_radius,
        oracle_stride,
        0.95,
        0.04,
        0.01,
    )?;
    print!("{}", render_token_c3c7_report(&report));
    Ok(())
}

fn audit_token_c3c7(args: &mut dyn Iterator<Item = String>) -> Result<(), String> {
    let data_root = args
        .next()
        .ok_or_else(|| "audit-token-c3c7 requires a data root".to_string())?;
    let train_tokens = parse_usize_flag(args.next(), "train_tokens", 1_000_000)?;
    let trigram_buckets = parse_usize_flag(args.next(), "trigram_buckets", 16_384)?;
    let val_tokens = parse_usize_flag(args.next(), "val_tokens", 16_384)?;
    let train_stride = parse_usize_flag(args.next(), "train_stride", 4)?;
    let train_steps = parse_usize_flag(args.next(), "train_steps", 2_200)?;
    let teacher_start_step = parse_usize_flag(args.next(), "teacher_start_step", 1_100)?;
    let oracle_radius = parse_usize_flag(args.next(), "oracle_radius", 2)?;
    let oracle_stride = parse_usize_flag(args.next(), "oracle_stride", 4)?;
    let chunk_size = parse_usize_flag(args.next(), "chunk_size", 64)?;
    let max_chunks = parse_usize_flag(args.next(), "max_chunks", 8)?;
    if args.next().is_some() {
        return Err(
            "audit-token-c3c7 takes <data-root> [train_tokens] [trigram_buckets] [val_tokens] [train_stride] [train_steps] [teacher_start_step] [oracle_radius] [oracle_stride] [chunk_size] [max_chunks]"
                .to_string(),
        );
    }
    let trained = train_token_c3c7_from_data_root(
        Path::new(&data_root),
        train_tokens,
        trigram_buckets,
        val_tokens,
        train_stride,
        train_steps,
        teacher_start_step,
        oracle_radius,
        oracle_stride,
        0.95,
        0.04,
        0.01,
    )?;
    print!("{}", render_token_c3c7_report(trained.report()));
    let report = audit_parameter_golf(
        trained.runner(),
        trained.eval_tokens(),
        chunk_size,
        max_chunks,
    )?;
    println!("token_c3c7_audit:");
    print!(
        "{}",
        indent_block(&report.render(trained.runner().name()), "  ")
    );
    Ok(())
}

fn run_token_conker3(args: &mut dyn Iterator<Item = String>) -> Result<(), String> {
    let data_root = args
        .next()
        .ok_or_else(|| "run-token-conker3 requires a data root".to_string())?;
    let train_tokens = parse_usize_flag(args.next(), "train_tokens", 100_000)?;
    let val_tokens = parse_usize_flag(args.next(), "val_tokens", 4_096)?;
    let scale = parse_f64_flag(args.next(), "scale", 0.5)?;
    let train_stride = parse_usize_flag(args.next(), "train_stride", 2)?;
    let epochs = parse_usize_flag(args.next(), "epochs", 1)?;
    let negatives = parse_usize_flag(args.next(), "negatives", 32)?;
    let learning_rate = parse_f64_flag(args.next(), "learning_rate", 0.02)?;
    if args.next().is_some() {
        return Err(
            "run-token-conker3 takes <data-root> [train_tokens] [val_tokens] [scale] [train_stride] [epochs] [negatives] [learning_rate]"
                .to_string(),
        );
    }
    let report = run_token_conker3_from_data_root(
        Path::new(&data_root),
        train_tokens,
        val_tokens,
        scale,
        train_stride,
        epochs,
        negatives,
        learning_rate,
    )?;
    print!("{}", render_token_conker3_report(&report));
    Ok(())
}

fn audit_token_conker3(args: &mut dyn Iterator<Item = String>) -> Result<(), String> {
    let data_root = args
        .next()
        .ok_or_else(|| "audit-token-conker3 requires a data root".to_string())?;
    let train_tokens = parse_usize_flag(args.next(), "train_tokens", 100_000)?;
    let val_tokens = parse_usize_flag(args.next(), "val_tokens", 4_096)?;
    let scale = parse_f64_flag(args.next(), "scale", 0.5)?;
    let train_stride = parse_usize_flag(args.next(), "train_stride", 2)?;
    let epochs = parse_usize_flag(args.next(), "epochs", 1)?;
    let negatives = parse_usize_flag(args.next(), "negatives", 32)?;
    let learning_rate = parse_f64_flag(args.next(), "learning_rate", 0.02)?;
    let chunk_size = parse_usize_flag(args.next(), "chunk_size", 64)?;
    let max_chunks = parse_usize_flag(args.next(), "max_chunks", 8)?;
    if args.next().is_some() {
        return Err(
            "audit-token-conker3 takes <data-root> [train_tokens] [val_tokens] [scale] [train_stride] [epochs] [negatives] [learning_rate] [chunk_size] [max_chunks]"
                .to_string(),
        );
    }
    let trained = train_token_conker3_from_data_root(
        Path::new(&data_root),
        train_tokens,
        val_tokens,
        scale,
        train_stride,
        epochs,
        negatives,
        learning_rate,
    )?;
    print!("{}", render_token_conker3_report(trained.report()));
    let report = audit_parameter_golf(
        trained.runner(),
        trained.eval_tokens(),
        chunk_size,
        max_chunks,
    )?;
    println!("token_conker3_audit:");
    print!(
        "{}",
        indent_block(&report.render(trained.runner().name()), "  ")
    );
    Ok(())
}

fn analyze_token_matchskip_oracle_entropy(
    args: &mut dyn Iterator<Item = String>,
) -> Result<(), String> {
    let data_root = args
        .next()
        .ok_or_else(|| "analyze-token-matchskip-oracle-entropy requires a data root".to_string())?;
    let train_tokens = parse_usize_flag(args.next(), "train_tokens", 1_000_000)?;
    let trigram_buckets = parse_usize_flag(args.next(), "trigram_buckets", 16_384)?;
    let skip_buckets = parse_usize_flag(args.next(), "skip_buckets", 16_384)?;
    let val_tokens = parse_usize_flag(args.next(), "val_tokens", 16_384)?;
    let match_depth = parse_usize_flag(args.next(), "match_depth", 12)?;
    let candidate_k = parse_usize_flag(args.next(), "candidate_k", 16)?;
    let train_stride = parse_usize_flag(args.next(), "train_stride", 4)?;
    let oracle_radius = parse_usize_flag(args.next(), "oracle_radius", 2)?;
    let oracle_stride = parse_usize_flag(args.next(), "oracle_stride", 1)?;
    if args.next().is_some() {
        return Err(
            "analyze-token-matchskip-oracle-entropy takes <data-root> [train_tokens] [trigram_buckets] [skip_buckets] [val_tokens] [match_depth] [candidate_k] [train_stride] [oracle_radius] [oracle_stride]"
                .to_string(),
        );
    }
    let resolved = resolve_data_root(Some(&data_root))?;
    let trained = train_token_matchskip_bridge_from_data_root(
        Path::new(&resolved.selected_path),
        train_tokens,
        trigram_buckets,
        skip_buckets,
        val_tokens,
        match_depth,
        candidate_k,
        train_stride,
        4.0,
        2.0,
        2.0,
    )?;
    let full_val_tokens = take_val_tokens(Path::new(&resolved.selected_path), val_tokens)?;
    let split = (full_val_tokens.len() / 2).max(1);
    let eval_tokens = full_val_tokens[split..].to_vec();
    let oracle_dataset = token_oracle_target_dataset_from_tokens(
        "eval_split",
        &eval_tokens,
        oracle_radius,
        oracle_stride,
    )?;
    let sample_positions: Vec<usize> = (0..eval_tokens.len()).collect();
    let outputs = trained
        .runner()
        .score_chunk(&eval_tokens, &sample_positions)?;

    let mut total_count = 0usize;
    let mut total_entropy = 0.0;
    let mut total_top1 = 0.0;
    let mut leq4_count = 0usize;
    let mut leq4_entropy = 0.0;
    let mut leq4_top1 = 0.0;
    let mut leq8_count = 0usize;
    let mut leq8_entropy = 0.0;
    let mut leq8_top1 = 0.0;
    let mut gt8_count = 0usize;
    let mut gt8_entropy = 0.0;
    let mut gt8_top1 = 0.0;

    for row in &oracle_dataset.rows {
        let position = row.label.position;
        if position >= outputs.sample_predictions.len() {
            continue;
        }
        let distribution = &outputs.sample_predictions[position];
        let entropy_bits = distribution_entropy_bits(distribution);
        let top1_prob = distribution_top1_prob(distribution);
        total_count += 1;
        total_entropy += entropy_bits;
        total_top1 += top1_prob;
        if row.label.bidi_leaveout_support_size <= 4 {
            leq4_count += 1;
            leq4_entropy += entropy_bits;
            leq4_top1 += top1_prob;
        }
        if row.label.bidi_leaveout_support_size <= 8 {
            leq8_count += 1;
            leq8_entropy += entropy_bits;
            leq8_top1 += top1_prob;
        } else {
            gt8_count += 1;
            gt8_entropy += entropy_bits;
            gt8_top1 += top1_prob;
        }
    }

    let mean = |sum: f64, count: usize| {
        if count == 0 { 0.0 } else { sum / count as f64 }
    };

    println!("data_root_spec: {data_root}");
    println!("data_root_selected_path: {}", resolved.selected_path);
    println!("train_token_budget: {train_tokens}");
    println!("val_token_budget: {val_tokens}");
    println!("match_depth: {match_depth}");
    println!("candidate_k: {candidate_k}");
    println!("oracle_radius: {oracle_radius}");
    println!("oracle_stride: {oracle_stride}");
    println!(
        "selected_runtime_gate: {}",
        trained.report().selected_runtime_gate
    );
    println!(
        "selected_runtime_lambda: {:.3}",
        trained.report().selected_runtime_lambda
    );
    println!("oracle_rows: {total_count}");
    println!(
        "overall_mean_entropy_bits: {:.6}",
        mean(total_entropy, total_count)
    );
    println!(
        "overall_mean_top1_prob: {:.6}",
        mean(total_top1, total_count)
    );
    println!("leq4_count: {leq4_count}");
    println!(
        "leq4_mean_entropy_bits: {:.6}",
        mean(leq4_entropy, leq4_count)
    );
    println!("leq4_mean_top1_prob: {:.6}", mean(leq4_top1, leq4_count));
    println!("leq8_count: {leq8_count}");
    println!(
        "leq8_mean_entropy_bits: {:.6}",
        mean(leq8_entropy, leq8_count)
    );
    println!("leq8_mean_top1_prob: {:.6}", mean(leq8_top1, leq8_count));
    println!("gt8_count: {gt8_count}");
    println!("gt8_mean_entropy_bits: {:.6}", mean(gt8_entropy, gt8_count));
    println!("gt8_mean_top1_prob: {:.6}", mean(gt8_top1, gt8_count));
    Ok(())
}

fn audit_token_matchskip_bridge(args: &mut dyn Iterator<Item = String>) -> Result<(), String> {
    let data_root = args
        .next()
        .ok_or_else(|| "audit-token-matchskip-bridge requires a data root".to_string())?;
    let train_tokens = parse_usize_flag(args.next(), "train_tokens", 1_000_000)?;
    let trigram_buckets = parse_usize_flag(args.next(), "trigram_buckets", 2_048)?;
    let skip_buckets = parse_usize_flag(args.next(), "skip_buckets", 2_048)?;
    let val_tokens = parse_usize_flag(args.next(), "val_tokens", 32_768)?;
    let match_depth = parse_usize_flag(args.next(), "match_depth", 8)?;
    let candidate_k = parse_usize_flag(args.next(), "candidate_k", 4)?;
    let train_stride = parse_usize_flag(args.next(), "train_stride", 1)?;
    let chunk_size = parse_usize_flag(args.next(), "chunk_size", 64)?;
    let max_chunks = parse_usize_flag(args.next(), "max_chunks", 8)?;
    if args.next().is_some() {
        return Err(
            "audit-token-matchskip-bridge takes <data-root> [train_tokens] [trigram_buckets] [skip_buckets] [val_tokens] [match_depth] [candidate_k] [train_stride] [chunk_size] [max_chunks]"
                .to_string(),
        );
    }
    let trained = train_token_matchskip_bridge_from_data_root(
        Path::new(&data_root),
        train_tokens,
        trigram_buckets,
        skip_buckets,
        val_tokens,
        match_depth,
        candidate_k,
        train_stride,
        4.0,
        2.0,
        2.0,
    )?;
    print!("{}", render_token_matchskip_bridge_report(trained.report()));
    let report = audit_parameter_golf(
        trained.runner(),
        trained.eval_tokens(),
        chunk_size,
        max_chunks,
    )?;
    println!("token_matchskip_bridge_audit:");
    print!(
        "{}",
        indent_block(&report.render(trained.runner().name()), "  ")
    );
    Ok(())
}

fn run_token_matchskip_bundle_json(args: &mut dyn Iterator<Item = String>) -> Result<(), String> {
    let parsed = parse_matchskip_export_args(args, "run-token-matchskip-bundle-json")?;
    let export = build_matchskip_export_context(parsed, "run-token-matchskip-bundle-json")?;
    let bundle = BridgeReportBundle {
        manifest: export.manifest,
        report: &export.report,
        audit: &export.audit,
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&bundle)
            .map_err(|err| format!("serialize matchskip bundle: {err}"))?
    );
    Ok(())
}

fn run_token_matchskip_manifest_json(args: &mut dyn Iterator<Item = String>) -> Result<(), String> {
    let parsed = parse_matchskip_export_args(args, "run-token-matchskip-manifest-json")?;
    let export = build_matchskip_export_context(parsed, "run-token-matchskip-manifest-json")?;
    println!(
        "{}",
        serde_json::to_string_pretty(&export.manifest)
            .map_err(|err| format!("serialize matchskip manifest: {err}"))?
    );
    Ok(())
}

fn run_token_experiment_matrix(args: &mut dyn Iterator<Item = String>) -> Result<(), String> {
    let data_root = args
        .next()
        .ok_or_else(|| "run-token-experiment-matrix requires a data root".to_string())?;
    let train_token_budget = parse_usize_flag(args.next(), "train_token_budget", 1_000_000)?;
    let trigram_buckets = parse_usize_flag(args.next(), "trigram_buckets", 8_192)?;
    let skip_buckets = parse_usize_flag(args.next(), "skip_buckets", 8_192)?;
    let val_token_budget = parse_usize_flag(args.next(), "val_token_budget", 32_768)?;
    let match_depth = parse_usize_flag(args.next(), "match_depth", 8)?;
    let copy_window = parse_usize_flag(args.next(), "copy_window", 256)?;
    let candidate_k = parse_usize_flag(args.next(), "candidate_k", 4)?;
    let train_stride = parse_usize_flag(args.next(), "train_stride", 4)?;
    let copy_decay_bp = parse_usize_flag(args.next(), "copy_decay_bp", 980)?;
    if args.next().is_some() {
        return Err(
            "run-token-experiment-matrix takes <data-root> [train_token_budget] [trigram_buckets] [skip_buckets] [val_token_budget] [match_depth] [copy_window] [candidate_k] [train_stride] [copy_decay_bp]"
                .to_string(),
        );
    }
    let report = run_token_experiment_matrix_from_data_root(
        &data_root,
        TokenExperimentMatrixConfig {
            train_token_budget,
            trigram_buckets,
            skip_buckets,
            val_token_budget,
            match_depth,
            copy_window,
            candidate_k,
            train_stride,
            copy_decay_bp,
            ..TokenExperimentMatrixConfig::default()
        },
    )?;
    eprint!("{}", report.render_compact());
    println!(
        "{}",
        serde_json::to_string_pretty(&report)
            .map_err(|err| format!("serialize token experiment matrix: {err}"))?
    );
    Ok(())
}

fn run_token_matchskipcopy_bridge(args: &mut dyn Iterator<Item = String>) -> Result<(), String> {
    let data_root = args
        .next()
        .ok_or_else(|| "run-token-matchskipcopy-bridge requires a data root".to_string())?;
    let train_tokens = parse_usize_flag(args.next(), "train_tokens", 1_000_000)?;
    let trigram_buckets = parse_usize_flag(args.next(), "trigram_buckets", 2_048)?;
    let skip_buckets = parse_usize_flag(args.next(), "skip_buckets", 2_048)?;
    let val_tokens = parse_usize_flag(args.next(), "val_tokens", 32_768)?;
    let match_depth = parse_usize_flag(args.next(), "match_depth", 8)?;
    let copy_window = parse_usize_flag(args.next(), "copy_window", 256)?;
    let candidate_k = parse_usize_flag(args.next(), "candidate_k", 4)?;
    let train_stride = parse_usize_flag(args.next(), "train_stride", 1)?;
    let copy_decay_bp = parse_usize_flag(args.next(), "copy_decay_bp", 980)?;
    if args.next().is_some() {
        return Err(
            "run-token-matchskipcopy-bridge takes <data-root> [train_tokens] [trigram_buckets] [skip_buckets] [val_tokens] [match_depth] [copy_window] [candidate_k] [train_stride] [copy_decay_bp]"
                .to_string(),
        );
    }
    let report = run_token_matchskipcopy_bridge_from_data_root(
        Path::new(&data_root),
        train_tokens,
        trigram_buckets,
        skip_buckets,
        val_tokens,
        match_depth,
        copy_window,
        candidate_k,
        train_stride,
        4.0,
        2.0,
        2.0,
        (copy_decay_bp as f64) / 1000.0,
    )?;
    print!("{}", render_token_matchskipcopy_bridge_report(&report));
    Ok(())
}

fn audit_token_matchskipcopy_bridge(args: &mut dyn Iterator<Item = String>) -> Result<(), String> {
    let data_root = args
        .next()
        .ok_or_else(|| "audit-token-matchskipcopy-bridge requires a data root".to_string())?;
    let train_tokens = parse_usize_flag(args.next(), "train_tokens", 1_000_000)?;
    let trigram_buckets = parse_usize_flag(args.next(), "trigram_buckets", 2_048)?;
    let skip_buckets = parse_usize_flag(args.next(), "skip_buckets", 2_048)?;
    let val_tokens = parse_usize_flag(args.next(), "val_tokens", 32_768)?;
    let match_depth = parse_usize_flag(args.next(), "match_depth", 8)?;
    let copy_window = parse_usize_flag(args.next(), "copy_window", 256)?;
    let candidate_k = parse_usize_flag(args.next(), "candidate_k", 4)?;
    let train_stride = parse_usize_flag(args.next(), "train_stride", 1)?;
    let copy_decay_bp = parse_usize_flag(args.next(), "copy_decay_bp", 980)?;
    let chunk_size = parse_usize_flag(args.next(), "chunk_size", 64)?;
    let max_chunks = parse_usize_flag(args.next(), "max_chunks", 8)?;
    if args.next().is_some() {
        return Err(
            "audit-token-matchskipcopy-bridge takes <data-root> [train_tokens] [trigram_buckets] [skip_buckets] [val_tokens] [match_depth] [copy_window] [candidate_k] [train_stride] [copy_decay_bp] [chunk_size] [max_chunks]"
                .to_string(),
        );
    }
    let trained = train_token_matchskipcopy_bridge_from_data_root(
        Path::new(&data_root),
        train_tokens,
        trigram_buckets,
        skip_buckets,
        val_tokens,
        match_depth,
        copy_window,
        candidate_k,
        train_stride,
        4.0,
        2.0,
        2.0,
        (copy_decay_bp as f64) / 1000.0,
    )?;
    print!(
        "{}",
        render_token_matchskipcopy_bridge_report(trained.report())
    );
    let report = audit_parameter_golf(
        trained.runner(),
        trained.eval_tokens(),
        chunk_size,
        max_chunks,
    )?;
    println!("token_matchskipcopy_bridge_audit:");
    print!(
        "{}",
        indent_block(&report.render(trained.runner().name()), "  ")
    );
    Ok(())
}

fn run_token_decay_bridge(args: &mut dyn Iterator<Item = String>) -> Result<(), String> {
    let data_root = args
        .next()
        .ok_or_else(|| "run-token-decay-bridge requires a data root".to_string())?;
    let train_tokens = parse_usize_flag(args.next(), "train_tokens", 1_000_000)?;
    let trigram_buckets = parse_usize_flag(args.next(), "trigram_buckets", 2_048)?;
    let val_tokens = parse_usize_flag(args.next(), "val_tokens", 32_768)?;
    let decay_bp = parse_usize_flag(args.next(), "decay_bp", 980)?;
    let candidate_k = parse_usize_flag(args.next(), "candidate_k", 4)?;
    let train_stride = parse_usize_flag(args.next(), "train_stride", 1)?;
    if args.next().is_some() {
        return Err(
            "run-token-decay-bridge takes <data-root> [train_tokens] [trigram_buckets] [val_tokens] [decay_bp] [candidate_k] [train_stride]"
                .to_string(),
        );
    }
    let report = run_token_decay_bridge_from_data_root(
        Path::new(&data_root),
        train_tokens,
        trigram_buckets,
        val_tokens,
        (decay_bp as f64) / 1000.0,
        candidate_k,
        train_stride,
        4.0,
        2.0,
    )?;
    print!("{}", render_token_decay_bridge_report(&report));
    Ok(())
}

fn audit_token_decay_bridge(args: &mut dyn Iterator<Item = String>) -> Result<(), String> {
    let data_root = args
        .next()
        .ok_or_else(|| "audit-token-decay-bridge requires a data root".to_string())?;
    let train_tokens = parse_usize_flag(args.next(), "train_tokens", 1_000_000)?;
    let trigram_buckets = parse_usize_flag(args.next(), "trigram_buckets", 2_048)?;
    let val_tokens = parse_usize_flag(args.next(), "val_tokens", 32_768)?;
    let decay_bp = parse_usize_flag(args.next(), "decay_bp", 980)?;
    let candidate_k = parse_usize_flag(args.next(), "candidate_k", 4)?;
    let train_stride = parse_usize_flag(args.next(), "train_stride", 1)?;
    let chunk_size = parse_usize_flag(args.next(), "chunk_size", 64)?;
    let max_chunks = parse_usize_flag(args.next(), "max_chunks", 8)?;
    if args.next().is_some() {
        return Err(
            "audit-token-decay-bridge takes <data-root> [train_tokens] [trigram_buckets] [val_tokens] [decay_bp] [candidate_k] [train_stride] [chunk_size] [max_chunks]"
                .to_string(),
        );
    }
    let trained = train_token_decay_bridge_from_data_root(
        Path::new(&data_root),
        train_tokens,
        trigram_buckets,
        val_tokens,
        (decay_bp as f64) / 1000.0,
        candidate_k,
        train_stride,
        4.0,
        2.0,
    )?;
    print!("{}", render_token_decay_bridge_report(trained.report()));
    let report = audit_parameter_golf(
        trained.runner(),
        trained.eval_tokens(),
        chunk_size,
        max_chunks,
    )?;
    println!("token_decay_bridge_audit:");
    print!(
        "{}",
        indent_block(&report.render(trained.runner().name()), "  ")
    );
    Ok(())
}

fn run_token_word_bridge(args: &mut dyn Iterator<Item = String>) -> Result<(), String> {
    let data_root = args
        .next()
        .ok_or_else(|| "run-token-word-bridge requires a data root".to_string())?;
    let train_tokens = parse_usize_flag(args.next(), "train_tokens", 1_000_000)?;
    let trigram_buckets = parse_usize_flag(args.next(), "trigram_buckets", 2_048)?;
    let word_buckets = parse_usize_flag(args.next(), "word_buckets", 8_192)?;
    let val_tokens = parse_usize_flag(args.next(), "val_tokens", 32_768)?;
    let word_window = parse_usize_flag(args.next(), "word_window", 24)?;
    let suffix_len = parse_usize_flag(args.next(), "suffix_len", 4)?;
    let boundary_markers = parse_usize_flag(args.next(), "boundary_markers", 64)?;
    let candidate_k = parse_usize_flag(args.next(), "candidate_k", 4)?;
    let train_stride = parse_usize_flag(args.next(), "train_stride", 1)?;
    if args.next().is_some() {
        return Err(
            "run-token-word-bridge takes <data-root> [train_tokens] [trigram_buckets] [word_buckets] [val_tokens] [word_window] [suffix_len] [boundary_markers] [candidate_k] [train_stride]"
                .to_string(),
        );
    }
    let report = run_token_word_bridge_from_data_root(
        Path::new(&data_root),
        train_tokens,
        trigram_buckets,
        word_buckets,
        val_tokens,
        word_window,
        suffix_len,
        boundary_markers,
        candidate_k,
        train_stride,
        4.0,
        2.0,
        2.0,
    )?;
    print!("{}", render_token_word_bridge_report(&report));
    Ok(())
}

fn audit_token_word_bridge(args: &mut dyn Iterator<Item = String>) -> Result<(), String> {
    let data_root = args
        .next()
        .ok_or_else(|| "audit-token-word-bridge requires a data root".to_string())?;
    let train_tokens = parse_usize_flag(args.next(), "train_tokens", 1_000_000)?;
    let trigram_buckets = parse_usize_flag(args.next(), "trigram_buckets", 2_048)?;
    let word_buckets = parse_usize_flag(args.next(), "word_buckets", 8_192)?;
    let val_tokens = parse_usize_flag(args.next(), "val_tokens", 32_768)?;
    let word_window = parse_usize_flag(args.next(), "word_window", 24)?;
    let suffix_len = parse_usize_flag(args.next(), "suffix_len", 4)?;
    let boundary_markers = parse_usize_flag(args.next(), "boundary_markers", 64)?;
    let candidate_k = parse_usize_flag(args.next(), "candidate_k", 4)?;
    let train_stride = parse_usize_flag(args.next(), "train_stride", 1)?;
    let chunk_size = parse_usize_flag(args.next(), "chunk_size", 64)?;
    let max_chunks = parse_usize_flag(args.next(), "max_chunks", 8)?;
    if args.next().is_some() {
        return Err(
            "audit-token-word-bridge takes <data-root> [train_tokens] [trigram_buckets] [word_buckets] [val_tokens] [word_window] [suffix_len] [boundary_markers] [candidate_k] [train_stride] [chunk_size] [max_chunks]"
                .to_string(),
        );
    }
    let trained = train_token_word_bridge_from_data_root(
        Path::new(&data_root),
        train_tokens,
        trigram_buckets,
        word_buckets,
        val_tokens,
        word_window,
        suffix_len,
        boundary_markers,
        candidate_k,
        train_stride,
        4.0,
        2.0,
        2.0,
    )?;
    print!("{}", render_token_word_bridge_report(trained.report()));
    let report = audit_parameter_golf(
        trained.runner(),
        trained.eval_tokens(),
        chunk_size,
        max_chunks,
    )?;
    println!("token_word_bridge_audit:");
    print!(
        "{}",
        indent_block(&report.render(trained.runner().name()), "  ")
    );
    Ok(())
}

fn run_token_column_bridge(args: &mut dyn Iterator<Item = String>) -> Result<(), String> {
    let data_root = args
        .next()
        .ok_or_else(|| "run-token-column-bridge requires a data root".to_string())?;
    let train_tokens = parse_usize_flag(args.next(), "train_tokens", 1_000_000)?;
    let trigram_buckets = parse_usize_flag(args.next(), "trigram_buckets", 2_048)?;
    let slot_buckets = parse_usize_flag(args.next(), "slot_buckets", 8_192)?;
    let slot_period = parse_usize_flag(args.next(), "slot_period", 16)?;
    let val_tokens = parse_usize_flag(args.next(), "val_tokens", 32_768)?;
    let candidate_k = parse_usize_flag(args.next(), "candidate_k", 4)?;
    let train_stride = parse_usize_flag(args.next(), "train_stride", 1)?;
    if args.next().is_some() {
        return Err(
            "run-token-column-bridge takes <data-root> [train_tokens] [trigram_buckets] [slot_buckets] [slot_period] [val_tokens] [candidate_k] [train_stride]"
                .to_string(),
        );
    }
    let report = run_token_column_bridge_from_data_root(
        Path::new(&data_root),
        train_tokens,
        trigram_buckets,
        slot_buckets,
        slot_period,
        val_tokens,
        candidate_k,
        train_stride,
        4.0,
        2.0,
        2.0,
    )?;
    print!("{}", render_token_column_bridge_report(&report));
    Ok(())
}

fn audit_token_column_bridge(args: &mut dyn Iterator<Item = String>) -> Result<(), String> {
    let data_root = args
        .next()
        .ok_or_else(|| "audit-token-column-bridge requires a data root".to_string())?;
    let train_tokens = parse_usize_flag(args.next(), "train_tokens", 1_000_000)?;
    let trigram_buckets = parse_usize_flag(args.next(), "trigram_buckets", 2_048)?;
    let slot_buckets = parse_usize_flag(args.next(), "slot_buckets", 8_192)?;
    let slot_period = parse_usize_flag(args.next(), "slot_period", 16)?;
    let val_tokens = parse_usize_flag(args.next(), "val_tokens", 32_768)?;
    let candidate_k = parse_usize_flag(args.next(), "candidate_k", 4)?;
    let train_stride = parse_usize_flag(args.next(), "train_stride", 1)?;
    let chunk_size = parse_usize_flag(args.next(), "chunk_size", 64)?;
    let max_chunks = parse_usize_flag(args.next(), "max_chunks", 8)?;
    if args.next().is_some() {
        return Err(
            "audit-token-column-bridge takes <data-root> [train_tokens] [trigram_buckets] [slot_buckets] [slot_period] [val_tokens] [candidate_k] [train_stride] [chunk_size] [max_chunks]"
                .to_string(),
        );
    }
    let trained = train_token_column_bridge_from_data_root(
        Path::new(&data_root),
        train_tokens,
        trigram_buckets,
        slot_buckets,
        slot_period,
        val_tokens,
        candidate_k,
        train_stride,
        4.0,
        2.0,
        2.0,
    )?;
    print!("{}", render_token_column_bridge_report(trained.report()));
    let report = audit_parameter_golf(
        trained.runner(),
        trained.eval_tokens(),
        chunk_size,
        max_chunks,
    )?;
    println!("token_column_bridge_audit:");
    print!(
        "{}",
        indent_block(&report.render(trained.runner().name()), "  ")
    );
    Ok(())
}
