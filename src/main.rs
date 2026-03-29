use std::env;
use std::fs;
use std::path::Path;

use chronohorn::audit::audit_parameter_golf;
use chronohorn::bridge::{bridge_doctrine, render_bridge_doctrine};
use chronohorn::byte_bridge::{
    render_byte_bridge_codec_report, render_byte_bridge_report, run_byte_bridge,
    run_byte_bridge_codec,
};
use chronohorn::checkpoint::{inspect_npz, render_entries};
use chronohorn::data::{data_home_report, resolve_data_root};
use chronohorn::demo::{DemoMode, PackedCacheDemo};
use chronohorn::oracle::{load_oracle_attack, render_oracle_clean_summary};
use chronohorn::packed_memory::{PackedMemoryRunner, compare_tables, render_table_diffs};
use chronohorn::protocol::Runner;
use chronohorn::token_bridge::{
    render_token_bridge_report, run_token_bridge_from_data_root, train_token_bridge_from_data_root,
};
use chronohorn::token_column_bridge::{
    render_token_column_bridge_report, run_token_column_bridge_from_data_root,
    train_token_column_bridge_from_data_root,
};
use chronohorn::token_copy_bridge::{
    render_token_copy_bridge_report, run_token_copy_bridge_from_data_root,
    train_token_copy_bridge_from_data_root,
};
use chronohorn::token_decay_bridge::{
    render_token_decay_bridge_report, run_token_decay_bridge_from_data_root,
    train_token_decay_bridge_from_data_root,
};
use chronohorn::token_experiment_matrix::{
    TokenExperimentMatrixConfig, run_token_experiment_matrix_from_data_root,
};
use chronohorn::token_match_bridge::{
    render_token_match_bridge_report, run_token_match_bridge_from_data_root,
    train_token_match_bridge_from_data_root,
};
use chronohorn::token_matchcopy_bridge::{
    render_token_matchcopy_bridge_report, run_token_matchcopy_bridge_from_data_root,
    train_token_matchcopy_bridge_from_data_root,
};
use chronohorn::token_matchskip_bridge::{
    render_token_matchskip_bridge_report, run_token_matchskip_bridge_from_data_root,
    train_token_matchskip_bridge_from_data_root,
};
use chronohorn::token_matchskipcopy_bridge::{
    render_token_matchskipcopy_bridge_report, run_token_matchskipcopy_bridge_from_data_root,
    train_token_matchskipcopy_bridge_from_data_root,
};
use chronohorn::token_skip_bridge::{
    render_token_skip_bridge_report, run_token_skip_bridge_from_data_root,
    train_token_skip_bridge_from_data_root,
};
use chronohorn::token_skipcopy_bridge::{
    render_token_skipcopy_bridge_report, run_token_skipcopy_bridge_from_data_root,
    train_token_skipcopy_bridge_from_data_root,
};
use chronohorn::token_word_bridge::{
    render_token_word_bridge_report, run_token_word_bridge_from_data_root,
    train_token_word_bridge_from_data_root,
};
mod export_schema;

use crate::export_schema::{
    ChronohornExportBundle, ChronohornExportManifest, TOKEN_MATCHSKIP_ALIASES,
};
use serde::Deserialize;

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
    manifest: ChronohornExportManifest,
    report: chronohorn::token_matchskip_bridge::TokenMatchSkipBridgeReport,
    audit: chronohorn::audit::LegalityReport,
}

fn parse_matchskip_export_args(
    args: &mut impl Iterator<Item = String>,
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
    let manifest = ChronohornExportManifest {
        schema: "chronohorn.export.bundle",
        schema_version: "1",
        family: "token_matchskip_bridge",
        aliases: TOKEN_MATCHSKIP_ALIASES,
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

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let Some(command) = args.next() else {
        print_usage();
        return Ok(());
    };
    match command.as_str() {
        "inspect-npz" => {
            let path = args
                .next()
                .ok_or_else(|| "inspect-npz requires a checkpoint path".to_string())?;
            if args.next().is_some() {
                return Err("inspect-npz takes exactly one path".to_string());
            }
            let entries = inspect_npz(&path)?;
            print!("{}", render_entries(&entries));
            Ok(())
        }
        "inspect-data-root" => {
            let data_root = args.next();
            if args.next().is_some() {
                return Err("inspect-data-root takes at most one path or alias".to_string());
            }
            let report = resolve_data_root(data_root.as_deref())?;
            println!(
                "{}",
                serde_json::to_string_pretty(&report)
                    .map_err(|err| format!("serialize data root resolution: {err}"))?
            );
            Ok(())
        }
        "print-data-home" => {
            if args.next().is_some() {
                return Err("print-data-home takes no arguments".to_string());
            }
            let report = data_home_report();
            println!(
                "{}",
                serde_json::to_string_pretty(&report)
                    .map_err(|err| format!("serialize data home report: {err}"))?
            );
            Ok(())
        }
        "audit-demo" => {
            let mode = DemoMode::parse(
                &args
                    .next()
                    .ok_or_else(|| "audit-demo requires a mode".to_string())?,
            )?;
            if args.next().is_some() {
                return Err("audit-demo takes exactly one mode".to_string());
            }
            let runner = PackedCacheDemo::new(8, 1.0, mode);
            let tokens = vec![0, 1, 0, 1, 2, 3, 1, 0, 2, 2, 1, 3, 0, 1, 2, 3];
            let report = audit_parameter_golf(&runner, &tokens, 8, 2)?;
            println!("mode: {}", mode.as_str());
            print!("{}", report.render(runner.name()));
            Ok(())
        }
        "audit-packed-memory" => {
            let data_root = args
                .next()
                .ok_or_else(|| "audit-packed-memory requires a data root".to_string())?;
            let token_budget = parse_usize_flag(args.next(), "token_budget", 200_000)?;
            let trigram_buckets = parse_usize_flag(args.next(), "trigram_buckets", 2_048)?;
            let (runner, val_tokens) = PackedMemoryRunner::from_data_root(
                Path::new(&data_root),
                1024,
                token_budget,
                trigram_buckets,
                4.0,
                2.0,
            )?;
            let report = audit_parameter_golf(&runner, &val_tokens, 64, 4)?;
            println!("data_root: {data_root}");
            println!("token_budget: {token_budget}");
            println!("trigram_buckets: {trigram_buckets}");
            print!("{}", report.render(runner.name()));
            Ok(())
        }
        "compare-packed-memory" => {
            let checkpoint_path = args
                .next()
                .ok_or_else(|| "compare-packed-memory requires a checkpoint path".to_string())?;
            let summary_path = args
                .next()
                .ok_or_else(|| "compare-packed-memory requires a summary path".to_string())?;
            let data_root = match args.next() {
                Some(root) => root,
                None => infer_data_root_from_summary(&summary_path)?,
            };
            if args.next().is_some() {
                return Err(
                    "compare-packed-memory takes <checkpoint.npz> <summary.json> [data-root]"
                        .to_string(),
                );
            }
            let cfg = load_summary(&summary_path)?;
            let (rebuilt_runner, val_tokens) = PackedMemoryRunner::from_data_root(
                Path::new(&data_root),
                1024,
                cfg.model.packed_tokens,
                cfg.model.trigram_buckets,
                cfg.model.alpha_bigram,
                cfg.model.alpha_trigram,
            )?;
            let checkpoint_runner = PackedMemoryRunner::from_checkpoint(
                &checkpoint_path,
                cfg.model.alpha_bigram,
                cfg.model.alpha_trigram,
            )?;
            let diffs = compare_tables(checkpoint_runner.tables(), rebuilt_runner.tables())?;
            let checkpoint_report = audit_parameter_golf(&checkpoint_runner, &val_tokens, 64, 4)?;
            let rebuilt_report = audit_parameter_golf(&rebuilt_runner, &val_tokens, 64, 4)?;
            println!("checkpoint: {checkpoint_path}");
            println!("summary: {summary_path}");
            println!("data_root: {data_root}");
            println!("packed_tokens: {}", cfg.model.packed_tokens);
            println!("trigram_buckets: {}", cfg.model.trigram_buckets);
            println!("alpha_bigram: {}", cfg.model.alpha_bigram);
            println!("alpha_trigram: {}", cfg.model.alpha_trigram);
            print!("{}", render_table_diffs(&diffs));
            println!("checkpoint_runner_audit:");
            print!(
                "{}",
                indent_block(&checkpoint_report.render(checkpoint_runner.name()), "  ")
            );
            println!("rebuilt_runner_audit:");
            print!(
                "{}",
                indent_block(&rebuilt_report.render(rebuilt_runner.name()), "  ")
            );
            Ok(())
        }
        "oracle-clean-summary" => {
            let attack_path = args
                .next()
                .ok_or_else(|| "oracle-clean-summary requires an attack JSON path".to_string())?;
            let top_n = parse_usize_flag(args.next(), "top_n", 8)?;
            if args.next().is_some() {
                return Err("oracle-clean-summary takes <attack.json> [top-n]".to_string());
            }
            let corpus = load_oracle_attack(&attack_path)?;
            print!("{}", render_oracle_clean_summary(&corpus, top_n));
            Ok(())
        }
        "train-byte-bridge" => {
            let radius = parse_usize_flag(args.next(), "radius", 4)?;
            let stride = parse_usize_flag(args.next(), "stride", 16)?;
            let max_files = parse_usize_flag(args.next(), "max_files", 80)?;
            if args.next().is_some() {
                return Err("train-byte-bridge takes [radius] [stride] [max-files]".to_string());
            }
            let report = run_byte_bridge(Path::new("."), radius, stride, max_files)?;
            print!("{}", render_byte_bridge_report(&report));
            Ok(())
        }
        "run-byte-bridge-codec" => {
            let radius = parse_usize_flag(args.next(), "radius", 4)?;
            let stride = parse_usize_flag(args.next(), "stride", 16)?;
            let max_files = parse_usize_flag(args.next(), "max_files", 80)?;
            if args.next().is_some() {
                return Err("run-byte-bridge-codec takes [radius] [stride] [max-files]".to_string());
            }
            let report = run_byte_bridge_codec(Path::new("."), radius, stride, max_files)?;
            print!("{}", render_byte_bridge_codec_report(&report));
            Ok(())
        }
        "run-token-bridge" => {
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
        "audit-token-bridge" => {
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
        "run-token-copy-bridge" => {
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
        "audit-token-copy-bridge" => {
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
        "run-token-skip-bridge" => {
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
        "audit-token-skip-bridge" => {
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
        "run-token-skipcopy-bridge" => {
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
        "audit-token-skipcopy-bridge" => {
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
        "run-token-match-bridge" => {
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
        "audit-token-match-bridge" => {
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
        "run-token-matchcopy-bridge" => {
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
        "audit-token-matchcopy-bridge" => {
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
        "run-token-matchskip-bridge" => {
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
        "audit-token-matchskip-bridge" => {
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
        "run-token-matchskip-bundle-json" => {
            let args = parse_matchskip_export_args(&mut args, "run-token-matchskip-bundle-json")?;
            let export = build_matchskip_export_context(args, "run-token-matchskip-bundle-json")?;
            let bundle = ChronohornExportBundle {
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
        "run-token-matchskip-manifest-json" => {
            let args =
                parse_matchskip_export_args(&mut args, "run-token-matchskip-manifest-json")?;
            let export =
                build_matchskip_export_context(args, "run-token-matchskip-manifest-json")?;
            println!(
                "{}",
                serde_json::to_string_pretty(&export.manifest)
                    .map_err(|err| format!("serialize matchskip manifest: {err}"))?
            );
            Ok(())
        }
        "run-token-experiment-matrix" => {
            let data_root = args
                .next()
                .ok_or_else(|| "run-token-experiment-matrix requires a data root".to_string())?;
            let train_token_budget =
                parse_usize_flag(args.next(), "train_token_budget", 1_000_000)?;
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
        "run-token-matchskipcopy-bridge" => {
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
        "audit-token-matchskipcopy-bridge" => {
            let data_root = args.next().ok_or_else(|| {
                "audit-token-matchskipcopy-bridge requires a data root".to_string()
            })?;
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
        "run-token-decay-bridge" => {
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
        "audit-token-decay-bridge" => {
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
        "run-token-word-bridge" => {
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
        "audit-token-word-bridge" => {
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
        "run-token-column-bridge" => {
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
        "audit-token-column-bridge" => {
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
        "design" => {
            println!("Chronohorn");
            println!();
            println!("A clean Rust reset with hard boundaries:");
            println!("  oracle: offline only");
            println!("  runtime: causal only");
            println!("  audit: adversarial by default");
            println!("  checkpoint: artifact boundary owned outside Python");
            println!();
            println!("Current anti-cheat surface:");
            println!("  normalization");
            println!("  repeatability");
            println!("  future suffix invariance");
            println!("  answer mask invariance");
            println!("  prefix truncation parity");
            println!("  stream rechunk parity");
            println!("  gold logprob consistency");
            Ok(())
        }
        "doctrine" => {
            print!("{}", render_bridge_doctrine());
            Ok(())
        }
        "doctrine-json" => {
            println!(
                "{}",
                serde_json::to_string_pretty(&bridge_doctrine())
                    .map_err(|err| format!("serialize doctrine: {err}"))?
            );
            Ok(())
        }
        _ => Err(format!("unknown command: {command}")),
    }
}

fn print_usage() {
    println!("chronohorn");
    println!();
    println!("Usage:");
    println!("  chronohorn inspect-npz <checkpoint.npz>");
    println!("  chronohorn inspect-data-root [@alias|path]");
    println!("  chronohorn print-data-home");
    println!("  chronohorn doctrine");
    println!("  chronohorn doctrine-json");
    println!(
        "  chronohorn audit-demo <legal|self-include|future-peek|length-peek|boundary-double-update|reported-gold-cheat>"
    );
    println!("  chronohorn audit-packed-memory <data-root> [token-budget] [trigram-buckets]");
    println!("  chronohorn compare-packed-memory <checkpoint.npz> <summary.json> [data-root]");
    println!("  chronohorn oracle-clean-summary <attack.json> [top-n]");
    println!("  chronohorn train-byte-bridge [radius] [stride] [max-files]");
    println!("  chronohorn run-byte-bridge-codec [radius] [stride] [max-files]");
    println!(
        "  chronohorn run-token-bridge <data-root> [train_tokens] [trigram_buckets] [val_tokens] [train_stride] [candidate_k]"
    );
    println!(
        "  chronohorn audit-token-bridge <data-root> [train_tokens] [trigram_buckets] [val_tokens] [train_stride] [candidate_k] [chunk_size] [max_chunks]"
    );
    println!(
        "  chronohorn run-token-copy-bridge <data-root> [train_tokens] [trigram_buckets] [val_tokens] [copy_window] [candidate_k] [train_stride] [copy_decay_bp]"
    );
    println!(
        "  chronohorn audit-token-copy-bridge <data-root> [train_tokens] [trigram_buckets] [val_tokens] [copy_window] [candidate_k] [train_stride] [copy_decay_bp] [chunk_size] [max_chunks]"
    );
    println!(
        "  chronohorn run-token-skip-bridge <data-root> [train_tokens] [trigram_buckets] [skip_buckets] [val_tokens] [train_stride] [candidate_k]"
    );
    println!(
        "  chronohorn audit-token-skip-bridge <data-root> [train_tokens] [trigram_buckets] [skip_buckets] [val_tokens] [train_stride] [candidate_k] [chunk_size] [max_chunks]"
    );
    println!(
        "  chronohorn run-token-skipcopy-bridge <data-root> [train_tokens] [trigram_buckets] [skip_buckets] [val_tokens] [copy_window] [candidate_k] [train_stride] [copy_decay_bp]"
    );
    println!(
        "  chronohorn audit-token-skipcopy-bridge <data-root> [train_tokens] [trigram_buckets] [skip_buckets] [val_tokens] [copy_window] [candidate_k] [train_stride] [copy_decay_bp] [chunk_size] [max_chunks]"
    );
    println!(
        "  chronohorn run-token-match-bridge <data-root> [train_tokens] [trigram_buckets] [val_tokens] [match_depth] [candidate_k] [train_stride]"
    );
    println!(
        "  chronohorn audit-token-match-bridge <data-root> [train_tokens] [trigram_buckets] [val_tokens] [match_depth] [candidate_k] [train_stride] [chunk_size] [max_chunks]"
    );
    println!(
        "  chronohorn run-token-matchcopy-bridge <data-root> [train_tokens] [trigram_buckets] [val_tokens] [match_depth] [copy_window] [candidate_k] [train_stride] [copy_decay_bp]"
    );
    println!(
        "  chronohorn audit-token-matchcopy-bridge <data-root> [train_tokens] [trigram_buckets] [val_tokens] [match_depth] [copy_window] [candidate_k] [train_stride] [copy_decay_bp] [chunk_size] [max_chunks]"
    );
    println!(
        "  chronohorn run-token-matchskip-bridge <data-root> [train_tokens] [trigram_buckets] [skip_buckets] [val_tokens] [match_depth] [candidate_k] [train_stride]"
    );
    println!(
        "  chronohorn audit-token-matchskip-bridge <data-root> [train_tokens] [trigram_buckets] [skip_buckets] [val_tokens] [match_depth] [candidate_k] [train_stride] [chunk_size] [max_chunks]"
    );
    println!(
        "  chronohorn run-token-matchskip-bundle-json <data-root> [train_tokens] [trigram_buckets] [skip_buckets] [val_tokens] [match_depth] [candidate_k] [train_stride] [chunk_size] [max_chunks]"
    );
    println!(
        "  chronohorn run-token-matchskip-manifest-json <data-root> [train_tokens] [trigram_buckets] [skip_buckets] [val_tokens] [match_depth] [candidate_k] [train_stride] [chunk_size] [max_chunks]"
    );
    println!(
        "  chronohorn run-token-experiment-matrix <data-root> [train_token_budget] [trigram_buckets] [skip_buckets] [val_token_budget] [match_depth] [copy_window] [candidate_k] [train_stride] [copy_decay_bp]"
    );
    println!(
        "  chronohorn run-token-matchskipcopy-bridge <data-root> [train_tokens] [trigram_buckets] [skip_buckets] [val_tokens] [match_depth] [copy_window] [candidate_k] [train_stride] [copy_decay_bp]"
    );
    println!(
        "  chronohorn audit-token-matchskipcopy-bridge <data-root> [train_tokens] [trigram_buckets] [skip_buckets] [val_tokens] [match_depth] [copy_window] [candidate_k] [train_stride] [copy_decay_bp] [chunk_size] [max_chunks]"
    );
    println!(
        "  chronohorn run-token-decay-bridge <data-root> [train_tokens] [trigram_buckets] [val_tokens] [decay_bp] [candidate_k] [train_stride]"
    );
    println!(
        "  chronohorn audit-token-decay-bridge <data-root> [train_tokens] [trigram_buckets] [val_tokens] [decay_bp] [candidate_k] [train_stride] [chunk_size] [max_chunks]"
    );
    println!(
        "  chronohorn run-token-word-bridge <data-root> [train_tokens] [trigram_buckets] [word_buckets] [val_tokens] [word_window] [suffix_len] [boundary_markers] [candidate_k] [train_stride]"
    );
    println!(
        "  chronohorn audit-token-word-bridge <data-root> [train_tokens] [trigram_buckets] [word_buckets] [val_tokens] [word_window] [suffix_len] [boundary_markers] [candidate_k] [train_stride] [chunk_size] [max_chunks]"
    );
    println!(
        "  chronohorn run-token-column-bridge <data-root> [train_tokens] [trigram_buckets] [slot_buckets] [slot_period] [val_tokens] [candidate_k] [train_stride]"
    );
    println!(
        "  chronohorn audit-token-column-bridge <data-root> [train_tokens] [trigram_buckets] [slot_buckets] [slot_period] [val_tokens] [candidate_k] [train_stride] [chunk_size] [max_chunks]"
    );
    println!("  chronohorn design");
}

fn parse_usize_flag(value: Option<String>, label: &str, default: usize) -> Result<usize, String> {
    match value {
        Some(raw) => raw
            .parse::<usize>()
            .map_err(|err| format!("invalid {label} {raw}: {err}")),
        None => Ok(default),
    }
}

#[derive(Debug, Deserialize)]
struct SummaryModel {
    packed_tokens: usize,
    trigram_buckets: usize,
    alpha_bigram: f64,
    alpha_trigram: f64,
}

#[derive(Debug, Deserialize)]
struct SummaryDataset {
    source_path: String,
}

#[derive(Debug, Deserialize)]
struct SummaryRoot {
    model: SummaryModel,
    dataset: SummaryDataset,
}

fn load_summary(path: &str) -> Result<SummaryRoot, String> {
    let raw = fs::read_to_string(path).map_err(|err| format!("read {path}: {err}"))?;
    serde_json::from_str(&raw).map_err(|err| format!("parse summary {path}: {err}"))
}

fn infer_data_root_from_summary(path: &str) -> Result<String, String> {
    let summary = load_summary(path)?;
    let first = summary
        .dataset
        .source_path
        .split("::")
        .next()
        .ok_or_else(|| format!("could not parse source_path in {path}"))?
        .trim();
    let cleaned = first.trim_end_matches('*').trim_end_matches('/');
    let parent = Path::new(cleaned)
        .parent()
        .ok_or_else(|| format!("could not infer data root from {cleaned}"))?;
    Ok(parent.display().to_string())
}

fn indent_block(text: &str, prefix: &str) -> String {
    let mut out = String::new();
    for line in text.lines() {
        out.push_str(prefix);
        out.push_str(line);
        out.push('\n');
    }
    out
}
