use std::path::Path;

use chronohorn::archive::token_ngram_checkpoint::{
    load_token_conker3_ngram_checkpoint_from_data_root, render_token_conker3_ngram_bundle,
    render_token_conker3_ngram_checkpoint_report,
    run_token_conker3_ngram_checkpoint_fast_from_data_root,
    run_token_conker3_ngram_checkpoint_from_data_root,
};
use chronohorn::causal_bank::checkpoint::{
    load_token_conker3_checkpoint_from_data_root,
    load_token_conker3_exact_checkpoint_from_data_root,
    render_token_conker3_boundary_checkpoint_report, render_token_conker3_checkpoint_report,
    render_token_conker3_exact_checkpoint_report,
    run_token_conker3_boundary_checkpoint_from_data_root,
    run_token_conker3_checkpoint_from_data_root, run_token_conker3_exact_checkpoint_from_data_root,
};
use chronohorn::causal_bank::exact_ngram_checkpoint::{
    load_token_conker3_exact_ngram_checkpoint_from_data_root,
    render_token_conker3_exact_ngram_checkpoint_report,
    run_token_conker3_exact_ngram_checkpoint_from_data_root,
};
use chronohorn::causal_bank::ngram_bulk::{
    build_token_conker3_ngram_oracle_budgeted_table_from_data_root,
    build_token_conker3_ngram_oracle_row_stats_from_data_root,
    pack_token_conker3_ngram_oracle_row_stats_artifact, render_token_conker3_ngram_bulk_report,
    run_token_conker3_ngram_bulk_from_data_root, run_token_conker3_ngram_bulk_from_table_artifact,
    run_token_conker3_ngram_bulk_oracle_budgeted_prebuilt_from_data_root,
    run_token_conker3_ngram_bulk_oracle_pruned_prebuilt_from_data_root,
    run_token_conker3_ngram_bulk_oracle_trust_prebuilt_from_data_root,
    run_token_conker3_ngram_bulk_prebuilt_from_data_root,
    run_token_conker3_ngram_bulk_priority_cache_prebuilt_from_data_root,
    run_token_conker3_oracle_table_bulk_from_data_root,
};
use chronohorn::core::audit::audit_parameter_golf;
use chronohorn::core::data::resolve_data_root;
use chronohorn::core::protocol::Runner;

use crate::shell_support::{indent_block, parse_usize_flag};

fn resolve_selected_path(data_root: &str) -> Result<String, String> {
    resolve_data_root(Some(data_root)).map(|resolved| resolved.selected_path)
}

fn print_runner_audit(
    label: &str,
    runner: &impl Runner,
    eval_tokens: &[usize],
    chunk_size: usize,
    max_chunks: usize,
) -> Result<(), String> {
    let report = audit_parameter_golf(runner, eval_tokens, chunk_size, max_chunks)?;
    println!("{label}:");
    print!("{}", indent_block(&report.render(runner.name()), "  "));
    Ok(())
}

pub(crate) fn handle(
    command: &str,
    args: &mut dyn Iterator<Item = String>,
) -> Option<Result<(), String>> {
    let result = match command {
        "run-causal-bank-checkpoint" | "run-token-conker3-checkpoint" => {
            let checkpoint_path = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "run-causal-bank-checkpoint requires a checkpoint path".to_string()
                    ));
                }
            };
            let summary_path = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "run-causal-bank-checkpoint requires a summary path".to_string()
                    ));
                }
            };
            let data_root = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "run-causal-bank-checkpoint requires a data root".to_string()
                    ));
                }
            };
            let val_tokens = match parse_usize_flag(args.next(), "val_tokens", 16_384) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            if args.next().is_some() {
                return Some(Err(
                    "run-causal-bank-checkpoint takes <checkpoint-path|bundle-dir> <summary.json> <data-root> [val_tokens]"
                        .to_string(),
                ));
            }
            Some(resolve_selected_path(&data_root).and_then(|selected_path| {
                let report = run_token_conker3_checkpoint_from_data_root(
                    &checkpoint_path,
                    &summary_path,
                    Path::new(&selected_path),
                    val_tokens,
                )?;
                print!("{}", render_token_conker3_checkpoint_report(&report));
                Ok(())
            }))
        }
        "run-causal-bank-boundary-checkpoint" | "run-token-conker3-boundary-checkpoint" => {
            let checkpoint_path = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "run-causal-bank-boundary-checkpoint requires a checkpoint path"
                            .to_string(),
                    ));
                }
            };
            let summary_path = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "run-causal-bank-boundary-checkpoint requires a summary path".to_string(),
                    ));
                }
            };
            let data_root = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "run-causal-bank-boundary-checkpoint requires a data root".to_string(),
                    ));
                }
            };
            let val_tokens = match parse_usize_flag(args.next(), "val_tokens", 16_384) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            if args.next().is_some() {
                return Some(Err(
                    "run-causal-bank-boundary-checkpoint takes <checkpoint-path|bundle-dir> <summary.json> <data-root> [val_tokens]"
                        .to_string(),
                ));
            }
            Some(resolve_selected_path(&data_root).and_then(|selected_path| {
                let report = run_token_conker3_boundary_checkpoint_from_data_root(
                    &checkpoint_path,
                    &summary_path,
                    Path::new(&selected_path),
                    val_tokens,
                )?;
                print!(
                    "{}",
                    render_token_conker3_boundary_checkpoint_report(&report)
                );
                Ok(())
            }))
        }
        "audit-causal-bank-checkpoint" | "audit-token-conker3-checkpoint" => {
            let checkpoint_path = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "audit-causal-bank-checkpoint requires a checkpoint path".to_string(),
                    ));
                }
            };
            let summary_path = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "audit-causal-bank-checkpoint requires a summary path".to_string()
                    ));
                }
            };
            let data_root = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "audit-causal-bank-checkpoint requires a data root".to_string()
                    ));
                }
            };
            let val_tokens = match parse_usize_flag(args.next(), "val_tokens", 16_384) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            let chunk_size = match parse_usize_flag(args.next(), "chunk_size", 64) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            let max_chunks = match parse_usize_flag(args.next(), "max_chunks", 8) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            if args.next().is_some() {
                return Some(Err(
                    "audit-causal-bank-checkpoint takes <checkpoint-path|bundle-dir> <summary.json> <data-root> [val_tokens] [chunk_size] [max_chunks]"
                        .to_string(),
                ));
            }
            Some(resolve_selected_path(&data_root).and_then(|selected_path| {
                let loaded = load_token_conker3_checkpoint_from_data_root(
                    &checkpoint_path,
                    &summary_path,
                    Path::new(&selected_path),
                    val_tokens,
                )?;
                print!(
                    "{}",
                    render_token_conker3_checkpoint_report(loaded.report())
                );
                print_runner_audit(
                    "causal_bank_checkpoint_audit",
                    loaded.runner(),
                    loaded.eval_tokens(),
                    chunk_size,
                    max_chunks,
                )
            }))
        }
        "run-causal-bank-exact-checkpoint" | "run-token-conker3-exact-checkpoint" => {
            let checkpoint_path = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "run-causal-bank-exact-checkpoint requires a checkpoint path".to_string(),
                    ));
                }
            };
            let summary_path = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "run-causal-bank-exact-checkpoint requires a summary path".to_string(),
                    ));
                }
            };
            let data_root = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "run-causal-bank-exact-checkpoint requires a data root".to_string()
                    ));
                }
            };
            let val_tokens = match parse_usize_flag(args.next(), "val_tokens", 16_384) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            if args.next().is_some() {
                return Some(Err(
                    "run-causal-bank-exact-checkpoint takes <checkpoint-path|bundle-dir> <summary.json> <data-root> [val_tokens]"
                        .to_string(),
                ));
            }
            Some(resolve_selected_path(&data_root).and_then(|selected_path| {
                let report = run_token_conker3_exact_checkpoint_from_data_root(
                    &checkpoint_path,
                    &summary_path,
                    Path::new(&selected_path),
                    val_tokens,
                )?;
                print!("{}", render_token_conker3_exact_checkpoint_report(&report));
                Ok(())
            }))
        }
        "audit-causal-bank-exact-checkpoint" | "audit-token-conker3-exact-checkpoint" => {
            let checkpoint_path = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "audit-causal-bank-exact-checkpoint requires a checkpoint path".to_string(),
                    ));
                }
            };
            let summary_path = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "audit-causal-bank-exact-checkpoint requires a summary path".to_string(),
                    ));
                }
            };
            let data_root = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "audit-causal-bank-exact-checkpoint requires a data root".to_string(),
                    ));
                }
            };
            let val_tokens = match parse_usize_flag(args.next(), "val_tokens", 16_384) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            let chunk_size = match parse_usize_flag(args.next(), "chunk_size", 64) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            let max_chunks = match parse_usize_flag(args.next(), "max_chunks", 8) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            if args.next().is_some() {
                return Some(Err(
                    "audit-causal-bank-exact-checkpoint takes <checkpoint-path|bundle-dir> <summary.json> <data-root> [val_tokens] [chunk_size] [max_chunks]"
                        .to_string(),
                ));
            }
            Some(resolve_selected_path(&data_root).and_then(|selected_path| {
                let loaded = load_token_conker3_exact_checkpoint_from_data_root(
                    &checkpoint_path,
                    &summary_path,
                    Path::new(&selected_path),
                    val_tokens,
                )?;
                print!(
                    "{}",
                    render_token_conker3_exact_checkpoint_report(loaded.report())
                );
                print_runner_audit(
                    "causal_bank_exact_checkpoint_audit",
                    loaded.runner(),
                    loaded.eval_tokens(),
                    chunk_size,
                    max_chunks,
                )
            }))
        }
        "run-causal-bank-exact-ngram-checkpoint" | "run-token-conker3-exact-ngram-checkpoint" => {
            let checkpoint_path = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "run-causal-bank-exact-ngram-checkpoint requires a checkpoint path"
                            .to_string(),
                    ));
                }
            };
            let summary_path = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "run-causal-bank-exact-ngram-checkpoint requires a summary path"
                            .to_string(),
                    ));
                }
            };
            let data_root = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "run-causal-bank-exact-ngram-checkpoint requires a data root".to_string(),
                    ));
                }
            };
            let train_tokens = match parse_usize_flag(args.next(), "train_tokens", 100_000_000) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            let val_tokens = match parse_usize_flag(args.next(), "val_tokens", 16_384) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            if args.next().is_some() {
                return Some(Err(
                    "run-causal-bank-exact-ngram-checkpoint takes <checkpoint-path|bundle-dir> <summary.json> <data-root> [train_tokens] [val_tokens]"
                        .to_string(),
                ));
            }
            Some(resolve_selected_path(&data_root).and_then(|selected_path| {
                let report = run_token_conker3_exact_ngram_checkpoint_from_data_root(
                    &checkpoint_path,
                    &summary_path,
                    Path::new(&selected_path),
                    train_tokens,
                    val_tokens,
                )?;
                print!(
                    "{}",
                    render_token_conker3_exact_ngram_checkpoint_report(&report)
                );
                Ok(())
            }))
        }
        "audit-causal-bank-exact-ngram-checkpoint"
        | "audit-token-conker3-exact-ngram-checkpoint" => {
            let checkpoint_path = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "audit-causal-bank-exact-ngram-checkpoint requires a checkpoint path"
                            .to_string(),
                    ));
                }
            };
            let summary_path = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "audit-causal-bank-exact-ngram-checkpoint requires a summary path"
                            .to_string(),
                    ));
                }
            };
            let data_root = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "audit-causal-bank-exact-ngram-checkpoint requires a data root".to_string(),
                    ));
                }
            };
            let train_tokens = match parse_usize_flag(args.next(), "train_tokens", 100_000_000) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            let val_tokens = match parse_usize_flag(args.next(), "val_tokens", 16_384) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            let chunk_size = match parse_usize_flag(args.next(), "chunk_size", 64) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            let max_chunks = match parse_usize_flag(args.next(), "max_chunks", 8) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            if args.next().is_some() {
                return Some(Err(
                    "audit-causal-bank-exact-ngram-checkpoint takes <checkpoint-path|bundle-dir> <summary.json> <data-root> [train_tokens] [val_tokens] [chunk_size] [max_chunks]"
                        .to_string(),
                ));
            }
            Some(resolve_selected_path(&data_root).and_then(|selected_path| {
                let loaded = load_token_conker3_exact_ngram_checkpoint_from_data_root(
                    &checkpoint_path,
                    &summary_path,
                    Path::new(&selected_path),
                    train_tokens,
                    val_tokens,
                )?;
                print!(
                    "{}",
                    render_token_conker3_exact_ngram_checkpoint_report(loaded.report())
                );
                print_runner_audit(
                    "causal_bank_exact_ngram_checkpoint_audit",
                    loaded.runner(),
                    loaded.eval_tokens(),
                    chunk_size,
                    max_chunks,
                )
            }))
        }
        "run-causal-bank-ngram-checkpoint" | "run-token-conker3-ngram-checkpoint" => {
            let checkpoint_path = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "run-causal-bank-ngram-checkpoint requires a checkpoint path".to_string(),
                    ));
                }
            };
            let summary_path = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "run-causal-bank-ngram-checkpoint requires a summary path".to_string(),
                    ));
                }
            };
            let data_root = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "run-causal-bank-ngram-checkpoint requires a data root".to_string()
                    ));
                }
            };
            let val_tokens = match parse_usize_flag(args.next(), "val_tokens", 16_384) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            if args.next().is_some() {
                return Some(Err(
                    "run-causal-bank-ngram-checkpoint takes <checkpoint-path|bundle-dir> <summary.json> <data-root> [val_tokens]"
                        .to_string(),
                ));
            }
            Some(resolve_selected_path(&data_root).and_then(|selected_path| {
                let report = run_token_conker3_ngram_checkpoint_from_data_root(
                    &checkpoint_path,
                    &summary_path,
                    Path::new(&selected_path),
                    val_tokens,
                )?;
                print!("{}", render_token_conker3_ngram_checkpoint_report(&report));
                Ok(())
            }))
        }
        "run-causal-bank-ngram-checkpoint-fast" | "run-token-conker3-ngram-checkpoint-fast" => {
            let checkpoint_path = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "run-causal-bank-ngram-checkpoint-fast requires a checkpoint path"
                            .to_string(),
                    ));
                }
            };
            let summary_path = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "run-causal-bank-ngram-checkpoint-fast requires a summary path".to_string(),
                    ));
                }
            };
            let data_root = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "run-causal-bank-ngram-checkpoint-fast requires a data root".to_string(),
                    ));
                }
            };
            let val_tokens = match parse_usize_flag(args.next(), "val_tokens", 16_384) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            if args.next().is_some() {
                return Some(Err(
                    "run-causal-bank-ngram-checkpoint-fast takes <checkpoint-path|bundle-dir> <summary.json> <data-root> [val_tokens]"
                        .to_string(),
                ));
            }
            Some(resolve_selected_path(&data_root).and_then(|selected_path| {
                let report = run_token_conker3_ngram_checkpoint_fast_from_data_root(
                    &checkpoint_path,
                    &summary_path,
                    Path::new(&selected_path),
                    val_tokens,
                )?;
                print!("{}", render_token_conker3_ngram_checkpoint_report(&report));
                Ok(())
            }))
        }
        "run-causal-bank-ngram-bulk" | "run-token-conker3-ngram-bulk" => {
            let checkpoint_path = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "run-causal-bank-ngram-bulk requires a checkpoint path".to_string()
                    ));
                }
            };
            let summary_path = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "run-causal-bank-ngram-bulk requires a summary path".to_string()
                    ));
                }
            };
            let data_root = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "run-causal-bank-ngram-bulk requires a data root".to_string()
                    ));
                }
            };
            let val_tokens = match parse_usize_flag(args.next(), "val_tokens", 62_000_000) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            let report_every = match parse_usize_flag(args.next(), "report_every", 1_000_000) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            if args.next().is_some() {
                return Some(Err(
                    "run-causal-bank-ngram-bulk takes <checkpoint-path|bundle-dir> <summary.json> <data-root> [val_tokens] [report_every]"
                        .to_string(),
                ));
            }
            Some(resolve_selected_path(&data_root).and_then(|selected_path| {
                let report = run_token_conker3_ngram_bulk_from_data_root(
                    &checkpoint_path,
                    &summary_path,
                    Path::new(&selected_path),
                    val_tokens,
                    report_every,
                )?;
                print!("{}", render_token_conker3_ngram_bulk_report(&report));
                Ok(())
            }))
        }
        "run-causal-bank-ngram-bulk-prebuilt" | "run-token-conker3-ngram-bulk-prebuilt" => {
            let checkpoint_path = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "run-causal-bank-ngram-bulk-prebuilt requires a checkpoint path"
                            .to_string(),
                    ));
                }
            };
            let summary_path = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "run-causal-bank-ngram-bulk-prebuilt requires a summary path".to_string(),
                    ));
                }
            };
            let data_root = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "run-causal-bank-ngram-bulk-prebuilt requires a data root".to_string(),
                    ));
                }
            };
            let train_tokens = match parse_usize_flag(args.next(), "train_tokens", usize::MAX) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            let val_tokens = match parse_usize_flag(args.next(), "val_tokens", 62_000_000) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            let report_every = match parse_usize_flag(args.next(), "report_every", 1_000_000) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            let profile = args.next().unwrap_or_else(|| "tiny".to_string());
            if args.next().is_some() {
                return Some(Err(
                    "run-causal-bank-ngram-bulk-prebuilt takes <checkpoint-path|bundle-dir> <summary.json> <data-root> [train_tokens] [val_tokens] [report_every] [profile]"
                        .to_string(),
                ));
            }
            Some(resolve_selected_path(&data_root).and_then(|selected_path| {
                let report = run_token_conker3_ngram_bulk_prebuilt_from_data_root(
                    &checkpoint_path,
                    &summary_path,
                    Path::new(&selected_path),
                    train_tokens,
                    val_tokens,
                    report_every,
                    &profile,
                )?;
                print!("{}", render_token_conker3_ngram_bulk_report(&report));
                Ok(())
            }))
        }
        "run-causal-bank-ngram-bulk-priority-cache-prebuilt"
        | "run-token-conker3-ngram-bulk-priority-cache-prebuilt" => {
            let checkpoint_path = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "run-causal-bank-ngram-bulk-priority-cache-prebuilt requires a checkpoint path"
                            .to_string(),
                    ));
                }
            };
            let summary_path = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "run-causal-bank-ngram-bulk-priority-cache-prebuilt requires a summary path"
                            .to_string(),
                    ));
                }
            };
            let data_root = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "run-causal-bank-ngram-bulk-priority-cache-prebuilt requires a data root"
                            .to_string(),
                    ));
                }
            };
            let train_tokens = match parse_usize_flag(args.next(), "train_tokens", usize::MAX) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            let val_tokens = match parse_usize_flag(args.next(), "val_tokens", 62_000_000) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            let report_every = match parse_usize_flag(args.next(), "report_every", 1_000_000) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            let profile = args.next().unwrap_or_else(|| "tiny".to_string());
            if args.next().is_some() {
                return Some(Err(
                    "run-causal-bank-ngram-bulk-priority-cache-prebuilt takes <checkpoint-path|bundle-dir> <summary.json> <data-root> [train_tokens] [val_tokens] [report_every] [profile]"
                        .to_string(),
                ));
            }
            Some(resolve_selected_path(&data_root).and_then(|selected_path| {
                let report = run_token_conker3_ngram_bulk_priority_cache_prebuilt_from_data_root(
                    &checkpoint_path,
                    &summary_path,
                    Path::new(&selected_path),
                    train_tokens,
                    val_tokens,
                    report_every,
                    &profile,
                )?;
                print!("{}", render_token_conker3_ngram_bulk_report(&report));
                Ok(())
            }))
        }
        "run-causal-bank-ngram-bulk-oracle-trust-prebuilt"
        | "run-token-conker3-ngram-bulk-oracle-trust-prebuilt" => {
            let checkpoint_path = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "run-causal-bank-ngram-bulk-oracle-trust-prebuilt requires a checkpoint path"
                            .to_string(),
                    ));
                }
            };
            let summary_path = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "run-causal-bank-ngram-bulk-oracle-trust-prebuilt requires a summary path"
                            .to_string(),
                    ));
                }
            };
            let data_root = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "run-causal-bank-ngram-bulk-oracle-trust-prebuilt requires a data root"
                            .to_string(),
                    ));
                }
            };
            let train_tokens = match parse_usize_flag(args.next(), "train_tokens", usize::MAX) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            let val_tokens = match parse_usize_flag(args.next(), "val_tokens", 62_000_000) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            let report_every = match parse_usize_flag(args.next(), "report_every", 1_000_000) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            let profile = args.next().unwrap_or_else(|| "tiny".to_string());
            let oracle_stride = match parse_usize_flag(args.next(), "oracle_stride", 1) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            if args.next().is_some() {
                return Some(Err(
                    "run-causal-bank-ngram-bulk-oracle-trust-prebuilt takes <checkpoint-path|bundle-dir> <summary.json> <data-root> [train_tokens] [val_tokens] [report_every] [profile] [oracle_stride]"
                        .to_string(),
                ));
            }
            Some(resolve_selected_path(&data_root).and_then(|selected_path| {
                let report = run_token_conker3_ngram_bulk_oracle_trust_prebuilt_from_data_root(
                    &checkpoint_path,
                    &summary_path,
                    Path::new(&selected_path),
                    train_tokens,
                    val_tokens,
                    report_every,
                    &profile,
                    oracle_stride,
                )?;
                print!("{}", render_token_conker3_ngram_bulk_report(&report));
                Ok(())
            }))
        }
        "run-causal-bank-ngram-bulk-oracle-pruned-prebuilt"
        | "run-token-conker3-ngram-bulk-oracle-pruned-prebuilt" => {
            let checkpoint_path = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "run-causal-bank-ngram-bulk-oracle-pruned-prebuilt requires a checkpoint path"
                            .to_string(),
                    ));
                }
            };
            let summary_path = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "run-causal-bank-ngram-bulk-oracle-pruned-prebuilt requires a summary path"
                            .to_string(),
                    ));
                }
            };
            let data_root = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "run-causal-bank-ngram-bulk-oracle-pruned-prebuilt requires a data root"
                            .to_string(),
                    ));
                }
            };
            let train_tokens = match parse_usize_flag(args.next(), "train_tokens", usize::MAX) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            let val_tokens = match parse_usize_flag(args.next(), "val_tokens", 62_000_000) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            let report_every = match parse_usize_flag(args.next(), "report_every", 1_000_000) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            let profile = args.next().unwrap_or_else(|| "tiny".to_string());
            let oracle_stride = match parse_usize_flag(args.next(), "oracle_stride", 1) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            if args.next().is_some() {
                return Some(Err(
                    "run-causal-bank-ngram-bulk-oracle-pruned-prebuilt takes <checkpoint-path|bundle-dir> <summary.json> <data-root> [train_tokens] [val_tokens] [report_every] [profile] [oracle_stride]"
                        .to_string(),
                ));
            }
            Some(resolve_selected_path(&data_root).and_then(|selected_path| {
                let report = run_token_conker3_ngram_bulk_oracle_pruned_prebuilt_from_data_root(
                    &checkpoint_path,
                    &summary_path,
                    Path::new(&selected_path),
                    train_tokens,
                    val_tokens,
                    report_every,
                    &profile,
                    oracle_stride,
                )?;
                print!("{}", render_token_conker3_ngram_bulk_report(&report));
                Ok(())
            }))
        }
        "run-causal-bank-ngram-bulk-oracle-budgeted-prebuilt"
        | "run-token-conker3-ngram-bulk-oracle-budgeted-prebuilt" => {
            let checkpoint_path = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "run-causal-bank-ngram-bulk-oracle-budgeted-prebuilt requires a checkpoint path"
                            .to_string(),
                    ));
                }
            };
            let summary_path = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "run-causal-bank-ngram-bulk-oracle-budgeted-prebuilt requires a summary path"
                            .to_string(),
                    ));
                }
            };
            let data_root = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "run-causal-bank-ngram-bulk-oracle-budgeted-prebuilt requires a data root"
                            .to_string(),
                    ));
                }
            };
            let train_tokens = match parse_usize_flag(args.next(), "train_tokens", usize::MAX) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            let val_tokens = match parse_usize_flag(args.next(), "val_tokens", 62_000_000) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            let report_every = match parse_usize_flag(args.next(), "report_every", 1_000_000) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            let profile = args.next().unwrap_or_else(|| "tiny".to_string());
            let oracle_stride = match parse_usize_flag(args.next(), "oracle_stride", 1) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            if args.next().is_some() {
                return Some(Err(
                    "run-causal-bank-ngram-bulk-oracle-budgeted-prebuilt takes <checkpoint-path|bundle-dir> <summary.json> <data-root> [train_tokens] [val_tokens] [report_every] [profile] [oracle_stride]"
                        .to_string(),
                ));
            }
            Some(resolve_selected_path(&data_root).and_then(|selected_path| {
                let report = run_token_conker3_ngram_bulk_oracle_budgeted_prebuilt_from_data_root(
                    &checkpoint_path,
                    &summary_path,
                    Path::new(&selected_path),
                    train_tokens,
                    val_tokens,
                    report_every,
                    &profile,
                    oracle_stride,
                )?;
                print!("{}", render_token_conker3_ngram_bulk_report(&report));
                Ok(())
            }))
        }
        "build-causal-bank-ngram-oracle-budgeted-table"
        | "build-token-conker3-ngram-oracle-budgeted-table" => {
            let data_root = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "build-causal-bank-ngram-oracle-budgeted-table requires a data root"
                            .to_string(),
                    ));
                }
            };
            let artifact_path = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "build-causal-bank-ngram-oracle-budgeted-table requires an artifact path"
                            .to_string(),
                    ));
                }
            };
            let train_tokens = match parse_usize_flag(args.next(), "train_tokens", usize::MAX) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            let report_every = match parse_usize_flag(args.next(), "report_every", 1_000_000) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            let profile = args.next().unwrap_or_else(|| "tiny".to_string());
            let oracle_stride = match parse_usize_flag(args.next(), "oracle_stride", 1) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            if args.next().is_some() {
                return Some(Err(
                    "build-causal-bank-ngram-oracle-budgeted-table takes <data-root> <artifact-path> [train_tokens] [report_every] [profile] [oracle_stride]"
                        .to_string(),
                ));
            }
            Some(resolve_selected_path(&data_root).and_then(|selected_path| {
                let report = build_token_conker3_ngram_oracle_budgeted_table_from_data_root(
                    Path::new(&selected_path),
                    Path::new(&artifact_path),
                    train_tokens,
                    report_every,
                    &profile,
                    oracle_stride,
                )?;
                print!("{report}");
                Ok(())
            }))
        }
        "build-causal-bank-ngram-oracle-row-stats"
        | "build-token-conker3-ngram-oracle-row-stats" => {
            let data_root = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "build-causal-bank-ngram-oracle-row-stats requires a data root".to_string(),
                    ));
                }
            };
            let artifact_path = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "build-causal-bank-ngram-oracle-row-stats requires an artifact path"
                            .to_string(),
                    ));
                }
            };
            let train_tokens = match parse_usize_flag(args.next(), "train_tokens", usize::MAX) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            let report_every = match parse_usize_flag(args.next(), "report_every", 1_000_000) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            let profile = args.next().unwrap_or_else(|| "tiny".to_string());
            let oracle_stride = match parse_usize_flag(args.next(), "oracle_stride", 1) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            if args.next().is_some() {
                return Some(Err(
                    "build-causal-bank-ngram-oracle-row-stats takes <data-root> <artifact-path> [train_tokens] [report_every] [profile] [oracle_stride]"
                        .to_string(),
                ));
            }
            Some(resolve_selected_path(&data_root).and_then(|selected_path| {
                let report = build_token_conker3_ngram_oracle_row_stats_from_data_root(
                    Path::new(&selected_path),
                    Path::new(&artifact_path),
                    train_tokens,
                    report_every,
                    &profile,
                    oracle_stride,
                )?;
                print!("{report}");
                Ok(())
            }))
        }
        "pack-causal-bank-ngram-oracle-row-stats" | "pack-token-conker3-ngram-oracle-row-stats" => {
            let stats_artifact_path = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "pack-causal-bank-ngram-oracle-row-stats requires a stats artifact path"
                            .to_string(),
                    ));
                }
            };
            let packed_artifact_path = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "pack-causal-bank-ngram-oracle-row-stats requires a packed artifact path"
                            .to_string(),
                    ));
                }
            };
            let target_bytes = match parse_usize_flag(args.next(), "target_bytes", 8 * 1024 * 1024)
            {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            if args.next().is_some() {
                return Some(Err(
                    "pack-causal-bank-ngram-oracle-row-stats takes <stats-artifact> <packed-artifact> [target_bytes]"
                        .to_string(),
                ));
            }
            Some(
                pack_token_conker3_ngram_oracle_row_stats_artifact(
                    Path::new(&stats_artifact_path),
                    Path::new(&packed_artifact_path),
                    target_bytes,
                )
                .map(|report| {
                    print!("{report}");
                }),
            )
        }
        "run-causal-bank-ngram-bulk-from-table" | "run-token-conker3-ngram-bulk-from-table" => {
            let checkpoint_path = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "run-causal-bank-ngram-bulk-from-table requires a checkpoint path"
                            .to_string(),
                    ));
                }
            };
            let summary_path = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "run-causal-bank-ngram-bulk-from-table requires a summary path".to_string(),
                    ));
                }
            };
            let data_root = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "run-causal-bank-ngram-bulk-from-table requires a data root".to_string(),
                    ));
                }
            };
            let artifact_path = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "run-causal-bank-ngram-bulk-from-table requires an artifact path"
                            .to_string(),
                    ));
                }
            };
            let val_tokens = match parse_usize_flag(args.next(), "val_tokens", 62_000_000) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            let report_every = match parse_usize_flag(args.next(), "report_every", 1_000_000) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            if args.next().is_some() {
                return Some(Err(
                    "run-causal-bank-ngram-bulk-from-table takes <checkpoint-path|bundle-dir> <summary.json> <data-root> <artifact-path> [val_tokens] [report_every]"
                        .to_string(),
                ));
            }
            Some(resolve_selected_path(&data_root).and_then(|selected_path| {
                let report = run_token_conker3_ngram_bulk_from_table_artifact(
                    &checkpoint_path,
                    &summary_path,
                    Path::new(&selected_path),
                    Path::new(&artifact_path),
                    val_tokens,
                    report_every,
                )?;
                print!("{}", render_token_conker3_ngram_bulk_report(&report));
                Ok(())
            }))
        }
        "run-causal-bank-oracle-table-bulk" | "run-token-conker3-oracle-table-bulk" => {
            let checkpoint_path = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "run-causal-bank-oracle-table-bulk requires a checkpoint path".to_string(),
                    ));
                }
            };
            let summary_path = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "run-causal-bank-oracle-table-bulk requires a summary path".to_string(),
                    ));
                }
            };
            let data_root = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "run-causal-bank-oracle-table-bulk requires a data root".to_string(),
                    ));
                }
            };
            let train_tokens = match parse_usize_flag(args.next(), "train_tokens", 1_000_000) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            let val_tokens = match parse_usize_flag(args.next(), "val_tokens", 10_000) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            let report_every = match parse_usize_flag(args.next(), "report_every", 1_000) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            let profile = args.next().unwrap_or_else(|| "tiny".to_string());
            let oracle_radius = match parse_usize_flag(args.next(), "oracle_radius", 2) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            let oracle_stride = match parse_usize_flag(args.next(), "oracle_stride", 1) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            if args.next().is_some() {
                return Some(Err(
                    "run-causal-bank-oracle-table-bulk takes <checkpoint-path|bundle-dir> <summary.json> <data-root> [train_tokens] [val_tokens] [report_every] [profile] [oracle_radius] [oracle_stride]"
                        .to_string(),
                ));
            }
            Some(resolve_selected_path(&data_root).and_then(|selected_path| {
                let report = run_token_conker3_oracle_table_bulk_from_data_root(
                    &checkpoint_path,
                    &summary_path,
                    Path::new(&selected_path),
                    train_tokens,
                    val_tokens,
                    report_every,
                    &profile,
                    oracle_radius,
                    oracle_stride,
                )?;
                print!("{}", render_token_conker3_ngram_bulk_report(&report));
                Ok(())
            }))
        }
        "audit-causal-bank-ngram-checkpoint" | "audit-token-conker3-ngram-checkpoint" => {
            let checkpoint_path = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "audit-causal-bank-ngram-checkpoint requires a checkpoint path".to_string(),
                    ));
                }
            };
            let summary_path = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "audit-causal-bank-ngram-checkpoint requires a summary path".to_string(),
                    ));
                }
            };
            let data_root = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "audit-causal-bank-ngram-checkpoint requires a data root".to_string(),
                    ));
                }
            };
            let val_tokens = match parse_usize_flag(args.next(), "val_tokens", 16_384) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            let chunk_size = match parse_usize_flag(args.next(), "chunk_size", 64) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            let max_chunks = match parse_usize_flag(args.next(), "max_chunks", 8) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            if args.next().is_some() {
                return Some(Err(
                    "audit-causal-bank-ngram-checkpoint takes <checkpoint-path|bundle-dir> <summary.json> <data-root> [val_tokens] [chunk_size] [max_chunks]"
                        .to_string(),
                ));
            }
            Some(resolve_selected_path(&data_root).and_then(|selected_path| {
                let loaded = load_token_conker3_ngram_checkpoint_from_data_root(
                    &checkpoint_path,
                    &summary_path,
                    Path::new(&selected_path),
                    val_tokens,
                )?;
                print!("{}", render_token_conker3_ngram_bundle(&loaded));
                print_runner_audit(
                    "causal_bank_ngram_checkpoint_audit",
                    loaded.runner(),
                    loaded.eval_tokens(),
                    chunk_size,
                    max_chunks,
                )
            }))
        }
        _ => None,
    };
    result
}
