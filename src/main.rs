use std::env;
use std::fs;
use std::path::Path;

use chronohorn::audit::audit_parameter_golf;
use chronohorn::checkpoint::{inspect_npz, render_entries};
use chronohorn::demo::{DemoMode, PackedCacheDemo};
use chronohorn::oracle::{load_oracle_attack, render_oracle_clean_summary};
use chronohorn::packed_memory::{PackedMemoryRunner, compare_tables, render_table_diffs};
use chronohorn::protocol::Runner;
use serde::Deserialize;

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
            println!("  gold logprob consistency");
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
    println!(
        "  chronohorn audit-demo <legal|self-include|future-peek|length-peek|reported-gold-cheat>"
    );
    println!("  chronohorn audit-packed-memory <data-root> [token-budget] [trigram-buckets]");
    println!("  chronohorn compare-packed-memory <checkpoint.npz> <summary.json> [data-root]");
    println!("  chronohorn oracle-clean-summary <attack.json> [top-n]");
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
