use std::path::Path;

use chronohorn::archive::packed_memory::{PackedMemoryRunner, compare_tables, render_table_diffs};
use chronohorn::core::audit::audit_parameter_golf;
use chronohorn::core::bridge::{bridge_doctrine, render_bridge_doctrine};
use chronohorn::core::checkpoint::{inspect_npz, render_entries};
use chronohorn::core::data::{data_home_report, resolve_data_root};
use chronohorn::core::demo::{DemoMode, PackedCacheDemo};
use chronohorn::core::oracle::{load_oracle_attack, render_oracle_clean_summary};
use chronohorn::core::protocol::Runner;

use crate::shell_support::{
    indent_block, infer_data_root_from_summary, load_summary, parse_usize_flag,
};

pub(crate) fn handle(
    command: &str,
    args: &mut dyn Iterator<Item = String>,
    parallel_runtime_report: &str,
) -> Option<Result<(), String>> {
    let result = match command {
        "print-parallel-runtime" => {
            if args.next().is_some() {
                return Some(Err("print-parallel-runtime takes no arguments".to_string()));
            }
            print!("{parallel_runtime_report}");
            Some(Ok(()))
        }
        "inspect-npz" => {
            let path = match args.next() {
                Some(path) => path,
                None => return Some(Err("inspect-npz requires a checkpoint path".to_string())),
            };
            if args.next().is_some() {
                return Some(Err("inspect-npz takes exactly one path".to_string()));
            }
            Some(inspect_npz(&path).map(|entries| print!("{}", render_entries(&entries))))
        }
        "inspect-data-root" => {
            let data_root = args.next();
            if args.next().is_some() {
                return Some(Err(
                    "inspect-data-root takes at most one path or alias".to_string()
                ));
            }
            Some(resolve_data_root(data_root.as_deref()).and_then(|report| {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&report)
                        .map_err(|err| format!("serialize data root resolution: {err}"))?
                );
                Ok(())
            }))
        }
        "print-data-home" => {
            if args.next().is_some() {
                return Some(Err("print-data-home takes no arguments".to_string()));
            }
            let report = data_home_report();
            Some(
                serde_json::to_string_pretty(&report)
                    .map_err(|err| format!("serialize data home report: {err}"))
                    .map(|json| println!("{json}")),
            )
        }
        "audit-demo" => {
            let mode_raw = match args.next() {
                Some(mode) => mode,
                None => return Some(Err("audit-demo requires a mode".to_string())),
            };
            if args.next().is_some() {
                return Some(Err("audit-demo takes exactly one mode".to_string()));
            }
            Some(DemoMode::parse(&mode_raw).and_then(|mode| {
                let runner = PackedCacheDemo::new(8, 1.0, mode);
                let tokens = vec![0, 1, 0, 1, 2, 3, 1, 0, 2, 2, 1, 3, 0, 1, 2, 3];
                let report = audit_parameter_golf(&runner, &tokens, 8, 2)?;
                println!("mode: {}", mode.as_str());
                print!("{}", report.render(runner.name()));
                Ok(())
            }))
        }
        "audit-packed-memory" => {
            let data_root = match args.next() {
                Some(root) => root,
                None => return Some(Err("audit-packed-memory requires a data root".to_string())),
            };
            let token_budget = match parse_usize_flag(args.next(), "token_budget", 200_000) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            let trigram_buckets = match parse_usize_flag(args.next(), "trigram_buckets", 2_048) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            Some(
                PackedMemoryRunner::from_data_root(
                    Path::new(&data_root),
                    1024,
                    token_budget,
                    trigram_buckets,
                    4.0,
                    2.0,
                )
                .and_then(|(runner, val_tokens)| {
                    let report = audit_parameter_golf(&runner, &val_tokens, 64, 4)?;
                    println!("data_root: {data_root}");
                    println!("token_budget: {token_budget}");
                    println!("trigram_buckets: {trigram_buckets}");
                    print!("{}", report.render(runner.name()));
                    Ok(())
                }),
            )
        }
        "compare-packed-memory" => {
            let checkpoint_path = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "compare-packed-memory requires a checkpoint path".to_string()
                    ));
                }
            };
            let summary_path = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "compare-packed-memory requires a summary path".to_string()
                    ));
                }
            };
            let data_root = match args.next() {
                Some(root) => root,
                None => match infer_data_root_from_summary(&summary_path) {
                    Ok(root) => root,
                    Err(err) => return Some(Err(err)),
                },
            };
            if args.next().is_some() {
                return Some(Err(
                    "compare-packed-memory takes <checkpoint-path|bundle-dir> <summary.json> [data-root]"
                        .to_string(),
                ));
            }
            Some(load_summary(&summary_path).and_then(|cfg| {
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
                let checkpoint_report =
                    audit_parameter_golf(&checkpoint_runner, &val_tokens, 64, 4)?;
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
            }))
        }
        "oracle-clean-summary" => {
            let attack_path = match args.next() {
                Some(path) => path,
                None => {
                    return Some(Err(
                        "oracle-clean-summary requires an attack JSON path".to_string()
                    ));
                }
            };
            let top_n = match parse_usize_flag(args.next(), "top_n", 8) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            if args.next().is_some() {
                return Some(Err(
                    "oracle-clean-summary takes <attack.json> [top-n]".to_string()
                ));
            }
            Some(load_oracle_attack(&attack_path).map(|corpus| {
                print!("{}", render_oracle_clean_summary(&corpus, top_n));
            }))
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
            Some(Ok(()))
        }
        "doctrine" => {
            print!("{}", render_bridge_doctrine());
            Some(Ok(()))
        }
        "doctrine-json" => Some(
            serde_json::to_string_pretty(&bridge_doctrine())
                .map_err(|err| format!("serialize doctrine: {err}"))
                .map(|json| println!("{json}")),
        ),
        _ => None,
    };
    result
}
