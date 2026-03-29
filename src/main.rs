use std::env;

use chronohorn::audit::audit_parameter_golf;
use chronohorn::checkpoint::{inspect_npz, render_entries};
use chronohorn::demo::{DemoMode, PackedCacheDemo};
use chronohorn::protocol::Runner;

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
    println!("  chronohorn audit-demo <legal|self-include|future-peek|reported-gold-cheat>");
    println!("  chronohorn design");
}
