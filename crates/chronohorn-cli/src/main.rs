use std::env;

use chronohorn::causal_bank::checkpoint::{
    probe_token_conker3_export_bundle, render_token_conker3_export_bundle_probe_report,
};
use chronohorn_runtime::{
    ChronohornExportInspectRequest, ChronohornReplayPrepRequest,
    ChronohornTensorInventoryInspectRequest, ChronohornTensorProbeVerifyRequest,
    inspect_export_bundle, inspect_manifest_json, inspect_tensor_inventory, prepare_replay_bundle,
    verify_tensor_probe,
};

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
    if matches!(command.as_str(), "-h" | "--help") {
        print_usage();
        return Ok(());
    }

    match command.as_str() {
        "abi" | "schema" => {
            if args.next().is_some() {
                return Err("abi takes no arguments".to_string());
            }
            print_schema();
            Ok(())
        }
        "inspect-export" => {
            let export_root = args
                .next()
                .ok_or_else(|| "inspect-export requires an export root".to_string())?;
            if args.next().is_some() {
                return Err("inspect-export takes exactly one export root".to_string());
            }
            let report = inspect_export_bundle(&ChronohornExportInspectRequest {
                export_root: export_root.into(),
            })?;
            print!("{}", report.render());
            Ok(())
        }
        "inspect-manifest" => {
            let manifest_path = args
                .next()
                .ok_or_else(|| "inspect-manifest requires a manifest path".to_string())?;
            if args.next().is_some() {
                return Err("inspect-manifest takes exactly one manifest path".to_string());
            }
            let report = inspect_manifest_json(manifest_path)?;
            print!("{}", report.render());
            Ok(())
        }
        "inspect-inventory" => {
            let export_root = args
                .next()
                .ok_or_else(|| "inspect-inventory requires an export root".to_string())?;
            if args.next().is_some() {
                return Err("inspect-inventory takes exactly one export root".to_string());
            }
            let report = inspect_tensor_inventory(&ChronohornTensorInventoryInspectRequest {
                export_root: export_root.into(),
            })?;
            print!("{}", report.render());
            Ok(())
        }
        "prepare-replay" => {
            let export_root = args
                .next()
                .ok_or_else(|| "prepare-replay requires an export root".to_string())?;
            if args.next().is_some() {
                return Err("prepare-replay takes exactly one export root".to_string());
            }
            let report = prepare_replay_bundle(&ChronohornReplayPrepRequest {
                export_root: export_root.into(),
            })?;
            print!("{}", report.render());
            Ok(())
        }
        "verify-probe" => {
            let export_root = args
                .next()
                .ok_or_else(|| "verify-probe requires an export root".to_string())?;
            if args.next().is_some() {
                return Err("verify-probe takes exactly one export root".to_string());
            }
            let report = verify_tensor_probe(&ChronohornTensorProbeVerifyRequest {
                export_root: export_root.into(),
            })?;
            print!("{}", report.render());
            Ok(())
        }
        "probe-causal-bank-export-bundle" => {
            let export_root = args.next().ok_or_else(|| {
                "probe-causal-bank-export-bundle requires an export root".to_string()
            })?;
            if args.next().is_some() {
                return Err(
                    "probe-causal-bank-export-bundle takes exactly one export root".to_string(),
                );
            }
            let report = probe_token_conker3_export_bundle(&export_root)?;
            print!(
                "{}",
                render_token_conker3_export_bundle_probe_report(&report)
            );
            Ok(())
        }
        _ => Err(format!("unknown command: {command}")),
    }
}

fn print_usage() {
    println!("chronohorn-cli");
    println!();
    println!("Usage:");
    println!("  chronohorn-cli abi");
    println!("  chronohorn-cli inspect-export <export-root>");
    println!("  chronohorn-cli inspect-manifest <manifest.json>");
    println!("  chronohorn-cli inspect-inventory <export-root>");
    println!("  chronohorn-cli prepare-replay <export-root>");
    println!("  chronohorn-cli verify-probe <export-root>");
    println!("  chronohorn-cli probe-causal-bank-export-bundle <export-root>");
    println!();
    println!("This CLI is the Chronohorn Rust surface for export bundle inspection.");
}

fn print_schema() {
    println!(
        "abi_name: {}",
        chronohorn_schema::CHRONOHORN_EXPORT_ABI_NAME
    );
    println!(
        "abi_version: {}",
        chronohorn_schema::CHRONOHORN_EXPORT_ABI_VERSION
    );
}
