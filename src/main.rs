use std::env;

mod shell_archive;
mod shell_causal_bank;
mod shell_core;
mod shell_support;
mod shell_usage;

use chronohorn::core::runtime::{configure_parallel_runtime, render_parallel_runtime_report};

use crate::shell_usage::print_usage;

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let Some(command_raw) = args.next() else {
        print_usage();
        return Ok(());
    };
    if command_raw == "--help" || command_raw == "-h" || command_raw == "help" {
        print_usage();
        return Ok(());
    }
    let command = command_raw.as_str();
    let parallel_runtime_report = render_parallel_runtime_report(configure_parallel_runtime()?);
    if let Some(result) = shell_causal_bank::handle(command, &mut args) {
        return result;
    }
    if let Some(result) = shell_core::handle(command, &mut args, &parallel_runtime_report) {
        return result;
    }
    if let Some(result) = shell_archive::handle(command, &mut args) {
        return result;
    }
    Err(format!("unknown command: {command}"))
}
