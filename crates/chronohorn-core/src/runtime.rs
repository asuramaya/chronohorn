use std::sync::OnceLock;

use rayon::ThreadPoolBuilder;
use serde::Serialize;

pub const CHRONOHORN_THREADS_ENV: &str = "CHRONOHORN_THREADS";

#[derive(Debug, Clone, Serialize)]
pub struct ParallelRuntimeReport {
    pub available_threads: usize,
    pub configured_threads: usize,
    pub env_override: Option<usize>,
}

static PARALLEL_RUNTIME: OnceLock<ParallelRuntimeReport> = OnceLock::new();

pub fn configure_parallel_runtime() -> Result<&'static ParallelRuntimeReport, String> {
    if let Some(report) = PARALLEL_RUNTIME.get() {
        return Ok(report);
    }
    let report = {
        let available_threads = std::thread::available_parallelism()
            .map(|width| width.get())
            .unwrap_or(1);
        let env_override = std::env::var(CHRONOHORN_THREADS_ENV)
            .ok()
            .and_then(|raw| raw.parse::<usize>().ok())
            .filter(|&threads| threads > 0);
        let configured_threads = env_override.unwrap_or(available_threads);
        ThreadPoolBuilder::new()
            .num_threads(configured_threads)
            .build_global()
            .map_err(|err| format!("configure rayon thread pool: {err}"))?;
        ParallelRuntimeReport {
            available_threads,
            configured_threads,
            env_override,
        }
    };
    let _ = PARALLEL_RUNTIME.set(report);
    PARALLEL_RUNTIME
        .get()
        .ok_or_else(|| "parallel runtime initialization failed".to_string())
}

pub fn parallel_runtime_report() -> Option<&'static ParallelRuntimeReport> {
    PARALLEL_RUNTIME.get()
}

pub fn render_parallel_runtime_report(report: &ParallelRuntimeReport) -> String {
    let env_override = report
        .env_override
        .map(|value| value.to_string())
        .unwrap_or_else(|| "none".to_string());
    format!(
        "chronohorn_parallel_runtime\navailable_threads: {}\nconfigured_threads: {}\nenv_override: {}\nenv_var: {}\n",
        report.available_threads, report.configured_threads, env_override, CHRONOHORN_THREADS_ENV,
    )
}
