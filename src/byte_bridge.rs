use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

const DEFAULT_SCAN_ROOTS: [&str; 4] = [
    "blinx/README.md",
    "blinx/conker/docs",
    "blinx/conker/src",
    "blinx/conker/scripts",
];
const FEATURE_DIM: usize = 6;

type Support = HashMap<u8, u32>;
type ContextMap = HashMap<Vec<u8>, Support>;

#[derive(Debug, Clone)]
struct FileBytes {
    bytes: Vec<u8>,
}

#[derive(Debug, Clone)]
struct Sample {
    features: [f64; FEATURE_DIM],
    label: f64,
}

#[derive(Debug, Clone)]
struct Standardizer {
    mean: [f64; FEATURE_DIM],
    std: [f64; FEATURE_DIM],
}

#[derive(Debug, Clone)]
struct LogisticModel {
    weights: [f64; FEATURE_DIM],
    bias: f64,
}

#[derive(Debug, Clone)]
pub struct ByteBridgeReport {
    pub radius: usize,
    pub stride: usize,
    pub max_files: usize,
    pub train_file_count: usize,
    pub eval_file_count: usize,
    pub train_samples: usize,
    pub eval_samples: usize,
    pub train_positive_rate: f64,
    pub eval_positive_rate: f64,
    pub train_loss: f64,
    pub eval_loss: f64,
    pub train_accuracy: f64,
    pub eval_accuracy: f64,
    pub train_precision: f64,
    pub eval_precision: f64,
    pub train_recall: f64,
    pub eval_recall: f64,
    pub majority_eval_accuracy: f64,
    pub heuristic_eval_accuracy: f64,
}

pub fn run_byte_bridge(
    workspace_root: &Path,
    radius: usize,
    stride: usize,
    max_files: usize,
) -> Result<ByteBridgeReport, String> {
    if radius == 0 {
        return Err("radius must be positive".to_string());
    }
    if stride == 0 {
        return Err("stride must be positive".to_string());
    }
    let mut files = collect_default_files(workspace_root)?;
    if max_files > 0 && files.len() > max_files {
        files.truncate(max_files);
    }
    if files.len() < 8 {
        return Err(format!("need at least 8 files, found {}", files.len()));
    }

    let mut train_files = Vec::new();
    let mut eval_files = Vec::new();
    for (index, file) in files.into_iter().enumerate() {
        if index % 5 == 0 {
            eval_files.push(file);
        } else {
            train_files.push(file);
        }
    }
    if train_files.is_empty() || eval_files.is_empty() {
        return Err("need both train and eval files".to_string());
    }

    let train_bidi = build_context_map(&train_files, radius, true);
    let train_left = build_context_map(&train_files, radius, false);

    let mut train_samples = Vec::new();
    for file in &train_files {
        let local_bidi = build_context_map_for_file(file, radius, true);
        let local_left = build_context_map_for_file(file, radius, false);
        train_samples.extend(samples_for_file(
            file,
            radius,
            stride,
            &train_bidi,
            Some(&local_bidi),
            &train_left,
            Some(&local_left),
        ));
    }
    let mut eval_samples = Vec::new();
    for file in &eval_files {
        eval_samples.extend(samples_for_file(
            file,
            radius,
            stride,
            &train_bidi,
            None,
            &train_left,
            None,
        ));
    }
    if train_samples.is_empty() || eval_samples.is_empty() {
        return Err("bridge dataset came back empty".to_string());
    }

    let standardizer = fit_standardizer(&train_samples);
    let train_scaled = apply_standardizer(&train_samples, &standardizer);
    let eval_scaled = apply_standardizer(&eval_samples, &standardizer);

    let model = train_logistic(&train_scaled, 80, 0.35, 1e-4);
    let train_metrics = evaluate(&model, &train_scaled);
    let eval_metrics = evaluate(&model, &eval_scaled);

    let eval_positive_rate = positive_rate(&eval_scaled);
    let majority_eval_accuracy = majority_baseline_accuracy(&eval_scaled);
    let heuristic_eval_accuracy = heuristic_top4_mass_accuracy(&eval_samples);

    Ok(ByteBridgeReport {
        radius,
        stride,
        max_files,
        train_file_count: train_files.len(),
        eval_file_count: eval_files.len(),
        train_samples: train_scaled.len(),
        eval_samples: eval_scaled.len(),
        train_positive_rate: positive_rate(&train_scaled),
        eval_positive_rate,
        train_loss: train_metrics.loss,
        eval_loss: eval_metrics.loss,
        train_accuracy: train_metrics.accuracy,
        eval_accuracy: eval_metrics.accuracy,
        train_precision: train_metrics.precision,
        eval_precision: eval_metrics.precision,
        train_recall: train_metrics.recall,
        eval_recall: eval_metrics.recall,
        majority_eval_accuracy,
        heuristic_eval_accuracy,
    })
}

pub fn render_byte_bridge_report(report: &ByteBridgeReport) -> String {
    let mut out = String::new();
    out.push_str("chronohorn_byte_bridge\n");
    out.push_str(&format!("radius: {}\n", report.radius));
    out.push_str(&format!("stride: {}\n", report.stride));
    out.push_str(&format!("max_files: {}\n", report.max_files));
    out.push_str(&format!("train_file_count: {}\n", report.train_file_count));
    out.push_str(&format!("eval_file_count: {}\n", report.eval_file_count));
    out.push_str(&format!("train_samples: {}\n", report.train_samples));
    out.push_str(&format!("eval_samples: {}\n", report.eval_samples));
    out.push_str(&format!(
        "train_positive_rate: {:.6}\n",
        report.train_positive_rate
    ));
    out.push_str(&format!(
        "eval_positive_rate: {:.6}\n",
        report.eval_positive_rate
    ));
    out.push_str(&format!("train_loss: {:.6}\n", report.train_loss));
    out.push_str(&format!("eval_loss: {:.6}\n", report.eval_loss));
    out.push_str(&format!("train_accuracy: {:.6}\n", report.train_accuracy));
    out.push_str(&format!("eval_accuracy: {:.6}\n", report.eval_accuracy));
    out.push_str(&format!("train_precision: {:.6}\n", report.train_precision));
    out.push_str(&format!("eval_precision: {:.6}\n", report.eval_precision));
    out.push_str(&format!("train_recall: {:.6}\n", report.train_recall));
    out.push_str(&format!("eval_recall: {:.6}\n", report.eval_recall));
    out.push_str(&format!(
        "majority_eval_accuracy: {:.6}\n",
        report.majority_eval_accuracy
    ));
    out.push_str(&format!(
        "heuristic_eval_accuracy: {:.6}\n",
        report.heuristic_eval_accuracy
    ));
    out
}

fn collect_default_files(workspace_root: &Path) -> Result<Vec<FileBytes>, String> {
    let mut paths = Vec::new();
    for root in DEFAULT_SCAN_ROOTS {
        let absolute = workspace_root.join(root);
        collect_files_under(&absolute, &mut paths)?;
    }
    paths.sort();
    paths.dedup();
    let mut files = Vec::new();
    for path in paths {
        let bytes = fs::read(&path).map_err(|err| format!("read {}: {err}", path.display()))?;
        if bytes.len() < 32 {
            continue;
        }
        files.push(FileBytes { bytes });
    }
    Ok(files)
}

fn collect_files_under(path: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    if !path.exists() {
        return Ok(());
    }
    if path.is_file() {
        out.push(path.to_path_buf());
        return Ok(());
    }
    for entry in fs::read_dir(path).map_err(|err| format!("read_dir {}: {err}", path.display()))? {
        let entry = entry.map_err(|err| format!("read_dir entry {}: {err}", path.display()))?;
        let child = entry.path();
        if child
            .file_name()
            .and_then(|name| name.to_str())
            .map(|name| name == "__pycache__" || name == ".git")
            .unwrap_or(false)
        {
            continue;
        }
        if child.as_os_str().to_string_lossy().contains("/out/") {
            continue;
        }
        if child.is_dir() {
            collect_files_under(&child, out)?;
        } else {
            out.push(child);
        }
    }
    Ok(())
}

fn build_context_map(files: &[FileBytes], radius: usize, bidirectional: bool) -> ContextMap {
    let mut map = HashMap::new();
    for file in files {
        add_file_contexts(&mut map, &file.bytes, radius, bidirectional);
    }
    map
}

fn build_context_map_for_file(file: &FileBytes, radius: usize, bidirectional: bool) -> ContextMap {
    let mut map = HashMap::new();
    add_file_contexts(&mut map, &file.bytes, radius, bidirectional);
    map
}

fn add_file_contexts(map: &mut ContextMap, bytes: &[u8], radius: usize, bidirectional: bool) {
    if bytes.len() < 2 * radius + 1 {
        return;
    }
    for pos in radius..(bytes.len() - radius) {
        let context = if bidirectional {
            let mut ctx = Vec::with_capacity(radius * 2);
            ctx.extend_from_slice(&bytes[pos - radius..pos]);
            ctx.extend_from_slice(&bytes[pos + 1..pos + 1 + radius]);
            ctx
        } else {
            bytes[pos - radius..pos].to_vec()
        };
        *map.entry(context)
            .or_default()
            .entry(bytes[pos])
            .or_insert(0) += 1;
    }
}

fn samples_for_file(
    file: &FileBytes,
    radius: usize,
    stride: usize,
    bidi_global: &ContextMap,
    bidi_local: Option<&ContextMap>,
    left_global: &ContextMap,
    left_local: Option<&ContextMap>,
) -> Vec<Sample> {
    let bytes = &file.bytes;
    if bytes.len() < 2 * radius + 1 {
        return Vec::new();
    }
    let mut out = Vec::new();
    for pos in radius..(bytes.len() - radius) {
        if (pos - radius) % stride != 0 {
            continue;
        }
        let mut bidi_context = Vec::with_capacity(radius * 2);
        bidi_context.extend_from_slice(&bytes[pos - radius..pos]);
        bidi_context.extend_from_slice(&bytes[pos + 1..pos + 1 + radius]);
        let left_context = bytes[pos - radius..pos].to_vec();

        let bidi_stats = support_stats(
            bidi_global.get(&bidi_context),
            bidi_local.and_then(|map| map.get(&bidi_context)),
        );
        let left_stats = support_stats(
            left_global.get(&left_context),
            left_local.and_then(|map| map.get(&left_context)),
        );
        let label = if bidi_stats.support_size > 0 && bidi_stats.support_size <= 4 {
            1.0
        } else {
            0.0
        };
        let features = [
            left_stats.log_total,
            left_stats.support_fraction,
            left_stats.top1_prob,
            left_stats.top4_mass,
            left_stats.normalized_entropy,
            left_stats.margin,
        ];
        out.push(Sample { features, label });
    }
    out
}

#[derive(Debug, Clone, Copy)]
struct SupportStats {
    support_size: usize,
    log_total: f64,
    support_fraction: f64,
    top1_prob: f64,
    top4_mass: f64,
    normalized_entropy: f64,
    margin: f64,
}

fn support_stats(global: Option<&Support>, local: Option<&Support>) -> SupportStats {
    let mut total = 0u32;
    let mut counts = Vec::new();
    if let Some(global_support) = global {
        for (&token, &count) in global_support {
            let local_count = local.and_then(|row| row.get(&token)).copied().unwrap_or(0);
            let remaining = count.saturating_sub(local_count);
            if remaining > 0 {
                total += remaining;
                counts.push(remaining as f64);
            }
        }
    }
    counts.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let total_f = total as f64;
    let top1 = counts.first().copied().unwrap_or(0.0);
    let top2 = counts.get(1).copied().unwrap_or(0.0);
    let top4_mass = if total > 0 {
        counts.iter().take(4).sum::<f64>() / total_f
    } else {
        0.0
    };
    let entropy = if total > 0 {
        let mut h = 0.0;
        for count in &counts {
            let p = *count / total_f;
            h -= p * p.max(f64::MIN_POSITIVE).ln();
        }
        h
    } else {
        0.0
    };
    SupportStats {
        support_size: counts.len(),
        log_total: (1.0 + total_f).ln(),
        support_fraction: counts.len() as f64 / 256.0,
        top1_prob: if total > 0 { top1 / total_f } else { 0.0 },
        top4_mass,
        normalized_entropy: if counts.len() > 1 {
            entropy / (counts.len() as f64).ln().max(f64::MIN_POSITIVE)
        } else {
            0.0
        },
        margin: if total > 0 {
            (top1 - top2) / total_f
        } else {
            0.0
        },
    }
}

fn fit_standardizer(samples: &[Sample]) -> Standardizer {
    let mut mean = [0.0; FEATURE_DIM];
    for sample in samples {
        for (index, value) in sample.features.iter().enumerate() {
            mean[index] += *value;
        }
    }
    for value in &mut mean {
        *value /= samples.len() as f64;
    }
    let mut var = [0.0; FEATURE_DIM];
    for sample in samples {
        for (index, value) in sample.features.iter().enumerate() {
            let diff = *value - mean[index];
            var[index] += diff * diff;
        }
    }
    let mut std = [1.0; FEATURE_DIM];
    for (index, value) in var.iter().enumerate() {
        std[index] = (value / samples.len() as f64).sqrt().max(1e-6);
    }
    Standardizer { mean, std }
}

fn apply_standardizer(samples: &[Sample], standardizer: &Standardizer) -> Vec<Sample> {
    samples
        .iter()
        .map(|sample| {
            let mut features = [0.0; FEATURE_DIM];
            for index in 0..FEATURE_DIM {
                features[index] =
                    (sample.features[index] - standardizer.mean[index]) / standardizer.std[index];
            }
            Sample {
                features,
                label: sample.label,
            }
        })
        .collect()
}

fn train_logistic(samples: &[Sample], epochs: usize, lr: f64, l2: f64) -> LogisticModel {
    let mut model = LogisticModel {
        weights: [0.0; FEATURE_DIM],
        bias: 0.0,
    };
    for _ in 0..epochs {
        let mut grad_w = [0.0; FEATURE_DIM];
        let mut grad_b = 0.0;
        for sample in samples {
            let p = predict_probability(&model, &sample.features);
            let err = p - sample.label;
            for (index, value) in sample.features.iter().enumerate() {
                grad_w[index] += err * *value;
            }
            grad_b += err;
        }
        let inv_n = 1.0 / samples.len() as f64;
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

#[derive(Debug, Clone, Copy)]
struct Metrics {
    loss: f64,
    accuracy: f64,
    precision: f64,
    recall: f64,
}

fn evaluate(model: &LogisticModel, samples: &[Sample]) -> Metrics {
    let mut loss = 0.0;
    let mut correct = 0usize;
    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut fn_ = 0usize;
    for sample in samples {
        let p = predict_probability(model, &sample.features).clamp(1e-8, 1.0 - 1e-8);
        loss += -(sample.label * p.ln() + (1.0 - sample.label) * (1.0 - p).ln());
        let pred = if p >= 0.5 { 1.0 } else { 0.0 };
        if (pred - sample.label).abs() < 1e-9 {
            correct += 1;
        }
        match (pred as i32, sample.label as i32) {
            (1, 1) => tp += 1,
            (1, 0) => fp += 1,
            (0, 1) => fn_ += 1,
            _ => {}
        }
    }
    Metrics {
        loss: loss / samples.len() as f64,
        accuracy: correct as f64 / samples.len() as f64,
        precision: tp as f64 / (tp + fp).max(1) as f64,
        recall: tp as f64 / (tp + fn_).max(1) as f64,
    }
}

fn positive_rate(samples: &[Sample]) -> f64 {
    samples.iter().map(|sample| sample.label).sum::<f64>() / samples.len() as f64
}

fn majority_baseline_accuracy(samples: &[Sample]) -> f64 {
    let positive_rate = positive_rate(samples);
    positive_rate.max(1.0 - positive_rate)
}

fn heuristic_top4_mass_accuracy(samples: &[Sample]) -> f64 {
    let mut correct = 0usize;
    for sample in samples {
        let pred = if sample.features[3] >= 0.9 { 1.0 } else { 0.0 };
        if (pred - sample.label).abs() < 1e-9 {
            correct += 1;
        }
    }
    correct as f64 / samples.len() as f64
}
