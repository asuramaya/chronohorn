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

#[derive(Debug, Clone)]
pub struct ByteBridgeCodecReport {
    pub radius: usize,
    pub train_file_count: usize,
    pub tune_file_count: usize,
    pub eval_file_count: usize,
    pub train_samples: usize,
    pub tune_positions: usize,
    pub eval_positions: usize,
    pub tuned_bridge_lambda: f64,
    pub tuned_heuristic_lambda: f64,
    pub tuned_direct_lambda: f64,
    pub tune_bpb_base: f64,
    pub tune_bpb_bridge: f64,
    pub tune_bpb_heuristic: f64,
    pub tune_bpb_direct: f64,
    pub eval_bpb_base: f64,
    pub eval_bpb_bridge: f64,
    pub eval_bpb_heuristic: f64,
    pub eval_bpb_direct: f64,
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

pub fn run_byte_bridge_codec(
    workspace_root: &Path,
    radius: usize,
    stride: usize,
    max_files: usize,
) -> Result<ByteBridgeCodecReport, String> {
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
    if files.len() < 12 {
        return Err(format!("need at least 12 files, found {}", files.len()));
    }

    let mut train_files = Vec::new();
    let mut tune_files = Vec::new();
    let mut eval_files = Vec::new();
    for (index, file) in files.into_iter().enumerate() {
        match index % 5 {
            0 => eval_files.push(file),
            1 => tune_files.push(file),
            _ => train_files.push(file),
        }
    }
    if train_files.is_empty() || tune_files.is_empty() || eval_files.is_empty() {
        return Err("need non-empty train/tune/eval file splits".to_string());
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
    if train_samples.is_empty() {
        return Err("bridge training samples came back empty".to_string());
    }
    let standardizer = fit_standardizer(&train_samples);
    let train_scaled = apply_standardizer(&train_samples, &standardizer);
    let model = train_logistic(&train_scaled, 80, 0.35, 1e-4);

    let train_codec_records = collect_codec_records(
        &train_files,
        radius,
        &train_left,
        true,
        &standardizer,
        &model,
    );
    let direct_model = train_compression_gate(&train_codec_records, 80, 0.2, 1e-4);

    let tune_records = collect_codec_records(
        &tune_files,
        radius,
        &train_left,
        false,
        &standardizer,
        &model,
    );
    let eval_records = collect_codec_records(
        &eval_files,
        radius,
        &train_left,
        false,
        &standardizer,
        &model,
    );
    if tune_records.is_empty() || eval_records.is_empty() {
        return Err("codec records came back empty".to_string());
    }

    let tuned_bridge_lambda = tune_lambda(&tune_records, |record| record.bridge_gate);
    let tuned_heuristic_lambda = tune_lambda(&tune_records, |record| record.heuristic_gate);
    let tuned_direct_lambda = tune_lambda(&tune_records, |record| {
        predict_probability(&direct_model, &record.features)
    });

    Ok(ByteBridgeCodecReport {
        radius,
        train_file_count: train_files.len(),
        tune_file_count: tune_files.len(),
        eval_file_count: eval_files.len(),
        train_samples: train_scaled.len(),
        tune_positions: tune_records.len(),
        eval_positions: eval_records.len(),
        tuned_bridge_lambda,
        tuned_heuristic_lambda,
        tuned_direct_lambda,
        tune_bpb_base: mean_bits_per_byte(&tune_records, None),
        tune_bpb_bridge: mean_bits_per_byte(
            &tune_records,
            Some((tuned_bridge_lambda, GateKind::Bridge)),
        ),
        tune_bpb_heuristic: mean_bits_per_byte(
            &tune_records,
            Some((tuned_heuristic_lambda, GateKind::Heuristic)),
        ),
        tune_bpb_direct: mean_bits_per_byte_with_gate(
            &tune_records,
            tuned_direct_lambda,
            |record| predict_probability(&direct_model, &record.features),
        ),
        eval_bpb_base: mean_bits_per_byte(&eval_records, None),
        eval_bpb_bridge: mean_bits_per_byte(
            &eval_records,
            Some((tuned_bridge_lambda, GateKind::Bridge)),
        ),
        eval_bpb_heuristic: mean_bits_per_byte(
            &eval_records,
            Some((tuned_heuristic_lambda, GateKind::Heuristic)),
        ),
        eval_bpb_direct: mean_bits_per_byte_with_gate(
            &eval_records,
            tuned_direct_lambda,
            |record| predict_probability(&direct_model, &record.features),
        ),
    })
}

pub fn render_byte_bridge_codec_report(report: &ByteBridgeCodecReport) -> String {
    let mut out = String::new();
    out.push_str("chronohorn_byte_bridge_codec\n");
    out.push_str(&format!("radius: {}\n", report.radius));
    out.push_str(&format!("train_file_count: {}\n", report.train_file_count));
    out.push_str(&format!("tune_file_count: {}\n", report.tune_file_count));
    out.push_str(&format!("eval_file_count: {}\n", report.eval_file_count));
    out.push_str(&format!("train_samples: {}\n", report.train_samples));
    out.push_str(&format!("tune_positions: {}\n", report.tune_positions));
    out.push_str(&format!("eval_positions: {}\n", report.eval_positions));
    out.push_str(&format!(
        "tuned_bridge_lambda: {:.3}\n",
        report.tuned_bridge_lambda
    ));
    out.push_str(&format!(
        "tuned_heuristic_lambda: {:.3}\n",
        report.tuned_heuristic_lambda
    ));
    out.push_str(&format!(
        "tuned_direct_lambda: {:.3}\n",
        report.tuned_direct_lambda
    ));
    out.push_str(&format!("tune_bpb_base: {:.6}\n", report.tune_bpb_base));
    out.push_str(&format!("tune_bpb_bridge: {:.6}\n", report.tune_bpb_bridge));
    out.push_str(&format!(
        "tune_bpb_heuristic: {:.6}\n",
        report.tune_bpb_heuristic
    ));
    out.push_str(&format!("tune_bpb_direct: {:.6}\n", report.tune_bpb_direct));
    out.push_str(&format!("eval_bpb_base: {:.6}\n", report.eval_bpb_base));
    out.push_str(&format!("eval_bpb_bridge: {:.6}\n", report.eval_bpb_bridge));
    out.push_str(&format!(
        "eval_bpb_heuristic: {:.6}\n",
        report.eval_bpb_heuristic
    ));
    out.push_str(&format!("eval_bpb_direct: {:.6}\n", report.eval_bpb_direct));
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

#[derive(Debug, Clone, Copy)]
struct CodecRecord {
    base_gold_prob: f64,
    top4_gold_prob: f64,
    bridge_gate: f64,
    heuristic_gate: f64,
    features: [f64; FEATURE_DIM],
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

fn collect_codec_records(
    files: &[FileBytes],
    radius: usize,
    left_global: &ContextMap,
    exclude_local: bool,
    standardizer: &Standardizer,
    model: &LogisticModel,
) -> Vec<CodecRecord> {
    let mut rows = Vec::new();
    for file in files {
        let local_left = if exclude_local {
            Some(build_context_map_for_file(file, radius, false))
        } else {
            None
        };
        let bytes = &file.bytes;
        if bytes.len() < 2 * radius + 1 {
            continue;
        }
        for pos in radius..(bytes.len() - radius) {
            let left_context = bytes[pos - radius..pos].to_vec();
            let local_row = local_left.as_ref().and_then(|map| map.get(&left_context));
            let stats = support_stats(left_global.get(&left_context), local_row);
            let features = standardize_features(
                [
                    stats.log_total,
                    stats.support_fraction,
                    stats.top1_prob,
                    stats.top4_mass,
                    stats.normalized_entropy,
                    stats.margin,
                ],
                standardizer,
            );
            let bridge_gate = predict_probability(model, &features);
            let gold = bytes[pos];
            let (base_gold_prob, top4_gold_prob) =
                left_distribution_probs(left_global.get(&left_context), local_row, gold);
            rows.push(CodecRecord {
                base_gold_prob,
                top4_gold_prob,
                bridge_gate,
                heuristic_gate: stats.top4_mass,
                features,
            });
        }
    }
    rows
}

fn standardize_features(
    features: [f64; FEATURE_DIM],
    standardizer: &Standardizer,
) -> [f64; FEATURE_DIM] {
    let mut out = [0.0; FEATURE_DIM];
    for index in 0..FEATURE_DIM {
        out[index] = (features[index] - standardizer.mean[index]) / standardizer.std[index];
    }
    out
}

fn left_distribution_probs(
    global: Option<&Support>,
    local: Option<&Support>,
    gold: u8,
) -> (f64, f64) {
    let alpha = 1.0;
    let vocab = 256.0;
    let mut counts = vec![0.0; 256];
    let mut total = 0.0;
    if let Some(support) = global {
        for (&token, &count) in support {
            let local_count = local.and_then(|row| row.get(&token)).copied().unwrap_or(0);
            let remaining = count.saturating_sub(local_count) as f64;
            counts[token as usize] = remaining;
            total += remaining;
        }
    }
    let denom = total + alpha * vocab;
    let base_gold_prob = (counts[gold as usize] + alpha) / denom;

    let mut indexed = counts
        .iter()
        .enumerate()
        .map(|(index, &count)| (index, count + alpha))
        .collect::<Vec<_>>();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top4 = indexed.into_iter().take(4).collect::<Vec<_>>();
    let top4_total = top4.iter().map(|(_, value)| *value).sum::<f64>().max(1e-12);
    let top4_gold_prob = top4
        .iter()
        .find(|(index, _)| *index == gold as usize)
        .map(|(_, value)| *value / top4_total)
        .unwrap_or(0.0);

    (base_gold_prob.max(1e-12), top4_gold_prob.max(0.0))
}

#[derive(Debug, Clone, Copy)]
enum GateKind {
    Bridge,
    Heuristic,
}

fn tune_lambda(records: &[CodecRecord], gate_fn: impl Fn(&CodecRecord) -> f64) -> f64 {
    let mut best_lambda = 0.0;
    let mut best_bpb = f64::INFINITY;
    for step in 0..=12 {
        let lambda = step as f64 / 10.0;
        let bpb = mean_bits_per_byte_with_gate(records, lambda, &gate_fn);
        if bpb < best_bpb {
            best_bpb = bpb;
            best_lambda = lambda;
        }
    }
    best_lambda
}

fn mean_bits_per_byte(records: &[CodecRecord], gate: Option<(f64, GateKind)>) -> f64 {
    match gate {
        None => {
            let mut bits = 0.0;
            for record in records {
                bits += -record.base_gold_prob.max(1e-12).log2();
            }
            bits / records.len() as f64
        }
        Some((lambda, GateKind::Bridge)) => {
            mean_bits_per_byte_with_gate(records, lambda, |record| record.bridge_gate)
        }
        Some((lambda, GateKind::Heuristic)) => {
            mean_bits_per_byte_with_gate(records, lambda, |record| record.heuristic_gate)
        }
    }
}

fn mean_bits_per_byte_with_gate(
    records: &[CodecRecord],
    lambda: f64,
    gate_fn: impl Fn(&CodecRecord) -> f64,
) -> f64 {
    let mut bits = 0.0;
    for record in records {
        let gate = (lambda * gate_fn(record)).clamp(0.0, 1.0);
        let mixed = (1.0 - gate) * record.base_gold_prob + gate * record.top4_gold_prob.max(1e-12);
        bits += -mixed.max(1e-12).log2();
    }
    bits / records.len() as f64
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

fn train_compression_gate(
    records: &[CodecRecord],
    epochs: usize,
    lr: f64,
    l2: f64,
) -> LogisticModel {
    let mut model = LogisticModel {
        weights: [0.0; FEATURE_DIM],
        bias: 0.0,
    };
    let ln2 = std::f64::consts::LN_2;
    for _ in 0..epochs {
        let mut grad_w = [0.0; FEATURE_DIM];
        let mut grad_b = 0.0;
        for record in records {
            let gate = predict_probability(&model, &record.features).clamp(1e-6, 1.0 - 1e-6);
            let mixed =
                ((1.0 - gate) * record.base_gold_prob + gate * record.top4_gold_prob).max(1e-12);
            let dloss_dgate = (record.base_gold_prob - record.top4_gold_prob) / (mixed * ln2);
            let dloss_dz = dloss_dgate * gate * (1.0 - gate);
            for (index, value) in record.features.iter().enumerate() {
                grad_w[index] += dloss_dz * *value;
            }
            grad_b += dloss_dz;
        }
        let inv_n = 1.0 / records.len() as f64;
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
