use crate::protocol::Runner;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct CheckSummary {
    pub covered: bool,
    pub probe_count: usize,
    pub failure_count: usize,
    pub max_abs_diff: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConstructionClaim {
    pub declared: bool,
    pub basis: &'static str,
}

impl CheckSummary {
    fn empty() -> Self {
        Self {
            covered: false,
            probe_count: 0,
            failure_count: 0,
            max_abs_diff: 0.0,
        }
    }

    fn record(&mut self, passed: bool, max_abs_diff: f64) {
        self.covered = true;
        self.probe_count += 1;
        if !passed {
            self.failure_count += 1;
        }
        self.max_abs_diff = self.max_abs_diff.max(max_abs_diff);
    }

    fn pass_label(&self) -> &'static str {
        if !self.covered {
            "uncovered"
        } else if self.failure_count == 0 {
            "pass"
        } else {
            "fail"
        }
    }
}

impl ConstructionClaim {
    fn declared_only(basis: &'static str) -> Self {
        Self {
            declared: true,
            basis,
        }
    }

    fn status_label(&self) -> &'static str {
        if self.declared {
            "declared-only"
        } else {
            "missing"
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct LegalityReport {
    pub normalization: CheckSummary,
    pub repeatability: CheckSummary,
    pub future_suffix_invariance: CheckSummary,
    pub answer_mask_invariance: CheckSummary,
    pub prefix_truncation_parity: CheckSummary,
    pub stream_rechunk_parity: CheckSummary,
    pub sample_set_invariance: CheckSummary,
    pub gold_logprob_consistency: CheckSummary,
    pub oracle_artifact_runtime_exclusion: ConstructionClaim,
    pub target_side_lookup_absence_at_eval: ConstructionClaim,
    pub oracle_supervision_offline_only: ConstructionClaim,
}

impl LegalityReport {
    pub fn render(&self, runner_name: &str) -> String {
        let mut out = String::new();
        out.push_str(&format!("runner: {runner_name}\n"));
        out.push_str("checks:\n");
        for (name, check) in [
            ("normalization", &self.normalization),
            ("repeatability", &self.repeatability),
            ("future_suffix_invariance", &self.future_suffix_invariance),
            ("answer_mask_invariance", &self.answer_mask_invariance),
            ("prefix_truncation_parity", &self.prefix_truncation_parity),
            ("stream_rechunk_parity", &self.stream_rechunk_parity),
            ("sample_set_invariance", &self.sample_set_invariance),
            ("gold_logprob_consistency", &self.gold_logprob_consistency),
        ] {
            out.push_str(&format!(
                "  {name}: {} (probes={} failures={} max_abs_diff={:.6})\n",
                check.pass_label(),
                check.probe_count,
                check.failure_count,
                check.max_abs_diff
            ));
        }
        out.push_str("obligations:\n");
        out.push_str("  prefix_causal_distribution: partially_covered\n");
        out.push_str("  full_normalized_distribution: partially_covered\n");
        out.push_str(&format!(
            "  score_accounting_independent_of_answer: {}\n",
            if self.gold_logprob_consistency.covered {
                "partially_covered"
            } else {
                "out_of_scope"
            }
        ));
        out.push_str("  no_outcome_selection_across_runs: out_of_scope\n");
        out.push_str("bridge_claims:\n");
        out.push_str(&format!(
            "  oracle_artifact_runtime_exclusion: {} (basis={})\n",
            self.oracle_artifact_runtime_exclusion.status_label(),
            self.oracle_artifact_runtime_exclusion.basis
        ));
        out.push_str(&format!(
            "  target_side_lookup_absence_at_eval: {} (basis={})\n",
            self.target_side_lookup_absence_at_eval.status_label(),
            self.target_side_lookup_absence_at_eval.basis
        ));
        out.push_str(&format!(
            "  oracle_supervision_offline_only: {} (basis={})\n",
            self.oracle_supervision_offline_only.status_label(),
            self.oracle_supervision_offline_only.basis
        ));
        out
    }
}

pub fn audit_parameter_golf<R: Runner>(
    runner: &R,
    tokens: &[usize],
    chunk_size: usize,
    max_chunks: usize,
) -> Result<LegalityReport, String> {
    if chunk_size == 0 {
        return Err("chunk_size must be positive".to_string());
    }
    if tokens.is_empty() {
        return Err("tokens must be non-empty".to_string());
    }

    let mut report = LegalityReport {
        normalization: CheckSummary::empty(),
        repeatability: CheckSummary::empty(),
        future_suffix_invariance: CheckSummary::empty(),
        answer_mask_invariance: CheckSummary::empty(),
        prefix_truncation_parity: CheckSummary::empty(),
        stream_rechunk_parity: CheckSummary::empty(),
        sample_set_invariance: CheckSummary::empty(),
        gold_logprob_consistency: CheckSummary::empty(),
        oracle_artifact_runtime_exclusion: ConstructionClaim::declared_only(
            "declared-only: bridge export keeps oracle artifacts out of the runtime runner path",
        ),
        target_side_lookup_absence_at_eval: ConstructionClaim::declared_only(
            "declared-only: audit eval only exercises local score_chunk/adapt_chunk calls on the runner",
        ),
        oracle_supervision_offline_only: ConstructionClaim::declared_only(
            "declared-only: oracle supervision is separated into offline doctrine and training-time shaping",
        ),
    };

    let mut live_runner = runner.clone();
    let mut chunk_index = 0usize;
    for chunk in tokens.chunks(chunk_size) {
        if chunk_index >= max_chunks {
            break;
        }
        let sample_positions = pick_positions(chunk.len());
        if !sample_positions.is_empty() {
            audit_chunk(&live_runner, chunk, &sample_positions, &mut report)?;
        }
        live_runner.score_chunk(chunk, &[])?;
        chunk_index += 1;
        if chunk_index < max_chunks && chunk_index * chunk_size < tokens.len() {
            live_runner.adapt_chunk(chunk)?;
        }
    }
    let span_len = tokens
        .len()
        .min(chunk_size.saturating_mul(max_chunks.max(1)));
    let alt_chunk_size = alternate_chunk_size(chunk_size, span_len);
    if span_len > 0 && alt_chunk_size != chunk_size {
        let wanted_positions = stream_sample_positions(span_len, chunk_size, max_chunks);
        if !wanted_positions.is_empty() {
            let base_rows =
                score_stream_positions(runner, &tokens[..span_len], chunk_size, &wanted_positions)?;
            let alt_rows = score_stream_positions(
                runner,
                &tokens[..span_len],
                alt_chunk_size,
                &wanted_positions,
            )?;
            let rechunk_check = compare_prediction_sets(&base_rows, &alt_rows);
            report
                .stream_rechunk_parity
                .record(rechunk_check.0, rechunk_check.1);
        }
    }
    Ok(report)
}

fn audit_chunk<R: Runner>(
    runner: &R,
    chunk: &[usize],
    sample_positions: &[usize],
    report: &mut LegalityReport,
) -> Result<(), String> {
    let base = runner.clone().score_chunk(chunk, sample_positions)?;
    let repeat = runner.clone().score_chunk(chunk, sample_positions)?;

    let vocab_size = runner.vocab_size();
    let normalization = check_normalization(&base.sample_predictions, vocab_size);
    report
        .normalization
        .record(normalization.0, normalization.1);

    if let Some(gold) = &base.sample_gold_logprobs {
        let gold_check =
            check_gold_logprob_consistency(&base.sample_predictions, gold, chunk, sample_positions);
        report
            .gold_logprob_consistency
            .record(gold_check.0, gold_check.1);
    }

    let repeat_check =
        compare_prediction_sets(&base.sample_predictions, &repeat.sample_predictions);
    report.repeatability.record(repeat_check.0, repeat_check.1);

    if sample_positions.len() > 1 {
        let mut reversed_positions = sample_positions.to_vec();
        reversed_positions.reverse();
        let reversed = runner.clone().score_chunk(chunk, &reversed_positions)?;
        let reordered_reversed = subset_predictions(
            &reversed.sample_predictions,
            &reversed_positions,
            sample_positions,
        )?;
        let order_check = compare_prediction_sets(&base.sample_predictions, &reordered_reversed);

        let subset_positions: Vec<usize> = sample_positions.iter().copied().step_by(2).collect();
        let subset_check =
            if subset_positions.is_empty() || subset_positions.len() == sample_positions.len() {
                (true, 0.0)
            } else {
                let subset_base = subset_predictions(
                    &base.sample_predictions,
                    sample_positions,
                    &subset_positions,
                )?;
                let subset_alt = runner.clone().score_chunk(chunk, &subset_positions)?;
                compare_prediction_sets(&subset_base, &subset_alt.sample_predictions)
            };

        report.sample_set_invariance.record(
            order_check.0 && subset_check.0,
            order_check.1.max(subset_check.1),
        );
    }

    for cutoff in future_cutoffs(chunk.len()) {
        let mut perturbed = chunk.to_vec();
        for (idx, value) in perturbed.iter_mut().enumerate().skip(cutoff) {
            *value = (idx * 7 + 3) % vocab_size;
        }
        let positions: Vec<usize> = sample_positions
            .iter()
            .copied()
            .filter(|pos| *pos < cutoff)
            .collect();
        if positions.is_empty() {
            continue;
        }
        let alt = runner.clone().score_chunk(&perturbed, &positions)?;
        let base_subset =
            subset_predictions(&base.sample_predictions, sample_positions, &positions)?;
        let future_check = compare_prediction_sets(&base_subset, &alt.sample_predictions);
        report
            .future_suffix_invariance
            .record(future_check.0, future_check.1);
    }

    for &pos in sample_positions.iter().take(2) {
        let mut perturbed = chunk.to_vec();
        for (idx, value) in perturbed.iter_mut().enumerate().skip(pos) {
            *value = (idx * 11 + 5) % vocab_size;
        }
        let alt = runner.clone().score_chunk(&perturbed, &[pos])?;
        let base_subset = subset_predictions(&base.sample_predictions, sample_positions, &[pos])?;
        let answer_check = compare_prediction_sets(&base_subset, &alt.sample_predictions);
        report
            .answer_mask_invariance
            .record(answer_check.0, answer_check.1);
    }

    for &pos in sample_positions {
        let prefix = &chunk[..=pos];
        let alt = runner.clone().score_chunk(prefix, &[pos])?;
        let base_subset = subset_predictions(&base.sample_predictions, sample_positions, &[pos])?;
        let prefix_check = compare_prediction_sets(&base_subset, &alt.sample_predictions);
        report
            .prefix_truncation_parity
            .record(prefix_check.0, prefix_check.1);
    }

    Ok(())
}

fn pick_positions(chunk_len: usize) -> Vec<usize> {
    if chunk_len == 0 {
        return Vec::new();
    }
    let mut picks = vec![0];
    if chunk_len > 1 {
        picks.push(chunk_len / 4);
        picks.push(chunk_len / 2);
        picks.push((3 * chunk_len) / 4);
        picks.push(chunk_len - 1);
    }
    picks.sort_unstable();
    picks.dedup();
    picks
}

fn future_cutoffs(chunk_len: usize) -> Vec<usize> {
    if chunk_len <= 2 {
        return Vec::new();
    }
    let mut cuts = vec![chunk_len / 3, (2 * chunk_len) / 3];
    cuts.retain(|cut| *cut > 0 && *cut < chunk_len);
    cuts.sort_unstable();
    cuts.dedup();
    cuts
}

fn alternate_chunk_size(chunk_size: usize, span_len: usize) -> usize {
    if span_len <= 1 {
        return chunk_size;
    }
    let half = (chunk_size / 2).max(1);
    if half != chunk_size {
        return half.min(span_len);
    }
    (chunk_size + 1).min(span_len)
}

fn stream_sample_positions(
    total_len: usize,
    base_chunk_size: usize,
    max_chunks: usize,
) -> Vec<usize> {
    let span_len = total_len.min(base_chunk_size.saturating_mul(max_chunks.max(1)));
    let mut positions = Vec::new();
    let mut start = 0usize;
    while start < span_len {
        let end = (start + base_chunk_size).min(span_len);
        for local in pick_positions(end - start) {
            positions.push(start + local);
        }
        start = end;
    }
    positions.sort_unstable();
    positions.dedup();
    positions
}

fn score_stream_positions<R: Runner>(
    runner: &R,
    tokens: &[usize],
    chunk_size: usize,
    wanted_positions: &[usize],
) -> Result<Vec<Vec<f64>>, String> {
    let mut live_runner = runner.clone();
    let mut rows = Vec::with_capacity(wanted_positions.len());
    let mut want_index = 0usize;
    let mut start = 0usize;
    while start < tokens.len() && want_index < wanted_positions.len() {
        let end = (start + chunk_size).min(tokens.len());
        let chunk = &tokens[start..end];
        let mut local_positions = Vec::new();
        while want_index < wanted_positions.len() && wanted_positions[want_index] < end {
            let global = wanted_positions[want_index];
            if global >= start {
                local_positions.push(global - start);
            }
            want_index += 1;
        }
        if !local_positions.is_empty() {
            let outputs = live_runner.score_chunk(chunk, &local_positions)?;
            rows.extend(outputs.sample_predictions);
        }
        if end < tokens.len() {
            live_runner.adapt_chunk(chunk)?;
        }
        start = end;
    }
    if rows.len() != wanted_positions.len() {
        return Err(format!(
            "stream scoring missed positions: wanted {} rows, got {}",
            wanted_positions.len(),
            rows.len()
        ));
    }
    Ok(rows)
}

fn check_normalization(rows: &[Vec<f64>], vocab_size: usize) -> (bool, f64) {
    let mut passed = true;
    let mut max_abs_diff: f64 = 0.0;
    for row in rows {
        if row.len() != vocab_size {
            passed = false;
            continue;
        }
        let sum: f64 = row.iter().sum();
        let local = (sum - 1.0).abs();
        max_abs_diff = max_abs_diff.max(local);
        if local > 1e-7 || row.iter().any(|value| *value < -1e-7) {
            passed = false;
        }
    }
    (passed, max_abs_diff)
}

fn check_gold_logprob_consistency(
    rows: &[Vec<f64>],
    gold_logprobs: &[f64],
    chunk: &[usize],
    sample_positions: &[usize],
) -> (bool, f64) {
    let mut passed = true;
    let mut max_abs_diff: f64 = 0.0;
    for ((row, &logged), &pos) in rows
        .iter()
        .zip(gold_logprobs.iter())
        .zip(sample_positions.iter())
    {
        let tok = chunk[pos];
        let implied = row
            .get(tok)
            .copied()
            .unwrap_or(f64::MIN_POSITIVE)
            .max(f64::MIN_POSITIVE)
            .ln();
        let diff = (implied - logged).abs();
        max_abs_diff = max_abs_diff.max(diff);
        if diff > 1e-7 {
            passed = false;
        }
    }
    (passed, max_abs_diff)
}

fn compare_prediction_sets(base: &[Vec<f64>], alt: &[Vec<f64>]) -> (bool, f64) {
    let mut passed = base.len() == alt.len();
    let mut max_abs_diff: f64 = 0.0;
    if base.len() != alt.len() {
        return (false, f64::INFINITY);
    }
    for (lhs, rhs) in base.iter().zip(alt.iter()) {
        if lhs.len() != rhs.len() {
            return (false, f64::INFINITY);
        }
        for (&a, &b) in lhs.iter().zip(rhs.iter()) {
            let diff = (a - b).abs();
            max_abs_diff = max_abs_diff.max(diff);
            if diff > 1e-7 {
                passed = false;
            }
        }
    }
    (passed, max_abs_diff)
}

fn subset_predictions(
    predictions: &[Vec<f64>],
    all_positions: &[usize],
    wanted_positions: &[usize],
) -> Result<Vec<Vec<f64>>, String> {
    let mut rows = Vec::with_capacity(wanted_positions.len());
    for &wanted in wanted_positions {
        let Some(index) = all_positions.iter().position(|pos| *pos == wanted) else {
            return Err(format!("missing sampled position {wanted}"));
        };
        rows.push(predictions[index].clone());
    }
    Ok(rows)
}
