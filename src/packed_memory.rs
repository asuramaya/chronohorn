use std::path::Path;

use crate::checkpoint::{F32Array, load_named_npz_f32_arrays};
use crate::data::{take_train_tokens, take_val_tokens};
use crate::protocol::{Runner, SampleOutputs};

const TRIGRAM_HASH_MUL_A: u64 = 1_315_423_911;
const TRIGRAM_HASH_MUL_B: u64 = 2_654_435_761;

#[derive(Debug, Clone)]
pub struct PackedTables {
    pub vocab_size: usize,
    pub unigram_probs: Vec<f64>,
    pub bigram_counts: Vec<f64>,
    pub bigram_totals: Vec<f64>,
    pub trigram_counts: Vec<f64>,
    pub trigram_totals: Vec<f64>,
    pub trigram_buckets: usize,
    pub token_budget: usize,
}

#[derive(Debug, Clone)]
pub struct PackedMemoryRunner {
    tables: PackedTables,
    alpha_bigram: f64,
    alpha_trigram: f64,
}

#[derive(Debug, Clone)]
pub struct TableDiff {
    pub name: &'static str,
    pub elem_count: usize,
    pub max_abs_diff: f64,
    pub mean_abs_diff: f64,
}

impl PackedMemoryRunner {
    pub fn from_data_root(
        root: &Path,
        vocab_size: usize,
        token_budget: usize,
        trigram_buckets: usize,
        alpha_bigram: f64,
        alpha_trigram: f64,
    ) -> Result<(Self, Vec<usize>), String> {
        let train = take_train_tokens(root, token_budget)?;
        let val = take_val_tokens(root, 4096)?;
        let tables = build_packed_tables(&train, vocab_size, trigram_buckets)?;
        Ok((
            Self {
                tables,
                alpha_bigram,
                alpha_trigram,
            },
            val,
        ))
    }

    pub fn from_checkpoint(
        checkpoint_path: &str,
        alpha_bigram: f64,
        alpha_trigram: f64,
    ) -> Result<Self, String> {
        let arrays = load_named_npz_f32_arrays(
            checkpoint_path,
            &[
                "packed_unigram_probs",
                "packed_bigram_counts",
                "packed_bigram_totals",
                "packed_trigram_counts",
                "packed_trigram_totals",
            ],
        )?;
        let tables = packed_tables_from_arrays(&arrays)?;
        Ok(Self {
            tables,
            alpha_bigram,
            alpha_trigram,
        })
    }

    pub fn tables(&self) -> &PackedTables {
        &self.tables
    }

    fn trigram_bucket(&self, prev2: usize, prev1: usize) -> usize {
        ((prev2 as u64 * TRIGRAM_HASH_MUL_A + prev1 as u64 * TRIGRAM_HASH_MUL_B)
            % self.tables.trigram_buckets as u64) as usize
    }

    fn score_position(&self, prev2: Option<usize>, prev1: Option<usize>) -> Vec<f64> {
        let vocab = self.tables.vocab_size;
        let p_uni = &self.tables.unigram_probs;
        let p_bigram = if let Some(p1) = prev1 {
            let row_start = p1 * vocab;
            let total = self.tables.bigram_totals[p1];
            let denom = (total + self.alpha_bigram).max(1e-8);
            let mut out = vec![0.0; vocab];
            for tok in 0..vocab {
                out[tok] = (self.tables.bigram_counts[row_start + tok]
                    + self.alpha_bigram * p_uni[tok])
                    / denom;
            }
            out
        } else {
            p_uni.clone()
        };
        if let (Some(p2), Some(p1)) = (prev2, prev1) {
            let bucket = self.trigram_bucket(p2, p1);
            let row_start = bucket * vocab;
            let total = self.tables.trigram_totals[bucket];
            let denom = (total + self.alpha_trigram).max(1e-8);
            let mut out = vec![0.0; vocab];
            for tok in 0..vocab {
                out[tok] = (self.tables.trigram_counts[row_start + tok]
                    + self.alpha_trigram * p_bigram[tok])
                    / denom;
            }
            normalize(&mut out);
            out
        } else {
            let mut out = p_bigram;
            normalize(&mut out);
            out
        }
    }
}

impl Runner for PackedMemoryRunner {
    fn name(&self) -> &'static str {
        "PackedMemoryRunner"
    }

    fn vocab_size(&self) -> usize {
        self.tables.vocab_size
    }

    fn score_chunk(
        &self,
        tokens: &[usize],
        sample_positions: &[usize],
    ) -> Result<SampleOutputs, String> {
        let mut sample_predictions = Vec::with_capacity(sample_positions.len());
        let mut sample_gold_logprobs = Vec::with_capacity(sample_positions.len());
        for &pos in sample_positions {
            if pos >= tokens.len() {
                return Err(format!("sample position {pos} out of bounds"));
            }
            let prev1 = if pos >= 1 {
                Some(tokens[pos - 1])
            } else {
                None
            };
            let prev2 = if pos >= 2 {
                Some(tokens[pos - 2])
            } else {
                None
            };
            let dist = self.score_position(prev2, prev1);
            let tok = tokens[pos];
            let gold = dist
                .get(tok)
                .copied()
                .unwrap_or(f64::MIN_POSITIVE)
                .max(f64::MIN_POSITIVE)
                .ln();
            sample_gold_logprobs.push(gold);
            sample_predictions.push(dist);
        }
        Ok(SampleOutputs {
            sample_predictions,
            sample_gold_logprobs: Some(sample_gold_logprobs),
        })
    }

    fn adapt_chunk(&mut self, _tokens: &[usize]) -> Result<(), String> {
        Ok(())
    }
}

pub fn build_packed_tables(
    tokens: &[usize],
    vocab_size: usize,
    trigram_buckets: usize,
) -> Result<PackedTables, String> {
    if trigram_buckets == 0 {
        return Err("trigram_buckets must be positive".to_string());
    }
    if tokens.len() < 4 {
        return Err("need at least 4 tokens".to_string());
    }

    let mut unigram_counts = vec![0.0; vocab_size];
    let mut bigram_counts = vec![0.0; vocab_size * vocab_size];
    let mut trigram_counts = vec![0.0; trigram_buckets * vocab_size];

    for window in tokens.windows(2) {
        let prev1 = window[0];
        let next = window[1];
        if prev1 >= vocab_size || next >= vocab_size {
            return Err(format!("token out of vocab: {prev1} -> {next}"));
        }
        unigram_counts[next] += 1.0;
        bigram_counts[prev1 * vocab_size + next] += 1.0;
    }

    for window in tokens.windows(3) {
        let prev2 = window[0];
        let prev1 = window[1];
        let next = window[2];
        if prev2 >= vocab_size || prev1 >= vocab_size || next >= vocab_size {
            return Err(format!(
                "trigram token out of vocab: {prev2},{prev1}->{next}"
            ));
        }
        let bucket = ((prev2 as u64 * TRIGRAM_HASH_MUL_A + prev1 as u64 * TRIGRAM_HASH_MUL_B)
            % trigram_buckets as u64) as usize;
        trigram_counts[bucket * vocab_size + next] += 1.0;
    }

    let unigram_sum: f64 = unigram_counts.iter().sum();
    let mut unigram_probs = unigram_counts;
    for value in &mut unigram_probs {
        *value /= unigram_sum.max(1.0);
    }

    let mut bigram_totals = vec![0.0; vocab_size];
    for prev1 in 0..vocab_size {
        let row = &bigram_counts[prev1 * vocab_size..(prev1 + 1) * vocab_size];
        bigram_totals[prev1] = row.iter().sum();
    }

    let mut trigram_totals = vec![0.0; trigram_buckets];
    for bucket in 0..trigram_buckets {
        let row = &trigram_counts[bucket * vocab_size..(bucket + 1) * vocab_size];
        trigram_totals[bucket] = row.iter().sum();
    }

    Ok(PackedTables {
        vocab_size,
        unigram_probs,
        bigram_counts,
        bigram_totals,
        trigram_counts,
        trigram_totals,
        trigram_buckets,
        token_budget: tokens.len(),
    })
}

fn normalize(values: &mut [f64]) {
    let total: f64 = values.iter().sum();
    let denom = total.max(f64::EPSILON);
    for value in values {
        *value /= denom;
    }
}

fn packed_tables_from_arrays(
    arrays: &std::collections::BTreeMap<String, F32Array>,
) -> Result<PackedTables, String> {
    let unigram = arrays
        .get("packed_unigram_probs")
        .ok_or_else(|| "missing packed_unigram_probs".to_string())?;
    let bigram_counts = arrays
        .get("packed_bigram_counts")
        .ok_or_else(|| "missing packed_bigram_counts".to_string())?;
    let bigram_totals = arrays
        .get("packed_bigram_totals")
        .ok_or_else(|| "missing packed_bigram_totals".to_string())?;
    let trigram_counts = arrays
        .get("packed_trigram_counts")
        .ok_or_else(|| "missing packed_trigram_counts".to_string())?;
    let trigram_totals = arrays
        .get("packed_trigram_totals")
        .ok_or_else(|| "missing packed_trigram_totals".to_string())?;

    let vocab_size = match unigram.shape.as_slice() {
        [vocab] => *vocab,
        other => {
            return Err(format!(
                "unexpected packed_unigram_probs shape: {:?}",
                other
            ));
        }
    };
    match bigram_counts.shape.as_slice() {
        [rows, cols] if *rows == vocab_size && *cols == vocab_size => {}
        other => {
            return Err(format!(
                "unexpected packed_bigram_counts shape: {:?}",
                other
            ));
        }
    }
    match bigram_totals.shape.as_slice() {
        [rows] if *rows == vocab_size => {}
        other => {
            return Err(format!(
                "unexpected packed_bigram_totals shape: {:?}",
                other
            ));
        }
    }
    let trigram_buckets = match trigram_counts.shape.as_slice() {
        [rows, cols] if *cols == vocab_size => *rows,
        other => {
            return Err(format!(
                "unexpected packed_trigram_counts shape: {:?}",
                other
            ));
        }
    };
    match trigram_totals.shape.as_slice() {
        [rows] if *rows == trigram_buckets => {}
        other => {
            return Err(format!(
                "unexpected packed_trigram_totals shape: {:?}",
                other
            ));
        }
    }

    Ok(PackedTables {
        vocab_size,
        unigram_probs: unigram.values.iter().map(|&v| v as f64).collect(),
        bigram_counts: bigram_counts.values.iter().map(|&v| v as f64).collect(),
        bigram_totals: bigram_totals.values.iter().map(|&v| v as f64).collect(),
        trigram_counts: trigram_counts.values.iter().map(|&v| v as f64).collect(),
        trigram_totals: trigram_totals.values.iter().map(|&v| v as f64).collect(),
        trigram_buckets,
        token_budget: 0,
    })
}

pub fn compare_tables(left: &PackedTables, right: &PackedTables) -> Result<Vec<TableDiff>, String> {
    if left.vocab_size != right.vocab_size {
        return Err(format!(
            "vocab mismatch: {} vs {}",
            left.vocab_size, right.vocab_size
        ));
    }
    if left.trigram_buckets != right.trigram_buckets {
        return Err(format!(
            "trigram bucket mismatch: {} vs {}",
            left.trigram_buckets, right.trigram_buckets
        ));
    }
    let mut diffs = Vec::new();
    for (name, l, r) in [
        (
            "packed_unigram_probs",
            &left.unigram_probs,
            &right.unigram_probs,
        ),
        (
            "packed_bigram_counts",
            &left.bigram_counts,
            &right.bigram_counts,
        ),
        (
            "packed_bigram_totals",
            &left.bigram_totals,
            &right.bigram_totals,
        ),
        (
            "packed_trigram_counts",
            &left.trigram_counts,
            &right.trigram_counts,
        ),
        (
            "packed_trigram_totals",
            &left.trigram_totals,
            &right.trigram_totals,
        ),
    ] {
        if l.len() != r.len() {
            return Err(format!(
                "length mismatch for {name}: {} vs {}",
                l.len(),
                r.len()
            ));
        }
        let mut max_abs_diff: f64 = 0.0;
        let mut sum_abs_diff = 0.0;
        for (&lv, &rv) in l.iter().zip(r.iter()) {
            let diff = (lv - rv).abs();
            max_abs_diff = max_abs_diff.max(diff);
            sum_abs_diff += diff;
        }
        diffs.push(TableDiff {
            name,
            elem_count: l.len(),
            max_abs_diff,
            mean_abs_diff: if l.is_empty() {
                0.0
            } else {
                sum_abs_diff / l.len() as f64
            },
        });
    }
    Ok(diffs)
}

pub fn render_table_diffs(diffs: &[TableDiff]) -> String {
    let mut out = String::new();
    out.push_str("table_diffs:\n");
    for diff in diffs {
        out.push_str(&format!(
            "  {:<22} elems={:<10} max_abs_diff={:.9} mean_abs_diff={:.9}\n",
            diff.name, diff.elem_count, diff.max_abs_diff, diff.mean_abs_diff
        ));
    }
    out
}
