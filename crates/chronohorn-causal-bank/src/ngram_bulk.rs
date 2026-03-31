use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::fs::File;
use std::hash::{BuildHasher, Hasher};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::time::Instant;

use chronohorn_core::data::{
    ByteAccountingReport, HEADER_BYTES, PARAMETER_GOLF_MAGIC, PARAMETER_GOLF_VERSION,
    compute_tokens_per_byte, list_shards, take_val_tokens,
};
use chronohorn_core::protocol::Runner;
use rayon::prelude::*;
use serde::Serialize;

use crate::checkpoint::{
    TokenConker3CheckpointMetadata, TokenConker3CheckpointRunner,
    load_token_conker3_checkpoint_runner_and_metadata,
};
use crate::oracle::{
    token_oracle_target_dataset_from_tokens, token_oracle_teacher_candidate_pairs,
};

const DEFAULT_BUCKETS: usize = 4_194_304;
const DEFAULT_VOCAB_SIZE: usize = 1024;
const PROBE_BUCKET_2: usize = 16_777_216;
const PROBE_BUCKET_3: usize = 16_777_216;
const PROBE_BUCKET_4: usize = 4_194_304;
const PROBE_BUCKET_5: usize = 4_194_304;
const PROBE_BUCKET_6: usize = 4_194_304;
const PROBE_BUCKET_7: usize = 4_194_304;
const MEDIUM_BUCKET_2: usize = 33_554_432;
const MEDIUM_BUCKET_3: usize = 33_554_432;
const MEDIUM_BUCKET_4: usize = 8_388_608;
const MEDIUM_BUCKET_5: usize = 8_388_608;
const MEDIUM_BUCKET_6: usize = 8_388_608;
const MEDIUM_BUCKET_7: usize = 8_388_608;
const ABSURD_BUCKET_2: usize = 67_108_864;
const ABSURD_BUCKET_3: usize = 67_108_864;
const ABSURD_BUCKET_4: usize = 16_777_216;
const ABSURD_BUCKET_5: usize = 16_777_216;
const ABSURD_BUCKET_6: usize = 16_777_216;
const ABSURD_BUCKET_7: usize = 16_777_216;
const MIN_ORDER: usize = 2;
const MAX_ORDER: usize = 10;
const MIX_BASE: f64 = 0.05;
const MIX_SPAN: f64 = 0.25;
const ENTROPY_MIDPOINT: f64 = 4.5;
const ENTROPY_SLOPE: f64 = 1.5;
const MIN_ROW_TOTAL: u32 = 4;
const RESIDUAL_CAP: f64 = 4.0;
const ORACLE_TOP_K: usize = 4;
const ORACLE_QUANT_SCALE: u32 = 255;
const ORACLE_MIN_ROW_TOTAL: u32 = 512;
const TABLE_ARTIFACT_MAGIC: &[u8; 8] = b"CHNGRM01";
const ROW_STATS_ARTIFACT_MAGIC: &[u8; 8] = b"CHNRWS01";
const READOUT_BLOCK: usize = 128;

#[derive(Clone, Default)]
struct IdentityBuildHasher;

#[derive(Default)]
struct IdentityHasher(u64);

impl Hasher for IdentityHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        let mut acc = self.0;
        for &byte in bytes {
            acc ^= byte as u64;
            acc = acc.rotate_left(7);
            acc = acc.wrapping_mul(0x9e37_79b1_85eb_ca87);
        }
        self.0 = acc;
    }

    fn write_u64(&mut self, value: u64) {
        self.0 = value;
    }
}

impl BuildHasher for IdentityBuildHasher {
    type Hasher = IdentityHasher;

    fn build_hasher(&self) -> Self::Hasher {
        IdentityHasher::default()
    }
}

type U64FastMap<V> = HashMap<u64, V, IdentityBuildHasher>;

#[derive(Debug, Clone, Serialize)]
pub struct TokenConker3NgramBulkOrderStats {
    pub order: usize,
    pub longest_match_fires: u64,
    pub gold_argmax_hits: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct TokenConker3NgramBulkCheckpoint {
    pub tokens_scored: usize,
    pub interval_tokens: usize,
    pub elapsed_sec: f64,
    pub tokens_per_sec: f64,
    pub cumulative_bpt_neural: f64,
    pub cumulative_bpt_mixed: f64,
    pub cumulative_bpb_neural: Option<f64>,
    pub cumulative_bpb_mixed: Option<f64>,
    pub interval_bpt_neural: f64,
    pub interval_bpt_mixed: f64,
    pub interval_bpb_neural: Option<f64>,
    pub interval_bpb_mixed: Option<f64>,
    pub mean_alpha_on_match: f64,
    pub mean_row_total_on_match: f64,
    pub no_match_positions: u64,
    pub order_stats: Vec<TokenConker3NgramBulkOrderStats>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TokenConker3NgramBulkReport {
    pub checkpoint_path: String,
    pub summary_path: String,
    pub train_seq_len: usize,
    pub eval_tokens: usize,
    pub scoring_mode: String,
    pub collision_control: String,
    pub table_source: String,
    pub train_tokens_built: usize,
    pub train_build_elapsed_sec: f64,
    pub ngram_orders: Vec<usize>,
    pub ngram_bucket_profile: Vec<usize>,
    pub mix_base: f64,
    pub mix_span: f64,
    pub entropy_midpoint: f64,
    pub entropy_slope: f64,
    pub min_row_total: u32,
    pub residual_cap: f64,
    pub report_every: usize,
    pub eval_bpt_neural: f64,
    pub eval_bpt_mixed: f64,
    pub eval_tokens_per_byte: Option<f64>,
    pub eval_bytes_per_token: Option<f64>,
    pub eval_bpb_neural: Option<f64>,
    pub eval_bpb_mixed: Option<f64>,
    pub elapsed_sec: f64,
    pub tokens_per_sec: f64,
    pub checkpoints: Vec<TokenConker3NgramBulkCheckpoint>,
}

#[derive(Debug, Clone)]
struct BulkConfig {
    enabled_orders: Vec<usize>,
    bucket_counts: Vec<usize>,
    mix_base: f64,
    mix_span: f64,
    entropy_midpoint: f64,
    entropy_slope: f64,
    min_row_total: u32,
    residual_cap: f64,
    report_every: usize,
}

impl BulkConfig {
    fn new(report_every: usize) -> Self {
        let bucket_counts =
            vec![DEFAULT_BUCKETS; MAX_ORDER.saturating_sub(MIN_ORDER).saturating_add(1)];
        Self {
            enabled_orders: (MIN_ORDER..=MAX_ORDER).collect(),
            bucket_counts,
            mix_base: MIX_BASE,
            mix_span: MIX_SPAN,
            entropy_midpoint: ENTROPY_MIDPOINT,
            entropy_slope: ENTROPY_SLOPE,
            min_row_total: MIN_ROW_TOTAL,
            residual_cap: RESIDUAL_CAP,
            report_every: report_every.max(1),
        }
    }

    fn collision_probe(report_every: usize) -> Self {
        Self {
            enabled_orders: vec![2, 3, 4, 5, 6, 7],
            bucket_counts: vec![
                PROBE_BUCKET_2,
                PROBE_BUCKET_3,
                PROBE_BUCKET_4,
                PROBE_BUCKET_5,
                PROBE_BUCKET_6,
                PROBE_BUCKET_7,
            ],
            mix_base: MIX_BASE,
            mix_span: MIX_SPAN,
            entropy_midpoint: ENTROPY_MIDPOINT,
            entropy_slope: ENTROPY_SLOPE,
            min_row_total: MIN_ROW_TOTAL,
            residual_cap: RESIDUAL_CAP,
            report_every: report_every.max(1),
        }
    }

    fn medium_collision_probe(report_every: usize) -> Self {
        Self {
            enabled_orders: vec![2, 3, 4, 5, 6, 7],
            bucket_counts: vec![
                MEDIUM_BUCKET_2,
                MEDIUM_BUCKET_3,
                MEDIUM_BUCKET_4,
                MEDIUM_BUCKET_5,
                MEDIUM_BUCKET_6,
                MEDIUM_BUCKET_7,
            ],
            mix_base: MIX_BASE,
            mix_span: MIX_SPAN,
            entropy_midpoint: ENTROPY_MIDPOINT,
            entropy_slope: ENTROPY_SLOPE,
            min_row_total: MIN_ROW_TOTAL,
            residual_cap: RESIDUAL_CAP,
            report_every: report_every.max(1),
        }
    }

    fn absurd_collision_probe(report_every: usize) -> Self {
        Self {
            enabled_orders: vec![2, 3, 4, 5, 6, 7],
            bucket_counts: vec![
                ABSURD_BUCKET_2,
                ABSURD_BUCKET_3,
                ABSURD_BUCKET_4,
                ABSURD_BUCKET_5,
                ABSURD_BUCKET_6,
                ABSURD_BUCKET_7,
            ],
            mix_base: MIX_BASE,
            mix_span: MIX_SPAN,
            entropy_midpoint: ENTROPY_MIDPOINT,
            entropy_slope: ENTROPY_SLOPE,
            min_row_total: MIN_ROW_TOTAL,
            residual_cap: RESIDUAL_CAP,
            report_every: report_every.max(1),
        }
    }

    fn named_prebuilt_profile(profile: &str, report_every: usize) -> Result<Self, String> {
        match profile {
            "tiny" => Ok(Self::collision_probe(report_every)),
            "medium" => Ok(Self::medium_collision_probe(report_every)),
            "absurd" => Ok(Self::absurd_collision_probe(report_every)),
            other => Err(format!(
                "unknown ngram prebuilt profile {other}; expected one of: tiny, medium, absurd"
            )),
        }
    }

    fn max_order(&self) -> usize {
        self.enabled_orders.last().copied().unwrap_or(1)
    }

    fn order_index(&self, order: usize) -> Option<usize> {
        self.enabled_orders
            .iter()
            .position(|&candidate| candidate == order)
    }

    fn bucket_count_for_order(&self, order: usize) -> usize {
        self.order_index(order)
            .map(|index| self.bucket_counts[index])
            .unwrap_or(1)
    }
}

#[derive(Debug, Clone)]
struct NgramBulkState {
    config: BulkConfig,
    vocab_size: usize,
    history: VecDeque<usize>,
    tables: Vec<HashMap<u32, Vec<FingerprintedCounter>>>,
}

#[derive(Debug, Clone, Default)]
struct SparseCounter {
    total: u32,
    counts: Vec<(usize, u32)>,
}

#[derive(Debug, Clone)]
struct FingerprintedCounter {
    fingerprint: u64,
    counter: SparseCounter,
}

#[derive(Debug, Clone, Default)]
struct OracleTrustCounter {
    examples: u32,
    top1_votes: SparseCounter,
    mass: SparseCounter,
}

#[derive(Debug, Clone)]
struct FingerprintedOracleTrustCounter {
    fingerprint: u64,
    counter: OracleTrustCounter,
}

#[derive(Debug, Clone, Copy)]
struct StoredOracleRowMetrics {
    examples: u32,
    support_size: u32,
    top1_agreement: f64,
    top_mass: f64,
}

#[derive(Debug, Clone)]
struct OracleBudgetedRowStat {
    bucket: u32,
    fingerprint: u64,
    row: SparseCounter,
    oracle: Option<StoredOracleRowMetrics>,
}

#[derive(Debug, Clone)]
struct OracleBudgetedRowStatsArtifact {
    config: BulkConfig,
    vocab_size: usize,
    train_tokens_built: usize,
    train_build_elapsed_sec: f64,
    tables: Vec<Vec<OracleBudgetedRowStat>>,
}

impl SparseCounter {
    fn increment(&mut self, token: usize) {
        self.total = self.total.saturating_add(1);
        if let Some((_, count)) = self
            .counts
            .iter_mut()
            .find(|(candidate, _)| *candidate == token)
        {
            *count = count.saturating_add(1);
        } else {
            self.counts.push((token, 1));
        }
    }

    fn add_weight(&mut self, token: usize, weight: u32) {
        if weight == 0 {
            return;
        }
        if let Some((_, count)) = self
            .counts
            .iter_mut()
            .find(|(candidate, _)| *candidate == token)
        {
            *count = count.saturating_add(weight);
        } else {
            self.counts.push((token, weight));
        }
    }

    fn prune_top_k(&mut self, keep: usize) {
        self.counts
            .sort_by(|(token_a, count_a), (token_b, count_b)| {
                count_b.cmp(count_a).then_with(|| token_a.cmp(token_b))
            });
        if self.counts.len() > keep {
            self.counts.truncate(keep);
        }
    }
}

impl OracleTrustCounter {
    fn prune_top_k(&mut self, keep: usize) {
        self.top1_votes.prune_top_k(keep);
        self.mass.prune_top_k(keep);
    }
}

impl NgramBulkState {
    fn new(vocab_size: usize, config: BulkConfig) -> Self {
        let context_lengths = config.bucket_counts.len();
        let history_capacity = config.max_order().saturating_sub(1);
        Self {
            config,
            vocab_size,
            history: VecDeque::with_capacity(history_capacity),
            tables: (0..context_lengths).map(|_| HashMap::new()).collect(),
        }
    }

    fn observe_token(&mut self, token: usize) {
        if token >= self.vocab_size {
            return;
        }
        let max_context = self
            .history
            .len()
            .min(self.config.max_order().saturating_sub(1));
        for (table_index, &order) in self.config.enabled_orders.iter().enumerate() {
            let context_len = order.saturating_sub(1);
            if context_len > max_context {
                continue;
            }
            let (bucket, fingerprint) =
                self.context_key_from_history(&self.history, context_len, order);
            let bucket_rows = self.tables[table_index].entry(bucket).or_default();
            if let Some(row) = bucket_rows
                .iter_mut()
                .find(|row| row.fingerprint == fingerprint)
            {
                row.counter.increment(token);
            } else {
                let mut counter = SparseCounter::default();
                counter.increment(token);
                bucket_rows.push(FingerprintedCounter {
                    fingerprint,
                    counter,
                });
            }
        }
        push_history(
            &mut self.history,
            self.config.max_order().saturating_sub(1),
            token,
        );
    }

    fn lookup_counter_from_history<'a>(
        &'a self,
        history: &VecDeque<usize>,
    ) -> Option<(&'a SparseCounter, usize, usize)> {
        let max_context = history.len().min(self.config.max_order().saturating_sub(1));
        for (table_index, &order) in self.config.enabled_orders.iter().enumerate().rev() {
            let context_len = order.saturating_sub(1);
            if context_len > max_context {
                continue;
            }
            let (bucket, fingerprint) = self.context_key_from_history(history, context_len, order);
            if let Some(rows) = self.tables[table_index].get(&bucket) {
                if let Some(row) = rows
                    .iter()
                    .find(|row| row.fingerprint == fingerprint)
                    .map(|row| &row.counter)
                {
                    if row.total >= self.config.min_row_total {
                        return Some((row, table_index, order));
                    }
                }
            }
        }
        None
    }

    fn context_key_from_history(
        &self,
        history: &VecDeque<usize>,
        context_len: usize,
        order: usize,
    ) -> (u32, u64) {
        let start = history.len().saturating_sub(context_len);
        self.context_key_from_iter(history.iter().skip(start).copied(), order)
    }

    fn context_key_from_slice(&self, context: &[usize], order: usize) -> (u32, u64) {
        context_key_from_iter(&self.config, context.iter().copied(), order)
    }

    fn prune_all_rows_top_k(&mut self, keep: usize) {
        for table in &mut self.tables {
            for rows in table.values_mut() {
                for row in rows {
                    row.counter.prune_top_k(keep);
                }
            }
        }
    }

    fn context_key_from_iter(
        &self,
        iter: impl IntoIterator<Item = usize>,
        order: usize,
    ) -> (u32, u64) {
        context_key_from_iter(&self.config, iter, order)
    }
}

impl OracleTrustState {
    fn lookup_for_order_from_history<'a>(
        &'a self,
        history: &VecDeque<usize>,
        order: usize,
    ) -> Option<&'a OracleTrustCounter> {
        let context_len = order.saturating_sub(1);
        if context_len == 0 || context_len > history.len() {
            return None;
        }
        let Some(table_index) = self.config.order_index(order) else {
            return None;
        };
        let start = history.len().saturating_sub(context_len);
        let (bucket, fingerprint) =
            context_key_from_iter(&self.config, history.iter().skip(start).copied(), order);
        self.tables[table_index]
            .get(&bucket)?
            .iter()
            .find(|row| row.fingerprint == fingerprint)
            .map(|row| &row.counter)
    }

    fn prune_all_rows_top_k(&mut self, keep: usize) {
        for table in &mut self.tables {
            for rows in table.values_mut() {
                for row in rows {
                    row.counter.prune_top_k(keep);
                }
            }
        }
    }
}

fn context_key_from_iter(
    config: &BulkConfig,
    iter: impl IntoIterator<Item = usize>,
    order: usize,
) -> (u32, u64) {
    let mut bucket_hash = 0xcbf2_9ce4_8422_2325u64 ^ (order as u64);
    let mut fingerprint = 0x9e37_79b9_7f4a_7c15u64 ^ ((order as u64) << 32);
    for token in iter {
        let token_u64 = token as u64;
        bucket_hash ^= token_u64 + 0x9e37_79b9;
        bucket_hash = bucket_hash.wrapping_mul(0x1000_0000_01b3);
        fingerprint ^= token_u64.wrapping_add(0x517c_c1b7_2722_0a95);
        fingerprint = fingerprint.rotate_left(13);
        fingerprint = fingerprint.wrapping_mul(0x9e37_79b1_85eb_ca87);
    }
    let buckets = config.bucket_count_for_order(order).max(1);
    fingerprint ^= fingerprint >> 33;
    fingerprint = fingerprint.wrapping_mul(0xff51_afd7_ed55_8ccd);
    fingerprint ^= fingerprint >> 33;
    fingerprint = fingerprint.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    fingerprint ^= fingerprint >> 33;
    ((bucket_hash % buckets as u64) as u32, fingerprint)
}

#[derive(Debug, Clone)]
struct IntervalStats {
    start_token: usize,
    enabled_orders: Vec<usize>,
    neural_nats: f64,
    mixed_nats: f64,
    alpha_sum: f64,
    row_total_sum: f64,
    matched_positions: u64,
    no_match_positions: u64,
    per_order_fires: Vec<u64>,
    per_order_gold_argmax_hits: Vec<u64>,
}

#[derive(Debug, Clone)]
struct OracleTrustState {
    config: BulkConfig,
    tables: Vec<HashMap<u32, Vec<FingerprintedOracleTrustCounter>>>,
}

#[derive(Debug, Clone, Copy)]
struct OracleTrustMetrics {
    examples: u32,
    support_size: usize,
    top1_agreement: f64,
    top_mass: f64,
}

#[derive(Debug, Clone, Copy)]
struct LogitSummary {
    gold_logit: f64,
    log_z: f64,
    entropy: f64,
    argmax_index: usize,
}

#[derive(Debug, Clone)]
struct BulkBlockToken {
    gold: usize,
    row: Option<(SparseCounter, usize)>,
}

impl IntervalStats {
    fn new(enabled_orders: &[usize]) -> Self {
        let width = enabled_orders.len();
        Self {
            start_token: 0,
            enabled_orders: enabled_orders.to_vec(),
            neural_nats: 0.0,
            mixed_nats: 0.0,
            alpha_sum: 0.0,
            row_total_sum: 0.0,
            matched_positions: 0,
            no_match_positions: 0,
            per_order_fires: vec![0; width],
            per_order_gold_argmax_hits: vec![0; width],
        }
    }

    fn reset(&mut self, start_token: usize) {
        self.start_token = start_token;
        self.neural_nats = 0.0;
        self.mixed_nats = 0.0;
        self.alpha_sum = 0.0;
        self.row_total_sum = 0.0;
        self.matched_positions = 0;
        self.no_match_positions = 0;
        self.per_order_fires.fill(0);
        self.per_order_gold_argmax_hits.fill(0);
    }
}

pub fn run_token_conker3_ngram_bulk_from_data_root(
    checkpoint_path: &str,
    summary_path: &str,
    root: &Path,
    val_token_budget: usize,
    report_every: usize,
) -> Result<TokenConker3NgramBulkReport, String> {
    let config = BulkConfig::new(report_every);
    let (runner, metadata) =
        load_token_conker3_checkpoint_runner_and_metadata(checkpoint_path, summary_path)?;
    let eval_tokens = take_val_tokens(root, val_token_budget)?;
    let byte_accounting = compute_tokens_per_byte(root, &eval_tokens)?;
    let ngram = NgramBulkState::new(runner.vocab_size(), config.clone());
    evaluate_bulk_report(
        &runner,
        metadata,
        eval_tokens,
        byte_accounting,
        ngram,
        config,
        "eval_prefix_incremental".to_string(),
        0,
        0.0,
        true,
    )
}

pub fn run_token_conker3_ngram_bulk_prebuilt_from_data_root(
    checkpoint_path: &str,
    summary_path: &str,
    root: &Path,
    train_token_budget: usize,
    val_token_budget: usize,
    report_every: usize,
    profile: &str,
) -> Result<TokenConker3NgramBulkReport, String> {
    let config = BulkConfig::named_prebuilt_profile(profile, report_every)?;
    let (runner, metadata) =
        load_token_conker3_checkpoint_runner_and_metadata(checkpoint_path, summary_path)?;
    let (ngram, train_tokens_built, train_build_elapsed_sec) =
        build_frozen_train_table(root, train_token_budget, runner.vocab_size(), &config)?;
    let eval_tokens = take_val_tokens(root, val_token_budget)?;
    let byte_accounting = compute_tokens_per_byte(root, &eval_tokens)?;
    evaluate_bulk_report(
        &runner,
        metadata,
        eval_tokens,
        byte_accounting,
        ngram,
        config,
        "prebuilt_train_frozen".to_string(),
        train_tokens_built,
        train_build_elapsed_sec,
        false,
    )
}

pub fn run_token_conker3_ngram_bulk_priority_cache_prebuilt_from_data_root(
    checkpoint_path: &str,
    summary_path: &str,
    root: &Path,
    train_token_budget: usize,
    val_token_budget: usize,
    report_every: usize,
    profile: &str,
) -> Result<TokenConker3NgramBulkReport, String> {
    let config = BulkConfig::named_prebuilt_profile(profile, report_every)?;
    let (runner, metadata) =
        load_token_conker3_checkpoint_runner_and_metadata(checkpoint_path, summary_path)?;
    let (frozen, train_tokens_built, train_build_elapsed_sec) =
        build_frozen_train_table(root, train_token_budget, runner.vocab_size(), &config)?;
    let eval_tokens = take_val_tokens(root, val_token_budget)?;
    let byte_accounting = compute_tokens_per_byte(root, &eval_tokens)?;
    evaluate_bulk_report_with_priority_cache(
        &runner,
        metadata,
        eval_tokens,
        byte_accounting,
        frozen,
        config,
        "prebuilt_train_plus_eval_cache_priority".to_string(),
        train_tokens_built,
        train_build_elapsed_sec,
    )
}

pub fn run_token_conker3_oracle_table_bulk_from_data_root(
    checkpoint_path: &str,
    summary_path: &str,
    root: &Path,
    train_token_budget: usize,
    val_token_budget: usize,
    report_every: usize,
    profile: &str,
    oracle_radius: usize,
    oracle_stride: usize,
) -> Result<TokenConker3NgramBulkReport, String> {
    let mut config = BulkConfig::named_prebuilt_profile(profile, report_every)?;
    config.min_row_total = ORACLE_MIN_ROW_TOTAL;
    let (runner, metadata) =
        load_token_conker3_checkpoint_runner_and_metadata(checkpoint_path, summary_path)?;
    let (oracle, train_tokens_built, train_build_elapsed_sec) = build_frozen_oracle_train_table(
        root,
        train_token_budget,
        runner.vocab_size(),
        &config,
        oracle_radius,
        oracle_stride,
    )?;
    let eval_tokens = take_val_tokens(root, val_token_budget)?;
    let byte_accounting = compute_tokens_per_byte(root, &eval_tokens)?;
    let mut report = evaluate_bulk_report(
        &runner,
        metadata,
        eval_tokens,
        byte_accounting,
        oracle,
        config,
        format!(
            "prebuilt_token_oracle_frozen:per_order_radius=stride={oracle_stride}:topk={ORACLE_TOP_K}"
        ),
        train_tokens_built,
        train_build_elapsed_sec,
        false,
    )?;
    report.scoring_mode = "oracle_top4_sparse_logit_residual_bulk".to_string();
    report.collision_control = "bucket_plus_fingerprint_plus_top4".to_string();
    Ok(report)
}

pub fn run_token_conker3_ngram_bulk_oracle_trust_prebuilt_from_data_root(
    checkpoint_path: &str,
    summary_path: &str,
    root: &Path,
    train_token_budget: usize,
    val_token_budget: usize,
    report_every: usize,
    profile: &str,
    oracle_stride: usize,
) -> Result<TokenConker3NgramBulkReport, String> {
    let config = BulkConfig::named_prebuilt_profile(profile, report_every)?;
    let (runner, metadata) =
        load_token_conker3_checkpoint_runner_and_metadata(checkpoint_path, summary_path)?;
    let build_start = Instant::now();
    let train_tokens = load_train_tokens(root, train_token_budget)?;
    let train_tokens_built = train_tokens.len();
    let (raw, trust) = rayon::join(
        || build_frozen_train_table_from_tokens(&train_tokens, runner.vocab_size(), &config),
        || {
            build_frozen_oracle_trust_table_from_tokens(
                &train_tokens,
                runner.vocab_size(),
                &config,
                oracle_stride,
            )
        },
    );
    let raw = raw?;
    let trust = trust?;
    let train_build_elapsed_sec = build_start.elapsed().as_secs_f64();
    let eval_tokens = take_val_tokens(root, val_token_budget)?;
    let byte_accounting = compute_tokens_per_byte(root, &eval_tokens)?;
    let mut report = evaluate_bulk_report_with_oracle_trust(
        &runner,
        metadata,
        eval_tokens,
        byte_accounting,
        raw,
        trust,
        config,
        format!("prebuilt_train_frozen_plus_oracle_trust:stride={oracle_stride}"),
        train_tokens_built,
        train_build_elapsed_sec,
    )?;
    report.scoring_mode = "raw_sparse_logit_residual_plus_oracle_trust".to_string();
    report.collision_control = "bucket_plus_fingerprint_plus_oracle_trust".to_string();
    Ok(report)
}

pub fn run_token_conker3_ngram_bulk_oracle_pruned_prebuilt_from_data_root(
    checkpoint_path: &str,
    summary_path: &str,
    root: &Path,
    train_token_budget: usize,
    val_token_budget: usize,
    report_every: usize,
    profile: &str,
    oracle_stride: usize,
) -> Result<TokenConker3NgramBulkReport, String> {
    let config = BulkConfig::named_prebuilt_profile(profile, report_every)?;
    let (runner, metadata) =
        load_token_conker3_checkpoint_runner_and_metadata(checkpoint_path, summary_path)?;
    let build_start = Instant::now();
    let train_tokens = load_train_tokens(root, train_token_budget)?;
    let train_tokens_built = train_tokens.len();
    let (raw, trust) = rayon::join(
        || build_frozen_train_table_from_tokens(&train_tokens, runner.vocab_size(), &config),
        || {
            build_frozen_oracle_trust_table_from_tokens(
                &train_tokens,
                runner.vocab_size(),
                &config,
                oracle_stride,
            )
        },
    );
    let mut raw = raw?;
    let trust = trust?;
    let train_build_elapsed_sec = build_start.elapsed().as_secs_f64();
    prune_raw_table_with_oracle_trust(&mut raw, &trust);
    let eval_tokens = take_val_tokens(root, val_token_budget)?;
    let byte_accounting = compute_tokens_per_byte(root, &eval_tokens)?;
    let mut report = evaluate_bulk_report(
        &runner,
        metadata,
        eval_tokens,
        byte_accounting,
        raw,
        config,
        format!("prebuilt_train_frozen_oracle_pruned:stride={oracle_stride}"),
        train_tokens_built,
        train_build_elapsed_sec,
        false,
    )?;
    report.scoring_mode = "raw_sparse_logit_residual_oracle_pruned".to_string();
    report.collision_control = "bucket_plus_fingerprint_plus_oracle_prune".to_string();
    Ok(report)
}

pub fn run_token_conker3_ngram_bulk_oracle_budgeted_prebuilt_from_data_root(
    checkpoint_path: &str,
    summary_path: &str,
    root: &Path,
    train_token_budget: usize,
    val_token_budget: usize,
    report_every: usize,
    profile: &str,
    oracle_stride: usize,
) -> Result<TokenConker3NgramBulkReport, String> {
    let config = BulkConfig::named_prebuilt_profile(profile, report_every)?;
    let (runner, metadata) =
        load_token_conker3_checkpoint_runner_and_metadata(checkpoint_path, summary_path)?;
    let build_start = Instant::now();
    let train_tokens = load_train_tokens(root, train_token_budget)?;
    let train_tokens_built = train_tokens.len();
    let (raw, trust) = rayon::join(
        || build_frozen_train_table_from_tokens(&train_tokens, runner.vocab_size(), &config),
        || {
            build_frozen_oracle_trust_table_from_tokens(
                &train_tokens,
                runner.vocab_size(),
                &config,
                oracle_stride,
            )
        },
    );
    let mut raw = raw?;
    let trust = trust?;
    let train_build_elapsed_sec = build_start.elapsed().as_secs_f64();
    budget_raw_table_with_oracle_trust(&mut raw, &trust);
    let eval_tokens = take_val_tokens(root, val_token_budget)?;
    let byte_accounting = compute_tokens_per_byte(root, &eval_tokens)?;
    let mut report = evaluate_bulk_report(
        &runner,
        metadata,
        eval_tokens,
        byte_accounting,
        raw,
        config,
        format!("prebuilt_train_frozen_oracle_budgeted:stride={oracle_stride}"),
        train_tokens_built,
        train_build_elapsed_sec,
        false,
    )?;
    report.scoring_mode = "raw_sparse_logit_residual_oracle_budgeted".to_string();
    report.collision_control = "bucket_plus_fingerprint_plus_oracle_budget".to_string();
    Ok(report)
}

pub fn build_token_conker3_ngram_oracle_budgeted_table_from_data_root(
    root: &Path,
    artifact_path: &Path,
    train_token_budget: usize,
    report_every: usize,
    profile: &str,
    oracle_stride: usize,
) -> Result<String, String> {
    let config = BulkConfig::named_prebuilt_profile(profile, report_every)?;
    let build_start = Instant::now();
    let train_tokens = load_train_tokens(root, train_token_budget)?;
    let train_tokens_built = train_tokens.len();
    let (raw, trust) = rayon::join(
        || build_frozen_train_table_from_tokens(&train_tokens, DEFAULT_VOCAB_SIZE, &config),
        || {
            build_frozen_oracle_trust_table_from_tokens(
                &train_tokens,
                DEFAULT_VOCAB_SIZE,
                &config,
                oracle_stride,
            )
        },
    );
    let mut raw = raw?;
    let trust = trust?;
    let train_build_elapsed_sec = build_start.elapsed().as_secs_f64();
    budget_raw_table_with_oracle_trust(&mut raw, &trust);
    save_ngram_table_artifact(
        artifact_path,
        &raw,
        train_tokens_built,
        train_build_elapsed_sec,
    )?;
    Ok(format!(
        "chronohorn_token_conker3_ngram_budgeted_table\nartifact_path: {}\ntable_source: prebuilt_train_frozen_oracle_budgeted:stride={oracle_stride}\ntrain_tokens_built: {}\ntrain_build_elapsed_sec: {:.6}\nngram_orders: {:?}\nngram_bucket_profile: {:?}\ncollision_control: bucket_plus_fingerprint_plus_oracle_budget\n",
        artifact_path.display(),
        train_tokens_built,
        train_build_elapsed_sec,
        raw.config.enabled_orders,
        raw.config.bucket_counts,
    ))
}

pub fn build_token_conker3_ngram_oracle_row_stats_from_data_root(
    root: &Path,
    artifact_path: &Path,
    train_token_budget: usize,
    report_every: usize,
    profile: &str,
    oracle_stride: usize,
) -> Result<String, String> {
    let config = BulkConfig::named_prebuilt_profile(profile, report_every)?;
    let build_start = Instant::now();
    let train_tokens = load_train_tokens(root, train_token_budget)?;
    let train_tokens_built = train_tokens.len();
    let (raw, trust) = rayon::join(
        || build_frozen_train_table_from_tokens(&train_tokens, DEFAULT_VOCAB_SIZE, &config),
        || {
            build_frozen_oracle_trust_table_from_tokens(
                &train_tokens,
                DEFAULT_VOCAB_SIZE,
                &config,
                oracle_stride,
            )
        },
    );
    let raw = raw?;
    let trust = trust?;
    let train_build_elapsed_sec = build_start.elapsed().as_secs_f64();
    let stats = build_oracle_budgeted_row_stats_artifact(
        &raw,
        &trust,
        train_tokens_built,
        train_build_elapsed_sec,
    );
    save_oracle_budgeted_row_stats_artifact(artifact_path, &stats)?;
    let total_rows: usize = stats.tables.iter().map(|rows| rows.len()).sum();
    Ok(format!(
        "chronohorn_token_conker3_ngram_row_stats\nartifact_path: {}\nrow_source: prebuilt_train_frozen_oracle_budgeted:stride={oracle_stride}\ntrain_tokens_built: {}\ntrain_build_elapsed_sec: {:.6}\nrow_count: {}\nngram_orders: {:?}\nngram_bucket_profile: {:?}\n",
        artifact_path.display(),
        train_tokens_built,
        train_build_elapsed_sec,
        total_rows,
        stats.config.enabled_orders,
        stats.config.bucket_counts,
    ))
}

pub fn pack_token_conker3_ngram_oracle_row_stats_artifact(
    stats_artifact_path: &Path,
    packed_artifact_path: &Path,
    target_bytes: usize,
) -> Result<String, String> {
    let stats = load_oracle_budgeted_row_stats_artifact(stats_artifact_path)?;
    let (packed, packed_bytes, kept_rows) =
        pack_oracle_budgeted_row_stats_artifact(&stats, target_bytes)?;
    save_ngram_table_artifact(
        packed_artifact_path,
        &packed,
        stats.train_tokens_built,
        stats.train_build_elapsed_sec,
    )?;
    Ok(format!(
        "chronohorn_token_conker3_ngram_packed_table\nstats_artifact_path: {}\npacked_artifact_path: {}\ntarget_bytes: {}\nestimated_packed_bytes: {}\nkept_rows: {}\nngram_orders: {:?}\nngram_bucket_profile: {:?}\ncollision_control: bucket_plus_fingerprint_plus_oracle_budget_packed\n",
        stats_artifact_path.display(),
        packed_artifact_path.display(),
        target_bytes,
        packed_bytes,
        kept_rows,
        packed.config.enabled_orders,
        packed.config.bucket_counts,
    ))
}

pub fn run_token_conker3_ngram_bulk_from_table_artifact(
    checkpoint_path: &str,
    summary_path: &str,
    root: &Path,
    artifact_path: &Path,
    val_token_budget: usize,
    report_every: usize,
) -> Result<TokenConker3NgramBulkReport, String> {
    let (runner, metadata) =
        load_token_conker3_checkpoint_runner_and_metadata(checkpoint_path, summary_path)?;
    let (mut ngram, train_tokens_built, train_build_elapsed_sec) =
        load_ngram_table_artifact(artifact_path)?;
    let mut config = ngram.config.clone();
    config.report_every = report_every.max(1);
    ngram.config.report_every = config.report_every;
    let eval_tokens = take_val_tokens(root, val_token_budget)?;
    let byte_accounting = compute_tokens_per_byte(root, &eval_tokens)?;
    evaluate_bulk_report(
        &runner,
        metadata,
        eval_tokens,
        byte_accounting,
        ngram,
        config,
        format!("artifact:{}", artifact_path.display()),
        train_tokens_built,
        train_build_elapsed_sec,
        false,
    )
}

fn evaluate_bulk_report(
    runner: &TokenConker3CheckpointRunner,
    metadata: TokenConker3CheckpointMetadata,
    eval_tokens: Vec<usize>,
    byte_accounting: Option<ByteAccountingReport>,
    mut ngram: NgramBulkState,
    config: BulkConfig,
    table_source: String,
    train_tokens_built: usize,
    train_build_elapsed_sec: f64,
    update_tables_during_eval: bool,
) -> Result<TokenConker3NgramBulkReport, String> {
    let (mut base_state, mut base_history) = runner.snapshot_runtime();
    let mut score_history = VecDeque::with_capacity(config.max_order().saturating_sub(1));
    let feature_dims = runner.feature_dims();
    let start = Instant::now();
    let mut total_neural_nats = 0.0f64;
    let mut total_mixed_nats = 0.0f64;
    let mut total_scored = 0usize;
    let mut interval = IntervalStats::new(&config.enabled_orders);
    let mut checkpoints = Vec::new();
    let mut block_tokens = Vec::with_capacity(READOUT_BLOCK);
    let mut linear_batch = Vec::with_capacity(READOUT_BLOCK * feature_dims.linear);
    let mut local_batch = Vec::with_capacity(READOUT_BLOCK * feature_dims.local);

    for chunk in eval_tokens.chunks(READOUT_BLOCK) {
        block_tokens.clear();
        linear_batch.clear();
        local_batch.clear();

        for &token in chunk {
            runner.append_feature_batches_from_prefix(
                &base_state,
                &base_history,
                &mut linear_batch,
                &mut local_batch,
            )?;
            let row = ngram
                .lookup_counter_from_history(&score_history)
                .map(|(row, order_index, _order)| (row.clone(), order_index));
            block_tokens.push(BulkBlockToken { gold: token, row });

            runner.advance_runtime_state(&mut base_state, &mut base_history, token)?;
            push_history(
                &mut score_history,
                config.max_order().saturating_sub(1),
                token,
            );
            if update_tables_during_eval {
                ngram.observe_token(token);
            }
        }

        let batched_logits = runner.predict_combined_logits_batch_from_features(
            &linear_batch,
            &local_batch,
            block_tokens.len(),
        )?;
        let vocab_size = runner.vocab_size();

        for (block_index, token_meta) in block_tokens.iter().enumerate() {
            let start_index = block_index * vocab_size;
            let end_index = start_index + vocab_size;
            let logits = &batched_logits[start_index..end_index];
            let summary = summarize_logits(logits, token_meta.gold);
            let log_p_neural = summary.gold_logit - summary.log_z;
            total_neural_nats -= log_p_neural;
            interval.neural_nats -= log_p_neural;

            let log_p_mixed = if let Some((row, order_index)) = &token_meta.row {
                let alpha = entropy_weight_from_summary(summary.entropy, &config);
                let (log_p, gold_is_argmax) = sparse_logit_residual_logprob(
                    logits,
                    token_meta.gold,
                    summary.gold_logit,
                    summary.log_z,
                    row,
                    vocab_size,
                    alpha,
                    config.residual_cap,
                );
                interval.alpha_sum += alpha;
                interval.row_total_sum += row.total as f64;
                interval.matched_positions += 1;
                if *order_index < interval.per_order_fires.len() {
                    interval.per_order_fires[*order_index] += 1;
                    if gold_is_argmax {
                        interval.per_order_gold_argmax_hits[*order_index] += 1;
                    }
                }
                log_p
            } else {
                interval.no_match_positions += 1;
                log_p_neural
            };
            total_mixed_nats -= log_p_mixed;
            interval.mixed_nats -= log_p_mixed;
            total_scored += 1;

            if total_scored % config.report_every == 0 {
                checkpoints.push(finish_interval(
                    &interval,
                    total_scored,
                    total_neural_nats,
                    total_mixed_nats,
                    start.elapsed().as_secs_f64(),
                    byte_accounting.as_ref().map(|row| row.tokens_per_byte),
                ));
                interval.reset(total_scored);
            }
        }
    }

    if total_scored > interval.start_token {
        checkpoints.push(finish_interval(
            &interval,
            total_scored,
            total_neural_nats,
            total_mixed_nats,
            start.elapsed().as_secs_f64(),
            byte_accounting.as_ref().map(|row| row.tokens_per_byte),
        ));
    }

    let elapsed_sec = start.elapsed().as_secs_f64();
    let denom = (total_scored as f64 * std::f64::consts::LN_2).max(f64::MIN_POSITIVE);
    let eval_bpt_neural = total_neural_nats / denom;
    let eval_bpt_mixed = total_mixed_nats / denom;

    Ok(TokenConker3NgramBulkReport {
        checkpoint_path: metadata.checkpoint_path,
        summary_path: metadata.summary_path,
        train_seq_len: metadata.train_seq_len,
        eval_tokens: total_scored,
        scoring_mode: "sparse_logit_residual_bulk".to_string(),
        collision_control: "bucket_plus_fingerprint".to_string(),
        table_source,
        train_tokens_built,
        train_build_elapsed_sec,
        ngram_orders: config.enabled_orders.clone(),
        ngram_bucket_profile: config.bucket_counts.clone(),
        mix_base: config.mix_base,
        mix_span: config.mix_span,
        entropy_midpoint: config.entropy_midpoint,
        entropy_slope: config.entropy_slope,
        min_row_total: config.min_row_total,
        residual_cap: config.residual_cap,
        report_every: config.report_every,
        eval_bpt_neural,
        eval_bpt_mixed,
        eval_tokens_per_byte: byte_accounting.as_ref().map(|row| row.tokens_per_byte),
        eval_bytes_per_token: byte_accounting.as_ref().map(|row| row.bytes_per_token),
        eval_bpb_neural: byte_accounting
            .as_ref()
            .map(|row| eval_bpt_neural * row.tokens_per_byte),
        eval_bpb_mixed: byte_accounting
            .as_ref()
            .map(|row| eval_bpt_mixed * row.tokens_per_byte),
        elapsed_sec,
        tokens_per_sec: total_scored as f64 / elapsed_sec.max(f64::MIN_POSITIVE),
        checkpoints,
    })
}

fn evaluate_bulk_report_with_priority_cache(
    runner: &TokenConker3CheckpointRunner,
    metadata: TokenConker3CheckpointMetadata,
    eval_tokens: Vec<usize>,
    byte_accounting: Option<ByteAccountingReport>,
    frozen: NgramBulkState,
    config: BulkConfig,
    table_source: String,
    train_tokens_built: usize,
    train_build_elapsed_sec: f64,
) -> Result<TokenConker3NgramBulkReport, String> {
    let (mut base_state, mut base_history) = runner.snapshot_runtime();
    let mut score_history = VecDeque::with_capacity(config.max_order().saturating_sub(1));
    let mut online = NgramBulkState::new(runner.vocab_size(), config.clone());
    let start = Instant::now();
    let mut total_neural_nats = 0.0f64;
    let mut total_mixed_nats = 0.0f64;
    let mut total_scored = 0usize;
    let mut interval = IntervalStats::new(&config.enabled_orders);
    let mut checkpoints = Vec::new();

    for &token in &eval_tokens {
        let logits = runner.predict_combined_logits_from(&base_state, &base_history)?;
        let summary = summarize_logits(&logits, token);
        let log_p_neural = summary.gold_logit - summary.log_z;
        total_neural_nats -= log_p_neural;
        interval.neural_nats -= log_p_neural;

        let frozen_hit = frozen.lookup_counter_from_history(&score_history);
        let online_hit = online.lookup_counter_from_history(&score_history);
        let selected = select_priority_row(summary, frozen_hit, online_hit);

        let log_p_mixed = if let Some(selected) = selected {
            let (log_p, gold_is_argmax) = sparse_logit_residual_logprob(
                &logits,
                token,
                summary.gold_logit,
                summary.log_z,
                selected.row,
                runner.vocab_size(),
                selected.alpha,
                config.residual_cap,
            );
            interval.alpha_sum += selected.alpha;
            interval.row_total_sum += selected.row.total as f64;
            interval.matched_positions += 1;
            if selected.order_index < interval.per_order_fires.len() {
                interval.per_order_fires[selected.order_index] += 1;
                if gold_is_argmax {
                    interval.per_order_gold_argmax_hits[selected.order_index] += 1;
                }
            }
            log_p
        } else {
            interval.no_match_positions += 1;
            log_p_neural
        };

        total_mixed_nats -= log_p_mixed;
        interval.mixed_nats -= log_p_mixed;
        total_scored += 1;

        runner.advance_runtime_state(&mut base_state, &mut base_history, token)?;
        push_history(
            &mut score_history,
            config.max_order().saturating_sub(1),
            token,
        );
        online.observe_token(token);

        if total_scored % config.report_every == 0 {
            checkpoints.push(finish_interval(
                &interval,
                total_scored,
                total_neural_nats,
                total_mixed_nats,
                start.elapsed().as_secs_f64(),
                byte_accounting.as_ref().map(|row| row.tokens_per_byte),
            ));
            interval.reset(total_scored);
        }
    }

    if total_scored > interval.start_token {
        checkpoints.push(finish_interval(
            &interval,
            total_scored,
            total_neural_nats,
            total_mixed_nats,
            start.elapsed().as_secs_f64(),
            byte_accounting.as_ref().map(|row| row.tokens_per_byte),
        ));
    }

    let elapsed_sec = start.elapsed().as_secs_f64();
    let denom = (total_scored as f64 * std::f64::consts::LN_2).max(f64::MIN_POSITIVE);
    let eval_bpt_neural = total_neural_nats / denom;
    let eval_bpt_mixed = total_mixed_nats / denom;

    Ok(TokenConker3NgramBulkReport {
        checkpoint_path: metadata.checkpoint_path,
        summary_path: metadata.summary_path,
        train_seq_len: metadata.train_seq_len,
        eval_tokens: total_scored,
        scoring_mode: "priority_cache_sparse_logit_residual_bulk".to_string(),
        collision_control: "bucket_plus_fingerprint".to_string(),
        table_source,
        train_tokens_built,
        train_build_elapsed_sec,
        ngram_orders: config.enabled_orders.clone(),
        ngram_bucket_profile: config.bucket_counts.clone(),
        mix_base: config.mix_base,
        mix_span: config.mix_span,
        entropy_midpoint: config.entropy_midpoint,
        entropy_slope: config.entropy_slope,
        min_row_total: config.min_row_total,
        residual_cap: config.residual_cap,
        report_every: config.report_every,
        eval_bpt_neural,
        eval_bpt_mixed,
        eval_tokens_per_byte: byte_accounting.as_ref().map(|row| row.tokens_per_byte),
        eval_bytes_per_token: byte_accounting.as_ref().map(|row| row.bytes_per_token),
        eval_bpb_neural: byte_accounting
            .as_ref()
            .map(|row| eval_bpt_neural * row.tokens_per_byte),
        eval_bpb_mixed: byte_accounting
            .as_ref()
            .map(|row| eval_bpt_mixed * row.tokens_per_byte),
        elapsed_sec,
        tokens_per_sec: total_scored as f64 / elapsed_sec.max(f64::MIN_POSITIVE),
        checkpoints,
    })
}

fn evaluate_bulk_report_with_oracle_trust(
    runner: &TokenConker3CheckpointRunner,
    metadata: TokenConker3CheckpointMetadata,
    eval_tokens: Vec<usize>,
    byte_accounting: Option<ByteAccountingReport>,
    raw: NgramBulkState,
    trust: OracleTrustState,
    config: BulkConfig,
    table_source: String,
    train_tokens_built: usize,
    train_build_elapsed_sec: f64,
) -> Result<TokenConker3NgramBulkReport, String> {
    let (mut base_state, mut base_history) = runner.snapshot_runtime();
    let mut score_history = VecDeque::with_capacity(config.max_order().saturating_sub(1));
    let start = Instant::now();
    let mut total_neural_nats = 0.0f64;
    let mut total_mixed_nats = 0.0f64;
    let mut total_scored = 0usize;
    let mut interval = IntervalStats::new(&config.enabled_orders);
    let mut checkpoints = Vec::new();

    for &token in &eval_tokens {
        let logits = runner.predict_combined_logits_from(&base_state, &base_history)?;
        let summary = summarize_logits(&logits, token);
        let log_p_neural = summary.gold_logit - summary.log_z;
        total_neural_nats -= log_p_neural;
        interval.neural_nats -= log_p_neural;

        let log_p_mixed = if let Some((raw_row, order_index, order)) =
            raw.lookup_counter_from_history(&score_history)
        {
            let trust_hit = trust.lookup_for_order_from_history(&score_history, order);
            let alpha = entropy_weight_from_summary(summary.entropy, &config);
            match oracle_trust_decision(raw_row, trust_hit, order, config.min_row_total) {
                OracleTrustDecision::UseMasked(masked) => {
                    let (log_p, gold_is_argmax) = sparse_logit_residual_logprob(
                        &logits,
                        token,
                        summary.gold_logit,
                        summary.log_z,
                        &masked,
                        runner.vocab_size(),
                        alpha,
                        config.residual_cap,
                    );
                    interval.alpha_sum += alpha;
                    interval.row_total_sum += masked.total as f64;
                    interval.matched_positions += 1;
                    if order_index < interval.per_order_fires.len() {
                        interval.per_order_fires[order_index] += 1;
                        if gold_is_argmax {
                            interval.per_order_gold_argmax_hits[order_index] += 1;
                        }
                    }
                    log_p
                }
                OracleTrustDecision::UseRaw => {
                    let (log_p, gold_is_argmax) = sparse_logit_residual_logprob(
                        &logits,
                        token,
                        summary.gold_logit,
                        summary.log_z,
                        raw_row,
                        runner.vocab_size(),
                        alpha,
                        config.residual_cap,
                    );
                    interval.alpha_sum += alpha;
                    interval.row_total_sum += raw_row.total as f64;
                    interval.matched_positions += 1;
                    if order_index < interval.per_order_fires.len() {
                        interval.per_order_fires[order_index] += 1;
                        if gold_is_argmax {
                            interval.per_order_gold_argmax_hits[order_index] += 1;
                        }
                    }
                    log_p
                }
            }
        } else {
            interval.no_match_positions += 1;
            log_p_neural
        };
        total_mixed_nats -= log_p_mixed;
        interval.mixed_nats -= log_p_mixed;
        total_scored += 1;

        runner.advance_runtime_state(&mut base_state, &mut base_history, token)?;
        push_history(
            &mut score_history,
            config.max_order().saturating_sub(1),
            token,
        );

        if total_scored % config.report_every == 0 {
            checkpoints.push(finish_interval(
                &interval,
                total_scored,
                total_neural_nats,
                total_mixed_nats,
                start.elapsed().as_secs_f64(),
                byte_accounting.as_ref().map(|row| row.tokens_per_byte),
            ));
            interval.reset(total_scored);
        }
    }

    if total_scored > interval.start_token {
        checkpoints.push(finish_interval(
            &interval,
            total_scored,
            total_neural_nats,
            total_mixed_nats,
            start.elapsed().as_secs_f64(),
            byte_accounting.as_ref().map(|row| row.tokens_per_byte),
        ));
    }

    let elapsed_sec = start.elapsed().as_secs_f64();
    let denom = (total_scored as f64 * std::f64::consts::LN_2).max(f64::MIN_POSITIVE);
    let eval_bpt_neural = total_neural_nats / denom;
    let eval_bpt_mixed = total_mixed_nats / denom;

    Ok(TokenConker3NgramBulkReport {
        checkpoint_path: metadata.checkpoint_path,
        summary_path: metadata.summary_path,
        train_seq_len: metadata.train_seq_len,
        eval_tokens: total_scored,
        scoring_mode: "raw_sparse_logit_residual_plus_oracle_trust".to_string(),
        collision_control: "bucket_plus_fingerprint_plus_oracle_trust".to_string(),
        table_source,
        train_tokens_built,
        train_build_elapsed_sec,
        ngram_orders: config.enabled_orders.clone(),
        ngram_bucket_profile: config.bucket_counts.clone(),
        mix_base: config.mix_base,
        mix_span: config.mix_span,
        entropy_midpoint: config.entropy_midpoint,
        entropy_slope: config.entropy_slope,
        min_row_total: config.min_row_total,
        residual_cap: config.residual_cap,
        report_every: config.report_every,
        eval_bpt_neural,
        eval_bpt_mixed,
        eval_tokens_per_byte: byte_accounting.as_ref().map(|row| row.tokens_per_byte),
        eval_bytes_per_token: byte_accounting.as_ref().map(|row| row.bytes_per_token),
        eval_bpb_neural: byte_accounting
            .as_ref()
            .map(|row| eval_bpt_neural * row.tokens_per_byte),
        eval_bpb_mixed: byte_accounting
            .as_ref()
            .map(|row| eval_bpt_mixed * row.tokens_per_byte),
        elapsed_sec,
        tokens_per_sec: total_scored as f64 / elapsed_sec.max(f64::MIN_POSITIVE),
        checkpoints,
    })
}

fn build_frozen_train_table(
    root: &Path,
    token_budget: usize,
    vocab_size: usize,
    config: &BulkConfig,
) -> Result<(NgramBulkState, usize, f64), String> {
    let start = Instant::now();
    let mut ngram = NgramBulkState::new(vocab_size, config.clone());
    let mut tokens_built = 0usize;
    for shard in list_shards(root, "fineweb_train_")? {
        if tokens_built >= token_budget {
            break;
        }
        let left = token_budget - tokens_built;
        let scanned = scan_shard_tokens(&shard, left, |token| ngram.observe_token(token))?;
        tokens_built += scanned;
    }
    if tokens_built < 4 {
        return Err("need at least 4 train tokens to build a frozen n-gram table".to_string());
    }
    ngram.history.clear();
    Ok((ngram, tokens_built, start.elapsed().as_secs_f64()))
}

fn build_frozen_train_table_from_tokens(
    train_tokens: &[usize],
    vocab_size: usize,
    config: &BulkConfig,
) -> Result<NgramBulkState, String> {
    if train_tokens.len() < 4 {
        return Err("need at least 4 train tokens to build a frozen n-gram table".to_string());
    }
    let tables = config
        .enabled_orders
        .par_iter()
        .map(|&order| build_raw_order_table(train_tokens, vocab_size, config, order))
        .collect::<Vec<_>>();
    Ok(NgramBulkState {
        config: config.clone(),
        vocab_size,
        history: VecDeque::with_capacity(config.max_order().saturating_sub(1)),
        tables,
    })
}

fn build_frozen_oracle_train_table(
    root: &Path,
    token_budget: usize,
    vocab_size: usize,
    config: &BulkConfig,
    _oracle_radius: usize,
    oracle_stride: usize,
) -> Result<(NgramBulkState, usize, f64), String> {
    let start = Instant::now();
    let train_tokens = load_train_tokens(root, token_budget)?;
    let mut oracle = NgramBulkState::new(vocab_size, config.clone());
    for (table_index, &order) in config.enabled_orders.iter().enumerate() {
        let radius = order.saturating_sub(1);
        if radius == 0 {
            continue;
        }
        if train_tokens.len() <= radius.saturating_mul(2) {
            continue;
        }
        let dataset = token_oracle_target_dataset_from_tokens(
            format!("chronohorn_oracle_table_train_o{order}_r{radius}"),
            &train_tokens,
            radius,
            oracle_stride,
        )?;
        for (position, pairs) in token_oracle_teacher_candidate_pairs(&dataset) {
            if position >= train_tokens.len() {
                continue;
            }
            let quantized = quantize_oracle_pairs(&pairs, vocab_size);
            if quantized.is_empty() {
                continue;
            }
            let context_len = order.saturating_sub(1);
            if position < context_len {
                continue;
            }
            let context = &train_tokens[position - context_len..position];
            let (bucket, fingerprint) = oracle.context_key_from_slice(context, order);
            let bucket_rows = oracle.tables[table_index].entry(bucket).or_default();
            let row = if let Some(row) = bucket_rows
                .iter_mut()
                .find(|row| row.fingerprint == fingerprint)
            {
                &mut row.counter
            } else {
                bucket_rows.push(FingerprintedCounter {
                    fingerprint,
                    counter: SparseCounter::default(),
                });
                &mut bucket_rows
                    .last_mut()
                    .expect("just pushed oracle row")
                    .counter
            };
            row.total = row.total.saturating_add(ORACLE_QUANT_SCALE);
            for &(token, weight) in &quantized {
                row.add_weight(token, weight);
            }
        }
    }
    oracle.prune_all_rows_top_k(ORACLE_TOP_K);
    oracle.history.clear();
    Ok((oracle, train_tokens.len(), start.elapsed().as_secs_f64()))
}

fn build_frozen_oracle_trust_table_from_tokens(
    train_tokens: &[usize],
    vocab_size: usize,
    config: &BulkConfig,
    oracle_stride: usize,
) -> Result<OracleTrustState, String> {
    if oracle_stride == 0 {
        return Err("oracle stride must be positive".to_string());
    }
    let tables = config
        .enabled_orders
        .par_iter()
        .map(|&order| {
            build_trust_order_table(train_tokens, vocab_size, config, order, oracle_stride)
        })
        .collect::<Vec<_>>();
    let mut trust = OracleTrustState {
        config: config.clone(),
        tables,
    };
    trust.prune_all_rows_top_k(ORACLE_TOP_K);
    Ok(trust)
}

fn repeated_span_hashes(train_tokens: &[usize], radius: usize) -> U64FastMap<u8> {
    let mut repeated = U64FastMap::with_hasher(IdentityBuildHasher);
    if radius == 0 || train_tokens.len() <= radius.saturating_mul(2) {
        return repeated;
    }
    for position in radius..(train_tokens.len() - radius) {
        let span_hash = oracle_span_hash(&train_tokens[position - radius..=position + radius]);
        match repeated.entry(span_hash) {
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                if *entry.get() < 2 {
                    *entry.get_mut() = 2;
                }
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(1);
            }
        }
    }
    repeated.retain(|_, seen| *seen > 1);
    repeated
}

fn build_raw_order_table(
    train_tokens: &[usize],
    vocab_size: usize,
    config: &BulkConfig,
    order: usize,
) -> HashMap<u32, Vec<FingerprintedCounter>> {
    let context_len = order.saturating_sub(1);
    let mut table: HashMap<u32, Vec<FingerprintedCounter>> = HashMap::new();
    if context_len == 0 || train_tokens.len() <= context_len {
        return table;
    }
    for position in context_len..train_tokens.len() {
        let token = train_tokens[position];
        if token >= vocab_size {
            continue;
        }
        let context = &train_tokens[position - context_len..position];
        let (bucket, fingerprint) = context_key_from_iter(config, context.iter().copied(), order);
        let bucket_rows = table.entry(bucket).or_default();
        if let Some(row) = bucket_rows
            .iter_mut()
            .find(|row| row.fingerprint == fingerprint)
        {
            row.counter.increment(token);
        } else {
            let mut counter = SparseCounter::default();
            counter.increment(token);
            bucket_rows.push(FingerprintedCounter {
                fingerprint,
                counter,
            });
        }
    }
    table
}

fn build_trust_order_table(
    train_tokens: &[usize],
    vocab_size: usize,
    config: &BulkConfig,
    order: usize,
    oracle_stride: usize,
) -> HashMap<u32, Vec<FingerprintedOracleTrustCounter>> {
    let radius = order.saturating_sub(1);
    let mut table: HashMap<u32, Vec<FingerprintedOracleTrustCounter>> = HashMap::new();
    if radius == 0 || train_tokens.len() <= radius.saturating_mul(2) {
        return table;
    }
    let repeated = repeated_span_hashes(train_tokens, radius);
    for position in radius..(train_tokens.len() - radius) {
        if (position - radius) % oracle_stride != 0 {
            continue;
        }
        let context = &train_tokens[position - radius..position];
        let (bucket, fingerprint) = context_key_from_iter(config, context.iter().copied(), order);
        let bucket_rows = table.entry(bucket).or_default();
        let row = if let Some(row) = bucket_rows
            .iter_mut()
            .find(|row| row.fingerprint == fingerprint)
        {
            &mut row.counter
        } else {
            bucket_rows.push(FingerprintedOracleTrustCounter {
                fingerprint,
                counter: OracleTrustCounter::default(),
            });
            &mut bucket_rows
                .last_mut()
                .expect("just pushed oracle trust row")
                .counter
        };
        row.examples = row.examples.saturating_add(1);
        let span_hash = oracle_span_hash(&train_tokens[position - radius..=position + radius]);
        if repeated.contains_key(&span_hash) {
            let center = train_tokens[position];
            if center < vocab_size {
                row.top1_votes.increment(center);
                row.mass.total = row.mass.total.saturating_add(ORACLE_QUANT_SCALE);
                row.mass.add_weight(center, ORACLE_QUANT_SCALE);
            }
        }
    }
    table
}

fn oracle_span_hash(span: &[usize]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &token in span {
        let token_u64 = token as u64;
        hash ^= token_u64.wrapping_add(0x517c_c1b7_2722_0a95);
        hash = hash.rotate_left(13);
        hash = hash.wrapping_mul(0x9e37_79b1_85eb_ca87);
        hash ^= hash >> 33;
        hash = hash.wrapping_mul(0xff51_afd7_ed55_8ccd);
        hash ^= hash >> 33;
    }
    hash
}

fn prune_raw_table_with_oracle_trust(raw: &mut NgramBulkState, trust: &OracleTrustState) {
    for (table_index, &order) in raw.config.enabled_orders.iter().enumerate() {
        let Some(trust_table) = trust.tables.get(table_index) else {
            continue;
        };
        for (bucket, rows) in raw.tables[table_index].iter_mut() {
            let Some(trust_rows) = trust_table.get(bucket) else {
                continue;
            };
            rows.retain(|row| {
                let Some(trust_row) = trust_rows
                    .iter()
                    .find(|candidate| candidate.fingerprint == row.fingerprint)
                    .map(|candidate| &candidate.counter)
                else {
                    return keep_without_oracle(order, &row.counter);
                };
                keep_with_oracle(order, &row.counter, trust_row)
            });
        }
    }
}

fn budget_raw_table_with_oracle_trust(raw: &mut NgramBulkState, trust: &OracleTrustState) {
    let budget = oracle_threshold_budget_count(raw, trust);
    let mut ranked = Vec::new();
    for (table_index, &order) in raw.config.enabled_orders.iter().enumerate() {
        let trust_table = trust.tables.get(table_index);
        for (bucket, rows) in &raw.tables[table_index] {
            let trust_rows = trust_table.and_then(|table| table.get(bucket));
            for row in rows {
                let trust_row = trust_rows.and_then(|candidates| {
                    candidates
                        .iter()
                        .find(|candidate| candidate.fingerprint == row.fingerprint)
                        .map(|candidate| &candidate.counter)
                });
                let score = oracle_row_budget_score(order, &row.counter, trust_row);
                if score.is_finite() && score > 0.0 {
                    ranked.push((score, table_index, *bucket, row.fingerprint));
                }
            }
        }
    }
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
    let keep = ranked
        .into_iter()
        .take(budget)
        .map(|(_, table_index, bucket, fingerprint)| (table_index, bucket, fingerprint))
        .collect::<HashSet<_>>();
    for (table_index, rows_by_bucket) in raw.tables.iter_mut().enumerate() {
        for (bucket, rows) in rows_by_bucket.iter_mut() {
            rows.retain(|row| keep.contains(&(table_index, *bucket, row.fingerprint)));
        }
    }
}

fn oracle_threshold_budget_count(raw: &NgramBulkState, trust: &OracleTrustState) -> usize {
    let mut kept = 0usize;
    for (table_index, &order) in raw.config.enabled_orders.iter().enumerate() {
        let trust_table = trust.tables.get(table_index);
        for (bucket, rows) in &raw.tables[table_index] {
            let trust_rows = trust_table.and_then(|table| table.get(bucket));
            for row in rows {
                let trust_row = trust_rows.and_then(|candidates| {
                    candidates
                        .iter()
                        .find(|candidate| candidate.fingerprint == row.fingerprint)
                        .map(|candidate| &candidate.counter)
                });
                let keep = match trust_row {
                    Some(trust_row) => keep_with_oracle(order, &row.counter, trust_row),
                    None => keep_without_oracle(order, &row.counter),
                };
                if keep {
                    kept += 1;
                }
            }
        }
    }
    kept.max(1)
}

fn keep_without_oracle(order: usize, row: &SparseCounter) -> bool {
    match order {
        2 => row.total >= 16,
        3 => row.total >= 10,
        4 | 5 => row.total >= 6,
        6 | 7 => row.total >= 8,
        _ => row.total >= MIN_ROW_TOTAL,
    }
}

fn keep_with_oracle(order: usize, row: &SparseCounter, trust_row: &OracleTrustCounter) -> bool {
    let Some(metrics) = oracle_trust_metrics(trust_row) else {
        return keep_without_oracle(order, row);
    };
    let row_total = row.total.max(1) as f64;
    let support_penalty = 1.0 + 0.20 * (metrics.support_size.saturating_sub(1)) as f64;
    let stability = metrics.top1_agreement * metrics.top_mass;
    let score = order_weight(order) * row_total.ln_1p() * stability / support_penalty;
    let threshold = match order {
        2 => 0.42,
        3 => 0.52,
        4 => 0.66,
        5 => 0.72,
        6 => 0.58,
        7 => 0.50,
        _ => 0.60,
    };
    let min_examples = match order {
        2 => 8,
        3 => 6,
        4 | 5 => 4,
        6 | 7 => 4,
        _ => 4,
    };
    metrics.examples >= min_examples && score >= threshold
}

fn oracle_row_budget_score(
    order: usize,
    row: &SparseCounter,
    trust_row: Option<&OracleTrustCounter>,
) -> f64 {
    let row_total = row.total.max(1) as f64;
    let row_support_penalty = 1.0 + 0.15 * (row.counts.len().saturating_sub(1)) as f64;
    let base = order_weight(order) * row_total.ln_1p() / row_support_penalty;
    let Some(metrics) = trust_row.and_then(oracle_trust_metrics) else {
        return 0.20 * base;
    };
    let support_penalty = 1.0 + 0.20 * (metrics.support_size.saturating_sub(1)) as f64;
    let stability = metrics.top1_agreement * metrics.top_mass;
    let example_bonus = (metrics.examples as f64).ln_1p();
    let order_bonus = match order {
        5 => 1.20,
        4 => 1.10,
        3 => 0.95,
        6 => 0.70,
        7 => 0.55,
        2 => 0.30,
        _ => 0.25,
    };
    order_bonus * base * stability * example_bonus / support_penalty
}

fn oracle_row_budget_score_from_metrics(
    order: usize,
    row: &SparseCounter,
    metrics: Option<OracleTrustMetrics>,
) -> f64 {
    let row_total = row.total.max(1) as f64;
    let row_support_penalty = 1.0 + 0.15 * (row.counts.len().saturating_sub(1)) as f64;
    let base = order_weight(order) * row_total.ln_1p() / row_support_penalty;
    let Some(metrics) = metrics else {
        return 0.20 * base;
    };
    let support_penalty = 1.0 + 0.20 * (metrics.support_size.saturating_sub(1)) as f64;
    let stability = metrics.top1_agreement * metrics.top_mass;
    let example_bonus = (metrics.examples as f64).ln_1p();
    let order_bonus = match order {
        5 => 1.20,
        4 => 1.10,
        3 => 0.95,
        6 => 0.70,
        7 => 0.55,
        2 => 0.30,
        _ => 0.25,
    };
    order_bonus * base * stability * example_bonus / support_penalty
}

fn order_weight(order: usize) -> f64 {
    match order {
        5 => 1.00,
        4 => 0.95,
        3 => 0.78,
        6 => 0.58,
        7 => 0.42,
        2 => 0.28,
        _ => 0.20,
    }
}

fn stored_oracle_metrics(counter: Option<&OracleTrustCounter>) -> Option<StoredOracleRowMetrics> {
    let metrics = counter.and_then(oracle_trust_metrics)?;
    Some(StoredOracleRowMetrics {
        examples: metrics.examples,
        support_size: metrics.support_size as u32,
        top1_agreement: metrics.top1_agreement,
        top_mass: metrics.top_mass,
    })
}

fn restore_oracle_metrics(metrics: Option<StoredOracleRowMetrics>) -> Option<OracleTrustMetrics> {
    metrics.map(|metrics| OracleTrustMetrics {
        examples: metrics.examples,
        support_size: metrics.support_size as usize,
        top1_agreement: metrics.top1_agreement,
        top_mass: metrics.top_mass,
    })
}

fn build_oracle_budgeted_row_stats_artifact(
    raw: &NgramBulkState,
    trust: &OracleTrustState,
    train_tokens_built: usize,
    train_build_elapsed_sec: f64,
) -> OracleBudgetedRowStatsArtifact {
    let mut tables = Vec::with_capacity(raw.tables.len());
    for (table_index, rows_by_bucket) in raw.tables.iter().enumerate() {
        let trust_table = trust.tables.get(table_index);
        let mut rows = Vec::new();
        for (&bucket, bucket_rows) in rows_by_bucket {
            let trust_rows = trust_table.and_then(|table| table.get(&bucket));
            for row in bucket_rows {
                let oracle = trust_rows.and_then(|candidates| {
                    candidates
                        .iter()
                        .find(|candidate| candidate.fingerprint == row.fingerprint)
                        .map(|candidate| &candidate.counter)
                        .and_then(|counter| stored_oracle_metrics(Some(counter)))
                });
                rows.push(OracleBudgetedRowStat {
                    bucket,
                    fingerprint: row.fingerprint,
                    row: row.counter.clone(),
                    oracle,
                });
            }
        }
        rows.sort_by(|a, b| {
            a.bucket
                .cmp(&b.bucket)
                .then_with(|| a.fingerprint.cmp(&b.fingerprint))
        });
        tables.push(rows);
    }
    OracleBudgetedRowStatsArtifact {
        config: raw.config.clone(),
        vocab_size: raw.vocab_size,
        train_tokens_built,
        train_build_elapsed_sec,
        tables,
    }
}

fn packed_runtime_artifact_base_bytes(config: &BulkConfig) -> usize {
    TABLE_ARTIFACT_MAGIC.len()
        + 4
        + 8
        + 8
        + 4
        + 4
        + 8
        + 8
        + 8
        + 8
        + 4
        + 8
        + 4
        + 4 * config.enabled_orders.len()
        + 8 * config.bucket_counts.len()
        + 4
        + 8 * config.enabled_orders.len()
}

fn packed_runtime_row_bytes(row: &OracleBudgetedRowStat) -> usize {
    8 + 4 + 4 + row.row.counts.len() * (4 + 4)
}

fn pack_oracle_budgeted_row_stats_artifact(
    stats: &OracleBudgetedRowStatsArtifact,
    target_bytes: usize,
) -> Result<(NgramBulkState, usize, usize), String> {
    let base_bytes = packed_runtime_artifact_base_bytes(&stats.config);
    if target_bytes < base_bytes {
        return Err(format!(
            "target bytes {target_bytes} is smaller than minimum runtime artifact header {base_bytes}"
        ));
    }
    let mut ranked = Vec::new();
    for (table_index, rows) in stats.tables.iter().enumerate() {
        let order = stats
            .config
            .enabled_orders
            .get(table_index)
            .copied()
            .unwrap_or(MIN_ORDER);
        for (row_index, row) in rows.iter().enumerate() {
            let score = oracle_row_budget_score_from_metrics(
                order,
                &row.row,
                restore_oracle_metrics(row.oracle),
            );
            if score.is_finite() && score > 0.0 {
                ranked.push((score, table_index, row_index));
            }
        }
    }
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));

    let mut selected_rows = HashSet::new();
    let mut selected_buckets = HashSet::new();
    let mut packed_bytes = base_bytes;
    for (_, table_index, row_index) in ranked {
        let row = &stats.tables[table_index][row_index];
        let bucket_key = (table_index, row.bucket);
        let bucket_overhead = if selected_buckets.contains(&bucket_key) {
            0
        } else {
            8
        };
        let row_bytes = packed_runtime_row_bytes(row);
        if packed_bytes + bucket_overhead + row_bytes > target_bytes {
            continue;
        }
        packed_bytes += bucket_overhead + row_bytes;
        selected_rows.insert((table_index, row_index));
        selected_buckets.insert(bucket_key);
    }

    let mut tables = Vec::with_capacity(stats.tables.len());
    for _rows in &stats.tables {
        tables.push(HashMap::new());
    }
    for (table_index, rows) in stats.tables.iter().enumerate() {
        for (row_index, row) in rows.iter().enumerate() {
            if !selected_rows.contains(&(table_index, row_index)) {
                continue;
            }
            tables[table_index]
                .entry(row.bucket)
                .or_insert_with(Vec::new)
                .push(FingerprintedCounter {
                    fingerprint: row.fingerprint,
                    counter: row.row.clone(),
                });
        }
    }
    Ok((
        NgramBulkState {
            config: stats.config.clone(),
            vocab_size: stats.vocab_size,
            history: VecDeque::with_capacity(stats.config.max_order().saturating_sub(1)),
            tables,
        },
        packed_bytes,
        selected_rows.len(),
    ))
}

fn load_train_tokens(root: &Path, token_budget: usize) -> Result<Vec<usize>, String> {
    let mut tokens = Vec::new();
    for shard in list_shards(root, "fineweb_train_")? {
        if tokens.len() >= token_budget {
            break;
        }
        let left = token_budget - tokens.len();
        scan_shard_tokens(&shard, left, |token| tokens.push(token))?;
    }
    if tokens.is_empty() {
        return Err("no train tokens loaded".to_string());
    }
    Ok(tokens)
}

fn quantize_oracle_pairs(pairs: &[(usize, usize)], vocab_size: usize) -> Vec<(usize, u32)> {
    let valid = pairs
        .iter()
        .copied()
        .filter(|(token, count)| *token < vocab_size && *count > 0)
        .collect::<Vec<_>>();
    let total = valid.iter().map(|(_, count)| *count).sum::<usize>();
    if total == 0 {
        return Vec::new();
    }
    valid
        .into_iter()
        .take(ORACLE_TOP_K)
        .filter_map(|(token, count)| {
            let weight = ((count as f64 / total as f64) * ORACLE_QUANT_SCALE as f64).round() as u32;
            (weight > 0).then_some((token, weight))
        })
        .collect()
}

fn scan_shard_tokens(
    path: &Path,
    limit: usize,
    mut on_token: impl FnMut(usize),
) -> Result<usize, String> {
    let blob = fs::read(path).map_err(|err| format!("read {}: {err}", path.display()))?;
    let (payload, token_count) = if blob.len() >= HEADER_BYTES {
        let header0 = i32::from_le_bytes([blob[0], blob[1], blob[2], blob[3]]);
        let header1 = i32::from_le_bytes([blob[4], blob[5], blob[6], blob[7]]);
        let header2 = i32::from_le_bytes([blob[8], blob[9], blob[10], blob[11]]);
        if header0 == PARAMETER_GOLF_MAGIC && header1 == PARAMETER_GOLF_VERSION {
            let declared = usize::try_from(header2).map_err(|_| {
                format!(
                    "negative token count in shard header for {}",
                    path.display()
                )
            })?;
            (&blob[HEADER_BYTES..], declared.min(limit))
        } else {
            (&blob[..], (blob.len() / 2).min(limit))
        }
    } else {
        (&blob[..], (blob.len() / 2).min(limit))
    };
    for chunk in payload.chunks_exact(2).take(token_count) {
        on_token(u16::from_le_bytes([chunk[0], chunk[1]]) as usize);
    }
    Ok(token_count)
}

pub fn render_token_conker3_ngram_bulk_report(report: &TokenConker3NgramBulkReport) -> String {
    let mut out = String::new();
    out.push_str("chronohorn_token_conker3_ngram_bulk\n");
    out.push_str(&format!("checkpoint_path: {}\n", report.checkpoint_path));
    out.push_str(&format!("summary_path: {}\n", report.summary_path));
    out.push_str(&format!("train_seq_len: {}\n", report.train_seq_len));
    out.push_str(&format!("eval_tokens: {}\n", report.eval_tokens));
    out.push_str(&format!("scoring_mode: {}\n", report.scoring_mode));
    out.push_str(&format!(
        "collision_control: {}\n",
        report.collision_control
    ));
    out.push_str(&format!("table_source: {}\n", report.table_source));
    out.push_str(&format!(
        "train_tokens_built: {}\n",
        report.train_tokens_built
    ));
    out.push_str(&format!(
        "train_build_elapsed_sec: {:.3}\n",
        report.train_build_elapsed_sec
    ));
    out.push_str(&format!(
        "ngram_orders: {}\n",
        report
            .ngram_orders
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(",")
    ));
    out.push_str(&format!(
        "ngram_bucket_profile: {}\n",
        report
            .ngram_bucket_profile
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(",")
    ));
    out.push_str(&format!(
        "entropy_mix: base={:.4} span={:.4} midpoint={:.4} slope={:.4}\n",
        report.mix_base, report.mix_span, report.entropy_midpoint, report.entropy_slope
    ));
    out.push_str(&format!(
        "ngram_residual: min_row_total={} residual_cap={:.4}\n",
        report.min_row_total, report.residual_cap
    ));
    out.push_str(&format!("report_every: {}\n", report.report_every));
    out.push_str(&format!("eval_bpt_neural: {:.6}\n", report.eval_bpt_neural));
    out.push_str(&format!("eval_bpt_mixed: {:.6}\n", report.eval_bpt_mixed));
    if let Some(value) = report.eval_tokens_per_byte {
        out.push_str(&format!("eval_tokens_per_byte: {:.6}\n", value));
    }
    if let Some(value) = report.eval_bytes_per_token {
        out.push_str(&format!("eval_bytes_per_token: {:.6}\n", value));
    }
    if let Some(value) = report.eval_bpb_neural {
        out.push_str(&format!("eval_bpb_neural: {:.6}\n", value));
    }
    if let Some(value) = report.eval_bpb_mixed {
        out.push_str(&format!("eval_bpb_mixed: {:.6}\n", value));
    }
    out.push_str(&format!("elapsed_sec: {:.3}\n", report.elapsed_sec));
    out.push_str(&format!("tokens_per_sec: {:.3}\n", report.tokens_per_sec));
    for checkpoint in &report.checkpoints {
        out.push_str(&format!(
            "checkpoint tokens={} interval_tokens={} elapsed_sec={:.3} tok_per_sec={:.3} cum_bpb_neural={} cum_bpb_mixed={} interval_bpb_neural={} interval_bpb_mixed={} mean_alpha_on_match={:.6} mean_row_total_on_match={:.3} no_match_positions={}\n",
            checkpoint.tokens_scored,
            checkpoint.interval_tokens,
            checkpoint.elapsed_sec,
            checkpoint.tokens_per_sec,
            format_optional(checkpoint.cumulative_bpb_neural),
            format_optional(checkpoint.cumulative_bpb_mixed),
            format_optional(checkpoint.interval_bpb_neural),
            format_optional(checkpoint.interval_bpb_mixed),
            checkpoint.mean_alpha_on_match,
            checkpoint.mean_row_total_on_match,
            checkpoint.no_match_positions,
        ));
        for order in &checkpoint.order_stats {
            out.push_str(&format!(
                "  order={} fires={} gold_argmax_hits={}\n",
                order.order, order.longest_match_fires, order.gold_argmax_hits
            ));
        }
    }
    out
}

fn finish_interval(
    interval: &IntervalStats,
    tokens_scored: usize,
    cumulative_neural_nats: f64,
    cumulative_mixed_nats: f64,
    elapsed_sec: f64,
    tokens_per_byte: Option<f64>,
) -> TokenConker3NgramBulkCheckpoint {
    let interval_tokens = tokens_scored.saturating_sub(interval.start_token);
    let interval_denom = (interval_tokens as f64 * std::f64::consts::LN_2).max(f64::MIN_POSITIVE);
    let cumulative_denom = (tokens_scored as f64 * std::f64::consts::LN_2).max(f64::MIN_POSITIVE);
    let interval_bpt_neural = interval.neural_nats / interval_denom;
    let interval_bpt_mixed = interval.mixed_nats / interval_denom;
    let cumulative_bpt_neural = cumulative_neural_nats / cumulative_denom;
    let cumulative_bpt_mixed = cumulative_mixed_nats / cumulative_denom;
    TokenConker3NgramBulkCheckpoint {
        tokens_scored,
        interval_tokens,
        elapsed_sec,
        tokens_per_sec: tokens_scored as f64 / elapsed_sec.max(f64::MIN_POSITIVE),
        cumulative_bpt_neural,
        cumulative_bpt_mixed,
        cumulative_bpb_neural: tokens_per_byte.map(|ratio| cumulative_bpt_neural * ratio),
        cumulative_bpb_mixed: tokens_per_byte.map(|ratio| cumulative_bpt_mixed * ratio),
        interval_bpt_neural,
        interval_bpt_mixed,
        interval_bpb_neural: tokens_per_byte.map(|ratio| interval_bpt_neural * ratio),
        interval_bpb_mixed: tokens_per_byte.map(|ratio| interval_bpt_mixed * ratio),
        mean_alpha_on_match: if interval.matched_positions > 0 {
            interval.alpha_sum / interval.matched_positions as f64
        } else {
            0.0
        },
        mean_row_total_on_match: if interval.matched_positions > 0 {
            interval.row_total_sum / interval.matched_positions as f64
        } else {
            0.0
        },
        no_match_positions: interval.no_match_positions,
        order_stats: interval
            .enabled_orders
            .iter()
            .enumerate()
            .map(|(index, &order)| TokenConker3NgramBulkOrderStats {
                order,
                longest_match_fires: interval.per_order_fires[index],
                gold_argmax_hits: interval.per_order_gold_argmax_hits[index],
            })
            .collect(),
    }
}

#[derive(Clone, Copy)]
struct PriorityRow<'a> {
    row: &'a SparseCounter,
    order_index: usize,
    alpha: f64,
}

#[derive(Clone, Copy)]
struct CounterStats {
    top_token: usize,
    top_count: u32,
    second_count: u32,
    support_size: usize,
    total: u32,
}

enum OracleTrustDecision {
    UseRaw,
    UseMasked(SparseCounter),
}

fn oracle_trust_decision(
    raw_row: &SparseCounter,
    trust_row: Option<&OracleTrustCounter>,
    order: usize,
    min_row_total: u32,
) -> OracleTrustDecision {
    let Some(trust_row) = trust_row else {
        return OracleTrustDecision::UseRaw;
    };
    if !(4..=5).contains(&order) {
        return OracleTrustDecision::UseRaw;
    }
    let Some(metrics) = oracle_trust_metrics(trust_row) else {
        return OracleTrustDecision::UseRaw;
    };
    let stable = metrics.examples >= 4
        && metrics.support_size <= ORACLE_TOP_K
        && metrics.top1_agreement >= 0.30
        && metrics.top_mass >= 0.45;
    if !stable {
        return OracleTrustDecision::UseRaw;
    }
    let mut filtered = filter_sparse_counter(raw_row, &trust_row.mass.counts);
    if filtered.total < min_row_total {
        return OracleTrustDecision::UseRaw;
    }
    filtered.prune_top_k(ORACLE_TOP_K);
    OracleTrustDecision::UseMasked(filtered)
}

fn oracle_trust_metrics(counter: &OracleTrustCounter) -> Option<OracleTrustMetrics> {
    if counter.examples == 0 || counter.mass.total == 0 || counter.mass.counts.is_empty() {
        return None;
    }
    let top1_stats = counter_stats(&counter.top1_votes)?;
    let mass_stats = counter_stats(&counter.mass)?;
    Some(OracleTrustMetrics {
        examples: counter.examples,
        support_size: counter.mass.counts.len(),
        top1_agreement: top1_stats.top_count as f64 / counter.examples as f64,
        top_mass: mass_stats.top_count as f64 / counter.mass.total.max(1) as f64,
    })
}

fn filter_sparse_counter(row: &SparseCounter, allowed: &[(usize, u32)]) -> SparseCounter {
    let mut filtered = SparseCounter::default();
    for &(token, count) in &row.counts {
        if allowed
            .iter()
            .any(|(allowed_token, _)| *allowed_token == token)
        {
            filtered.total = filtered.total.saturating_add(count);
            filtered.counts.push((token, count));
        }
    }
    filtered
}

fn select_priority_row<'a>(
    summary: LogitSummary,
    frozen_hit: Option<(&'a SparseCounter, usize, usize)>,
    online_hit: Option<(&'a SparseCounter, usize, usize)>,
) -> Option<PriorityRow<'a>> {
    let frozen_candidate = frozen_hit.and_then(|(row, order_index, order)| {
        score_priority_candidate(
            row,
            order_index,
            order,
            summary.argmax_index,
            summary.entropy,
            false,
        )
        .map(|alpha| PriorityRow {
            row,
            order_index,
            alpha,
        })
    });
    let online_candidate = online_hit.and_then(|(row, order_index, order)| {
        score_priority_candidate(
            row,
            order_index,
            order,
            summary.argmax_index,
            summary.entropy,
            true,
        )
        .map(|alpha| PriorityRow {
            row,
            order_index,
            alpha,
        })
    });

    match (frozen_candidate, online_candidate) {
        (Some(frozen), Some(online)) => {
            let frozen_stats = counter_stats(frozen.row)?;
            let online_stats = counter_stats(online.row)?;
            if online_stats.top_token == frozen_stats.top_token {
                if online_stats.total >= frozen_stats.total {
                    Some(PriorityRow {
                        alpha: online.alpha.max(0.85),
                        ..online
                    })
                } else {
                    Some(PriorityRow {
                        alpha: frozen.alpha.max(0.80),
                        ..frozen
                    })
                }
            } else if online.alpha >= frozen.alpha + 0.05 {
                Some(online)
            } else if frozen.alpha >= online.alpha + 0.05 {
                Some(frozen)
            } else if online_stats.total >= frozen_stats.total {
                Some(online)
            } else {
                Some(frozen)
            }
        }
        (Some(frozen), None) => Some(frozen),
        (None, Some(online)) => Some(online),
        (None, None) => None,
    }
}

fn score_priority_candidate(
    row: &SparseCounter,
    _order_index: usize,
    order: usize,
    neural_top1: usize,
    entropy: f64,
    is_online: bool,
) -> Option<f64> {
    let stats = counter_stats(row)?;
    let margin = if stats.total > 0 {
        (stats.top_count.saturating_sub(stats.second_count)) as f64 / stats.total as f64
    } else {
        0.0
    };
    let top_mass = if stats.total > 0 {
        stats.top_count as f64 / stats.total as f64
    } else {
        0.0
    };

    let threshold = match order {
        4 | 5 => 0.16,
        6 | 7 => 0.22,
        3 => 0.28,
        2 => 0.36,
        _ => 0.40,
    };
    let total_floor = match order {
        4 | 5 => 6,
        6 | 7 => 8,
        3 => 10,
        2 => 16,
        _ => u32::MAX,
    };
    if stats.total < total_floor || margin < threshold {
        return None;
    }

    let agreement_bonus = if stats.top_token == neural_top1 {
        0.08
    } else {
        0.0
    };
    let online_bonus = if is_online { 0.10 } else { 0.0 };
    let entropy_bonus = ((entropy - 4.2).max(0.0) * 0.03).min(0.08);
    let support_penalty = if stats.support_size > 24 { 0.08 } else { 0.0 };
    let order_base = match order {
        5 => 0.62,
        4 => 0.58,
        6 => 0.52,
        7 => 0.46,
        3 => 0.40,
        2 => 0.32,
        _ => 0.24,
    };
    let total_bonus = ((stats.total as f64).ln_1p() * 0.04).min(0.10);
    let alpha = (order_base
        + 0.55 * margin
        + 0.18 * top_mass
        + total_bonus
        + agreement_bonus
        + online_bonus
        + entropy_bonus
        - support_penalty)
        .clamp(0.0, 0.95);
    if alpha >= 0.45 { Some(alpha) } else { None }
}

fn counter_stats(row: &SparseCounter) -> Option<CounterStats> {
    let mut top_token = 0usize;
    let mut top_count = 0u32;
    let mut second_count = 0u32;
    for &(token, count) in &row.counts {
        if count > top_count {
            second_count = top_count;
            top_count = count;
            top_token = token;
        } else if count > second_count {
            second_count = count;
        }
    }
    if top_count == 0 {
        None
    } else {
        Some(CounterStats {
            top_token,
            top_count,
            second_count,
            support_size: row.counts.len(),
            total: row.total,
        })
    }
}

fn save_ngram_table_artifact(
    path: &Path,
    state: &NgramBulkState,
    train_tokens_built: usize,
    train_build_elapsed_sec: f64,
) -> Result<(), String> {
    let file = File::create(path).map_err(|err| format!("create {}: {err}", path.display()))?;
    let mut writer = BufWriter::new(file);
    writer
        .write_all(TABLE_ARTIFACT_MAGIC)
        .map_err(|err| format!("write {}: {err}", path.display()))?;
    write_u32(&mut writer, state.vocab_size as u32, path)?;
    write_u64(&mut writer, train_tokens_built as u64, path)?;
    write_f64(&mut writer, train_build_elapsed_sec, path)?;
    write_u32(&mut writer, state.config.enabled_orders.len() as u32, path)?;
    write_u32(&mut writer, state.config.bucket_counts.len() as u32, path)?;
    write_f64(&mut writer, state.config.mix_base, path)?;
    write_f64(&mut writer, state.config.mix_span, path)?;
    write_f64(&mut writer, state.config.entropy_midpoint, path)?;
    write_f64(&mut writer, state.config.entropy_slope, path)?;
    write_u32(&mut writer, state.config.min_row_total, path)?;
    write_f64(&mut writer, state.config.residual_cap, path)?;
    write_u32(&mut writer, state.config.report_every as u32, path)?;
    for &order in &state.config.enabled_orders {
        write_u32(&mut writer, order as u32, path)?;
    }
    for &bucket_count in &state.config.bucket_counts {
        write_u64(&mut writer, bucket_count as u64, path)?;
    }
    write_u32(&mut writer, state.tables.len() as u32, path)?;
    for table in &state.tables {
        write_u64(&mut writer, table.len() as u64, path)?;
        for (&bucket, rows) in table {
            write_u32(&mut writer, bucket, path)?;
            write_u32(&mut writer, rows.len() as u32, path)?;
            for row in rows {
                write_u64(&mut writer, row.fingerprint, path)?;
                write_u32(&mut writer, row.counter.total, path)?;
                write_u32(&mut writer, row.counter.counts.len() as u32, path)?;
                for &(token, count) in &row.counter.counts {
                    write_u32(&mut writer, token as u32, path)?;
                    write_u32(&mut writer, count, path)?;
                }
            }
        }
    }
    writer
        .flush()
        .map_err(|err| format!("flush {}: {err}", path.display()))
}

fn save_oracle_budgeted_row_stats_artifact(
    path: &Path,
    stats: &OracleBudgetedRowStatsArtifact,
) -> Result<(), String> {
    let file = File::create(path).map_err(|err| format!("create {}: {err}", path.display()))?;
    let mut writer = BufWriter::new(file);
    writer
        .write_all(ROW_STATS_ARTIFACT_MAGIC)
        .map_err(|err| format!("write {}: {err}", path.display()))?;
    write_u32(&mut writer, stats.vocab_size as u32, path)?;
    write_u64(&mut writer, stats.train_tokens_built as u64, path)?;
    write_f64(&mut writer, stats.train_build_elapsed_sec, path)?;
    write_u32(&mut writer, stats.config.enabled_orders.len() as u32, path)?;
    write_u32(&mut writer, stats.config.bucket_counts.len() as u32, path)?;
    write_f64(&mut writer, stats.config.mix_base, path)?;
    write_f64(&mut writer, stats.config.mix_span, path)?;
    write_f64(&mut writer, stats.config.entropy_midpoint, path)?;
    write_f64(&mut writer, stats.config.entropy_slope, path)?;
    write_u32(&mut writer, stats.config.min_row_total, path)?;
    write_f64(&mut writer, stats.config.residual_cap, path)?;
    write_u32(&mut writer, stats.config.report_every as u32, path)?;
    for &order in &stats.config.enabled_orders {
        write_u32(&mut writer, order as u32, path)?;
    }
    for &bucket_count in &stats.config.bucket_counts {
        write_u64(&mut writer, bucket_count as u64, path)?;
    }
    write_u32(&mut writer, stats.tables.len() as u32, path)?;
    for rows in &stats.tables {
        write_u64(&mut writer, rows.len() as u64, path)?;
        for row in rows {
            write_u32(&mut writer, row.bucket, path)?;
            write_u64(&mut writer, row.fingerprint, path)?;
            write_u32(&mut writer, row.row.total, path)?;
            write_u32(&mut writer, row.row.counts.len() as u32, path)?;
            for &(token, count) in &row.row.counts {
                write_u16(&mut writer, token as u16, path)?;
                write_u32(&mut writer, count, path)?;
            }
            if let Some(metrics) = row.oracle {
                write_u8(&mut writer, 1, path)?;
                write_u32(&mut writer, metrics.examples, path)?;
                write_u32(&mut writer, metrics.support_size, path)?;
                write_f64(&mut writer, metrics.top1_agreement, path)?;
                write_f64(&mut writer, metrics.top_mass, path)?;
            } else {
                write_u8(&mut writer, 0, path)?;
            }
        }
    }
    writer
        .flush()
        .map_err(|err| format!("flush {}: {err}", path.display()))
}

fn load_ngram_table_artifact(path: &Path) -> Result<(NgramBulkState, usize, f64), String> {
    let file = File::open(path).map_err(|err| format!("open {}: {err}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut magic = [0u8; 8];
    reader
        .read_exact(&mut magic)
        .map_err(|err| format!("read {}: {err}", path.display()))?;
    if &magic != TABLE_ARTIFACT_MAGIC {
        return Err(format!("table artifact {} has wrong magic", path.display()));
    }
    let vocab_size = read_u32(&mut reader, path)? as usize;
    let train_tokens_built = read_u64(&mut reader, path)? as usize;
    let train_build_elapsed_sec = read_f64(&mut reader, path)?;
    let orders_len = read_u32(&mut reader, path)? as usize;
    let buckets_len = read_u32(&mut reader, path)? as usize;
    let mix_base = read_f64(&mut reader, path)?;
    let mix_span = read_f64(&mut reader, path)?;
    let entropy_midpoint = read_f64(&mut reader, path)?;
    let entropy_slope = read_f64(&mut reader, path)?;
    let min_row_total = read_u32(&mut reader, path)?;
    let residual_cap = read_f64(&mut reader, path)?;
    let report_every = read_u32(&mut reader, path)? as usize;
    let mut enabled_orders = Vec::with_capacity(orders_len);
    for _ in 0..orders_len {
        enabled_orders.push(read_u32(&mut reader, path)? as usize);
    }
    let mut bucket_counts = Vec::with_capacity(buckets_len);
    for _ in 0..buckets_len {
        bucket_counts.push(read_u64(&mut reader, path)? as usize);
    }
    let config = BulkConfig {
        enabled_orders,
        bucket_counts,
        mix_base,
        mix_span,
        entropy_midpoint,
        entropy_slope,
        min_row_total,
        residual_cap,
        report_every,
    };
    let table_len = read_u32(&mut reader, path)? as usize;
    let mut tables = Vec::with_capacity(table_len);
    for _ in 0..table_len {
        let bucket_entries = read_u64(&mut reader, path)? as usize;
        let mut table = HashMap::with_capacity(bucket_entries);
        for _ in 0..bucket_entries {
            let bucket = read_u32(&mut reader, path)?;
            let row_count = read_u32(&mut reader, path)? as usize;
            let mut rows = Vec::with_capacity(row_count);
            for _ in 0..row_count {
                let fingerprint = read_u64(&mut reader, path)?;
                let total = read_u32(&mut reader, path)?;
                let count_len = read_u32(&mut reader, path)? as usize;
                let mut counts = Vec::with_capacity(count_len);
                for _ in 0..count_len {
                    let token = read_u32(&mut reader, path)? as usize;
                    let count = read_u32(&mut reader, path)?;
                    counts.push((token, count));
                }
                rows.push(FingerprintedCounter {
                    fingerprint,
                    counter: SparseCounter { total, counts },
                });
            }
            table.insert(bucket, rows);
        }
        tables.push(table);
    }
    Ok((
        NgramBulkState {
            config,
            vocab_size,
            history: VecDeque::new(),
            tables,
        },
        train_tokens_built,
        train_build_elapsed_sec,
    ))
}

fn load_oracle_budgeted_row_stats_artifact(
    path: &Path,
) -> Result<OracleBudgetedRowStatsArtifact, String> {
    let file = File::open(path).map_err(|err| format!("open {}: {err}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut magic = [0u8; 8];
    reader
        .read_exact(&mut magic)
        .map_err(|err| format!("read {}: {err}", path.display()))?;
    if &magic != ROW_STATS_ARTIFACT_MAGIC {
        return Err(format!(
            "row stats artifact {} has wrong magic",
            path.display()
        ));
    }
    let vocab_size = read_u32(&mut reader, path)? as usize;
    let train_tokens_built = read_u64(&mut reader, path)? as usize;
    let train_build_elapsed_sec = read_f64(&mut reader, path)?;
    let orders_len = read_u32(&mut reader, path)? as usize;
    let buckets_len = read_u32(&mut reader, path)? as usize;
    let mix_base = read_f64(&mut reader, path)?;
    let mix_span = read_f64(&mut reader, path)?;
    let entropy_midpoint = read_f64(&mut reader, path)?;
    let entropy_slope = read_f64(&mut reader, path)?;
    let min_row_total = read_u32(&mut reader, path)?;
    let residual_cap = read_f64(&mut reader, path)?;
    let report_every = read_u32(&mut reader, path)? as usize;
    let mut enabled_orders = Vec::with_capacity(orders_len);
    for _ in 0..orders_len {
        enabled_orders.push(read_u32(&mut reader, path)? as usize);
    }
    let mut bucket_counts = Vec::with_capacity(buckets_len);
    for _ in 0..buckets_len {
        bucket_counts.push(read_u64(&mut reader, path)? as usize);
    }
    let config = BulkConfig {
        enabled_orders,
        bucket_counts,
        mix_base,
        mix_span,
        entropy_midpoint,
        entropy_slope,
        min_row_total,
        residual_cap,
        report_every,
    };
    let table_len = read_u32(&mut reader, path)? as usize;
    let mut tables = Vec::with_capacity(table_len);
    for _ in 0..table_len {
        let row_count = read_u64(&mut reader, path)? as usize;
        let mut rows = Vec::with_capacity(row_count);
        for _ in 0..row_count {
            let bucket = read_u32(&mut reader, path)?;
            let fingerprint = read_u64(&mut reader, path)?;
            let total = read_u32(&mut reader, path)?;
            let count_len = read_u32(&mut reader, path)? as usize;
            let mut counts = Vec::with_capacity(count_len);
            for _ in 0..count_len {
                let token = read_u16(&mut reader, path)? as usize;
                let count = read_u32(&mut reader, path)?;
                counts.push((token, count));
            }
            let oracle = if read_u8(&mut reader, path)? == 1 {
                Some(StoredOracleRowMetrics {
                    examples: read_u32(&mut reader, path)?,
                    support_size: read_u32(&mut reader, path)?,
                    top1_agreement: read_f64(&mut reader, path)?,
                    top_mass: read_f64(&mut reader, path)?,
                })
            } else {
                None
            };
            rows.push(OracleBudgetedRowStat {
                bucket,
                fingerprint,
                row: SparseCounter { total, counts },
                oracle,
            });
        }
        tables.push(rows);
    }
    Ok(OracleBudgetedRowStatsArtifact {
        config,
        vocab_size,
        train_tokens_built,
        train_build_elapsed_sec,
        tables,
    })
}

fn write_u32(writer: &mut impl Write, value: u32, path: &Path) -> Result<(), String> {
    writer
        .write_all(&value.to_le_bytes())
        .map_err(|err| format!("write {}: {err}", path.display()))
}

fn write_u16(writer: &mut impl Write, value: u16, path: &Path) -> Result<(), String> {
    writer
        .write_all(&value.to_le_bytes())
        .map_err(|err| format!("write {}: {err}", path.display()))
}

fn write_u8(writer: &mut impl Write, value: u8, path: &Path) -> Result<(), String> {
    writer
        .write_all(&[value])
        .map_err(|err| format!("write {}: {err}", path.display()))
}

fn write_u64(writer: &mut impl Write, value: u64, path: &Path) -> Result<(), String> {
    writer
        .write_all(&value.to_le_bytes())
        .map_err(|err| format!("write {}: {err}", path.display()))
}

fn write_f64(writer: &mut impl Write, value: f64, path: &Path) -> Result<(), String> {
    writer
        .write_all(&value.to_le_bytes())
        .map_err(|err| format!("write {}: {err}", path.display()))
}

fn read_u32(reader: &mut impl Read, path: &Path) -> Result<u32, String> {
    let mut buf = [0u8; 4];
    reader
        .read_exact(&mut buf)
        .map_err(|err| format!("read {}: {err}", path.display()))?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u16(reader: &mut impl Read, path: &Path) -> Result<u16, String> {
    let mut buf = [0u8; 2];
    reader
        .read_exact(&mut buf)
        .map_err(|err| format!("read {}: {err}", path.display()))?;
    Ok(u16::from_le_bytes(buf))
}

fn read_u8(reader: &mut impl Read, path: &Path) -> Result<u8, String> {
    let mut buf = [0u8; 1];
    reader
        .read_exact(&mut buf)
        .map_err(|err| format!("read {}: {err}", path.display()))?;
    Ok(buf[0])
}

fn read_u64(reader: &mut impl Read, path: &Path) -> Result<u64, String> {
    let mut buf = [0u8; 8];
    reader
        .read_exact(&mut buf)
        .map_err(|err| format!("read {}: {err}", path.display()))?;
    Ok(u64::from_le_bytes(buf))
}

fn read_f64(reader: &mut impl Read, path: &Path) -> Result<f64, String> {
    let mut buf = [0u8; 8];
    reader
        .read_exact(&mut buf)
        .map_err(|err| format!("read {}: {err}", path.display()))?;
    Ok(f64::from_le_bytes(buf))
}

fn entropy_weight_from_summary(entropy: f64, config: &BulkConfig) -> f64 {
    config.mix_base
        + config.mix_span * sigmoid(config.entropy_slope * (entropy - config.entropy_midpoint))
}

fn summarize_logits(logits: &[f32], gold: usize) -> LogitSummary {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut sum = 0.0f64;
    let mut centered_weighted_sum = 0.0f64;
    let mut argmax_index = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (index, &value) in logits.iter().enumerate() {
        if value > best_value {
            best_value = value;
            argmax_index = index;
        }
        let centered = (value as f64) - max;
        let weight = centered.exp();
        sum += weight;
        centered_weighted_sum += weight * centered;
    }
    let sum = sum.max(f64::MIN_POSITIVE);
    let log_sum = sum.ln();
    let gold_logit = logits.get(gold).copied().unwrap_or(-1e9) as f64;
    LogitSummary {
        gold_logit,
        log_z: max + log_sum,
        entropy: log_sum - centered_weighted_sum / sum,
        argmax_index,
    }
}

fn sparse_logit_residual_logprob(
    logits: &[f32],
    gold: usize,
    gold_logit: f64,
    log_z_base: f64,
    row: &SparseCounter,
    vocab_size: usize,
    alpha: f64,
    residual_cap: f64,
) -> (f64, bool) {
    let uniform_log = -((vocab_size.max(1)) as f64).ln();
    let total = row.total.max(1) as f64;
    let base_sum = log_z_base.exp();
    let mut delta_sum = 0.0f64;
    let mut gold_residual = 0.0f64;
    let mut max_count = 0u32;
    let mut gold_count = 0u32;

    for &(token, count) in &row.counts {
        if count > max_count {
            max_count = count;
        }
        if token == gold {
            gold_count = count;
        }
        if token >= logits.len() {
            continue;
        }
        let log_prob = ((count as f64) / total).max(f64::MIN_POSITIVE).ln();
        let raw = alpha * (log_prob - uniform_log);
        let residual = residual_cap * (raw / residual_cap).tanh();
        let base_mass = (logits[token] as f64 - log_z_base).exp();
        delta_sum += base_mass * (residual.exp() - 1.0);
        if token == gold {
            gold_residual = residual;
        }
    }

    let adjusted_sum = (base_sum * (1.0 + delta_sum)).max(f64::MIN_POSITIVE);
    let log_z_adj = adjusted_sum.ln();
    let log_p = gold_logit + gold_residual - log_z_adj;
    (log_p, gold_count > 0 && gold_count == max_count)
}

fn push_history(history: &mut VecDeque<usize>, max_len: usize, token: usize) {
    history.push_back(token);
    while history.len() > max_len {
        history.pop_front();
    }
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        let z = (-value).exp();
        1.0 / (1.0 + z)
    } else {
        let z = value.exp();
        z / (1.0 + z)
    }
}

fn format_optional(value: Option<f64>) -> String {
    value
        .map(|v| format!("{v:.6}"))
        .unwrap_or_else(|| "na".to_string())
}
