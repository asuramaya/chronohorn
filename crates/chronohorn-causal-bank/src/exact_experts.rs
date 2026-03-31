use std::collections::{BTreeSet, HashMap, VecDeque};
use std::hash::Hash;

const MAX_CONTEXT: usize = 3;

type FollowRow = HashMap<usize, u32>;
type Exact2Key = (usize, usize);
type Exact3Key = (usize, usize, usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExactExpertKind {
    Exact1,
    Exact2,
    Exact3,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExactFollowCounts {
    pub exact1: Vec<(usize, u32)>,
    pub exact2: Vec<(usize, u32)>,
    pub exact3: Vec<(usize, u32)>,
}

impl ExactFollowCounts {
    pub fn is_empty(&self) -> bool {
        self.exact1.is_empty() && self.exact2.is_empty() && self.exact3.is_empty()
    }

    pub fn support_size(&self, kind: ExactExpertKind) -> usize {
        match kind {
            ExactExpertKind::Exact1 => self.exact1.len(),
            ExactExpertKind::Exact2 => self.exact2.len(),
            ExactExpertKind::Exact3 => self.exact3.len(),
        }
    }

    pub fn support_tokens(&self, exact1_opens_candidates: bool) -> Vec<usize> {
        let mut tokens = BTreeSet::new();
        if exact1_opens_candidates {
            for &(token, _) in &self.exact1 {
                tokens.insert(token);
            }
        }
        for &(token, _) in &self.exact2 {
            tokens.insert(token);
        }
        for &(token, _) in &self.exact3 {
            tokens.insert(token);
        }
        tokens.into_iter().collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExactResidualSource {
    pub gate: f32,
    pub count_weight: f32,
    pub flag_weight: f32,
    pub opens_candidates: bool,
}

impl ExactResidualSource {
    pub const fn new(
        gate: f32,
        count_weight: f32,
        flag_weight: f32,
        opens_candidates: bool,
    ) -> Self {
        Self {
            gate,
            count_weight,
            flag_weight,
            opens_candidates,
        }
    }

    pub const fn conker_ungated(opens_candidates: bool) -> Self {
        Self::new(1.0, 1.0, 1.0, opens_candidates)
    }

    fn score_count(self, count: u32) -> f32 {
        if count == 0 {
            return 0.0;
        }
        self.gate * (self.count_weight * (count as f32).ln_1p() + self.flag_weight)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExactResidualConfig {
    pub exact1: ExactResidualSource,
    pub exact2: ExactResidualSource,
    pub exact3: ExactResidualSource,
    pub residual_cap: Option<f32>,
}

impl ExactResidualConfig {
    pub const fn conker4b(exact1_opens_candidates: bool, residual_cap: Option<f32>) -> Self {
        Self {
            exact1: ExactResidualSource::conker_ungated(exact1_opens_candidates),
            exact2: ExactResidualSource::conker_ungated(true),
            exact3: ExactResidualSource::conker_ungated(true),
            residual_cap,
        }
    }
}

impl Default for ExactResidualConfig {
    fn default() -> Self {
        Self::conker4b(true, Some(4.0))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExactResidualScores {
    pub pre_activation: Vec<f32>,
    pub residual_logits: Vec<f32>,
    pub combined_logits: Vec<f32>,
    pub candidate_mask: Vec<bool>,
}

#[derive(Debug, Clone)]
pub struct ExactFollowCounter {
    vocab_size: usize,
    history: VecDeque<usize>,
    exact1_rows: HashMap<usize, FollowRow>,
    exact2_rows: HashMap<Exact2Key, FollowRow>,
    exact3_rows: HashMap<Exact3Key, FollowRow>,
}

impl ExactFollowCounter {
    pub fn new(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            history: VecDeque::with_capacity(MAX_CONTEXT),
            exact1_rows: HashMap::new(),
            exact2_rows: HashMap::new(),
            exact3_rows: HashMap::new(),
        }
    }

    pub fn from_history(tokens: &[usize], vocab_size: usize) -> Result<Self, String> {
        let mut counter = Self::new(vocab_size);
        for &token in tokens {
            counter.observe(token)?;
        }
        Ok(counter)
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn clear(&mut self) {
        self.history.clear();
        self.exact1_rows.clear();
        self.exact2_rows.clear();
        self.exact3_rows.clear();
    }

    pub fn observe(&mut self, token: usize) -> Result<(), String> {
        if token >= self.vocab_size {
            return Err(format!(
                "token {token} out of range for vocab size {}",
                self.vocab_size
            ));
        }
        if let Some(&prev1) = self.history.back() {
            increment_row(&mut self.exact1_rows, prev1, token);
        }
        if self.history.len() >= 2 {
            let prev2 = self.history[self.history.len() - 2];
            let prev1 = self.history[self.history.len() - 1];
            increment_row(&mut self.exact2_rows, (prev2, prev1), token);
        }
        if self.history.len() >= 3 {
            let prev3 = self.history[self.history.len() - 3];
            let prev2 = self.history[self.history.len() - 2];
            let prev1 = self.history[self.history.len() - 1];
            increment_row(&mut self.exact3_rows, (prev3, prev2, prev1), token);
        }
        self.history.push_back(token);
        if self.history.len() > MAX_CONTEXT {
            self.history.pop_front();
        }
        Ok(())
    }

    pub fn counts_for_next(&self) -> ExactFollowCounts {
        let exact1 = self
            .history
            .back()
            .and_then(|token| self.exact1_rows.get(token));
        let exact2 = if self.history.len() >= 2 {
            Some((
                self.history[self.history.len() - 2],
                self.history[self.history.len() - 1],
            ))
        } else {
            None
        }
        .and_then(|key| self.exact2_rows.get(&key));
        let exact3 = if self.history.len() >= 3 {
            Some((
                self.history[self.history.len() - 3],
                self.history[self.history.len() - 2],
                self.history[self.history.len() - 1],
            ))
        } else {
            None
        }
        .and_then(|key| self.exact3_rows.get(&key));
        ExactFollowCounts {
            exact1: snapshot_row(exact1),
            exact2: snapshot_row(exact2),
            exact3: snapshot_row(exact3),
        }
    }

    pub fn score_next(
        &self,
        base_logits: &[f32],
        config: &ExactResidualConfig,
    ) -> Result<ExactResidualScores, String> {
        residual_candidate_scores(base_logits, &self.counts_for_next(), config)
    }
}

pub fn causal_count_rows(
    tokens: &[usize],
    vocab_size: usize,
) -> Result<Vec<ExactFollowCounts>, String> {
    let mut counter = ExactFollowCounter::new(vocab_size);
    let mut rows = Vec::with_capacity(tokens.len());
    for &token in tokens {
        rows.push(counter.counts_for_next());
        counter.observe(token)?;
    }
    Ok(rows)
}

pub fn score_causal_rows(
    base_logits_rows: &[Vec<f32>],
    tokens: &[usize],
    config: &ExactResidualConfig,
) -> Result<Vec<ExactResidualScores>, String> {
    if base_logits_rows.len() != tokens.len() {
        return Err(format!(
            "base row/token length mismatch: {} vs {}",
            base_logits_rows.len(),
            tokens.len()
        ));
    }
    let vocab_size = base_logits_rows.first().map(|row| row.len()).unwrap_or(0);
    for (row_index, row) in base_logits_rows.iter().enumerate() {
        if row.len() != vocab_size {
            return Err(format!(
                "base logits row {row_index} has width {}, expected {vocab_size}",
                row.len()
            ));
        }
    }
    let mut counter = ExactFollowCounter::new(vocab_size);
    let mut scored = Vec::with_capacity(base_logits_rows.len());
    for (row, &token) in base_logits_rows.iter().zip(tokens.iter()) {
        scored.push(counter.score_next(row, config)?);
        counter.observe(token)?;
    }
    Ok(scored)
}

pub fn residual_candidate_scores(
    base_logits: &[f32],
    counts: &ExactFollowCounts,
    config: &ExactResidualConfig,
) -> Result<ExactResidualScores, String> {
    let mut pre_activation = vec![0.0; base_logits.len()];
    let mut candidate_mask = vec![false; base_logits.len()];
    apply_source(
        "exact1",
        &mut pre_activation,
        &mut candidate_mask,
        &counts.exact1,
        config.exact1,
    )?;
    apply_source(
        "exact2",
        &mut pre_activation,
        &mut candidate_mask,
        &counts.exact2,
        config.exact2,
    )?;
    apply_source(
        "exact3",
        &mut pre_activation,
        &mut candidate_mask,
        &counts.exact3,
        config.exact3,
    )?;

    let mut residual_logits = vec![0.0; base_logits.len()];
    let mut combined_logits = base_logits.to_vec();
    for token in 0..base_logits.len() {
        if !candidate_mask[token] {
            continue;
        }
        let residual = cap_residual(pre_activation[token], config.residual_cap);
        residual_logits[token] = residual;
        combined_logits[token] += residual;
    }

    Ok(ExactResidualScores {
        pre_activation,
        residual_logits,
        combined_logits,
        candidate_mask,
    })
}

fn increment_row<K>(table: &mut HashMap<K, FollowRow>, key: K, next_token: usize)
where
    K: Eq + Hash,
{
    let row = table.entry(key).or_default();
    *row.entry(next_token).or_insert(0) += 1;
}

fn snapshot_row(row: Option<&FollowRow>) -> Vec<(usize, u32)> {
    let mut out: Vec<(usize, u32)> = row
        .map(|row| row.iter().map(|(&token, &count)| (token, count)).collect())
        .unwrap_or_default();
    out.sort_unstable_by_key(|&(token, _)| token);
    out
}

fn apply_source(
    source_name: &str,
    pre_activation: &mut [f32],
    candidate_mask: &mut [bool],
    counts: &[(usize, u32)],
    source: ExactResidualSource,
) -> Result<(), String> {
    for &(token, count) in counts {
        if token >= pre_activation.len() {
            return Err(format!(
                "{source_name} token {token} out of range for vocab size {}",
                pre_activation.len()
            ));
        }
        pre_activation[token] += source.score_count(count);
        if source.opens_candidates {
            candidate_mask[token] = true;
        }
    }
    Ok(())
}

fn cap_residual(score: f32, residual_cap: Option<f32>) -> f32 {
    match residual_cap {
        Some(cap) if cap.is_finite() && cap > 0.0 => cap * (score / cap).tanh(),
        _ => score,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counter_collects_repeated_exact_ngrams_causally() {
        let counter = ExactFollowCounter::from_history(&[1, 2, 3, 4, 1, 2, 3], 8).unwrap();
        let counts = counter.counts_for_next();
        assert_eq!(counts.exact1, vec![(4, 1)]);
        assert_eq!(counts.exact2, vec![(4, 1)]);
        assert_eq!(counts.exact3, vec![(4, 1)]);
    }

    #[test]
    fn causal_count_rows_match_online_state() {
        let tokens = [1, 2, 3, 4, 1, 2, 3, 5];
        let rows = causal_count_rows(&tokens, 8).unwrap();
        assert!(rows[0].is_empty());
        assert_eq!(rows[7].exact1, vec![(4, 1)]);
        assert_eq!(rows[7].exact2, vec![(4, 1)]);
        assert_eq!(rows[7].exact3, vec![(4, 1)]);
    }

    #[test]
    fn support_only_exact1_needs_another_candidate_opener() {
        let counts = ExactFollowCounts {
            exact1: vec![(2, 2), (3, 2)],
            exact2: vec![(3, 1)],
            exact3: Vec::new(),
        };
        let config = ExactResidualConfig::conker4b(false, None);
        let scored = residual_candidate_scores(&vec![0.0; 6], &counts, &config).unwrap();

        let exact1_term = (2.0f32).ln_1p() + 1.0;
        let exact2_term = (1.0f32).ln_1p() + 1.0;

        assert!((scored.pre_activation[2] - exact1_term).abs() < 1e-6);
        assert_eq!(scored.candidate_mask[2], false);
        assert_eq!(scored.residual_logits[2], 0.0);

        assert_eq!(scored.candidate_mask[3], true);
        assert!((scored.residual_logits[3] - (exact1_term + exact2_term)).abs() < 1e-6);
    }

    #[test]
    fn residual_cap_clamps_large_scores() {
        let counts = ExactFollowCounts {
            exact1: Vec::new(),
            exact2: vec![(1, 1000)],
            exact3: Vec::new(),
        };
        let config = ExactResidualConfig::conker4b(true, Some(1.0));
        let scored = residual_candidate_scores(&vec![0.5, 0.0, 0.0], &counts, &config).unwrap();

        assert!(scored.pre_activation[1] > 1.0);
        assert!(scored.residual_logits[1] < 1.0);
        assert!((scored.residual_logits[1] - scored.pre_activation[1].tanh()).abs() < 1e-6);
        assert!((scored.combined_logits[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn score_rows_tracks_causal_exact_support_over_base_logits() {
        let tokens = [1, 2, 3, 4, 1, 2, 3, 5];
        let base = vec![vec![0.0; 8]; tokens.len()];
        let scored = score_causal_rows(&base, &tokens, &ExactResidualConfig::default()).unwrap();

        assert!(scored[7].candidate_mask[4]);
        assert!(scored[7].combined_logits[4] > 0.0);
    }
}
