use crate::protocol::{Runner, SampleOutputs};

#[derive(Debug, Clone, Copy)]
pub enum DemoMode {
    Legal,
    SelfInclude,
    FuturePeek,
    LengthPeek,
    BoundaryDoubleUpdate,
    ReportedGoldCheat,
}

impl DemoMode {
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw {
            "legal" => Ok(Self::Legal),
            "self-include" => Ok(Self::SelfInclude),
            "future-peek" => Ok(Self::FuturePeek),
            "length-peek" => Ok(Self::LengthPeek),
            "boundary-double-update" => Ok(Self::BoundaryDoubleUpdate),
            "reported-gold-cheat" => Ok(Self::ReportedGoldCheat),
            _ => Err(format!("unknown demo mode: {raw}")),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Legal => "legal",
            Self::SelfInclude => "self-include",
            Self::FuturePeek => "future-peek",
            Self::LengthPeek => "length-peek",
            Self::BoundaryDoubleUpdate => "boundary-double-update",
            Self::ReportedGoldCheat => "reported-gold-cheat",
        }
    }
}

#[derive(Debug, Clone)]
pub struct PackedCacheDemo {
    vocab_size: usize,
    alpha: f64,
    mode: DemoMode,
    global_counts: Vec<f64>,
}

impl PackedCacheDemo {
    pub fn new(vocab_size: usize, alpha: f64, mode: DemoMode) -> Self {
        Self {
            vocab_size,
            alpha,
            mode,
            global_counts: vec![0.0; vocab_size],
        }
    }

    fn posterior(&self, counts: &[f64]) -> Vec<f64> {
        let mut probs: Vec<f64> = counts.iter().map(|value| value + self.alpha).collect();
        normalize(&mut probs);
        probs
    }

    fn apply_length_peek(&self, probs: &[f64], chunk_len: usize) -> Vec<f64> {
        let mut out = probs.to_vec();
        if self.vocab_size == 0 {
            return out;
        }
        let slot = chunk_len % self.vocab_size;
        out[slot] += 0.05;
        normalize(&mut out);
        out
    }
}

impl Runner for PackedCacheDemo {
    fn name(&self) -> &'static str {
        "PackedCacheDemo"
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn score_chunk(
        &self,
        tokens: &[usize],
        sample_positions: &[usize],
    ) -> Result<SampleOutputs, String> {
        if tokens.is_empty() {
            return Ok(SampleOutputs {
                sample_predictions: Vec::new(),
                sample_gold_logprobs: Some(Vec::new()),
            });
        }

        let mut rolling = self.global_counts.clone();
        let mut full_chunk = rolling.clone();
        for &tok in tokens {
            if tok >= self.vocab_size {
                return Err(format!(
                    "token {tok} exceeds vocab size {}",
                    self.vocab_size
                ));
            }
            full_chunk[tok] += 1.0;
        }

        let mut all_predictions: Vec<Vec<f64>> = Vec::with_capacity(tokens.len());
        let mut all_gold: Vec<f64> = Vec::with_capacity(tokens.len());

        for &tok in tokens {
            let visible = match self.mode {
                DemoMode::Legal => self.posterior(&rolling),
                DemoMode::SelfInclude => {
                    let mut with_self = rolling.clone();
                    with_self[tok] += 1.0;
                    self.posterior(&with_self)
                }
                DemoMode::FuturePeek => self.posterior(&full_chunk),
                DemoMode::LengthPeek => {
                    self.apply_length_peek(&self.posterior(&rolling), tokens.len())
                }
                DemoMode::BoundaryDoubleUpdate => self.posterior(&rolling),
                DemoMode::ReportedGoldCheat => self.posterior(&rolling),
            };
            let scored = match self.mode {
                DemoMode::ReportedGoldCheat => self.posterior(&full_chunk),
                _ => visible.clone(),
            };

            all_gold.push(scored[tok].max(f64::MIN_POSITIVE).ln());
            all_predictions.push(visible);
            rolling[tok] += 1.0;
        }

        let mut sample_predictions = Vec::with_capacity(sample_positions.len());
        let mut sample_gold = Vec::with_capacity(sample_positions.len());
        for &pos in sample_positions {
            if pos >= all_predictions.len() {
                return Err(format!("sample position {pos} out of bounds"));
            }
            sample_predictions.push(all_predictions[pos].clone());
            sample_gold.push(all_gold[pos]);
        }

        Ok(SampleOutputs {
            sample_predictions,
            sample_gold_logprobs: Some(sample_gold),
        })
    }

    fn adapt_chunk(&mut self, tokens: &[usize]) -> Result<(), String> {
        for &tok in tokens {
            if tok >= self.vocab_size {
                return Err(format!(
                    "token {tok} exceeds vocab size {}",
                    self.vocab_size
                ));
            }
            self.global_counts[tok] += 1.0;
        }
        if matches!(self.mode, DemoMode::BoundaryDoubleUpdate) {
            if let Some(&tok) = tokens.last() {
                self.global_counts[tok] += 1.0;
            }
        }
        Ok(())
    }
}

fn normalize(values: &mut [f64]) {
    let total: f64 = values.iter().sum();
    let denom = total.max(f64::EPSILON);
    for value in values {
        *value /= denom;
    }
}
