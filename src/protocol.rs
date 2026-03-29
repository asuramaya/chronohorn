#[derive(Debug, Clone)]
pub struct SampleOutputs {
    pub sample_predictions: Vec<Vec<f64>>,
    pub sample_gold_logprobs: Option<Vec<f64>>,
}

pub trait Runner: Clone {
    fn name(&self) -> &'static str;
    fn vocab_size(&self) -> usize;
    fn score_chunk(
        &self,
        tokens: &[usize],
        sample_positions: &[usize],
    ) -> Result<SampleOutputs, String>;
    fn adapt_chunk(&mut self, tokens: &[usize]) -> Result<(), String>;
}

