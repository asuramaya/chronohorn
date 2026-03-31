use std::array::from_fn;

/// Prefix-only signals used to derive an adaptive control surface.
///
/// The caller owns the feature extraction. This module only consumes
/// prefix-derived values and never consults future tokens, model state,
/// or any global oracle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrefixAdaptiveGateFeatures {
    /// Observed prefix length in tokens.
    pub prefix_len: usize,
    /// Budget or reference length used to normalize progress.
    pub prefix_budget: usize,
    /// Support size seen at the current prefix position.
    pub support_size: usize,
    /// Number of candidates currently available to the local branch.
    pub candidate_count: usize,
    /// Top-1 probability or confidence estimate from the local branch.
    pub top1_prob: f64,
    /// Mass assigned to the top-k candidate set, if available.
    pub topk_mass: f64,
    /// Local margin signal, expected in [0, 1] after caller-side normalization.
    pub margin: f64,
    /// Branch agreement or consensus score, expected in [0, 1].
    pub agreement: f64,
}

/// Bounded policy knobs for turning prefix features into control values.
///
/// The defaults are intentionally conservative: the scalar gate stays in a
/// narrow usable band, and the per-mode decay template is derived from that
/// scalar rather than inventing a second opaque control path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrefixAdaptiveGatePolicy {
    pub scalar_min: f64,
    pub scalar_max: f64,
    pub decay_min: f64,
    pub decay_max: f64,
    pub scalar_bias: f64,
    pub progress_weight: f64,
    pub support_weight: f64,
    pub candidate_weight: f64,
    pub confidence_weight: f64,
    pub margin_weight: f64,
    pub disagreement_weight: f64,
    pub decay_shape_weight: f64,
}

impl Default for PrefixAdaptiveGatePolicy {
    fn default() -> Self {
        Self {
            scalar_min: 0.10,
            scalar_max: 0.90,
            decay_min: 0.50,
            decay_max: 1.25,
            scalar_bias: -0.10,
            progress_weight: 0.20,
            support_weight: 0.25,
            candidate_weight: 0.10,
            confidence_weight: 0.20,
            margin_weight: 0.15,
            disagreement_weight: 0.30,
            decay_shape_weight: 0.45,
        }
    }
}

/// Compact adaptive control output for a fixed number of local modes.
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveControl<const MODES: usize> {
    pub scalar_modulation: f64,
    pub decay_template: [f64; MODES],
}

/// Convenience alias for the native Conker-3 path.
pub type Conker3AdaptiveControl = AdaptiveControl<3>;

/// Normalize a raw prefix feature into [0, 1].
pub fn clamp01(value: f64) -> f64 {
    if value.is_nan() {
        0.0
    } else {
        value.clamp(0.0, 1.0)
    }
}

fn clamp_bounds(min: f64, max: f64, value: f64) -> f64 {
    let (lo, hi) = if min <= max { (min, max) } else { (max, min) };
    value.clamp(lo, hi)
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

impl PrefixAdaptiveGateFeatures {
    pub fn prefix_progress(&self) -> f64 {
        if self.prefix_budget == 0 {
            0.0
        } else {
            clamp01(self.prefix_len as f64 / self.prefix_budget as f64)
        }
    }

    pub fn support_pressure(&self) -> f64 {
        1.0 / (1.0 + self.support_size as f64)
    }

    pub fn candidate_pressure(&self) -> f64 {
        1.0 / (1.0 + self.candidate_count as f64)
    }

    pub fn confidence(&self) -> f64 {
        clamp01(self.top1_prob.max(self.topk_mass))
    }

    pub fn margin(&self) -> f64 {
        clamp01(self.margin)
    }

    pub fn agreement(&self) -> f64 {
        clamp01(self.agreement)
    }

    pub fn disagreement(&self) -> f64 {
        1.0 - self.agreement()
    }
}

pub fn scalar_modulation_factor(
    features: &PrefixAdaptiveGateFeatures,
    policy: &PrefixAdaptiveGatePolicy,
) -> f64 {
    let raw = policy.scalar_bias
        + policy.progress_weight * features.prefix_progress()
        + policy.support_weight * features.support_pressure()
        + policy.candidate_weight * features.candidate_pressure()
        + policy.confidence_weight * features.confidence()
        + policy.margin_weight * features.margin()
        - policy.disagreement_weight * features.disagreement();
    let unit = sigmoid(raw);
    clamp_bounds(policy.scalar_min, policy.scalar_max, unit)
}

pub fn decay_modulation_template<const MODES: usize>(
    features: &PrefixAdaptiveGateFeatures,
    policy: &PrefixAdaptiveGatePolicy,
) -> [f64; MODES] {
    assert!(MODES > 0, "MODES must be positive");
    let scalar = scalar_modulation_factor(features, policy);
    let progress = features.prefix_progress();
    let support_pressure = features.support_pressure();
    let confidence = features.confidence();
    let disagreement = features.disagreement();
    let shape = 1.0
        + policy.decay_shape_weight
            * (0.50 * progress + 0.25 * support_pressure + 0.15 * confidence - 0.20 * disagreement);
    from_fn(|mode| {
        let mode_fraction = if MODES == 1 {
            0.0
        } else {
            mode as f64 / (MODES as f64 - 1.0)
        };
        let local_scale = (1.0 - 0.75 * mode_fraction).max(0.0);
        let value = scalar * shape * local_scale;
        clamp_bounds(policy.decay_min, policy.decay_max, value)
    })
}

pub fn build_adaptive_control<const MODES: usize>(
    features: &PrefixAdaptiveGateFeatures,
    policy: &PrefixAdaptiveGatePolicy,
) -> AdaptiveControl<MODES> {
    AdaptiveControl {
        scalar_modulation: scalar_modulation_factor(features, policy),
        decay_template: decay_modulation_template::<MODES>(features, policy),
    }
}

pub fn conker3_adaptive_control(
    features: &PrefixAdaptiveGateFeatures,
    policy: &PrefixAdaptiveGatePolicy,
) -> Conker3AdaptiveControl {
    build_adaptive_control::<3>(features, policy)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_features() -> PrefixAdaptiveGateFeatures {
        PrefixAdaptiveGateFeatures {
            prefix_len: 48,
            prefix_budget: 128,
            support_size: 4,
            candidate_count: 3,
            top1_prob: 0.41,
            topk_mass: 0.73,
            margin: 0.22,
            agreement: 0.64,
        }
    }

    #[test]
    fn scalar_and_template_are_bounded() {
        let features = sample_features();
        let policy = PrefixAdaptiveGatePolicy::default();
        let control = build_adaptive_control::<3>(&features, &policy);

        assert!(control.scalar_modulation >= policy.scalar_min);
        assert!(control.scalar_modulation <= policy.scalar_max);
        assert_eq!(control.decay_template.len(), 3);
        for value in control.decay_template {
            assert!(value >= policy.decay_min);
            assert!(value <= policy.decay_max);
        }
    }

    #[test]
    fn same_prefix_inputs_produce_same_control() {
        let features = sample_features();
        let policy = PrefixAdaptiveGatePolicy::default();
        let first = conker3_adaptive_control(&features, &policy);
        let second = conker3_adaptive_control(&features, &policy);
        assert_eq!(first, second);
    }

    #[test]
    fn more_confident_prefixes_raise_scalar() {
        let policy = PrefixAdaptiveGatePolicy::default();
        let mut weak = sample_features();
        weak.top1_prob = 0.12;
        weak.topk_mass = 0.18;
        weak.agreement = 0.10;

        let mut strong = sample_features();
        strong.top1_prob = 0.84;
        strong.topk_mass = 0.91;
        strong.agreement = 0.92;

        let weak_scalar = scalar_modulation_factor(&weak, &policy);
        let strong_scalar = scalar_modulation_factor(&strong, &policy);
        assert!(strong_scalar >= weak_scalar);
    }
}
