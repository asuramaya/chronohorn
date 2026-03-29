# Doctrine

`Chronohorn` is built around four hard roles:

1. `oracle`
2. `compressor`
3. `bridge`
4. `audit`

These are not style preferences. They are the project boundary.

## Oracle

The oracle is:

- offline only
- noncausal
- BLINX-style leave-one-out structural analysis
- candidate-set and trust-target production
- contamination-audited before use

The oracle is not:

- a runtime scorer
- an eval-time feature source
- a hidden fallback path

If it touches eval-time scoring directly, the result is invalid.

## Compressor

The compressor is:

- the only deployed scorer
- fully causal
- packed-memory first
- allowed to use orthogonal experts
- measured by actual held-out `bpb`

The compressor must stand on its own.
Any runtime gain has to survive even if the oracle disappears.

## Bridge

The bridge is the seam between oracle and compressor.

Its job is to convert oracle-side structure into legal runtime leverage:

- target generation
- feature prediction
- trust estimation
- gating and control learning

This is the only place where oracle knowledge is allowed to influence the runtime line.

So the central Chronohorn question is:

Can a causal compressor learn useful behavior from noncausal structure without importing that structure into runtime?

## Audit

The audit exists to attack the oracle/runtime boundary, not just the scorer in isolation.

It must prove:

- the bridge did not smuggle oracle information into runtime
- the scorer is still one causal prequential process
- the artifact boundary is honest

That is why `Chronohorn` keeps checks like:

- normalization
- repeatability
- future-suffix invariance
- answer-mask invariance
- prefix-truncation parity
- stream-rechunk parity
- sample-set invariance
- gold-logprob consistency

## Parent Lines

`BLINX` is not “the failed codec.”
It is the first oracle.

`Conker` is not “the old runtime to port.”
It is the parent compressor line.

`Chronohorn` is the first system where both are first-class but separated correctly.

## Build Path

The build path follows the same roles:

1. make oracle targets a formal input
2. make bridge learning a formal subsystem
3. keep compressor runtime pure
4. keep audit pressure on the oracle/runtime boundary

If those four drift back together, the project regresses.
