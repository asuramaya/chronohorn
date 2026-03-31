# Chronohorn C3/C7 Next

Status: research note for a teacher/distillation branch. This is not the current promoted runtime/compiler path.

This is the next chapter after the current `match+skip` line.

The immediate result that triggered it:

- real `@fineweb`, `1M / 16384 / 16384 / depth12 / k16`
- direct `match+skip`: `3.124678 bpb`
- weighted direct: `3.124680 bpb`
- ranked-support distilled direct: `3.124803 bpb`

So the three-term teacher on the current tiny gate is legal, but not enough. The bottleneck is no longer "do we have a real oracle?" We do. The bottleneck is the student/runtime family.

## Thesis

The right merge line is:

- `Conker-3` legal base architecture
- `Conker-7` narrow-teacher training method
- ranked BLINX oracle distilled through a three-term loss

Not:

- more sweeps on the existing `match+skip` gate
- more scalar oracle targets into the same small control head

## What To Port From Conker-3

Essential:

- frozen linear multiscale substrate
- parallel local residual coder
- `window4` local path
- additive mixing, not learned per-token gate as the mainline
- static bank gate for oscillatory/non-oscillatory banks
- short/medium reservoir, not broad half-life tails

Concrete `Conker-3` frontier traits from the docs/code:

- best replicated under-cap branch:
  - `17x / 2200 / staticgate`
- strong legal packed anchors:
  - `16x / 2200 / staticgate`
  - `17x / 2200 / staticgate`
- local residual is real; `window4` is the stable winner
- `half_life_max` wants to stay narrow, roughly `8 .. 16`
- oscillatory fraction around `0.875` was the best late packed direction

Relevant source:

- `/Users/asuramaya/Code/carving_machine_v3/conker/src/conker3.py`
- `/Users/asuramaya/Code/carving_machine_v3/conker-standalone/conker/docs/CONKER3.md`

Nonessential baggage to reject:

- old branch naming
- MLX-specific training wrappers
- historical quantization machinery in the first port
- broad branch matrix search before the port works

## What To Port From Conker-7

Essential:

- narrow teacher only
- weak teacher weight
- delayed teacher start
- warm-start from a legal causal base
- no teacher artifact at eval

Concrete `Conker-7` recipe:

- narrow teacher:
  - `exact2 + exact3`
- bidirectional teacher slightly beat future-only on the narrow row
- weak weight wins:
  - around `teacher_weight=0.05`
- delayed start helps:
  - around `250 .. 500`
- warm-start from a legal student is the real lever

Relevant source:

- `/Users/asuramaya/Code/carving_machine_v3/conker/src/conker7.py`
- `/Users/asuramaya/Code/carving_machine_v3/conker-standalone/conker/docs/CONKER7.md`

Historical warning:

- the old `Conker-7` headline numbers were tied to the contaminated `Conker-4b` lineage
- port the method, not the historical result

## What BLINX Supplies

BLINX is the oracle, not the runtime.

The teacher should be:

- train-only
- leave-one-out
- contamination-aware
- ranked candidate distributions, not just binary small-candidate labels

What matters now:

- candidate identities
- candidate counts / ranked support
- ambiguity signal

Not:

- raw uniqueness
- direct runtime scoring
- codec claims

## Chronohorn Student

The student should stop being "tiny gate over `match+skip`" and become:

- a legal `Conker-3`-style causal scorer
- with a small residual/control surface trained by the oracle

That means the teacher distills into a real scorer, not just a logistic mixer.

## Three-Term Loss

Use the three-term ranked-support loss from the bridge discussion:

- gold NLL
- support-set mass term
- ranked teacher term inside the support

Recommended starting regime from the `Conker-7` lesson:

- `alpha = 0.95`
- `beta = 0.04`
- `gamma = 0.01`

with:

- ambiguity weighting on the oracle terms
- teacher off for singleton supports
- delayed activation after warm-up

## First Chronohorn C3/C7 Branch

Build `Chronohorn-C3C7-1`:

1. Port the `Conker-3` runtime core into Rust.
2. Train it causally without teacher first.
3. Save a legal warm-start checkpoint.
4. Turn on ranked BLINX teacher late:
   - narrow teacher only
   - weak weight
   - delayed start
5. Keep the current Chronohorn audit suite on every promoted run.

## Suggested First Recipe

Base:

- `window4`
- additive local residual
- static bank gate
- `half_life_max in {8, 16}`
- oscillatory fraction near `0.875`

Training:

- warm-start from the legal causal base
- teacher start around step `1100` of `2200`
- weak teacher:
  - total non-NLL weight `0.05`
- split as:
  - `alpha = 0.95`
  - `beta = 0.04`
  - `gamma = 0.01`

Teacher:

- ranked BLINX support distribution
- no broad lexical teacher families in the first pass

## What This Replaces

This does not kill the current `match+skip` line as an architecture probe.

It changes its role:

- `match+skip` remains the current best lightweight Chronohorn bridge line
- `Chronohorn-C3C7-1` becomes the main attempt to cash out the deeper Conker/BLINX/Giddy-Up findings

## Bottom Line

The next frontier attempt should not be:

- another gate
- another sweep
- another small target ablation

It should be:

- `Conker-3` legal base
- `Conker-7` training discipline
- BLINX ranked teacher
- Chronohorn audit boundary
