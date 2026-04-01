## Frontier Queue

This is the current score-first run order.

### Running now

- `chronohorn/manifests/frontier_queue_after_rowstats.jsonl`

Observer commands for the active frontier:

- `python -m chronohorn observe status --manifest chronohorn/manifests/frontier_long_slop_matrix.jsonl --probe-runtime`
- `python -m chronohorn observe frontier --manifest chronohorn/manifests/frontier_long_slop_matrix.jsonl --probe-runtime`
- `python -m chronohorn observe query-records --manifest chronohorn/manifests/frontier_long_slop_matrix.jsonl --probe-runtime --kind runtime_state`

The `frontier` view merges:

- active slop runs and forecasts
- tracked historical reference scores from `state/frontier_status.json`
- the best artifact-feasible small model baseline

Jobs:

- `chronohorn-causal-bank-18x-packed4-fullval`
- `chronohorn-causal-bank-18x-packed8-fullval`
- `chronohorn-causal-bank-18x-packed12-fullval`
- `chronohorn-causal-bank-18x-packed16-fullval`

These are the submission-shaped packed full-val comparisons on the promoted
`18x routed_sqrelu_e8` exported bundle.

### Long slop matrix

- `chronohorn/manifests/frontier_long_slop_matrix.jsonl`

This is the long-horizon two-slop program after the short pilot matrix.

Shape:

- Phase A: `2600`-step ranking pilots on the strongest pilot directions
  - `scale18`, `scale19`
  - `window4`, `window16`
  - `oscfrac=0.99`
  - `lr=1.5e-3` plus one `scale19 / lr=1e-3` stability control
- Phase B: `5200`-step seed confirmations on the two most plausible bases
  - `scale18 / window4 / lr=1.5e-3 / seeds 42,43,44`
  - `scale19 / window4 / lr=1.5e-3 / seeds 42,43,44`
- Phase C: `10400`-step stretch runs
  - `scale18 / window4 / lr=1.5e-3 / seed42`
  - `scale19 / window4 / lr=1.5e-3 / seed42`

All long rows use adaptive probes instead of one final-only probe:

- geometric probe schedule starting at `100`
- ratio `2.0`
- micro cutoff at `200`
- micro eval `2` batches
- standard eval `4` batches
- promotion eval `12` or `16` batches
- promotion count `2`

This is the right long-run slop shape:

- fewer rows than the short pilot scan
- deeper horizons
- explicit seed confirmation
- probe cost controlled by the runtime instead of hand-picked one-off steps

### Fast parity gate

- `chronohorn/manifests/frontier_fixed_batch_parity.jsonl`

Jobs:

- `causal-bank-parity-torch-cuda-routed-e8`
- `causal-bank-parity-torch-cuda-mlp`
- `causal-bank-parity-torch-cpu-routed-e8`
- `causal-bank-parity-torch-cpu-mlp`

These are the fastest structured backend diagnostics. Run them before spending
another long pilot on a backend whose fixed-batch behavior is already off.

Current status:

- fixed-batch MLX vs Torch is green locally for both `mlp` and `routed_sqrelu_e8`
- fixed-batch Torch CPU vs CUDA is green on the slops for both `mlp` and `routed_sqrelu_e8`
- parity rows now log explicit optimizer defaults, backend environment metadata, tokens/s, and estimated TFLOPs

### Run after row-stats exists

- `chronohorn/manifests/frontier_queue_after_rowstats.jsonl`

Jobs:

- `chronohorn-causal-bank-18x-packed4-fullval`
- `chronohorn-causal-bank-18x-packed8-fullval`
- `chronohorn-causal-bank-18x-packed12-fullval`
- `chronohorn-causal-bank-18x-packed16-fullval`

These are the submission-shaped full-val comparisons that matter.

### Exotic 16MB matrix (artifact-viable mutations)

- `chronohorn/manifests/frontier_exotic_16mb.jsonl`

42 short pilots (1000 steps) across the full architectural knob space, all
configured to fit within the 16MB int6 artifact budget:

- Group A: capacity vs routing (scale 4–17 MLP, e2, e4, e8, e16)
- Group B: oscillatory scheduling (mincorr_greedy, period_bucket_greedy)
- Group C: oscillatory fraction and period range (0.0–0.99, periods 1–512)
- Group D: input projection schemes (orthogonal, split_banks, kernel_energy)
- Group E: half-life range (8–256)
- Group F: mix mode (gated vs additive)
- Group G: local window and scale
- Group H: sequence length (128–512)
- Group I: learning rate (2e-3, 3e-3)
- Group J: interaction combos

Key result: **sequence length dominates all other knobs.** `seq_len=512` at
scale 14 MLP hit **2.0510 bpb** — beating the metal mutation reference (2.078).
All other architectural knobs clustered within 2.13–2.16 bpb noise at 1000 steps.

### Exotic deepening

- `chronohorn/manifests/frontier_exotic_deepen.jsonl`

Top 8 pilots deepened to 5200 steps, top 4 to 10000 steps, plus seed
confirmations for the seq512 winner. In progress.

### Current blockers

- The packed `4/8/12/16MB` full-val matrix was stopped (compression defeats
  the point on golf data — the right direction is architectural mutations).
- The exotic 16MB matrix completed pilot phase; deepening runs are in progress.
- The remaining live question is whether seq_len=512 gain holds at depth,
  and whether any architectural knobs separate from noise at longer horizons.

### Mac lane

The Mac should stay on short local probes and export work only.

Do not use the Mac for:

- large row-stats builds
- oversized table evals
- recursive readout experiments

### Priority order

1. Finish the exotic deepening matrix (5200 and 10000 step runs).
2. If seq512 holds at depth, promote it as the new artifact-viable reference.
3. Explore OPC mutations that combine with seq512: byte-class routing,
   online causal context cache, context-dependent decay modulation.
4. Use the forecaster's marginal_gain_per_tflop to guide which directions
   to invest further compute in.
