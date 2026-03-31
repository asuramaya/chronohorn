## Frontier Queue

This is the current score-first run order.

### Running now

- `chronohorn/manifests/frontier_queue_after_rowstats.jsonl`

Observer commands for the active frontier:

- `python -m chronohorn observe status --manifest chronohorn/manifests/frontier_long_slop_matrix.jsonl --probe-runtime`
- `python -m chronohorn observe query-records --manifest chronohorn/manifests/frontier_long_slop_matrix.jsonl --probe-runtime --kind runtime_state`

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

### Current blockers

- The fixed-batch trust gate is resolved.
- The promoted `18x` exported bundle exists and the `100M` row-stats artifact exists.
- The remaining live question is score: whether packed `4/8/12/16MB` closes the gap on the real `18x` base.

### Mac lane

The Mac should stay on short local probes and export work only.

Do not use the Mac for:

- large row-stats builds
- oversized table evals
- recursive readout experiments

### Priority order

1. Finish the packed `4/8/12/16MB` full-val matrix on the promoted `18x` exported base.
2. Promote the best packed configuration as the new submission-shaped reference.
3. Resume next pure-causal or packed-variant search only after those full-val numbers land.
