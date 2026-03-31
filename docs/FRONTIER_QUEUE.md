## Frontier Queue

This is the current score-first run order.

### Running now

- `chronohorn/manifests/frontier_queue_after_rowstats.jsonl`

Jobs:

- `chronohorn-causal-bank-18x-packed4-fullval`
- `chronohorn-causal-bank-18x-packed8-fullval`
- `chronohorn-causal-bank-18x-packed12-fullval`
- `chronohorn-causal-bank-18x-packed16-fullval`

These are the submission-shaped packed full-val comparisons on the promoted
`18x routed_sqrelu_e8` exported bundle.

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
