# GPU Roadmap: Making Causal Banks Competitive on CUDA

## The Opportunity

SSMs have a theoretical advantage on byte-level prediction — they maintain
compressed fixed-size state with constant memory regardless of context length,
while transformers pay quadratic cost for their "perfect recall" attention.
Research from Goomba Lab (2025) confirms this: SSMs excel on character/byte-level
data where transformers struggle.

Chronohorn's 1.806 BPB proves the causal-bank approach works. The gap to
competitive isn't the paradigm — it's that the implementation is tuned for
Apple Silicon, not for the hardware that dominates competition training.

## Why Current Code Underperforms on GPU

An H100 SXM delivers 989 TFLOPS FP16 with tensor cores and 3.35 TB/s HBM3
bandwidth. To saturate compute, operations need arithmetic intensity above
~10 FLOPS/byte. Here's how the current architecture maps:

| Operation | FLOPS/byte | H100 Utilization |
|-----------|-----------|-----------------|
| Embedding lookup | ~0 | ~0% |
| Input projection [batch*seq, 64] x [64, 256] | ~6 | ~60% |
| Chunked cumsum (K=32 sequential) | ~0.25 | ~2% |
| FFT kernel recompute | ~0.03 | ~0.3% |
| Kernel matmul [256, batch, 256] x [256, 256, 256] | ~2.7 | ~27% |
| **Readout MLP** [batch*seq, 320] x [320, vocab] | **~8.6** | **~86%** |
| Routed experts (8 sequential) | ~4 | ~40% |

**Weighted average: 1-3 FLOPS/byte — roughly 10x below H100 saturation.**

The readout is the only path that naturally fits GPU physics. Everything
upstream (substrate computation) is memory-bound sequential work.

## What's Been Done (This PR)

### 1. Batched Expert Readout

The routed expert readout previously ran 8 experts in a Python for-loop —
8 sequential matmuls where one batched matmul would do. Now uses `torch.bmm`
for both expert stages plus `torch.einsum` for the routing reduction.

This isn't CUDA-specific: batched matmul is faster on Metal too. Expected
2-3x speedup on the readout path across all backends.

### 2. CUDA Training Defaults

When `device=cuda`:
- TF32 enabled for matmul and cuDNN (~1.3x throughput, negligible accuracy impact)
- Fused AdamW (single kernel for optimizer step instead of per-param updates)
- `torch.compile(mode='max-autotune')` instead of default (30s trace overhead,
  then persistent kernel selection)

### 3. CUDA Profiler Integration

`--profile-cuda N` wraps the first N training steps in `torch.profiler` and
writes a Chrome trace to `out/profile/`. Load in `chrome://tracing` or
TensorBoard to see:
- Per-op CUDA time and memory
- Tensor core utilization
- Memory bandwidth saturation
- Kernel launch overhead

This is the diagnostic tool needed before any deeper optimization work.

## Short-Term (Next 2 Weeks)

### Enable torch.compile for Full Graph

Current compile wraps the model but not the loss computation or optimizer
step. Wrapping the full train step in a compiled function enables:
- Operator fusion across forward + loss + backward
- Elimination of Python overhead in the training loop
- Automatic kernel selection for small ops

```python
@torch.compile(mode='max-autotune', fullgraph=True)
def train_step(model, x, y, optimizer):
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss
```

Expected: 1.5-2x over current compiled path.

### Batch Size Scaling

Metal training uses batch=16 (unified memory constraint). H100 HBM3 (80GB)
can handle batch=128-256 for this model size. Larger batches amortize kernel
launch overhead and improve tensor core occupancy.

## Medium-Term (Weeks 3-5)

### Parallel Scan via mamba-ssm

The chunked sequential scan (`_linear_states_recurrent`, K=32) has O(T)
sequential depth with inter-chunk state dependencies. The `mamba-ssm` package
provides GPU-optimized selective scan kernels with O(log T) depth using
the blelloch parallel prefix algorithm.

```python
from mamba_ssm import selective_scan_fn

# Replace chunked cumsum loop with:
states = selective_scan_fn(
    u=drive,       # [batch, seq, modes]
    delta=gates,   # [batch, seq, modes]
    A=...,         # decay parameters
    B=...,         # input projection
    C=...,         # output projection
)
```

Mamba-2's SSD framework achieves 2-8x speedup over Mamba-1 by fusing
parameter fetching + recurrence + discretization into a single kernel.

Expected: 2-4x speedup on the recurrence path (from ~2% to ~15% H100
utilization for the substrate).

### Mixed Precision Training (AMP)

The current training loop is FP32 throughout. Adding `torch.cuda.amp` with
FP16 forward/backward and FP32 optimizer state would:
- 2x memory bandwidth efficiency (FP16 tensors)
- Access tensor core FP16 peak (989 TFLOPS vs 495 for FP32)
- Enable larger batch sizes

## Long-Term (Weeks 5+)

### Fused Triton Kernel for Core Path

Combine embedding -> input projection -> recurrence -> readout into one
kernel. Eliminates intermediate HBM round-trips:

- Current: 5 separate kernel launches, each reading/writing HBM
- Fused: 1 kernel, intermediate values stay in registers/L1
- Arithmetic intensity: 1-3 -> 5-8 FLOPS/byte
- Expected: 3-5x forward pass speedup

Trade-off: kernel is locked to specific config. Architecture changes require
kernel rewrites. Only worth it once the architecture stabilizes.

### Hybrid SSM + Selective Attention

Goomba Lab found optimal ratio is ~3:1 SSM:attention for byte-level tasks.
Replace some causal-bank layers with cross-sequence attention (XSA) to add
fine-grained recall where the SSM's compressed state loses information.

This keeps the linear-time core but adds targeted O(n^2) attention only where
the model needs it — a principled tradeoff informed by the profiler data.

### Quantization-Aware Training

The Parameter Golf competition has proven that int5/int6 quantization with
straight-through estimator (STE) gradients is viable:
- Train with simulated quantization noise in the forward pass
- Backward pass uses STE to flow gradients through the rounding
- Final model exports at int6 (6 bits per parameter)
- Reduces artifact size by ~4x vs FP32

This is especially relevant for the 16MB artifact budget constraint.

## Reference Data

- Goomba Lab SSM vs Transformer: https://goombalab.github.io/blog/2025/tradeoffs/
- Mamba-2 SSD Framework: https://pli.princeton.edu/blog/2024/mamba-2-algorithms-and-systems
- Parameter Golf leaderboard: 1.1147 BPB (transformer, int6+GPTQ)
- Chronohorn current best: 1.806 BPB (causal-bank, learned recurrence)
- L3TC (RWKV compression, AAAI 2025): 48% savings vs gzip using SSM
