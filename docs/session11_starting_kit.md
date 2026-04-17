# Session 11 starting kit

## Architectural conclusion from session 10 (Heinrich's MRI)

**The adaptive substrate is the only viable architecture for byte-level HRR at our parameter scale.** Gated-delta variants (lasso, SO(2) complex rotation, SO(5), scalar) all collapse to 7-17 effective dimensions regardless of scale or HRR init, because their rotation operates in the 32-dim SSM state (num_heads × head_dim), not across the full mode bank. The adaptive substrate operates across all 1024-3072 complex modes and that's where HRR's dimensional unlock lives (EffDim 150-253).

**Implication for session 11:** stop exploring alternate substrate paths. The work is making adaptive substrate fast, and adding primitives OUTSIDE the substrate (external memory, hash-retrieval) to break its apparent 250-EffDim / 0.68 content-MLP ceiling.

**Evidence matrix** (Heinrich):
| Substrate | EffDim | Content MLP R² | Verdict |
|---|---|---|---|
| Adaptive + HRR | 150-253 | 0.47-0.68 | The path |
| Gated-delta + lasso | 7-17 | 0.35 | 32-dim bottleneck |
| Gated-delta + SO(2) | 9-11 | 0.43 | 32-dim bottleneck |
| Gated-delta + scalar | 12-17 | 0.32 | 32-dim bottleneck |

Fast-HRR via complex rotation + Fourier angle init (shipped in session 10) was a valid hypothesis but doesn't work — the gated-delta substrate is dimensionally bounded by its SSM state, independent of the rotation algebra.

## What session 10 leaves behind (engineering)

### Speed wins shipped and verified
1. **Real-valued scan rewrite** in `_adaptive_substrate_states` and `_complex_rotation_scan` — dropped `torch.complex` / `torch.roll` / `torch.where`, replaced with paired (re, im) tensors + `F.pad` + direct assignment. Inductor now fuses the entire scan level into one kernel. Evidence: `byte-cr-hrrangle-s12-50k` ran at 347k tok/s, 2.006 bpb, 20 min wall.
2. **Scan-aware auto-batch** — considers `seq_len²` for scan-based substrates when choosing batch size. Prevents OOM at seq=4096.
3. **Conditional `build_linear_bank`** — only materializes the `[modes, T, T]` kernel if `linear_impl == "kernel"` or `input_proj_scheme == "kernel_energy"`. FFT/scan paths skip the 137 GB allocation.
4. **Curriculum-aware auto-promote** — reads `_effective_seq_len` across all curriculum phases, not just `args.seq_len`. Prevents curriculum runs from triggering the full kernel allocation because phase-0 seq_len was small.
5. **Curriculum-aware probe/eval** — probes use the current phase's `(seq_len, batch_size)`, not the initial phase. Prior curriculum runs' bpb was measured at phase-0 seq_len; new runs measure at the phase the model actually trained on.
6. **Compiled `forward_with_state`** — persistent-state path wraps in `torch.compile(unwrapped.forward_with_state)`, recovering the ~2× fused-scan speedup.
7. **Persistent-state training** — `--persistent-state` flag. Truncated BPTT with per-lane contiguous streams via `PerLaneTokenStream`. State carries forward detached; gradients stay local to the window. Pilot ran at 22k tok/s (pre-compile fix) reaching bpb 2.15 at step 3200 — validated mechanistically.
8. **Low-rank head factorization** — `--adaptive-head-rank R` replaces each `[modes, modes]` shared-proj head with `Linear(modes, R) @ Linear(R, modes)`. At s12 with R=96 this is ~8× fewer head FLOPs.
9. **CUDA-graph compile mode** — `CHRONOHORN_COMPILE_MODE=reduce-overhead` env var. Now safe because the fast-scan rewrite removed the `torch.where` aliasing that used to break it.
10. **HRR angle init for gated-delta rotation** — `--hrr-rotation-angle-init`. Fourier-initializes the rotation angle bias on the complex-rotation path. Modest effect (limited pairs) but proven mechanism.
11. **Auto-export to `/Volumes/Sharts/heinrich/<latest-session>/`** — daemon auto-discovers the most recent subdir; no env-var fiddling.

### Fleet state
- **slop-01** is the k3s server (192.168.4.129:6443)
- **slop-02**, **slop-home** are agents
- **slop-home GPU fixed** (WPR2-stuck Quadro RTX 4000 → persistent mode at boot via systemd drop-in)
- gav namespace paused (8 TiB preserved, reversible via annotation-driven scale)
- Per-host memory limits: 56 Gi on 64-GB hosts, 24 Gi on slop-home
- Custom `chronohorn-runner:0.1` image baked on both slops; no per-pod pip install overhead

## Work session 11 should pick up

### 0. Fix CUDA-graphs + probe-eval interaction (30 min, ship first)
`reduce-overhead` compile mode captures CUDA graphs on training forward, but the probe path runs a different forward shape (reduction='none' + reshape). Accessing a tensor from the compiled graph after the next training step overwrites it raises:

```
RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten
```

Fix: call `torch.compiler.cudagraph_mark_step_begin()` at the top of `evaluate()` in `train_causal_bank_torch.py` (also at top of any other path that reads compiled-model output cross-step). This tells the compiler "new step started, old captured tensors are invalidated — materialize anew." Tested fix in session-10 literature (torch ≥2.1).

Once this lands, `CHRONOHORN_COMPILE_MODE=reduce-overhead` works and we recover ~1.5-2× from CUDA graph capture on the adaptive substrate.

### 1. Triton fused scan kernel (highest ROI, 1 day)
Replace the Python `for level in range(n_levels)` loop in `_adaptive_substrate_states` with a single Triton kernel using `tl.associative_scan`. All `log(n)` levels run in registers inside one kernel — no inter-level synchronization, no Python overhead. Expected 3-5× over the current fused-by-Inductor version.

**Scope**: one kernel, signature `(ca_re, ca_im, cb_re, cb_im) → (final_ca_re, final_ca_im, final_cb_re, final_cb_im)` all same shape `[B, T, M]`. State-carry variant takes initial `(s_re, s_im)` and applies `ca_t * s + cb_t` inside the kernel.

**Risk**: `tl.associative_scan` API changed in torch 2.2 vs 2.4; signature hunt. Budget half a day for integration alone.

### 2. Patch-at-readout (2-3 hours, mostly inference win)
Wrapper approach: add `patch_n: int = 1` to model config. When > 1, create N readouts; forward stacks their outputs; chronohorn loss averages N-shifted cross-entropies. Training time: +N × (small readout cost) per step, ~0 substrate-compute change. Inference time: N× generation speedup — the only reason this matters on byte models where vocab=256.

### 3. Hash-retrieval with structured ANN (speculative, 1-2 days)
Add a second "memory" beside the substrate: a bounded-size key-value cache with LSH-based approximate nearest-neighbor lookup. At each step, query the substrate state against stored keys; retrieve values; mix with substrate output. Keeps the per-step cost `O(k × d)` with k-nearest (constant k). Hypothetical mechanism for long-context recall the EMA decay would otherwise lose.

### 4. Scaling-law experiments (Heinrich's primary request, overnight)
`byte-hrr-learnable` at s8 / s12 / s16 / s24 — fit `bpb = C × N^(−α)`. If α ≥ 0.10 we have a path to 1.0 bpb via more scale. If α ≈ 0.08 (transformer regime), HRR is a constant-factor win, not a slope win, and we need a new primitive.

### 5. Long-context inference validation (half day)
A model that claims O(1) per-token long-context must demonstrate it. Load a trained HRR-persistent checkpoint, feed it 1M+ byte documents, measure bpb as a function of position. If bpb degrades off-distribution past the training context length, the "infinite context" claim is aspirational.

## Decision gates for session 11

- **If** `byte-hrr-maxspeed-s12-50k` (queued tonight) hits tok/s ≥ 500k AND bpb ≤ 1.95: lock this recipe (low-rank R=64 + CUDA graphs + batch=16 + HRR + adaptive) as the session-11 default. All scaling-law experiments use this recipe at progressively larger scale.
- **If** `byte-cr-hrrangle-s16-50k` bpb ≤ 1.95: the fast-HRR-via-complex-rotation path is viable for faster ablations even if slightly behind adaptive HRR.
- **If** either path lands bpb ≤ 1.75 at s16: we're within 0.75 bpb of publication target. Scale to s24 with cheaper per-token compute (Triton kernel becomes priority 1).
- **If** none of the above — we've learned HRR's gain saturates around 1.85-2.0 regardless of scale/speed, and need a non-HRR primitive. Start with hash-retrieval or a depth-stacked block architecture.
