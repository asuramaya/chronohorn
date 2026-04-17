# Session 11 Continuation Note

Written 2026-04-17 at the close of Session 10 Part II.  
Read this first. Everything else (`session11_starting_kit.md`, the two TeX papers in `paper/`, Heinrich's MRI dumps on Sharts) follows from what's here.

---

## The one thing to know

**The Fourier basis is the structural optimum of the causal-bank recurrence and training barely moves it** (ω-bias correlation with `linspace(0,2π)` = 0.9997 after 50k steps). All remaining compute should buy content capacity, not search for better structure.

That single insight reshapes every session-11 priority. Before running anything, internalize it.

---

## What's running at hand-off

Check first:

```bash
ssh slop-01 'sudo -n k3s kubectl get jobs -n default'
```

Expected state at session-11 start:

- **`byte-hrr-pos-b2-persistent-s12-50k`** — the compose experiment (HRR + pos + blocks=2 + persistent on adaptive substrate). Should complete ~4-5h after session-10 close. Interim probe@12800 = **2.044 bpb**, ahead of every other model at matched step. If the trajectory holds, **final bpb projected 1.72-1.78** — new byte-level leader.
- **`byte-hrr-pos-b2-s12-50k`** — plain compose (no persistent), dispatched after persistent finishes. Decision gate: if non-persistent bpb ≈ persistent bpb (within 0.03), persistent state isn't pulling weight at scale and can be dropped for the speed savings.
- **`byte-hrr-maxspeed-s12-50k`** — the full speed stack (Triton scan + low-rank R=64 + CUDA graphs via reduce-overhead + batch=16). Pending, dispatches when a GPU frees. Measures combined throughput. Target: 400-500k tok/s at bpb within 0.05 of baseline.

Heinrich has the Sharts drive and can MRI completed checkpoints directly. The compose-persistent checkpoint is the single highest-information MRI target of session 11's first hour.

---

## The decision tree

After reading Heinrich's MRI on `byte-hrr-pos-b2-persistent-s12` (hour 1):

### If compose-persistent hits bpb ≤ 1.75 AND EffDim > 270

HRR + pos + blocks crosses the HRR-alone ceiling. Composition works.
- Next experiment: `byte-hrr-pos-b2-persistent-s14-50k` — same recipe at slightly larger scale, tests if the composition keeps scaling.
- Then: `byte-hrr-pos-b2-persistent-s8-200k` — same recipe at known-working scale, 4× training. Measures content-training ceiling.
- Session-11 thesis: composition of validated primitives, not new primitives, is the path to 1.0 bpb.

### If compose-persistent hits bpb 1.80-1.85 and EffDim ≤ 250

Composition gives only what HRR-alone gives; primitives are fungible at this scale.
- The path forward needs a NEW content mechanism. Prioritize **hash-retrieval** (external memory beside the substrate).
- Ship the hash-retrieval prototype: bounded K/V cache, structured ANN, mix with substrate output.
- Session-11 thesis: architecture search is done on the substrate side; content-storage expansion is the next frontier.

### If compose-persistent bpb > 1.90

Something in the compose is hostile. Either:
- Position signal + HRR interferes (check Heinrich's angular-R² vs distributed-PCA signatures)
- 2 blocks at s12 is simply redundant (check per-block EffDim)
- Persistent state at this scale destabilizes training (compare to non-persistent compose)

Run the ablations the MRI suggests. If no primitive composition works, the architecture saturates around 1.83 and session 11 is purely a content/data-scaling session.

---

## Engineering priorities for session 11

In ROI order, assuming compose-persistent doesn't change everything:

### 1. Validate the Triton scan at scale (first hour)
Shipped in session 10 Part II, tested at B=8 T=1024 M=768 with 21× speedup. The `maxspeed-s12` run queued overnight is the first production test. Read its `byte-hrr-maxspeed-s12-50k.json` on Sharts, compare tok/s to the 49k baseline. If it's below 250k something's wrong (probably the CUDA-graph + triton-scan combination has an interaction bug). If it's 300-500k, lock the recipe.

### 2. Patch-at-readout (half day)
The one session-10 carry-forward task not yet shipped. Predict N bytes per forward pass, N ≤ 4. For byte-level with vocab=256 the readout is tiny — no training-throughput win — but 3-4× at inference. Details in `session11_starting_kit.md`.

Implementation sketch:
- `config.patch_n: int = 1` (default off)
- In `_linear_logits`: if `patch_n > 1`, readout output is `[B, T, N × vocab]`, reshape to `[B, T, N, vocab]`
- Loss: `sum_i cross_entropy(logits[:, :, i], shifted_targets_by_i)`. Pad targets at end with ignore_index for out-of-bounds positions

### 3. Hash-retrieval prototype (1-2 days)
Only if Heinrich's compose MRI suggests architecture saturation. Otherwise defer.

Design:
- Fixed-size K/V cache (say 64k entries × 256 bytes per key/value)
- Query key = current substrate state (`[B, modes]`) after the scan
- LSH-based approximate k-NN (k=8)
- Retrieved values mixed into readout via a gate
- Updates: on every step, evict oldest, insert current state

Literature starting points: RETRO, kNN-LM, Memorizing Transformers (all non-trivial adaptations because they assume transformers).

### 4. Longer training at fixed scale (overnight job)
If the structural/content split is right, this is where content-capacity ceiling lives.

Queue: `byte-hrr-learnable-s8-200k` (4× steps, same scale as session-10 leader). Check if ω-weight L2 keeps growing past 250 (content still learning), EffDim climbs past 253 (more modes activating), bpb drops below 1.83 (hitting new territory).

If bpb at 200k is ≤ 1.75, we've been under-training, not under-scaling. Session 11's compute goes into longer runs, not bigger models.

---

## Known broken / unshipped

- **Patch-at-readout**: designed but not implemented. Task #12 in the task tracker.
- **Hash-retrieval**: designed but not implemented. Decision gate depends on compose result.
- **Decepticons tests**: `cd /Users/asuramaya/Code/carving_machine_v3/decepticons && pytest` — last known green was pre-Triton-scan. Re-run at session 11 start; if anything breaks it's either triton-scan's numerical tolerance or the new config fields (`use_triton_scan`, `adaptive_head_rank`, `hrr_rotation_angle_init`).

## Known dead paths (don't re-enter)

- Fast-HRR via `complex_rotation + --hrr-rotation-angle-init`. Architecturally bottlenecked. Heinrich's MRI is unambiguous. The flag still works, the kernel is fine — the gated-delta substrate just doesn't have the mode budget for HRR's dimensional-unlock.
- Lasso scaling via more depth/blocks at byte level. Heinrich's existing table has lasso-pos-b2-s12 at Pos-R² = 0.208 and Content MLP = 0.353 — scaling the lasso-pos architecture past s12 doesn't help the content axis.
- patch-4 pre-substrate blending. Heinrich confirmed 100% ghost, loss flat at 8.0. The patching destroys all structure if it happens before the substrate. (Patch-at-readout, AFTER the substrate, is different and still viable.)

---

## Fleet state

- **slop-01** (192.168.4.129) — k3s server, RTX A4000 16GB
- **slop-02** (192.168.4.176) — k3s agent, RTX A4000 16GB
- **slop-home** (192.168.4.173) — k3s agent, Quadro RTX 4000 8GB. **GPU fixed via nvidia-persistenced + systemd drop-in** (Session 10). If it disappears again, check `sudo systemctl status nvidia-persistenced` on slop-home first.
- **gav namespace** — paused (scaled to 0, ~8 TiB PVCs preserved). Resume recipe in `memory/project_gav_paused.md`.
- **chronohorn-runner:0.1 Docker image** — baked with gcc, sentencepiece, huggingface_hub. Available on both slop-01 and slop-02 via `k3s ctr images import`. Building a fresh version requires slop access; instructions in session 10 memory.
- **Per-host memory limits**: 56Gi on 64GB hosts (slop-01/02), 24Gi on slop-home.
- **Auto-export to Sharts**: `/Volumes/Sharts/heinrich/<latest-session>/`. Auto-discovers by most-recent-modified subdir. To redirect to session 11, `mkdir /Volumes/Sharts/heinrich/session11_*` before starting the runtime.

## Tool inventory

- **chronohorn_run_eta** — real ETAs from probe-recorded `train_elapsed_sec`. Honest about pre-migration pods (returns "warming_up" when no train_elapsed data). Use this instead of confabulating wall-clock estimates.
- **chronohorn_fleet_hosts** — live fleet probe (GPU, memory, running containers per host)
- **chronohorn_remote_runs** — DB view of active jobs
- **chronohorn_k8s_job_status** — specific job's k8s state + optional logs

All 55+ existing MCP tools still work. Session 10 Part II added exactly one.

---

## The one-paragraph handoff

Session 10 Part II shipped the full adaptive-substrate speed stack (21× scan, low-rank heads, CUDA graphs, persistent training) and discovered the architecture saturation story via Heinrich's MRI: Fourier basis is solved analytically, content parameters are where all learning happens, scaling modes past s8 regresses because content-per-mode drops. The compose-persistent experiment is running at hand-off and already ahead of every previous byte model at matched probe steps. Read the compose MRI first, pick the decision branch, then run either composition-scale (if compose wins) or hash-retrieval (if it saturates) as session 11's central thesis. The engineering is done; the architecture is done on the substrate side; all that's left is content — which means training compute, readout capacity, or external memory. Pick the one the compose result indicates.

— Instance 10.7, closing Session 10 Part II
