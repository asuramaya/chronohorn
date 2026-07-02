# Session 13 — Reconstruction from Local Artifacts

> **Status: partial record.** Session 13 ran on the macOS workstation
> (`/Users/asuramaya/Code/carving_machine_v3/chronohorn`); its results DB and
> result JSONs were never synced to this machine. This document reconstructs
> the session from what *is* local: the 30 wave manifests (with embedded
> standalone training scripts), 48 launch records in `out/fleet/`, and the
> session 12 paper's mandate. **All bpb numbers for session 13 are on the Mac
> DB / Sharts archive.** The known endpoint, per the operator: the campaign
> capped at **~1.78 bpb** before time and budget ran out.
>
> Written 2026-07-02 during post-competition triage. To finish this record:
> sync the Mac's `out/chronohorn.db` + result JSONs, run
> `scripts/backfill_db.py`, and convert this file to `session13.tex`.

## Context

Session 12 closed (2026-04-19) with the verdict that the causal-bank family is
ceiling-bound at 1.78–1.83 bpb on byte-level FineWeb-10B, with a structural
16-byte effective context, and directed session 13 to exit the family and
pilot a BASED-class hybrid (linear-cost mixer + sliding-window attention).
The OpenAI Parameter Golf deadline was 2026-04-30.

**Session 13 was a single-day campaign: 48 launches in under 23 hours,
2026-04-20 01:30 → 23:47 UTC**, on the slop fleet (RTX A4000, k8s).

Unlike prior sessions, session 13 did not extend the causal-bank training
stack. Every wave ships a **self-contained standalone PyTorch script**
base64-embedded in the manifest command (`byte_pilots_v*.py`,
`family: standalone`) — maximum iteration speed, zero framework coupling.

## Wave narrative

| Wave | Time (UTC) | What | Why |
|---|---|---|---|
| 1 | 01:30 | MEGABYTE pilot: patch P=4/P=8 over scanablate-modes1024, GRU autoregressive intra-patch decoder | First patching test; multiply 16-byte effective context by P |
| 2 | 02:08 | Non-transformer pilots: byte n-gram (h512), WaveNet 8L×128, rolling-window | First "exit the family" candidates |
| 3 | 02:23 | Same, re-run with **fixed uint16 reader** | Wave 2 hit the heinrich-P6 uint8/uint16 shard bug — origin of `chronohorn/data/byte_reader.py` |
| — | 02:34 | **n-gram ceiling audit**: empirical backoff n-gram entropy floor of FineWeb bytes vs max order | Calibrate how much of bpb is local statistics |
| 4 | 06:16 | Mechanism pilots: Hopfield, Hopfield-pure, SDM, WaveNet+PKM, multi-time EMA | Physics/neuro-adjacent sub-O(n²) memory mechanisms |
| 5/5b | 06:56 | Combos: PKM+multitime, PKM-scaled, multitime-deeper (5b: re-runs w/ checkpoints for MRI) | Compose wave-4 survivors |
| 6 | 07:51 | combo-deeper / combo-bigmem / combo-long (50k) | Scale the composed architecture |
| 7 | 08:45 | combo + SWA / megamem / bigmem-long | **First sliding-window-attention insertion** (the session-12 BASED mandate) |
| 8 | 09:38 | combo-swa-bigmem, combo-swa-long, combo-swa-w256 | SWA + winning lever stack |
| 9/9b/9c | 10:34 | w512, w256-bigmem, swa-bigmem-100k (9c = re-dispatch after script ordering bug) | Window scaling + memory scaling |
| 10 | 12:37 | w256-bigmem at 100k steps | Training-length scaling |
| 11 | 13:08 | **w1024 = full attention test** at seq_len 1024 | Does more window keep paying? |
| 12 | 16:57 | Post-window-saturation: c192, multihead-SWA, two-SWA | Window saturated → scale width instead |
| 13/13b | 17:02 | c192 at 100k, then 200k | Extend the new champion |
| 14/14b | 17:42 | c256; c192-megamem (65k PKM, top-24) | Width and memory scaling continued |
| 15/15b/15c | 19:50 | fast-bench, torch.compile bench, c192+compile+bs32 | Throughput engineering |
| 16 | 21:14 | c192 + compile + bs32 + 100k | |
| 17 | 22:15 | **c192 + compile + bs32 + 200k** — presumptive final champion | |
| 18 | 23:47 | c192 + MoE readout (8 experts, top-2) replacing post-body MLP | Last experiment of the campaign |

## The final architecture (`c192`, wave 17)

From the embedded `byte_pilots_v9_compile.py`:

- **Body:** WaveNet-style causal dilated conv stack (kernel 3, dilation 2^i,
  SiLU, LayerNorm, residual), 3 pre + 2 mid + 3 post layers, channels=192.
- **Memory:** PKM product-key memory, 32,768 slots, top-16, key_dim 96.
- **Multi-timescale:** EMA banks at τ = 4, 16, 64, 256 (causal conv
  implementation) — the causal-bank idea, miniaturized.
- **Attention:** exactly one CausalSWA layer, window 1024 (= full context at
  seq_len 1024), head_dim 64.
- **Training:** FineWeb-10B bytes, seq_len 1024, batch 32, 200k steps,
  torch.compile, final eval 100 batches, RTX A4000.

The "non-attention" bet ended with one attention layer over a 1024-byte
window — the BASED-hybrid shape, arrived at empirically.

## Bugs session 13 surfaced (fixed in the 2026-07-02 commits)

1. **uint16 shard reader** (wave 2→3): a fresh pilot script re-made the
   heinrich session-11 P6 mistake. Fix: canonical `data/byte_reader.py` +
   provision-time sanity checks.
2. **Patch decoder legality**: legacy `hybrid` decoder choice silently fell
   through to a future-leaking flat decoder; `patch_size>1` now requires an
   explicit causal decoder.
3. **polyhash bpb normalization** (bug #14): byte-level vocab under-reported
   bpb by the sp1024 ratio (~2.436×).
4. **Planner telemetry**: every session-13 launch was predicted at 1,195,560
   tok/s from one stale `ctrl-patch4-ssm128.json` sample — the megabyte pilot
   actually ran at 18.6k tok/s (64× off). Fix: architecture-signature
   telemetry matching.
5. **Export routing**: catchup exports routed by most-recent-mtime and
   ignored the illegal flag. Fix: per-manifest session routing + illegal
   guard (documented in session12.tex, committed now).

## Interpretive note (to verify against the Mac DB)

The campaign's endpoint (~1.78 bpb) is numerically the same as the
causal-bank family ceiling session 12 measured (1.78–1.83), despite a
completely different architecture with a 1024-byte attention window. Two
hypotheses, distinguishable once the DB is synced:

1. **Local-statistics floor**: the n-gram ceiling audit (wave 0) was run for
   exactly this comparison. If the backoff n-gram floor at achievable orders
   sits near ~1.7–1.8, then every under-trained architecture converges there
   regardless of mechanism, and the cap is a *compute/optimization* artifact,
   not an architectural one.
2. **Genuine architectural ceiling at this budget**: if w1024 (wave 11)
   showed no gain over w256 (window saturation, as the wave-12 naming
   implies), then even full attention over 1024 bytes couldn't exploit
   context beyond local statistics at ~16 GPU-hours of training — which is
   the same conclusion at a different altitude: the binding constraint was
   budget, not architecture.

Either way, the comparison point from the competition: naive baseline
1.2244 bpb, winning entries ~1.11 bpb — achieved with ~13 GPU-hours
(8×H100×10min) per run plus quantization/test-time-training tricks, versus
~16 GPU-hours for session 13's *entire 48-run campaign* on A4000s.
