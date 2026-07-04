# The State of Small Language Models (0.5B–32B), July 2026

*Compiled 2026-07-02 by a web-research agent (~150 searches, primary sources cross-verified). Target hardware: 12GB RTX A3000 mobile (336 GB/s, ≈ desktop RTX 3060 12GB − 7%) + 62GB system RAM. Vendor benchmarks marked **(v)**; independent numbers use Artificial Analysis Intelligence Index v4.1 (re-based in 2026; frontier open GLM-5.2 = 51 — not comparable to 2025 AA scores).*

**The one-paragraph answer.** The industry converged on: sparse MoE with tiny activated parameters (30–35B total / 3B active is the new "small"), hybrid linear-attention layouts at a 3:1 linear:full ratio (Qwen's Gated DeltaNet layout ships from 0.8B to 397B), distillation-first post-training (off-policy reasoning traces, then on-policy logit KL — RL demoted to a short finishing pass), QAT int4 release checkpoints, and multi-token prediction. It abandoned: pure SSM/attention-free production models, BitNet scale-up, RETRO-style retrieval pretraining, from-scratch dense models above ~32B, and Meta's open small-model line entirely. A 12GB + 62GB laptop sits in the sweet spot of the single biggest local-inference shift: MoE experts offloaded to system RAM.

---

## 1. Best small models by family (mid-2026)

### Google — Gemma 4 (Gemma 3 / 3n fully superseded)
Released March 31–April 2, 2026, license changed to Apache 2.0. Built on Gemini 3 research; MTP checkpoints April 16; Gemma 4 12B "Unified" (encoder-free text+image+audio) June 3.

| Model | Params | Notes | 12GB 4-bit fit |
|---|---|---|---|
| E2B | 5.1B total / 2.3B effective (Per-Layer Embeddings) | text+image+audio, 128K ctx; MMLU-Pro 60.0 (v) | Yes (~3GB QAT) |
| E4B | 8B / 4.5B effective | 128K ctx | Yes (~5GB QAT) |
| 12B Unified | 11.95B dense | 256K ctx, encoder-free multimodal | Yes (~7GB QAT) |
| 26B A4B | 25.2B / 3.8B active MoE | AA 26 | Offload only (~15GB) |
| 31B dense | 30.7B | MMLU-Pro 85.2, GPQA-D 84.3 (v); AA 29 | No (~18GB) |

QAT checkpoints for the whole family shipped June 5, 2026 — the 12B QAT (~7GB) is the default local vision model.

### Qwen — Qwen3.5 (Feb–Mar 2026) and Qwen3.6 (Apr 2026): won the small tier
All Apache 2.0, natively multimodal, **Gated DeltaNet hybrid (3:1 linear:full attention) at every size**, 262K context, MTP.

- **Qwen3.5 small series: 0.8B / 2B / 4B / 9B dense-hybrid.** Independent AA: 9B = 32, 4B = 27, 2B = 16, 0.8B = 9 — the 9B roughly doubles the next-best sub-10B (Falcon-H1R-7B = 16, Nemotron Nano 9B v2 = 15). GPQA-D 81.7 (v). Weakness: extreme token hunger (230–390M output tokens on the AA index). All four fit 12GB; 9B ≈ 6GB at 4-bit.
- **Qwen3.6-35B-A3B** (Apr 16): 35B/3B-active MoE; SWE-bench-V 73.4 (v), AA 32. ~19GB Q4 total but ~7GB VRAM with expert offload — the r/LocalLLaMA default for 12–16GB rigs.
- **Qwen3.6-27B** (Apr 22): dense-hybrid (64 layers, 3:1 GDN); AIME'26 94.1, GPQA-D 87.8, SWE-bench-V 77.2 (v); **AA 37 — #1 open model in the 4–40B class**. ~16.8GB Q4 — doesn't fit 12GB.
- Qwen3 dense and Qwen3-30B-A3B-2507 now legacy. Unconfirmed "Qwen3.7" preview (~57 AA), weights unreleased.

### DeepSeek — frontier only; small line still R1-Distill (Jan 2025)
- **DeepSeek V4** (open weights April 2026, MIT): V4-Pro 1.6T/49B-active, V4-Flash 284B/13B-active, 1M ctx. Not laptop-sized (Flash GGUF ≈ 146GB). No V4 distills exist; "V4-Lite" rumors are SEO noise.
- **Engram** (arXiv 2601.07372, Jan 12 2026) — hashed n-gram conditional-memory module — is research + code, not a model; did NOT ship in V4 (V4 uses CSA/HCA sparse attention).
- Small DeepSeek = R1-Distill-Qwen 1.5B/7B/14B (fit) + R1-0528-Qwen3-8B — all now behind Qwen3.5-9B-class.

### GLM — one great mid-size MoE, then nothing small
- **GLM-4.7-Flash** (Jan 19, 2026): 30B-A3B MoE, MIT, SWE-bench-V 59.2 (v) — best small GLM ever; top-3 local coding pick for 12GB-with-offload.
- **GLM-5** (Feb 2026) → **GLM-5.2** (June 13, 2026): ~744B/40B-active, MIT, AA 51 = #1 open-weights model in the world. **No GLM-5-Air exists** (open HF discussion begging for one). GLM-4.5-Air (106B-A12B) remains the smallest Air.

### Meta — exited the open small-model race
No small Llama 4; Scout (109B-A17B) is the smallest. Meta Superintelligence Labs' first model "Muse Spark" (April 2026) is closed-weight. Llama 4.5 promised "before end of 2026." For local users, Meta is currently irrelevant.

### Microsoft Phi — incremental; no Phi-5
Phi-4-reasoning-vision-15B (Mar 2026): selective reasoning (decides when to think), ~200B multimodal tokens (v), fits 12GB tightly. Phi-4 14B / mini / reasoning fit but are outclassed by Qwen3.5-9B. Phi-5 unreleased.

### Mistral
- **Ministral 3** (Dec 2025, Apache 2.0): dense 3B/8B/14B, base/instruct/reasoning + vision; 14B-reasoning claims AIME'25 85% (v). All fit 12GB.
- **Mistral Small 4** (Mar 2026, Apache 2.0): "Small" is now a 119B-total/6B-active MoE (256K ctx, configurable reasoning effort). Laptop-runnable only via RAM offload at ~Q3.

### The rest
- SmolLM3-3B (July 2025) still HF's flagship; no SmolLM4.
- **IBM Granite 4.1** (April 29, 2026): a return to dense pure transformers at 3B/8B/30B after the hybrid Granite 4.0 — IBM says the dense 8B matches its 32B-A9B hybrid and is easier to fine-tune.
- Apple (WWDC 2026): ~20B sparse on-device AFM activating 1–4B; not open weights.
- **NVIDIA Nemotron 3 Nano** (Dec 2025): 31.6B/3.6B-active hybrid Mamba-Transformer MoE, 1M ctx, beats Qwen3-30B-A3B at 3.3× throughput (v). Offload-viable.
- TII Falcon-H1R-7B (Jan 2026): 7B hybrid reasoner; vendor claims wins over Qwen3-32B; independent AA = 16 (case study in vendor inflation).
- Ai2 OLMo 3 / 3.1-32B: strongest fully-open (data+code+weights) models.
- OpenAI GPT-OSS-20B (Aug 2025, 21B/3.6B-active, MXFP4-native, ~o3-mini level): still the fastest capable model on 12GB.
- Also-rans: Apriel-1.6-15B-Thinker, ERNIE-4.5-21B-A3B, Seed-OSS-36B, Hunyuan 0.5–7B (+ native 2-bit 1.8B), EXAONE 4.5 33B (inflated; AA 23).

**Tier calibration:** GPT-4o (Nov 2024) ≈ MMLU-Pro 74 / GPQA-D 54. Best mid-2026 9B clears GPT-4o on reasoning benchmarks while trailing on breadth/robustness/world knowledge. The 15–27B tier is credibly at/above Claude Haiku 4.5 / GPT-5-mini class on many evals; frontier open (GLM-5.2 = 51 AA) remains far ahead of the 27B tier (37 AA).

---

## 2. The recipes

Consensus: **pretrain long on a curriculum, hybridize the attention, distill from a big sibling, short RL pass, then QAT.**

1. **Distillation from a huge teacher — the most-credited technique, two stages.** Pretraining logit distillation (Gemma 3: 256 sampled logits/token; Llama 3.2 from 8B/70B logits). Post-training strong-to-weak distillation (Qwen3 report, arXiv 2505.09388): off-policy trace SFT → **on-policy distillation** (student generates, reverse-KL to teacher logits) — matched direct RL at ~1/10 the GPU-hours, improved pass@k not just pass@1 (replicated by Thinking Machines, Oct 2025). DeepSeek V4's entire post-training replaces monolithic RL with ~10 per-domain expert teachers consolidated via on-policy distillation. Pure trace-SFT (R1-Distill's 800k samples, no RL) remains the cheap entry.
2. **Pruning + distillation is a commodity** (Minitron ~40× fewer tokens than from-scratch; Nemotron Nano 2 pruned 12B→9B after alignment; SlimMoE 41.9B→7.6B).
3. **QAT is release engineering** (Gemma 4 QAT all sizes; Kimi K2 Thinking native INT4; NVFP4 quantization-aware distillation).
4. **Tiny-active MoE conquered 20–35B-total**: OLMoE (MoE matches dense at <½ FLOPs); Qwen3-Next-80B-A3B matched Qwen3-32B at <10% training cost — the economic death sentence for mid-size dense. Below ~10B total, dense still rules.
5. **Hybrid linear attention is the default at small scale** (§4).
6. **Test-time reasoning on small models works; pathologies mapped.** Overthinking: wrong answers carry ~2× thinking tokens; mitigations = trained thinking budgets (Qwen modes, Mistral reasoning_effort, "decide when to think"). AA tracks token efficiency as first-class.
7. **RL demoted:** works atop a distilled base (DeepScaleR-1.5B: 43.1% AIME'24 for ~$4.5k) but sharpens pass@1 without expanding pass@k. Ordering: distill (traces → on-policy logits) → short RLVR → done.
8. **Data:** token counts climbing (Qwen3 36T, Nemotron 25T); synthetic data + multi-stage curricula with a reasoning-heavy mid-training stage are universal. "Curriculum" institutionalized as "mid-training."

---

## 3. The dead-end list

| Approach | Verdict | Evidence |
|---|---|---|
| Pure SSM/Mamba production models | Dead → absorbed into hybrids | Codestral Mamba retired June 2025; Falcon Mamba → hybrid Falcon-H1; RecurrentGemma no v2; NVIDIA keeps 8–25% attention because pure SSMs fail associative recall. Mamba-3 is a paper, not a product. Holdout: RWKV. |
| Linear attention in flagships | Walked back → re-routed to sparse | MiniMax "Why did M2 end up as a full attention model?" (Oct 2025): hybrids matched saturated benchmarks but had multi-hop reasoning deficits at scale; numerically precision-sensitive. 2026 answer = sparse attention (M3, DeepSeek DSA/CSA), not linear. |
| BitNet / 1.58-bit | Stalled | Nothing bigger than 2B4T; follow-up was BitNet Distillation (converting FP models). CPU/edge niche survives. No GPU serving path. |
| RETRO-style retrieval pretraining | Abandoned; motivation reincarnated as in-model memory | Nothing after InstructRetro. Heirs: Meta Memory Layers (unproductized), **DeepSeek Engram** (hashed n-gram conditional memory, +3–5 pts at 27–40B, DRAM tables at <3% throughput cost) — the most important unproductized idea of 2026. |
| From-scratch dense >32B (open) | Abandoned for MoE | Llama 4 all-MoE; Mistral "Small" = 119B-A6B. Dense survives below ~32B (single-GPU moat). |
| Hyena/StripedHyena for LLMs | Abandoned for text | Survives in genomics (Evo 2). |
| Griffin/RecurrentGemma | Abandoned | No successor since Aug 2024. |
| Byte-level / tokenizer-free (BLT, MambaByte) | Research-only | Meta still publishes (BLT-D, May 2026); zero production adoption. |
| Separate draft models (spec decode) | Displaced | EAGLE-3 heads + native MTP killed the category. |
| MatFormer/elastic inference | Absorbed quietly | Gemma 3n elasticity never fully shipped. |
| Diffusion LMs | NOT dead — niche | Mercury 2 commercial; LLaDA2.0 100B; DiffusionGemma open. Speed niche, no flagship displacement. |
| HRM/TRM recursive tiny models | Niche research | 7M-param TRM's 45% ARC-AGI; no production. |
| Note: Granite 4.1 | Hybrid → back to dense | IBM: dense 8B matches hybrid 32B-A9B, easier to fine-tune. Hybrids winning, not unanimously. |

---

## 4. Attention-free / hybrid small models in production

Pure full attention in every layer is effectively dead below ~80B. Three camps:

1. **Linear/SSM hybrids** — Qwen's entire 2026 line (3:1 GDN:gated-attention, 0.8B→397B including the dense 27B), Nemotron 3 Nano (≈4:1 Mamba2 + MoE), Falcon-H1/H1R (parallel SSM+attention per block), LFM2/2.5 (3:1 conv:GQA), Jamba2 (7:1), Kimi Linear (3 KDA:1 MLA, NoPE), Zamba2-VL, Hunyuan-TurboS.
2. **Cheap-softmax transformers** — Gemma 3/4 (5:1 sliding-window-1024:global), GPT-OSS (1:1, 128-token SWA), DeepSeek sparse (DSA → CSA/HCA). Interleaved SWA is not attention-free — it's the transformer camp's answer to the same KV-cache problem.
3. **Defectors back to dense attention** — MiniMax M2/M2.5 (full GQA all 62 layers), IBM Granite 4.1.

**Convergent ratios:** quality-first hybrids cluster at **3:1 linear:full** (Qwen, Kimi's ablated sweet spot, LFM2); throughput-first at 7:1–9:1; SWA camp at 5:1. RULER-style recall approaches transformer parity around 3:1–6:1 (HALO, arXiv 2601.22156).

**Quality:** at ≤32B, top-lab hybrids match or beat same-size transformers (Kimi Linear beat its full-attention twin in a controlled 1.4T-token study; Qwen3.6-27B is AA #1 small). Counter-evidence: MiniMax's multi-hop warning, Granite 4.1, and pure attention-free still pays a tax — RWKV7-G 7.2B: MMLU 65.1 vs low-80s for same-size Qwen (impressive per training-FLOP, not leaderboard-competitive). New 2026 failure mode: CoT fine-tuning can break long-range recall in hybrids ("Attention Amnesia," arXiv 2606.11052).

**RWKV:** RWKV-7 G-series at 7.2B/13.3B (13.3B: MMLU 76.5, GSM8K 92.3); RWKV-8 unreleased. Its most influential artifact is RADLADS (§6).

---

## 5. What the 12GB A3000 + 62GB RAM laptop can run (July 2026)

Calibration: A3000 mobile ≈ desktop 3060 12GB × 0.85–0.95 (thermals). Laptop RAM bandwidth caps offloaded MoE: DDR4-3200 ≈ 51 GB/s, DDR5-5600 ≈ 85 GB/s.

**Stack:** llama.cpp/GGUF won this class — MoE CPU-offload (`--n-cpu-moe`/`-ot "exps=CPU"`), quantized KV, MTP merged ~April 2026 (≈2× decode on Qwen3.6), Qwen3.5 DeltaNet-hybrid support in current builds. ik_llama.cpp for hybrid CPU+GPU squeezing. ExLlamaV3: skip on Ampere. Ollama leaves 30–50% on the table vs raw llama.cpp.

Ranking:
1. **Qwen3.6-35B-A3B Q4_K_M + `--n-cpu-moe ~32` + MTP** — ~7GB VRAM, ~22GB RAM, 64K ctx (q4 KV); 33–36 tok/s on 3060+DDR4 desktop; expect ~15–30 tok/s on the laptop. The daily driver.
2. **GPT-OSS-20B (MXFP4)** — ~75 tok/s @2K, ~56 @32K [3060]. Fastest capable; best tool-calling per token.
3. **GLM-4.7-Flash 30B-A3B Q4** — best local coder in class; ~15–30 tok/s.
4. **VRAM-resident dense: Qwen3.5-9B Q6_K (~9GB, consensus small-dense king) and Gemma 4 12B QAT (~7GB, multimodal)** — 25–55 tok/s.
5. **Big-MoE-on-DRAM (occasional):** GPT-OSS-120B ran 18–22 tok/s on a 3080Ti-12GB+DDR4 desktop, but ~60GB working set is marginal on 62GB — keep under ~50GB via smaller quant. Qwen3.5-122B-A10B: 17.5 tok/s reported on 64GB+12GB desktop. GLM-4.5-Air only at Q3, 4–8 tok/s. Long-prompt TTFT is the real pain.
6. **Skip:** DeepSeek V4-Flash (146GB), Qwen3-Next-80B (DeltaNet CPU path ~5× slow), vLLM.

KV: q8_0 near-lossless; 14B Q4 → ~16–24K ctx; 8–9B → 32–64K. Fine-tuning ceiling: QLoRA ~14B dense (Unsloth: 14B → ~9–10GB).

Engram-style DRAM offload: the paper demonstrated a 100B-param memory table in host DRAM at <3% throughput penalty — validating exactly this laptop shape — but it's a research demo; nothing to run today.

---

## 6. Distillation practicality for a solo researcher (custom non-transformer student)

**Logit access:** realistic teachers for logit-level work are 7B–32B open weights on rented GPUs via vLLM/SGLang (full logprobs). Offline top-k logit datasets are established (Arcee DistillKit, re-released Dec 2025; open Llama-405B logits dataset). Top-100 fp16 logits ≈ 400 bytes/token → 1B tokens ≈ 0.4TB.

**Traces come free:** OpenThoughts3-1.2M (Apache-2.0; SFT-only gave +15 AIME over R1-Distill-7B), NVIDIA Nemotron-Post-Training v1/v2 (CC-BY), OpenR1-Math-220k, Dolphin-R1. Nothing OpenThoughts-scale yet from GLM-5.x/V4 teachers.

**Cross-architecture distillation results:**

| Work | Direction | Tokens | Result |
|---|---|---|---|
| MOHAWK/Phi-Mamba (2024) | Phi-1.5 → Mamba2 | 3B | ~96.5% of teacher avg |
| MambaInLlama (2024) | Llama-3-8B → 25%-attention hybrid | ~20B | ≈ teacher on chat |
| LoLCATs (2024) | Llama 8B/70B/405B → linear | 40M | ~80% of gap closed; 5-shot & long-ctx lag |
| RADLADS/QRWKV (2025) | Qwen2.5 7B/32B/72B → RWKV-variant | 350–700M | 93–95% MMLU retention; 72B for <$2,500 |
| Llamba / M1 (2025) | Llama → Mamba (incl. reasoning distill + RL) | 12B | near-teacher; reasoning survives |
| Jet-Nemotron (2025) | PostNAS retrofit | 400B | industrial version |
| HALO / xLSTM distills / Priming (2026) | Qwen3/Llama/OLMo → RNN/xLSTM/GDN hybrids | 2.3–20B | parity-or-better; Priming's hybrid-32B beats Qwen3-32B +3.8 reasoning pts at 2.3× decode |

Kimi Linear and Qwen3-Next/3.5 were trained from scratch, not distilled. Consistent failure modes: math/multi-step reasoning drops first; long-context recall degrades badly in fully-linear students (every successful recipe keeps ~12–25% exact attention or adds a positional fix); some RNN variants unstable at 32B+.

**The realistic solo path (~$500–5,000 rented compute):**
- *Phase A — capability via traces (~$100–500):* SFT the custom architecture on OpenThoughts3 + Nemotron-v2 slice. Architecture-agnostic. ≤1B students feasible on the laptop itself; rented H100-days cover 1–3B.
- *Phase B — distribution via a mid-size teacher (~$300–3,000):* 7B–32B Apache/MIT teacher (Qwen3.x-9B/27B class); staged recipe: (1) per-layer hidden-state/mixer alignment (~100–700M tokens), (2) top-k logit KD (0.25–5B tokens, offline or on-policy), (3) short context extension. Expect 93–97% retention on knowledge benchmarks.
- *Catch for a genuinely custom substrate:* the cheapest recipes (RADLADS) depend on **weight transfer** — the student mixer must be QKV-shaped enough to inherit attention projections. A from-scratch substrate forfeits that; budget 3–20B tokens (~$2k–10k), keep a fraction of exact-attention or local-conv layers for recall, and **adopt the teacher's tokenizer**.
- *Licenses:* DeepSeek (MIT), Qwen (Apache-2.0), GLM (MIT) permit distillation freely; avoid Llama as teacher (naming clause).
- *Unproven:* nobody has published converting a huge MoE flagship (V4/GLM-5/K2) into a small non-transformer — all cross-arch results use ≤72B dense teachers.

---

## Uncertainty flags

(1) All tok/s figures are desktop proxies (−5–20% for A3000 mobile; DDR4 vs DDR5 swings offloaded MoE ~1.6×). (2) Vendor benchmarks aggressively inflated in 2026 (EXAONE 4.5, Falcon-H1R examples); trust AA v4.1 / LMArena. (3) Secondary-source-only: Mistral Small 4 exact split, Gemma 4 MatFormer internals, V4 OPD details, Muse Spark. (4) Rumored/unreleased: Phi-5, DeepSeek R2 (rumored 32B dense — would be the most interesting release for this hardware class), GLM-5-Air, SmolLM4, Qwen3.7 weights, RWKV-8.

**Bottom line for a solo researcher on this laptop:** the industry's converged bets — tiny-active MoE with experts in DRAM, ~3:1 linear:attention hybrids, hashed-memory sparsity (Engram), cheap cross-architecture distillation — all favor a 12GB-GPU + 62GB-RAM machine. Run Qwen3.6-35B-A3B / GPT-OSS-20B / GLM-4.7-Flash locally; to train a custom non-transformer, the proven path is trace-SFT from open reasoning corpora plus MOHAWK/RADLADS-style staged distillation from a 7–27B Apache/MIT teacher — with long-context recall as the one capability every 2026 result says you must architect for, not distill for.
