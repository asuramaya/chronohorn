# State of Attention-Free / Sub-Quadratic Sequence Modeling (2024–mid-2026)
## A comparative research report against the decepticons / chronohorn kernel

*Compiled 2026-07-02 by a web-research agent. Reliability note: frontier-2024/2025 items are anchored to arXiv/conference sources; several mid-2026 items (Gated DeltaNet-2, Mamba-3, DeepSeek Engram, the Parameter Golf leaderboard) are sourced from vendor pages, secondary blogs, and leaderboard scrapes and should be treated as directionally correct but not peer-reviewed. Where a number is soft it says so.*

---

## 1. Attention-free / sub-quadratic SOTA, 2024–2026

The field moved through three phases in this window: (a) 2024 proved pure sub-quadratic models can match transformers at 7B (Falcon Mamba); (b) 2024–2025 the *linear-attention-as-fast-weight* view (DeltaNet lineage) took over from the pure-SSM view; (c) 2025–2026 the consensus settled on **hybrids**, not pure attention-free models, as the production default.

**The core lineages:**

- **Mamba-2 / SSD (2024).** "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality," Dao & Gu, arXiv 2405.21060 (May 2024). Reframed selective SSMs and linear attention as the same object (structured state-space duality), which is the theoretical backbone everything after it builds on.

- **Griffin / Hawk (Feb 2024).** "Griffin: Mixing Gated Linear Recurrences with Local Attention," DeepMind, arXiv 2402.19427. Introduced the **RG-LRU** (real-gated linear recurrent unit). Hawk (pure recurrent) beat Mamba on downstream tasks with half the tokens; Griffin (recurrent + local attention, ratio ~3 recurrent : 1 local-MQA) slightly beat Llama-2 with ~6× fewer tokens. This is the template modern hybrids still use.

- **xLSTM (2024) → xLSTM-7B (Mar 2025) → scaling laws (Oct 2025).** NXAI. Original arXiv 2405.04517; xLSTM-7B arXiv 2503.13427; "xLSTM Scaling Laws," arXiv 2510.02228. sLSTM/mLSTM blocks with matrix memory and exponential gating; positioned as the strongest non-transformer 7B on next-token/MMLU with linear-time inference and lower time-to-first-token than transformers of equal size.

- **RWKV-7 "Goose" (Mar 2025).** arXiv 2503.14456. Pure RNN, constant memory, no KV cache. Key advance is a **generalized delta rule with vector-valued (channel-wise) gating and in-context learning rates**, plus a relaxed value-replacement rule — this gives it provable **state tracking** and recognition of all regular languages while staying parallelizable (linear attention cannot do this; it's a real expressivity result). Four models 0.19B–2.9B on a 3.1T-token multilingual corpus; new 3B SoTA on multilingual, parity on English at dramatically lower FLOPs. Apache-2.0.

- **Gated DeltaNet (ICLR 2025) → Gated DeltaNet-2 (May 2026).** "Gated Delta Networks: Improving Mamba2 with Delta Rule," Yang/Kautz et al., arXiv 2412.06464. DeltaNet fast-weight memory update + Mamba-style gating; beats both DeltaNet and Mamba-2. NVIDIA's **Gated DeltaNet-2** (MarkTechPost, May 2026) decouples "erase" and "write" in the delta rule and reportedly tops Mamba-3, KDA, and Gated DeltaNet on their suite (hybrid avg 53.97 vs Mamba-3-MIMO 52.72 — vendor numbers, unverified).

- **Titans / test-time-training (Jan 2025).** "Titans: Learning to Memorize at Test Time," Behrouz et al. (Google), arXiv 2501.00663. Long-term memory is a small neural net whose **weights are updated at inference by gradient descent on a "surprise" signal**. Three variants: Memory-as-Context (MAC), Memory-as-Gate (MAG), Memory-as-Layer (MAL). Near-perfect retrieval claimed to 2M tokens. Follow-up **MIRAS** generalizes the memory objective (Google Research blog, "Titans + MIRAS"). This is the academic anchor for the TTT layers that later dominate Parameter Golf (§5).

- **Hyena → StripedHyena 2 (Mar 2025).** "Systems and Algorithms for Convolutional Multi-Hybrid Language Models at Scale," arXiv 2503.01868. Three input-dependent convolution scales (short/medium/long Hyena operators) + attention; 1.2–2.9× faster end-to-end than optimized transformers at 40B. Powers **Evo 2** (genomics FM, 9.3T DNA bp, Nature s41586-026-10176-5).

- **Oscillatory SSMs — LinOSS (ICLR 2025 oral).** "Oscillatory State-Space Models," arXiv 2410.03943, + "Learning to Dissipate Energy in Oscillatory State-Space Models," arXiv 2505.12171. Forced harmonic oscillators, stable with only a **non-negative diagonal** state matrix, parallel scan. Beats Mamba/LRU ~2× on 50k-length sequence tasks. *Directly relevant to decepticons' oscillatory substrate mode.*

**2025–2026 frontier hybrids (the actual production story):**

- **Qwen3-Next (2025):** 3:1 layout — most layers Gated DeltaNet, some layers gated full attention.
- **Kimi Linear / Kimi Delta Attention (KDA, 2025–2026):** refines Gated DeltaNet with **channel-wise** (per-feature) gating instead of Qwen's scalar per-head gate.
- **Mamba-3 (ICLR 2026, OpenReview HwCvaJOiCj):** adds RoPE and further refinements; being slotted into hybrids in place of Gated DeltaNet.
- **Falcon Mamba-7B:** cited as the first 7B attention-free model to beat same-size transformers — the existence proof.
- Supporting surveys: "A Systematic Analysis of Hybrid Linear Attention," arXiv 2507.06457; "Log-Linear Attention," arXiv 2506.04761; "The End of Transformers? On Challenging Attention and the Rise of Sub-Quadratic Architectures," arXiv 2510.05364; "Comparing Transformers and Hybrid Models at the Token Level," arXiv 2606.20936.

**Where the 2026 consensus sits.** Honest summary of what the sources converge on:

1. **Frontier general-purpose leaderboards are still full-attention.** No top-10 LMSys-class model is known to be pure sub-quadratic. Pure attention-free has *not* displaced attention at the top.
2. **Hybrids won the pragmatic argument.** The debate is no longer "attention vs no attention" but "what ratio" (typically 3:1 to ~6:1 linear:attention). Blog title of the era: "Qwen3.5: Nobody Agrees on Attention Anymore."
3. **At small scale / edge / long-context / non-text modalities, attention-free is genuinely competitive to superior** (RWKV-7, xLSTM-7B, Falcon Mamba, Evo 2). This is exactly the regime a byte-level parameter-golf kernel lives in.
4. **Expressivity is the live research frontier:** pure linear attention provably cannot do state tracking; DeltaNet/RWKV-7-style delta rules and Titans-style test-time memory are the responses. The whole field is converging on *"cheap recurrence + a small amount of content-dependent, gradient-at-test-time or delta-rule memory."*

---

## 2. Frozen / fixed substrate + learned readout (reservoir computing)

Short answer: **yes, this is an active and reviving area — but almost entirely in time-series/forecasting/physical-systems, not language modeling. A frozen multi-timescale linear bank feeding a learned readout for a byte LM is close to, but not identical to, anything found published.**

**Reservoir computing is explicitly in revival:**
- **ESNv2 — "Resurrecting Reservoir Computing in the Deep Learning Era"** (OpenReview N6G2Mmz8qs). The title states the thesis.
- **FreezeTST — "Frozen in Time: Parameter-Efficient Time Series Transformers via Reservoir-Induced Feature Expansion and Fixed Random Dynamics,"** arXiv 2508.18130 (Oct 2025). Draws encoder blocks at random, **freezes them for the model's lifetime**, and lets surrounding learned layers query the fixed nonlinear state — a transformer that inherits reservoir long-memory bias without paying the recurrent optimization cost. Conceptually the nearest neighbor to "frozen substrate + learned readout" in a modern deep net.
- **ParalESN (Oct 2025):** parallel echo state network with **diagonal linear recurrence in complex space**, parallelizable in training, comparable accuracy to classical RC. *Strikingly close* to decepticons' `build_linear_bank()` (deterministic diagonal linear recurrences with fixed decay rates), just aimed at forecasting instead of bytes.
- Survey: Grezes, "Reservoir Computing: A New Paradigm for Neural Networks," arXiv 2504.02639. Frontiers editorial "Deep neural network architectures and reservoir computing" (2025, PMC12408670). Deep-ESN with reservoir interactions (2026).

**The "fixed dynamics" SSM lineage is the bridge to language:**
- **S4D / DSS:** "On the Parameterization and Initialization of Diagonal State Space Models," arXiv 2206.11893. The diagonal state matrix is **initialized from the HiPPO-LegS spectrum**; DSS showed that fixing to fully diagonal with HiPPO-N init preserves S4's dynamics. So the *initial* dynamics are essentially prescribed/fixed; standard S4D still learns `A` by gradient, but the community knows the init carries most of the value.
- **S5** (arXiv 2208.04933) and the **LRU** (Linear Recurrent Unit, Orvieto et al. 2023) sit on the same "structured, near-fixed diagonal recurrence" spectrum.

**How close is the field to "frozen linear bank + learned readout" for LM?** The gap is specific:
- The *pieces* are all published: diagonal linear recurrence with fixed multi-timescale decays (S4D/DSS/LinOSS/ParalESN), frozen-random substrates (FreezeTST, ESNv2, lottery-ticket/frozen-random-net literature), learned readouts (classical RC).
- But in practice **modern "fixed-init" SSMs still train the dynamics by gradient**, and **reservoir computing with a truly frozen substrate is almost never applied to language**. decepticons' `frozen` mode (nothing learns in the substrate; pure reservoir with fixed diagonal dynamics) applied to **byte-level LM** is on the sparse edge of the literature. Its `learnable_decays` / `learnable_mixing` / `learned_recurrence` ladder is exactly the RC→S4D→S5→Mamba spectrum made explicit and selectable — a nice framing but not itself novel.

---

## 3. Byte-level language models & where 1.78 bpb sits

**The canonical lineage and their bits-per-byte numbers:**

| Model | Paper / date | Key idea | Reported bpb (corpus) |
|---|---|---|---|
| Byte-level Transformer (baseline) | — | raw bytes | ~1.14 (PG-19) |
| **MegaByte** | arXiv 2305.07185 (2023) | multiscale patches (global + local) | ~1.00 (PG-19); competitive on enwik8 |
| **MambaByte** | arXiv 2401.13660, COLM 2024 | token-free selective SSM | **0.93 (PG-19, 353M, 8k ctx)** — beats MegaByte-758M (1.00) at equal compute |
| **SpaceByte** | arXiv 2404.14408, NeurIPS 2024 | dynamic patch boundaries at "space-like" bytes | **1.009 (PG-19)**; matches subword transformers; beats MegaByte trained on 2.7× more compute |
| **Byte Latent Transformer (BLT)** | arXiv 2412.09871, Meta (Dec 2024), ACL 2025 | **entropy-based dynamic patching**, learned byte→patch | matches Llama-3 up to 8B / 4T bytes; better FLOP-controlled scaling than tokenized models |
| **Fast BLT** | arXiv 2605.08044 (2026) | throughput-optimized BLT | efficiency-focused successor |

For reference on **enwik8** (bpc ≈ bpb for ASCII): strong *small* char-level models land near 0.94–1.06 bpc — e.g., "Focus" 22M at 0.94, Transformer-XL 41M at 1.06, GPT-2 1.5B at 0.94. These are heavily-trained char models, not tiny-budget from-scratch runs.

**Is 1.78 bpb competitive, and at what parameter count?** Honest assessment:

- **On any standard heavily-trained byte corpus (PG-19, enwik8), 1.78 bpb is not competitive** — ~0.8 bpb worse than MambaByte/SpaceByte (~0.93–1.0) and worse than a plain byte-transformer baseline (~1.14).
- **On FineWeb under the Parameter Golf regime (see §5), 1.78 is worse than the *naive baseline* of 1.2244** and far behind the ~1.11 SOTA. As an absolute number at the 16 MB / 10-minute budget it is not in the running.
- **The most charitable reading:** 1.78 bpb is plausible for a *very* small parameter count, very short training, or an unfinished run. At <1M params trained briefly on bytes, ~1.7–1.9 bpb is an ordinary "it's learning but under-trained" number. The value of the codebase is architectural exploration, not this figure. **The 1.78 number is likely an artifact of budget, not a statement about the architecture's ceiling.**

Bottom line: **1.78 bpb would need to reach ~1.15 or below to be competitive at the parameter-golf budget, or ~0.95–1.0 to be competitive with published small byte LMs.** That's a large gap, consistent with an incomplete run.

---

## 4. N-gram / hash-based augmentation

One of the **hottest and most directly relevant** areas to the codebase — it went mainstream in exactly the 2024–2026 window.

- **infini-gram / ∞-gram** — "Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens," arXiv 2401.17377 (2024, updated Apr 2025). Suffix-array backend, arbitrarily large `n` with backoff, 5T-token n-gram LM (largest ever). **Interpolating ∞-gram with a neural LM cuts perplexity by up to 73%, even against a 70B model**, and beats kNN-LM/RIC-LM as a retrieval strategy. Related: "The Role of n-gram Smoothing in the Age of Neural Networks," arXiv 2403.17240.

- **kNN-LM lineage** — Khandelwal et al. 2020 (kNN-LM); **Memorizing Transformers**, ICLR 2022, arXiv 2203.08913 (non-differentiable kNN memory of past KV pairs, learned gate blending local vs retrieved, scales to 262K memory); unbounded/continuous cache (Grave et al. 2017). Survey: "Memory-Augmented Transformers: A Systematic Review," arXiv 2508.10824.

- **DeepSeek Engram (Jan 12 2026)** — github.com/deepseek-ai/Engram, "Conditional Memory via Scalable Lookup: A New Axis of Sparsity." The big one for polyhash: it **modernizes classic hashed n-gram embeddings into a lookup-table module inserted in the transformer's middle layers** — compress token IDs, **hash suffix n-grams, multi-head hashing (K hash functions → K table rows, aggregated), O(1) retrieval, table lives in DRAM/CXL off-GPU.** Reported as a new sparsity axis decoupling static-knowledge memory from reasoning compute. (Sources: DeepSeek blog, Tom's Hardware, Introl, Shreyansh Singh paper summary — vendor/secondary, but consistent.)

- **nanoGPT speedrun "Bigram Hash Embedding" (record 62)** — explicitly described as combining a 2017 hash-embeddings idea with DeepSeek's Engram. Independent confirmation the pattern is real and useful at tiny scale (see §5).

- **Parameter Golf bigram-hashing entries** (leaderboard #7 thwu1, #8 Raahil Shah) pair bigram hashing with quantization/gating.

**Is anyone combining n-gram tables with neural scans specifically?** Yes — that is now a recognized pattern (Engram = hash n-gram lookup fused into a neural net; infini-gram = interpolate n-gram with neural LM; Parameter Golf's "Order-16 Frozen N-gram Oracle + Learned Gate + TTT" fuses a frozen n-gram table with a learned gate on top of a neural model). **decepticons/polyhash (O(1) hash n-gram + gated scan) and the online-n-gram memory (runtime accumulator injected at inference, i.e. a dynamic cache) are squarely inside this active cluster.** They look like **parallel/independent reinvention** of Engram + unbounded-cache + n-gram-oracle-with-gate, not something ahead of it.

---

## 5. Parameter-efficiency competitions

**OpenAI "Parameter Golf" (the direct match).** github.com/openai/parameter-golf; OpenAI writeup "What Parameter Golf taught us"; ran **March 18 – April 30, 2026**.
- **Rules:** entire self-contained artifact (weights + training + inference code) ≤ **16,000,000 bytes**; train ≤ **10 minutes on 8×H100**; judged by **tokenizer-agnostic bits-per-byte on the FineWeb validation set** (first 50k docs). New SOTA must beat prior by ≥0.005 nats at p<0.01.
- **Numbers:** naive baseline **1.2244 bpb** (2026-03-18); best confirmed **1.1147 bpb** (abaybektursun, "self-generated GPTQ + cross-sparse attention"); a pending/unconfirmed **0.8265** was reported. (Note: some scraped leaderboard rows like "0.0109 bpb" are almost certainly *improvement deltas in nats*, not absolute bpb.)
- **Winning architectural meta (the important part):** the same handful of ideas recur at the top —
  - **Quantization:** GPTQ / self-generated GPTQ, ternary weights, int5/int6 mixed precision, QAT, LZMA-compressed weights.
  - **Test-time training (TTT):** "score-first TTT," "long-context no-Q/V TTT," n-gram-oracle-+-learned-gate-+-TTT.
  - **N-gram / hashing:** bigram hashing, token-only n-gram + calibration ("Calib32 Token-Only N-gram + AsymLogit Stack" ~1.0565), packed causal n-gram + Dirichlet backoff.
  - **Architecture tricks:** cross-sparse attention (often only in the deepest few layers), partial/rotary RoPE, depth recurrence, parallel residuals, LeakyReLU²/squared activations.
  - **Optimizer:** Muon.

**nanoGPT speedrun (modded-nanoGPT, Keller Jordan).** Ongoing FineWeb val-loss speedrun; GPT-2(124M)-target now ~sub-90s (~30× over baseline). Birthplace of the **Muon optimizer** (Newton–Schulz orthogonalization, later adopted by Kimi K2, GLM-4.5). Later records *invent* architecture: **record 58 "Paired Head Attention,"** **record 62 "Bigram Hash Embedding."** METR's April 2026 analysis ("Evidence on AI R&D Progress from NanoGPT") and the "Automated LLM Speedrunning Benchmark" (arXiv 2506.22419) study reproducibility. Takeaway: at tiny scale, **imported ideas drove ~6.7× of the 31× cumulative speedup; invented ideas ~1.6×** — architecture novelty helps less than optimizer/data/quantization tricks.

**BabyLM Challenge (data-efficiency, not param-efficiency).** "Findings of the BabyLM Challenge," arXiv 2504.08165 (2025). Winners used **LTG-BERT** (a well-optimized *encoder transformer*), beating models trained on trillions of words on a fixed ~100M-word budget. Notably: **curriculum learning largely failed**, and exotic architectures did *not* win — clean optimization of a standard transformer did. Cautionary data point for "novel architecture wins the small-budget game."

**Net pattern of winning entries:** they are **not exotic architectures**. They are standard/hybrid transformers + aggressive **quantization** + **test-time training** + **n-gram/hash augmentation** + **Muon**, tuned hard. Architecture novelty is a minor term; quantization + TTT + data/optimizer are the major terms.

---

## 6. The gap: what in the codebase is done, reinvented, or genuinely open

### (a) Already done and published
- **Frozen diagonal linear bank with fixed multi-timescale decays** ≈ the reservoir-computing limit of S4D/DSS + ParalESN (diagonal complex-space linear recurrence) + FreezeTST (fixed random dynamics). Published, 2022–2025.
- **Learnable-decays / learnable-mixing / learned-recurrence ladder** = the RC → S4D → S5 → Mamba parameterization spectrum. Well-trodden; the codebase's contribution here is *packaging/selectability*, not the mechanism.
- **Local causal convolution** = short conv in H3 / Hyena / Mamba. Standard.
- **Selective-scan augment (content-dependent state on top of a frozen bank)** = Mamba/Mamba-2 selective SSM. Combining a fixed component with a selective one is conceptually a mini-hybrid (cf. SSD, Griffin's fixed-vs-gated split).
- **Banded readouts split by timescale** = multi-head / multi-timescale SSM (S5 heads; multi-resolution Hyena; the RWKV/xLSTM per-channel decay spectrum). Published.
- **Routed 4-expert squared-ReLU readout** = Primer's squared-ReLU (arXiv 2109.08668) + sparse MoE FFN (Shazeer 2017 →). Both published; the *combination as a readout head* is unusual but not novel in components.
- **Byte-level LM** = MegaByte / MambaByte / SpaceByte / BLT.
- **polyhash (O(1) hash n-gram + gated scan)** = DeepSeek Engram (hashed n-gram lookup tables, multi-head hashing, O(1)) + nanoGPT "Bigram Hash Embedding" + Parameter Golf bigram-hashing entries.
- **Online n-gram memory injected at inference** = infini-gram interpolation + unbounded/continuous cache + Memorizing-Transformers-style non-differentiable memory + Parameter Golf's "frozen n-gram oracle + learned gate."

### (b) Independently reinvented in parallel with the literature
- **The whole "cheap frozen/linear temporal mixer + n-gram/hash memory + learned gate + small efficient readout" recipe is exactly the meta that Parameter Golf converged on in March–May 2026** — concurrently with, and independently of, this codebase. polyhash ≈ their bigram hashing; the online n-gram accumulator ≈ their "n-gram oracle + learned gate + TTT." Genuine parallel invention — and mild bad news for novelty: the competition's public leaderboard already contains close analogues, combined with quantization + TTT that the codebase does not yet exploit.
- **Frozen multi-timescale substrate for sequence modeling** ≈ the reservoir-computing revival (ESNv2, FreezeTST, ParalESN), reinvented in parallel but for bytes/language rather than forecasting.
- **Routed squared-ReLU experts as readout** ≈ MoE-FFN + Primer, reassembled independently.

### (c) Genuinely underexplored in 2026 (the defensible novelty)
The individual bricks are all published; what nobody appears to be doing is the **specific full-stack integration**:

1. **A *truly frozen* (gradient-free) multi-timescale diagonal bank as the *primary* temporal mixer of a competitive byte-level LM.** Modern "fixed-init" SSMs still learn `A` by gradient; reservoir computing stays in time-series. Frozen-substrate-for-language is sparse. decepticons' `frozen` mode for bytes is on the thin edge.
2. **Nonlinear *routed-expert* readouts over a frozen reservoir.** Classical RC readouts are linear/ridge-regression; even the RC revival keeps readouts simple. A 4-expert squared-ReLU MoE readout over a fixed multi-timescale bank is a *readout-richness* direction the reservoir literature has basically not explored — "how much can a rich nonlinear readout recover from a deliberately fixed substrate?" is open.
3. **Oscillatory frozen banks for language.** LinOSS shows oscillatory diagonal SSMs are strong on long-range sequences, but nobody has used an oscillatory substrate *frozen* as a reservoir for byte LM. decepticons has oscillatory substrate modes — this specific combination is unclaimed.
4. **The triple combination — frozen multi-timescale substrate + routed-expert readout + online n-gram memory — as one system.** Each pair exists somewhere (frozen substrate + readout = RC; n-gram + neural = Engram/infini-gram; MoE readout = MoE), but the three together, for byte LM, has no published instance found. That is the codebase's most distinctive and least-covered footprint.

**Strategic honesty:**
- The novelty that survives scrutiny is **integration and the frozen-substrate-for-language angle**, not any single mechanism. Every component has a 2022–2026 publication.
- The empirical bar is unforgiving: at the Parameter Golf budget, SOTA is ~1.11 bpb using quantization + TTT + hashing; 1.78 is behind the naive 1.2244 baseline. **The fastest credible path to competitiveness is to adopt the leaderboard meta the codebase is currently missing — aggressive quantization (GPTQ/ternary/int5-6), test-time training on the online n-gram memory, and Muon — rather than to push the frozen-substrate architecture alone.** At tiny budgets, quantization + TTT + optimizer dominate architecture novelty (nanoGPT speedrun attribution; BabyLM's "boring transformer wins").

---

### Sources

Attention-free SOTA: [Mamba-2/SSD 2405.21060](https://arxiv.org/abs/2405.21060) · [Griffin/Hawk 2402.19427](https://arxiv.org/abs/2402.19427) · [xLSTM scaling 2510.02228](https://arxiv.org/html/2510.02228v1) · [xLSTM-7B (NXAI)](https://www.nx-ai.com/en/news/xlstm-7b-nxai-releases-its-new-xlstm-7b-model) · [RWKV-7 Goose 2503.14456](https://arxiv.org/abs/2503.14456) · [Gated DeltaNet 2412.06464 (ICLR25 PDF)](https://jankautz.com/publications/GatedDeltaNet_ICLR25.pdf) · [Gated DeltaNet-2 (MarkTechPost)](https://www.marktechpost.com/2026/05/24/nvidia-ai-releases-gated-deltanet-2-a-linear-attention-layer-that-decouples-erase-and-write-in-the-delta-rule/) · [Titans 2501.00663](https://arxiv.org/abs/2501.00663) · [Titans+MIRAS (Google)](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/) · [StripedHyena 2 / Evo 2 2503.01868](https://arxiv.org/abs/2503.01868) · [LinOSS 2410.03943](https://arxiv.org/abs/2410.03943) · [Mamba-3 (ICLR 2026)](https://openreview.net/pdf?id=HwCvaJOiCj) · [Qwen3.5 / hybrid attention (HF blog)](https://huggingface.co/blog/mlabonne/qwen35) · [Hybrid linear attention 2507.06457](https://arxiv.org/pdf/2507.06457) · [End of Transformers? 2510.05364](https://arxiv.org/html/2510.05364v1)

Frozen substrate / reservoir: [FreezeTST 2508.18130](https://arxiv.org/html/2508.18130v2) · [ESNv2 (OpenReview)](https://openreview.net/forum?id=N6G2Mmz8qs) · [RC survey 2504.02639](https://arxiv.org/pdf/2504.02639) · [S4D/DSS init 2206.11893](https://arxiv.org/abs/2206.11893) · [S5 2208.04933](https://arxiv.org/pdf/2208.04933) · [Frontiers RC editorial](https://pmc.ncbi.nlm.nih.gov/articles/PMC12408670/)

Byte-level LMs: [MegaByte 2305.07185](https://arxiv.org/abs/2305.07185) · [MambaByte 2401.13660](https://arxiv.org/html/2401.13660v1) · [SpaceByte 2404.14408 (NeurIPS24 PDF)](https://proceedings.neurips.cc/paper_files/paper/2024/file/e1f418450107c4a0ddc16d008d131573-Paper-Conference.pdf) · [BLT 2412.09871](https://arxiv.org/abs/2412.09871) · [BLT ACL 2025](https://aclanthology.org/2025.acl-long.453/) · [Fast BLT 2605.08044](https://arxiv.org/pdf/2605.08044)

N-gram / hash: [infini-gram 2401.17377](https://arxiv.org/abs/2401.17377) · [infini-gram.io](https://infini-gram.io/) · [Memorizing Transformers 2203.08913](https://arxiv.org/pdf/2203.08913) · [DeepSeek Engram (GitHub)](https://github.com/deepseek-ai/Engram) · [Engram (Tom's Hardware)](https://www.tomshardware.com/tech-industry/artificial-intelligence/deepseek-touts-memory-breakthrough-engram) · [n-gram smoothing 2403.17240](https://arxiv.org/pdf/2403.17240)

Competitions: [OpenAI Parameter Golf (GitHub)](https://github.com/openai/parameter-golf) · [What Parameter Golf taught us (OpenAI)](https://openai.com/index/what-parameter-golf-taught-us/) · [Parameter Golf leaderboard (CodeSOTA)](https://www.codesota.com/parameter-golf) · [modded-nanoGPT (GitHub)](https://github.com/KellerJordan/modded-nanogpt) · [METR nanoGPT progress (Apr 2026)](https://metr.org/notes/2026-04-21-ai-rd-nanogpt-progress/) · [Automated LLM Speedrun 2506.22419](https://arxiv.org/pdf/2506.22419) · [BabyLM findings 2504.08165](https://arxiv.org/abs/2504.08165)

Readout components: [Primer / squared-ReLU 2109.08668](https://arxiv.org/pdf/2109.08668)
