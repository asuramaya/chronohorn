"""PolyHash v12: Content-Addressed Memory.

Hash tables with FIXED addressing (polynomial hash) for n-gram patterns.
Product Key Memory with LEARNED addressing for content-dependent routing.
Induced Set Attention for global sequence summary.
Optional Mamba-style selection scan.

The hash says "you are at THIS context."
The PKM says "I NEED this information."
The ISAB says "here's what the full sequence contains."
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

HASH_PRIMES = [
    2654435761, 2246822519, 3266489917, 2028178513,
    1220703125, 1610612741, 805306457, 402653189,
]
SCALE_WINDOWS = [1, 2, 4, 8, 16, 32, 64, 128]


@dataclass(frozen=True)
class V12Config:
    vocab_size: int = 1024
    byte_embed_dim: int = 64
    # Hash tables (fixed addressing — proven)
    num_hash_tables: int = 8
    hash_buckets: int = 65536
    hash_embed_dim: int = 16
    # Product Key Memory (learned addressing)
    pkm_enabled: bool = True
    pkm_sub_keys: int = 256        # size of each sub-codebook
    pkm_top_k: int = 16            # top-k per sub-codebook
    pkm_key_dim: int = 32          # dimension of each sub-key
    pkm_value_dim: int = 32        # dimension of values
    # Induced Set Attention Block
    isab_enabled: bool = True
    isab_num_points: int = 16      # number of inducing points
    isab_dim: int = 128            # attention dimension
    isab_heads: int = 4
    # PKM variant
    pkm_xsa: bool = False          # Exclusive self-attention: weights sum to 0 (correction mode)
    # TTT (test-time training): inner model updated by gradient descent during forward
    ttt_enabled: bool = False
    ttt_dim: int = 64              # inner model dimension
    ttt_lr: float = 0.01           # inner learning rate
    ttt_mini_batch: int = 16       # process this many tokens per inner step
    # Scan
    scan_dim: int = 256
    scan_chunk_size: int = 32
    scan_selection: bool = False   # Mamba-style input-dependent gates
    scan_rotation: bool = False    # Rotation scan: preserve info as phase, not decay
    scan_mamba: bool = False       # Use mamba-ssm fused CUDA kernel (requires mamba-ssm package)
    scan_mamba_dstate: int = 16    # SSM state dimension for mamba scan
    # Multi-scale scan: parallel scans at different time scales
    scan_multiscale: tuple = ()    # Extra scan dims at different scales, e.g. (64, 64) for 2 extra scans
    # Causal self-attention (hybrid SSM+attention)
    attn_layers: tuple = ()        # Insert causal attention after these MLP layer indices (e.g. (2, 5))
    attn_heads: int = 4
    # Quantization-aware training
    qat_bits: int = 0              # 0=disabled, 6=int6 QAT with STE gradients
    qat_hash_only: bool = False    # Only quantize hash table embeddings
    # MLP
    hidden_dim: int = 512
    num_layers: int = 2
    pre_norm: bool = False         # Pre-norm (LN before block) vs post-norm (LN after residual add)
    resid_mix: bool = False        # Learned per-dimension gate on MLP output (resid_mix_mlp)
    # Patch encoding
    patch_size: int = 1            # 1=byte-level (default), 4=patch4 (4 bytes per token)
    # Local features
    conv_kernel: int = 8
    match_offsets: tuple = (1, 2, 3, 4, 5, 6, 7, 8)
    # Training
    dropout: float = 0.0
    max_seq_len: int = 512
    init_seed: int = 42
    hash_dropout: float = 0.0


class SwiGLU(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.w1 = nn.Linear(d, h, bias=False)
        self.w2 = nn.Linear(d, h, bias=False)
        self.proj = nn.Linear(h, d, bias=False)
    def forward(self, x):
        return self.proj(F.silu(self.w1(x)) * self.w2(x))


class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.0, pre_norm=False, resid_mix=False):
        super().__init__()
        self.block = SwiGLU(dim, dim)
        self.ln = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else None
        self.pre_norm = pre_norm
        self.mix = nn.Parameter(torch.ones(dim)) if resid_mix else None

    def forward(self, x):
        if self.pre_norm:
            h = self.block(self.ln(x))
        else:
            h = self.block(x)
        if self.drop:
            h = self.drop(h)
        if self.mix is not None:
            h = h * self.mix
        if self.pre_norm:
            return h + x
        return self.ln(h + x)


# ── Patch Encoding ──────────────────────────────────────────

class PatchEmbed(nn.Module):
    """Embed a patch of P consecutive bytes into a single vector.

    Each byte is embedded independently, then the P embeddings are
    concatenated and projected to the model dimension. This avoids
    a 256^P vocab table while still encoding the full patch content.
    """

    def __init__(self, patch_size, byte_vocab, byte_dim, out_dim):
        super().__init__()
        self.patch_size = patch_size
        self.byte_embeds = nn.Embedding(byte_vocab, byte_dim)
        self.proj = nn.Linear(byte_dim * patch_size, out_dim)

    def forward(self, bytes_flat):
        """bytes_flat: [B, T*P] raw byte IDs → [B, T, out_dim]"""
        B, L = bytes_flat.shape
        P = self.patch_size
        T = L // P
        # Reshape to [B, T, P], embed each byte, concat, project
        patches = bytes_flat[:, :T * P].reshape(B, T, P)
        embs = self.byte_embeds(patches)           # [B, T, P, byte_dim]
        flat = embs.reshape(B, T, -1)               # [B, T, P * byte_dim]
        return self.proj(flat), T


class PatchOutput(nn.Module):
    """Predict the next patch's bytes autoregressively within the patch.

    Given hidden state at position t (which encodes patch t), predict
    all P bytes of patch t+1. Factored: predict byte 0, then byte 1
    conditioned on byte 0, etc. This is P sequential predictions per
    step, not one massive softmax over 256^P.
    """

    def __init__(self, hidden_dim, patch_size, byte_vocab=256):
        super().__init__()
        self.patch_size = patch_size
        self.byte_vocab = byte_vocab
        # Each byte position gets its own output head + conditioning on prior bytes
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim + i * byte_vocab, byte_vocab)
            for i in range(patch_size)
        ])

    def forward(self, h, target_patches=None):
        """h: [B, T, hidden_dim]. Returns logits [B, T, P, 256].

        During training, target_patches [B, T, P] provides teacher forcing.
        During inference, samples autoregressively within each patch.
        """
        B, T, D = h.shape
        P = self.patch_size
        all_logits = []

        context = h  # [B, T, hidden_dim]
        for i in range(P):
            logits_i = self.heads[i](context)  # [B, T, 256]
            all_logits.append(logits_i)
            if target_patches is not None and i < P - 1:
                # Teacher forcing: condition next byte on true previous byte
                byte_onehot = F.one_hot(target_patches[:, :, i], self.byte_vocab).float()
                context = torch.cat([context, byte_onehot], dim=-1)
            elif i < P - 1:
                # Inference: condition on predicted byte
                pred_byte = logits_i.argmax(dim=-1)
                byte_onehot = F.one_hot(pred_byte, self.byte_vocab).float()
                context = torch.cat([context, byte_onehot], dim=-1)

        return torch.stack(all_logits, dim=2)  # [B, T, P, 256]


# ── Hash Tables (fixed addressing) ──────────────────────────

class HashTables(nn.Module):
    def __init__(self, num_tables, buckets, embed_dim):
        super().__init__()
        self.num_tables = num_tables
        self.buckets = buckets
        self.embed_dim = embed_dim
        self.tables = nn.ModuleList([
            nn.Embedding(buckets, embed_dim) for _ in range(num_tables)
        ])
        for t in self.tables:
            nn.init.normal_(t.weight, std=0.02)

    def forward(self, tokens):
        B, T = tokens.shape
        # Compute all hash indices in parallel — one tensor op per window offset,
        # not one per (table, offset) pair.
        # Pre-compute all shifted token tensors needed across all tables.
        max_window = max(
            (SCALE_WINDOWS[i] if i < len(SCALE_WINDOWS) else i + 1)
            for i in range(self.num_tables)
        )
        # shifts[j] = tokens shifted right by j+1 positions (causal)
        shifts = []
        primed_shifts = []
        for j in range(max_window):
            offset = j + 1
            if offset >= T:
                break
            shifted = torch.zeros_like(tokens)
            shifted[:, offset:] = tokens[:, :-offset]
            shifts.append(shifted)
            primed_shifts.append(shifted * HASH_PRIMES[j % len(HASH_PRIMES)])

        # Compute hash per table using cumulative XOR of pre-computed primed shifts
        all_indices = []
        for i in range(self.num_tables):
            w = SCALE_WINDOWS[i] if i < len(SCALE_WINDOWS) else i + 1
            h = torch.zeros(B, T, dtype=torch.long, device=tokens.device)
            for j in range(min(w, len(primed_shifts))):
                h = h ^ primed_shifts[j]
            all_indices.append(h % self.buckets)

        # Batched embedding lookups — each table still has its own weights,
        # but we avoid re-entering Python between lookups.
        embs = [self.tables[i](all_indices[i]) for i in range(self.num_tables)]
        return torch.cat(embs, dim=-1)


# ── Product Key Memory (learned addressing) ──────────────────

class ProductKeyMemory(nn.Module):
    """Content-dependent memory lookup via factored key search.

    Query is split into two halves, each matched against its sub-codebook.
    Top-k candidates from each half are combined (k² pairs), scored,
    and the final top-k values are read and aggregated.
    """
    def __init__(self, input_dim, sub_keys, top_k, key_dim, value_dim, xsa=False):
        super().__init__()
        self.sub_keys = sub_keys
        self.top_k = top_k
        self.key_dim = key_dim
        self.xsa = xsa
        self.total_slots = sub_keys * sub_keys

        # Query projection: input → 2 × key_dim
        self.query_proj = nn.Linear(input_dim, 2 * key_dim)

        # Two sub-codebooks
        self.codebook_a = nn.Parameter(torch.randn(sub_keys, key_dim) * 0.02)
        self.codebook_b = nn.Parameter(torch.randn(sub_keys, key_dim) * 0.02)

        # Value table: total_slots × value_dim
        self.values = nn.Embedding(self.total_slots, value_dim)
        nn.init.normal_(self.values.weight, std=0.02)

        self.output_proj = nn.Linear(value_dim, input_dim)
        self.ln = nn.LayerNorm(input_dim)

    def forward(self, x):
        B, T, D = x.shape
        k = self.top_k

        # Project to query
        q = self.query_proj(x)  # [B, T, 2*key_dim]
        qa, qb = q.chunk(2, dim=-1)  # each [B, T, key_dim]

        # Score against each codebook
        scores_a = torch.matmul(qa, self.codebook_a.T)  # [B, T, sub_keys]
        scores_b = torch.matmul(qb, self.codebook_b.T)  # [B, T, sub_keys]

        # Top-k per codebook
        topk_a = scores_a.topk(k, dim=-1)  # values [B,T,k], indices [B,T,k]
        topk_b = scores_b.topk(k, dim=-1)

        # Combine: k² candidate indices
        # idx = a_idx * sub_keys + b_idx
        idx_a = topk_a.indices.unsqueeze(-1).expand(-1, -1, -1, k)  # [B,T,k,k]
        idx_b = topk_b.indices.unsqueeze(-2).expand(-1, -1, k, -1)  # [B,T,k,k]
        combined_idx = (idx_a * self.sub_keys + idx_b).reshape(B, T, k * k)  # [B,T,k²]

        # Combined scores
        score_a_exp = topk_a.values.unsqueeze(-1).expand(-1, -1, -1, k)
        score_b_exp = topk_b.values.unsqueeze(-2).expand(-1, -1, k, -1)
        combined_scores = (score_a_exp + score_b_exp).reshape(B, T, k * k)  # [B,T,k²]

        # Final top-k from k² candidates
        final_topk = combined_scores.topk(k, dim=-1)  # [B,T,k]
        final_idx = combined_idx.gather(-1, final_topk.indices)  # [B,T,k]

        # Retrieve values
        vals = self.values(final_idx)  # [B,T,k,value_dim]

        # Softmax-weighted aggregation (XSA: subtract uniform → correction mode)
        weights = F.softmax(final_topk.values, dim=-1)  # [B,T,k]
        if self.xsa:
            weights = weights - 1.0 / k  # sums to 0: output is a correction, not a prediction
        weights = weights.unsqueeze(-1)  # [B,T,k,1]
        output = (vals * weights).sum(dim=-2)  # [B,T,value_dim]

        return self.ln(x + self.output_proj(output))


# ── Induced Set Attention Block ──────────────────────────────

class ISAB(nn.Module):
    """Induced Set Attention Block — O(T×m) global context via inducing points.

    m learned inducing points cross-attend to all T positions (global summary).
    Then each position cross-attends to the m summaries (read global context).
    Total cost: O(T × m × d) where m << T.
    """
    def __init__(self, dim, num_points, heads):
        super().__init__()
        self.num_points = num_points
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        # Learnable inducing points
        self.inducing = nn.Parameter(torch.randn(num_points, dim) * 0.02)

        # Cross-attention: inducing ← input (down-project)
        self.q_down = nn.Linear(dim, dim)
        self.k_down = nn.Linear(dim, dim)
        self.v_down = nn.Linear(dim, dim)

        # Cross-attention: input ← inducing (up-project)
        self.q_up = nn.Linear(dim, dim)
        self.k_up = nn.Linear(dim, dim)
        self.v_up = nn.Linear(dim, dim)

        self.out_proj = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)

    def _multihead_attn(self, Q, K, V, mask=None):
        B, T_q, _ = Q.shape
        T_k = K.shape[1]
        h, d = self.heads, self.head_dim

        Q = Q.view(B, T_q, h, d).transpose(1, 2)  # [B,h,T_q,d]
        K = K.view(B, T_k, h, d).transpose(1, 2)
        V = V.view(B, T_k, h, d).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores + mask
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        return out.transpose(1, 2).reshape(B, T_q, h * d)

    def forward(self, x):
        B, T, D = x.shape
        m = self.num_points

        # Expand inducing points for batch
        I = self.inducing.unsqueeze(0).expand(B, -1, -1)  # [B, m, D]

        # Step 1: inducing points attend to input (causal: each inducing point
        # sees all positions — this is the global summary)
        Q1 = self.q_down(I)       # [B, m, D]
        K1 = self.k_down(x)       # [B, T, D]
        V1 = self.v_down(x)       # [B, T, D]
        H = self._multihead_attn(Q1, K1, V1)  # [B, m, D] — global summary

        # Step 2: input positions attend to inducing summaries (read global)
        Q2 = self.q_up(x)         # [B, T, D]
        K2 = self.k_up(H)         # [B, m, D]
        V2 = self.v_up(H)         # [B, m, D]
        out = self._multihead_attn(Q2, K2, V2)  # [B, T, D]

        return self.ln(x + self.out_proj(out))


# ── Test-Time Training Layer ─────────────────────────────────

class TTTLayer(nn.Module):
    """Test-time training: a linear inner model updated by gradient descent during forward.

    The inner model W maps hidden states to predictions. At each position t,
    W is updated using the reconstruction loss from positions 0..t-1.
    The weights W_t encode the sequence history — weights ARE the memory.

    Strictly causal: position t's output uses W updated only from past tokens.
    Cost: O(T × d × ttt_dim) per forward — same order as a scan.

    Mini-batched: processes ttt_mini_batch tokens at once for efficiency,
    updating W after each mini-batch. This trades granularity for speed.
    """

    def __init__(self, input_dim, ttt_dim, inner_lr=0.1, mini_batch=16):
        super().__init__()
        self.ttt_dim = ttt_dim
        self.inner_lr = inner_lr
        self.mini_batch = mini_batch

        # Project to TTT space and back
        self.proj_in = nn.Linear(input_dim, ttt_dim)
        self.proj_out = nn.Linear(ttt_dim, input_dim)

        # Initial inner weights (learned slow weights)
        # W: ttt_dim → ttt_dim (the "fast" model updated per-sequence)
        self.W_init = nn.Parameter(torch.zeros(ttt_dim, ttt_dim))
        nn.init.orthogonal_(self.W_init)
        self.b_init = nn.Parameter(torch.zeros(ttt_dim))

        # Target projection: what the inner model tries to reconstruct
        self.target_proj = nn.Linear(input_dim, ttt_dim)

        self.ln = nn.LayerNorm(input_dim)

    def forward(self, x):
        B, T, D = x.shape
        mb = self.mini_batch
        lr = self.inner_lr

        z = self.proj_in(x)        # [B, T, ttt_dim] — input to inner model
        target = self.target_proj(x)  # [B, T, ttt_dim] — what to reconstruct

        # Initialize fast weights per batch element
        W = self.W_init.unsqueeze(0).expand(B, -1, -1).clone()  # [B, d, d]
        b = self.b_init.unsqueeze(0).expand(B, -1).clone()      # [B, d]

        outputs = torch.zeros_like(z)  # [B, T, ttt_dim]

        # Process in mini-batches for efficiency
        for t0 in range(0, T, mb):
            t1 = min(t0 + mb, T)
            chunk_z = z[:, t0:t1]          # [B, chunk, d]
            chunk_target = target[:, t0:t1]  # [B, chunk, d]

            # Apply current fast weights to get output
            # pred = z @ W^T + b
            pred = torch.bmm(chunk_z, W.transpose(1, 2)) + b.unsqueeze(1)  # [B, chunk, d]
            outputs[:, t0:t1] = pred

            # Compute reconstruction error and update fast weights
            # Loss = ||pred - target||² (per-element, mean over chunk and dim)
            err = pred - chunk_target  # [B, chunk, d]

            # Gradient of W: ∇W = (1/chunk) Σ_t err_t ⊗ z_t = err^T @ z / chunk
            chunk_len = t1 - t0
            grad_W = torch.bmm(err.transpose(1, 2), chunk_z) / chunk_len  # [B, d, d]
            grad_b = err.mean(dim=1)  # [B, d]

            # Clip gradients to prevent divergence
            grad_norm = (grad_W.norm(dim=(1, 2), keepdim=True).clamp(min=1e-8))
            grad_W = grad_W * (1.0 / grad_norm.clamp(min=1.0))  # clip to unit norm
            grad_b = grad_b.clamp(-1.0, 1.0)

            # SGD step on fast weights
            W = W - lr * grad_W
            b = b - lr * grad_b

        return self.ln(x + self.proj_out(outputs))


# ── Causal Self-Attention ────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """Standard causal (masked) multi-head self-attention.

    Inserted between MLP layers for hybrid SSM+attention architectures.
    Strictly causal: each position attends only to past positions.
    Cost: O(T² × d) — use sparingly (every Nth layer).
    """

    def __init__(self, dim, heads=4, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        # Zero-init output projection — attention contributes nothing at init,
        # pure residual passthrough. Model learns to use attention gradually.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.ln = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        B, T, D = x.shape
        h, d = self.heads, self.head_dim

        qkv = self.qkv(x).reshape(B, T, 3, h, d).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]  # each [B, h, T, d]

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, h, T, T]

        # Causal mask: prevent attending to future positions
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        if self.drop is not None:
            attn = self.drop(attn)

        out = torch.matmul(attn, V)  # [B, h, T, d]
        out = out.transpose(1, 2).reshape(B, T, D)

        return self.ln(x + self.out_proj(out))


# ── Int6 Quantization-Aware Training ────────────────────────

class FakeQuantize(torch.autograd.Function):
    """Fake quantization with straight-through estimator (STE).

    Forward: round to nearest int at the given bit width.
    Backward: pass gradient through as if rounding didn't happen.
    """

    @staticmethod
    def forward(ctx, x, bits):
        # Symmetric quantization: [-max_val, max_val]
        max_val = (1 << (bits - 1)) - 1
        scale = x.abs().max().clamp(min=1e-8) / max_val
        quantized = (x / scale).round().clamp(-max_val, max_val) * scale
        return quantized

    @staticmethod
    def backward(ctx, grad_output):
        # STE: pass gradient through unchanged
        return grad_output, None


def fake_quantize(x, bits=6):
    """Apply fake quantization for QAT."""
    return FakeQuantize.apply(x, bits)


def apply_qat(model, bits=6, hash_only=False):
    """Register forward hooks that quantize weights during training.

    Returns a handle list for removal after training.
    """
    handles = []

    def _make_hook(name, param_name):
        def hook(module, input):
            param = getattr(module, param_name)
            quantized = fake_quantize(param.data, bits)
            param.data.copy_(quantized)
        return hook

    for name, module in model.named_modules():
        if hash_only and "hash" not in name and "table" not in name:
            continue
        if isinstance(module, nn.Embedding) or isinstance(module, nn.Linear) and not hash_only:
            handles.append(module.register_forward_pre_hook(_make_hook(name, "weight")))

    return handles


# ── Gated Scan (with optional Mamba-style selection) ─────────

class GatedScan(nn.Module):
    def __init__(self, input_dim, scan_dim, chunk_size=32, selection=False):
        super().__init__()
        self.scan_dim = scan_dim
        self.chunk_size = chunk_size
        self.selection = selection

        if selection:
            # Mamba-style: gates are input-dependent
            self.gate_proj = nn.Linear(input_dim, scan_dim)
            self.dt_proj = nn.Linear(input_dim, scan_dim)  # discretization step
        else:
            self.gate_proj = nn.Linear(input_dim, scan_dim)

        self.input_proj = nn.Linear(input_dim, scan_dim)
        self.output_proj = nn.Linear(scan_dim, input_dim)
        self.ln = nn.LayerNorm(input_dim)

    def forward(self, x):
        B, T, D = x.shape
        K = min(self.chunk_size, T)

        if self.selection:
            dt = F.softplus(self.dt_proj(x))  # [B,T,scan_dim] — input-dependent step
            gates = torch.sigmoid(self.gate_proj(x) * dt)
        else:
            gates = torch.sigmoid(self.gate_proj(x))

        inp = self.input_proj(x)
        drive = (1.0 - gates) * inp

        n_chunks = (T + K - 1) // K
        states = torch.zeros(B, T, self.scan_dim, device=x.device, dtype=x.dtype)
        h = torch.zeros(B, self.scan_dim, device=x.device, dtype=x.dtype)

        for c in range(n_chunks):
            s, e = c * K, min((c + 1) * K, T)
            a = gates[:, s:e]
            b = drive[:, s:e]
            log_a = torch.log(a.clamp(min=1e-6))
            cum_a = torch.exp(torch.cumsum(log_a, dim=1))
            inv_cum_a = 1.0 / cum_a.clamp(min=1e-8)
            cum_wb = torch.cumsum(b * inv_cum_a, dim=1)
            chunk_out = cum_a * (h.unsqueeze(1) + cum_wb)
            states[:, s:e] = chunk_out
            h = chunk_out[:, -1]

        return self.ln(x + self.output_proj(states))


# ── Rotation Scan ────────────────────────────────────────────

class RotationScan(nn.Module):
    """Rotation scan: information persists as phase angles, not decaying amplitudes.

    Uses complex numbers: rotation by θ = multiplication by e^{iθ}.
    This makes the scan a LINEAR RECURRENCE in the complex plane:
        h[t] = a[t] * h[t-1] + b[t]
    where a[t] = retain[t] * e^{iθ[t]} (complex scalar per dim).
    This is parallelizable with the same chunked prefix-sum as the gated scan.

    scan_dim is the number of complex dims (= scan_dim real pairs).
    The output is 2 * scan_dim real values (real + imaginary parts).
    """

    def __init__(self, input_dim, scan_dim, chunk_size=32):
        super().__init__()
        self.scan_dim = scan_dim  # complex dims
        self.real_dim = scan_dim * 2  # real output dim
        self.chunk_size = chunk_size

        # Rotation angle: input-dependent
        self.theta_proj = nn.Linear(input_dim, scan_dim)
        # Retention magnitude (how much to keep vs write)
        self.retain_proj = nn.Linear(input_dim, scan_dim)
        # Input projection (to complex: real + imag)
        self.input_proj = nn.Linear(input_dim, self.real_dim)
        # Output
        self.output_proj = nn.Linear(self.real_dim, input_dim)
        self.ln = nn.LayerNorm(input_dim)

    def forward(self, x):
        B, T, D = x.shape
        K = min(self.chunk_size, T)
        d = self.scan_dim

        theta = self.theta_proj(x)  # [B, T, d] rotation angles
        retain = torch.sigmoid(self.retain_proj(x))  # [B, T, d] magnitude in [0,1]
        inp_real = self.input_proj(x)  # [B, T, 2d]
        inp_r, inp_i = inp_real[:, :, :d], inp_real[:, :, d:]  # real, imag parts

        # Complex recurrence coefficients:
        # a = retain * e^{iθ} = retain * (cos θ + i sin θ)
        a_r = retain * torch.cos(theta)  # [B, T, d]
        a_i = retain * torch.sin(theta)

        # Drive: (1 - retain) * input (complex)
        drive_scale = 1.0 - retain
        b_r = drive_scale * inp_r
        b_i = drive_scale * inp_i

        # Chunked parallel scan (same structure as gated scan but in complex domain)
        n_chunks = (T + K - 1) // K
        out_r = torch.zeros(B, T, d, device=x.device, dtype=x.dtype)
        out_i = torch.zeros(B, T, d, device=x.device, dtype=x.dtype)
        h_r = torch.zeros(B, d, device=x.device, dtype=x.dtype)
        h_i = torch.zeros(B, d, device=x.device, dtype=x.dtype)

        for c in range(n_chunks):
            s, e = c * K, min((c + 1) * K, T)
            ca_r, ca_i = a_r[:, s:e], a_i[:, s:e]  # [B, chunk, d]
            cb_r, cb_i = b_r[:, s:e], b_i[:, s:e]

            # Log-space cumulative product for the complex magnitude
            # |a| = retain, log|a| = log(retain)
            log_mag = torch.log(retain[:, s:e].clamp(min=1e-6))  # [B, chunk, d]
            cum_log_mag = torch.cumsum(log_mag, dim=1)  # [B, chunk, d]
            cum_mag = torch.exp(cum_log_mag)  # cumulative magnitude decay

            # Cumulative angle
            cum_theta = torch.cumsum(theta[:, s:e], dim=1)  # [B, chunk, d]
            cum_cos = torch.cos(cum_theta)
            cum_sin = torch.sin(cum_theta)

            # Cumulative complex coefficient: product of a[s..t]
            # = cum_mag * e^{i * cum_theta}
            cum_a_r = cum_mag * cum_cos
            cum_a_i = cum_mag * cum_sin

            # Inverse cumulative for the drive sum
            inv_mag = 1.0 / cum_mag.clamp(min=1e-8)
            inv_cos = torch.cos(-cum_theta)
            inv_sin = torch.sin(-cum_theta)
            inv_r = inv_mag * inv_cos
            inv_i = inv_mag * inv_sin

            # Complex multiply: inv * b (element-wise)
            scaled_b_r = inv_r * cb_r - inv_i * cb_i
            scaled_b_i = inv_r * cb_i + inv_i * cb_r

            # Cumulative sum of scaled drives
            cum_b_r = torch.cumsum(scaled_b_r, dim=1)
            cum_b_i = torch.cumsum(scaled_b_i, dim=1)

            # Combine: h[t] = cum_a[t] * (h_carry + cum_b[t])
            # Complex multiply: cum_a * (carry + cum_b)
            total_r = h_r.unsqueeze(1) + cum_b_r  # [B, chunk, d]
            total_i = h_i.unsqueeze(1) + cum_b_i
            chunk_r = cum_a_r * total_r - cum_a_i * total_i
            chunk_i = cum_a_r * total_i + cum_a_i * total_r

            out_r[:, s:e] = chunk_r
            out_i[:, s:e] = chunk_i
            h_r = chunk_r[:, -1]
            h_i = chunk_i[:, -1]

        # Concatenate real and imaginary parts
        output = torch.cat([out_r, out_i], dim=-1)  # [B, T, 2d]
        return self.ln(x + self.output_proj(output))


# ── Mamba Scan (fused CUDA kernel) ───────────────────────────

class MambaScan(nn.Module):
    """Selective scan using mamba-ssm's fused CUDA kernel.

    Same semantics as GatedScan with selection=True, but the scan runs
    entirely in GPU registers via a custom CUDA kernel — no HBM round-trips
    per step. This takes the scan from ~0.25 FLOPS/byte to ~15-40 FLOPS/byte.

    Falls back to GatedScan if mamba-ssm is not installed or on CPU.
    """

    def __init__(self, input_dim, scan_dim, chunk_size=32, dstate=16):
        super().__init__()
        self.scan_dim = scan_dim
        self.input_dim = input_dim
        self._has_mamba = False

        try:
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
            self._selective_scan_fn = selective_scan_fn
            self._has_mamba = True
        except ImportError:
            self._selective_scan_fn = None

        self.dstate = dstate

        # Mamba-style projections: input → (z, x, B, C, dt)
        self.in_proj = nn.Linear(input_dim, scan_dim * 2)  # x and z
        self.dt_proj = nn.Linear(input_dim, scan_dim)
        self.B_proj = nn.Linear(input_dim, self.dstate)
        self.C_proj = nn.Linear(input_dim, self.dstate)

        # A: [scan_dim, dstate] — learned diagonal in log-space
        self.A_log = nn.Parameter(torch.log(
            torch.linspace(1, self.dstate, self.dstate).unsqueeze(0).expand(scan_dim, -1).clone()
        ))

        self.output_proj = nn.Linear(scan_dim, input_dim)
        self.ln = nn.LayerNorm(input_dim)
        self.D = nn.Parameter(torch.ones(scan_dim))

        # Fallback for CPU/non-CUDA
        self._fallback = GatedScan(input_dim, scan_dim, chunk_size, selection=True)

    def forward(self, x):
        B, T, D = x.shape

        if not self._has_mamba or not x.is_cuda:
            return self._fallback(x)

        # Run scan in fp32 outside autocast — the fused kernel is register-bound,
        # so fp32 doesn't cost throughput. Avoids AMP dtype promotion conflicts.
        orig_dtype = x.dtype
        with torch.amp.autocast("cuda", enabled=False):
            x_f32 = x.float()
            xz = self.in_proj(x_f32)
            x_ssm, z = xz.chunk(2, dim=-1)
            dt = F.softplus(self.dt_proj(x_f32))
            B_mat = self.B_proj(x_f32)
            C_mat = self.C_proj(x_f32)
            A = -torch.exp(self.A_log.float())

            x_ssm = x_ssm.transpose(1, 2).contiguous()
            dt = dt.transpose(1, 2).contiguous()
            B_mat = B_mat.transpose(1, 2).unsqueeze(1).contiguous()
            C_mat = C_mat.transpose(1, 2).unsqueeze(1).contiguous()

            y = self._selective_scan_fn(
                x_ssm, dt, A, B_mat, C_mat, self.D.float(),
                z=None, delta_bias=None, delta_softplus=False,
                return_last_state=False,
            )
            y = y * F.silu(z.transpose(1, 2))
            y = y.transpose(1, 2).to(orig_dtype)

        return self.ln(x + self.output_proj(y))


# ── Multi-Scale Scan ─────────────────────────────────────────

class MultiScaleScan(nn.Module):
    """Parallel scans at different time scales.

    Each scale has its own gate bias initialization that controls
    how fast it forgets. Scale 0 = fast (local patterns),
    scale N = slow (long-range structure).

    All scans run in parallel on the same input. Outputs are
    concatenated and projected back to input_dim.
    """

    def __init__(self, input_dim, scale_dims, chunk_size=32):
        super().__init__()
        self.scans = nn.ModuleList()
        self.total_dim = sum(scale_dims)

        for i, dim in enumerate(scale_dims):
            scan = GatedScan(input_dim, dim, chunk_size, selection=False)
            # Initialize gate bias to control time scale:
            # scale 0 → bias -1 (gate ≈ 0.27, fast decay, half-life ~1 token)
            # scale N → bias +3 (gate ≈ 0.95, slow decay, half-life ~14 tokens)
            target_bias = -1.0 + (4.0 * i / max(len(scale_dims) - 1, 1))
            nn.init.constant_(scan.gate_proj.bias, target_bias)
            self.scans.append(scan)

        self.merge = nn.Linear(input_dim * len(scale_dims), input_dim)
        self.ln = nn.LayerNorm(input_dim)

    def forward(self, x):
        # Run all scans in parallel (same input, different decay rates)
        outputs = [scan(x) for scan in self.scans]
        # Each scan returns ln(x + proj(states)) — already residual
        # Concatenate and merge
        merged = torch.cat(outputs, dim=-1)
        return self.ln(x + self.merge(merged))


# ── Main Model ───────────────────────────────────────────────

class PolyHashV12(nn.Module):
    def __init__(self, config: V12Config = V12Config()):
        super().__init__()
        self.config = config
        V = config.vocab_size

        self.patch_size = config.patch_size
        self.patch_embed = None
        self.patch_output = None

        if config.patch_size > 1:
            # Patch mode: embed P bytes → one vector, predict P bytes at output
            self.patch_embed = PatchEmbed(
                config.patch_size, V, config.byte_embed_dim, config.byte_embed_dim,
            )
            self.patch_output = PatchOutput(config.hidden_dim, config.patch_size, V)
            self.byte_embed = None
        else:
            self.byte_embed = nn.Embedding(V, config.byte_embed_dim)

        # Hash tables (fixed addressing)
        self.hash_tables = None
        hash_feat_dim = 0
        if config.num_hash_tables > 0:
            self.hash_tables = HashTables(
                config.num_hash_tables, config.hash_buckets, config.hash_embed_dim
            )
            hash_feat_dim = config.num_hash_tables * config.hash_embed_dim

        # Conv
        self.dw_conv = None
        if config.conv_kernel > 0 and hash_feat_dim > 0:
            self.dw_conv = nn.Conv1d(
                hash_feat_dim, hash_feat_dim,
                kernel_size=config.conv_kernel, padding=0,
                groups=hash_feat_dim,
            )
            self.conv_pad = config.conv_kernel - 1

        # Match features
        n_match = len(config.match_offsets) if config.match_offsets else 0

        # Feature dim after hash + byte + match
        feat_dim = config.byte_embed_dim + hash_feat_dim + n_match

        # Input projection to hidden
        self.input_proj = nn.Linear(feat_dim, config.hidden_dim)

        # Product Key Memory (learned addressing)
        self.pkm = None
        if config.pkm_enabled:
            self.pkm = ProductKeyMemory(
                config.hidden_dim,
                config.pkm_sub_keys, config.pkm_top_k,
                config.pkm_key_dim, config.pkm_value_dim,
                xsa=config.pkm_xsa,
            )

        # TTT (test-time training)
        self.ttt = None
        if config.ttt_enabled:
            self.ttt = TTTLayer(
                config.hidden_dim, config.ttt_dim,
                inner_lr=config.ttt_lr,
                mini_batch=config.ttt_mini_batch,
            )

        # Induced Set Attention (global summary)
        self.isab = None
        if config.isab_enabled:
            self.isab = ISAB(config.hidden_dim, config.isab_num_points, config.isab_heads)

        # Scan (single or multi-scale)
        self.scan = None
        self.multiscale_scan = None
        if config.scan_multiscale:
            self.multiscale_scan = MultiScaleScan(
                config.hidden_dim, config.scan_multiscale,
                config.scan_chunk_size,
            )
        elif config.scan_dim > 0:
            if config.scan_mamba:
                self.scan = MambaScan(
                    config.hidden_dim, config.scan_dim,
                    config.scan_chunk_size,
                    dstate=config.scan_mamba_dstate,
                )
            elif config.scan_rotation:
                self.scan = RotationScan(
                    config.hidden_dim, config.scan_dim,
                    config.scan_chunk_size,
                )
            else:
                self.scan = GatedScan(
                    config.hidden_dim, config.scan_dim,
                    config.scan_chunk_size, config.scan_selection,
                )

        # MLP readout with optional causal attention layers
        self.mlp = nn.ModuleList([
            ResBlock(config.hidden_dim, config.dropout,
                     pre_norm=config.pre_norm, resid_mix=config.resid_mix)
            for _ in range(config.num_layers - 1)
        ])
        self.attn_layers_set = set(config.attn_layers)
        self.attn_modules = nn.ModuleDict({
            str(i): CausalSelfAttention(config.hidden_dim, config.attn_heads, config.dropout)
            for i in config.attn_layers
        })
        if config.patch_size <= 1:
            self.output_proj = nn.Linear(config.hidden_dim, V)
        else:
            self.output_proj = None  # PatchOutput handles this

    def _match_features(self, chars):
        B, T = chars.shape
        feats = []
        for k in self.config.match_offsets:
            if k >= T:
                feats.append(torch.zeros(B, T, device=chars.device))
            else:
                shifted = torch.zeros_like(chars)
                shifted[:, k:] = chars[:, :-k]
                m = (chars == shifted).float()
                m[:, :k] = 0.0
                feats.append(m)
        return torch.stack(feats, dim=-1) if feats else None

    def forward(self, chars, target_patches=None):
        cfg = self.config
        B, L = chars.shape
        P = self.patch_size

        # --- Input embedding (byte or patch) ---
        if P > 1:
            byte_emb, T = self.patch_embed(chars)  # [B, T, byte_embed_dim]
            # Hash tables operate on the original byte sequence
            hash_feat = self.hash_tables(chars) if self.hash_tables is not None else None
            if hash_feat is not None:
                # Downsample hash features to patch resolution: mean-pool over P
                hash_feat = hash_feat[:, :T * P].reshape(B, T, P, -1).mean(dim=2)
        else:
            T = L
            byte_emb = self.byte_embed(chars)
            hash_feat = self.hash_tables(chars) if self.hash_tables is not None else None

        if self.dw_conv is not None and hash_feat is not None:
            ht = hash_feat.transpose(1, 2)
            ht = F.pad(ht, (self.conv_pad, 0))
            hash_feat = F.silu(self.dw_conv(ht)).transpose(1, 2)

        parts = [byte_emb]
        if hash_feat is not None:
            parts.append(hash_feat)
        if P == 1:
            mf = self._match_features(chars) if cfg.match_offsets else None
            if mf is not None:
                parts.append(mf)

        h = F.silu(self.input_proj(torch.cat(parts, dim=-1)))

        # PKM: content-dependent memory lookup
        if self.pkm is not None:
            h = self.pkm(h)

        # TTT: test-time training — weights as memory
        if self.ttt is not None:
            h = self.ttt(h)

        # ISAB: global sequence summary
        if self.isab is not None:
            h = self.isab(h)

        # Scan: sequential state (single or multi-scale)
        if self.multiscale_scan is not None:
            h = self.multiscale_scan(h)
        elif self.scan is not None:
            h = self.scan(h)

        # MLP readout with optional causal attention at specified layers
        for i, block in enumerate(self.mlp):
            h = block(h)
            if str(i) in self.attn_modules:
                h = self.attn_modules[str(i)](h)

        # --- Output (byte or patch) ---
        if P > 1:
            return self.patch_output(h, target_patches)  # [B, T, P, 256]
        return self.output_proj(h)
