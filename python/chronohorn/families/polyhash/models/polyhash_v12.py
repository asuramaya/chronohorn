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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

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
    # MLP
    hidden_dim: int = 512
    num_layers: int = 2
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
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.block = SwiGLU(dim, dim)
        self.ln = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else None
    def forward(self, x):
        h = self.block(x)
        if self.drop: h = self.drop(h)
        return self.ln(h + x)


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

    def __init__(self, input_dim, scan_dim, chunk_size=32):
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

        # SSM state dimension — small, independent of scan_dim
        self.dstate = 16

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


# ── Main Model ───────────────────────────────────────────────

class PolyHashV12(nn.Module):
    def __init__(self, config: V12Config = V12Config()):
        super().__init__()
        self.config = config
        V = config.vocab_size

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

        # Scan
        self.scan = None
        if config.scan_dim > 0:
            if config.scan_mamba:
                self.scan = MambaScan(
                    config.hidden_dim, config.scan_dim,
                    config.scan_chunk_size,
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

        # MLP readout
        self.mlp = nn.ModuleList([
            ResBlock(config.hidden_dim, config.dropout)
            for _ in range(config.num_layers - 1)
        ])
        self.output_proj = nn.Linear(config.hidden_dim, V)

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

    def forward(self, chars):
        cfg = self.config
        B, T = chars.shape

        byte_emb = self.byte_embed(chars)
        hash_feat = self.hash_tables(chars) if self.hash_tables is not None else None

        if self.dw_conv is not None and hash_feat is not None:
            ht = hash_feat.transpose(1, 2)
            ht = F.pad(ht, (self.conv_pad, 0))
            hash_feat = F.silu(self.dw_conv(ht)).transpose(1, 2)

        parts = [byte_emb]
        if hash_feat is not None:
            parts.append(hash_feat)
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

        # Scan: sequential state
        if self.scan is not None:
            h = self.scan(h)

        # MLP readout
        for block in self.mlp:
            h = block(h)

        return self.output_proj(h)
