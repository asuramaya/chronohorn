"""PolyHash v4: Engram-inspired + Product Key Memory + rich recurrence.

New mechanisms:
  - Scalar context gate (Engram): sigmoid(rmsnorm(h) @ rmsnorm(e) / sqrt(d))
  - Multi-head hashing (Engram): K hashes per table, mean/max pool
  - Depthwise causal conv: learnable smoothing after hash lookup
  - Product Key Memory: learned split-codebook addressing, sqrt(N) search
  - Matrix-valued scan: d_s x d_s state matrix for rich recurrence
  - RWKV-style update: S = decay*S + v*k^T, o = S*q
  - Long causal conv: kernel 16-128 as parallelizable scan replacement
  - Match features: binary token[t] == token[t-k]
  - Count-augmented: bucket visit counts as confidence signal
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

HASH_PRIMES = [
    2654435761, 2246822519, 3266489917, 2028178513,
    1220703125, 1610612741, 805306457, 402653189,
    3674653429, 2860486313, 1073676287, 2971215073,
    1500450271, 3267000013, 2654435789, 4049292737,
    2246822531, 3266489927, 2028178519, 1220703133,
    1610612743, 805306459, 402653191, 3674653433,
    2654435771, 2246822527, 3266489933, 2028178529,
    1220703137, 1610612747, 805306463, 402653197,
]

FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21]


@dataclass(frozen=True)
class V4Config:
    vocab_size: int = 1024
    byte_embed_dim: int = 128
    # Hash tables
    num_tables: int = 8
    buckets_per_table: int = 32768
    embed_per_table: int = 16
    # MLP
    hidden_dim: int = 512
    num_layers: int = 2
    # Engram tricks
    scalar_gate: str = "none"        # none, pre, post
    num_hash_heads: int = 1          # multi-head hashing (1=off, 4/8=Engram)
    hash_head_pool: str = "mean"     # mean, max
    conv_kernel: int = 0             # depthwise causal conv after lookup (0=off)
    conv_dilation: int = 1
    # Product Key Memory
    pkm_mode: str = "none"           # none, replace, hybrid
    pkm_sub_keys: int = 512          # sub-codebook size (effective = sub_keys^2)
    pkm_topk: int = 8               # top-k per sub-codebook
    pkm_tables: int = 4             # number of PKM tables
    pkm_dim: int = 16               # PKM output dim per table
    # Recurrence
    scan_mode: str = "none"          # none, ema, matrix, rwkv
    scan_dim: int = 0               # state dim for ema; matrix size for matrix/rwkv
    # Long conv (parallelizable scan replacement)
    long_conv_kernel: int = 0        # 0=off, 16/32/64/128
    # Match features
    match_offsets: tuple = ()        # () = off
    # Count augmented
    count_augmented: bool = False
    # Training
    max_seq_len: int = 512
    init_seed: int = 42
    dropout: float = 0.0


class SwiGLU(nn.Module):
    def __init__(self, in_dim: int, hid: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(in_dim, hid, bias=False)
        self.w2 = nn.Linear(in_dim, hid, bias=False)
        self.proj = nn.Linear(hid, in_dim, bias=False)

    def forward(self, x):
        return self.proj(F.silu(self.w1(x)) * self.w2(x))


class ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = SwiGLU(dim, dim)
        self.ln = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        h = self.block(x)
        if self.drop is not None:
            h = self.drop(h)
        return self.ln(h + x)


def _hash_ctx(tokens, pattern, table_idx, buckets, head_offset=0):
    B, T = tokens.shape
    h = torch.zeros(B, T, dtype=torch.long, device=tokens.device)
    for k, offset in enumerate(pattern):
        pidx = (table_idx * 3 + k + head_offset * 7) % len(HASH_PRIMES)
        prime = HASH_PRIMES[pidx]
        shifted = torch.zeros_like(tokens)
        if 0 < offset < T:
            shifted[:, offset:] = tokens[:, :-offset]
        h = h ^ (shifted * prime)
    return h % buckets


class ProductKeyMemory(nn.Module):
    """Learned split-codebook addressing. O(1) per position."""
    def __init__(self, input_dim: int, sub_keys: int = 512,
                 topk: int = 8, value_dim: int = 16) -> None:
        super().__init__()
        self.sub_keys = sub_keys
        self.topk = topk
        self.half_dim = input_dim // 2
        # Two sub-codebooks
        self.codebook1 = nn.Parameter(torch.randn(sub_keys, self.half_dim) * 0.02)
        self.codebook2 = nn.Parameter(torch.randn(sub_keys, self.half_dim) * 0.02)
        # Value table: sub_keys^2 entries but stored sparsely via gather
        self.values = nn.Embedding(sub_keys * sub_keys, value_dim)
        nn.init.normal_(self.values.weight, std=0.02)

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        """query: [B, T, input_dim] -> [B, T, value_dim]"""
        B, T, D = query.shape
        q1 = query[..., :self.half_dim]  # [B, T, half]
        q2 = query[..., self.half_dim:self.half_dim * 2]

        # Scores against sub-codebooks
        s1 = torch.matmul(q1, self.codebook1.t())  # [B, T, sub_keys]
        s2 = torch.matmul(q2, self.codebook2.t())

        # Top-k per codebook
        tk1_vals, tk1_idx = s1.topk(self.topk, dim=-1)  # [B, T, topk]
        tk2_vals, tk2_idx = s2.topk(self.topk, dim=-1)

        # Cartesian product indices
        idx1 = tk1_idx.unsqueeze(-1).expand(-1, -1, -1, self.topk)  # [B,T,k,k]
        idx2 = tk2_idx.unsqueeze(-2).expand(-1, -1, self.topk, -1)
        full_idx = idx1 * self.sub_keys + idx2  # [B, T, k, k]
        full_idx = full_idx.reshape(B, T, -1)  # [B, T, k^2]

        # Weights from product of scores
        w1 = F.softmax(tk1_vals, dim=-1)  # [B, T, k]
        w2 = F.softmax(tk2_vals, dim=-1)
        weights = w1.unsqueeze(-1) * w2.unsqueeze(-2)  # [B, T, k, k]
        weights = weights.reshape(B, T, -1)  # [B, T, k^2]

        # Gather values and weighted sum
        vals = self.values(full_idx)  # [B, T, k^2, value_dim]
        out = (weights.unsqueeze(-1) * vals).sum(dim=-2)  # [B, T, value_dim]
        return out


class MatrixScan(nn.Module):
    """Matrix-valued recurrent scan. State is d_s x d_s."""
    def __init__(self, input_dim: int, state_dim: int, mode: str = "matrix") -> None:
        super().__init__()
        self.state_dim = state_dim
        self.mode = mode
        if mode == "rwkv":
            self.q_proj = nn.Linear(input_dim, state_dim)
            self.k_proj = nn.Linear(input_dim, state_dim)
            self.v_proj = nn.Linear(input_dim, state_dim)
            self.decay = nn.Parameter(torch.full((state_dim,), 0.95))
            self.out_proj = nn.Linear(state_dim, input_dim)
        else:  # matrix
            self.in_proj = nn.Linear(input_dim, state_dim * state_dim)
            self.decay = nn.Parameter(torch.full((state_dim, state_dim), 0.9))
            self.out_proj = nn.Linear(state_dim * state_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        ds = self.state_dim

        if self.mode == "rwkv":
            q = self.q_proj(x)  # [B, T, ds]
            k = self.k_proj(x)
            v = self.v_proj(x)
            decay = torch.sigmoid(self.decay)
            S = torch.zeros(B, ds, ds, device=x.device)
            outs = []
            for t in range(T):
                S = torch.diag_embed(decay) @ S + v[:, t].unsqueeze(-1) @ k[:, t].unsqueeze(-2)
                o = (S @ q[:, t].unsqueeze(-1)).squeeze(-1)
                outs.append(o)
            scan_out = torch.stack(outs, dim=1)
            return x + self.out_proj(scan_out)
        else:  # matrix
            inp = self.in_proj(x).reshape(B, T, ds, ds)
            decay = torch.sigmoid(self.decay)
            S = torch.zeros(B, ds, ds, device=x.device)
            outs = []
            for t in range(T):
                S = decay * S + inp[:, t]
                outs.append(S.reshape(B, -1))
            scan_out = torch.stack(outs, dim=1)
            return x + self.out_proj(scan_out)


class PolyHashV4(nn.Module):
    def __init__(self, config: V4Config = V4Config()) -> None:
        super().__init__()
        self.config = config
        V = config.vocab_size

        self.byte_embed = nn.Embedding(V, config.byte_embed_dim)

        # Skip patterns (fibonacci)
        self.skip_patterns = tuple(
            (FIBONACCI[i],) if i < len(FIBONACCI) else (1, i + 1)
            for i in range(config.num_tables)
        )

        # Hash tables
        self.hash_tables = nn.ModuleList()
        for _ in range(config.num_tables):
            t = nn.Embedding(config.buckets_per_table, config.embed_per_table)
            nn.init.normal_(t.weight, std=0.02)
            self.hash_tables.append(t)

        # Count tracking (non-learnable)
        if config.count_augmented:
            for i in range(config.num_tables):
                self.register_buffer(
                    f"_counts_{i}",
                    torch.zeros(config.buckets_per_table)
                )

        # Product Key Memory
        self.pkm_layers = nn.ModuleList()
        if config.pkm_mode != "none":
            pkm_input_dim = config.byte_embed_dim
            for _ in range(config.pkm_tables):
                self.pkm_layers.append(ProductKeyMemory(
                    pkm_input_dim, config.pkm_sub_keys,
                    config.pkm_topk, config.pkm_dim,
                ))

        # Depthwise causal conv
        hash_feat_dim = config.num_tables * config.embed_per_table
        self.dw_conv = None
        if config.conv_kernel > 0:
            self.dw_conv = nn.Conv1d(
                hash_feat_dim, hash_feat_dim,
                kernel_size=config.conv_kernel,
                padding=0,  # manual causal padding
                dilation=config.conv_dilation,
                groups=hash_feat_dim,
            )
            self.conv_pad = (config.conv_kernel - 1) * config.conv_dilation

        # Long conv (parallelizable scan replacement)
        self.long_conv = None
        if config.long_conv_kernel > 0:
            self.long_conv = nn.Conv1d(
                config.hidden_dim, config.hidden_dim,
                kernel_size=config.long_conv_kernel,
                padding=0,
                groups=config.hidden_dim,
            )
            self.long_conv_pad = config.long_conv_kernel - 1

        # Scalar context gate
        self.gate_ln_h = None
        if config.scalar_gate != "none":
            gate_dim = config.hidden_dim if config.scalar_gate == "post" else hash_feat_dim
            self.gate_ln_h = nn.LayerNorm(gate_dim, elementwise_affine=False)
            self.gate_ln_e = nn.LayerNorm(gate_dim, elementwise_affine=False)

        # Compute feature dim
        n_match = len(config.match_offsets)
        n_pkm = config.pkm_tables * config.pkm_dim if config.pkm_mode != "none" else 0
        n_count = config.num_tables if config.count_augmented else 0

        if config.pkm_mode == "replace":
            feat_dim = config.byte_embed_dim + n_pkm + n_match + n_count
        else:
            feat_dim = config.byte_embed_dim + hash_feat_dim + n_pkm + n_match + n_count

        self.input_proj = nn.Linear(feat_dim, config.hidden_dim)

        # Recurrence
        self.scan = None
        if config.scan_mode == "ema" and config.scan_dim > 0:
            self.scan_decay = nn.Parameter(torch.full((config.scan_dim,), 0.9))
            self.scan_in = nn.Linear(config.hidden_dim, config.scan_dim)
            self.scan_out = nn.Linear(config.scan_dim, config.hidden_dim)
        elif config.scan_mode in ("matrix", "rwkv") and config.scan_dim > 0:
            self.scan = MatrixScan(config.hidden_dim, config.scan_dim, config.scan_mode)

        # MLP
        self.mlp = nn.ModuleList([
            ResBlock(config.hidden_dim, config.dropout)
            for _ in range(config.num_layers - 1)
        ])
        self.output_proj = nn.Linear(config.hidden_dim, V)
        self._init_weights()

    def _init_weights(self):
        rng = torch.Generator().manual_seed(self.config.init_seed)
        nn.init.xavier_uniform_(self.input_proj.weight, generator=rng)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight, generator=rng)
        nn.init.zeros_(self.output_proj.bias)

    def _hash_features(self, chars):
        cfg = self.config
        all_embs = []
        all_counts = []
        for i, table in enumerate(self.hash_tables):
            pat = self.skip_patterns[i]
            if cfg.num_hash_heads > 1:
                head_embs = []
                for hh in range(cfg.num_hash_heads):
                    idx = _hash_ctx(chars, pat, i, cfg.buckets_per_table, head_offset=hh)
                    head_embs.append(table(idx))
                stacked = torch.stack(head_embs, dim=0)
                if cfg.hash_head_pool == "max":
                    emb = stacked.max(dim=0).values
                else:
                    emb = stacked.mean(dim=0)
                # Use first head's indices for counts
                if cfg.count_augmented:
                    idx0 = _hash_ctx(chars, pat, i, cfg.buckets_per_table, head_offset=0)
                    counts_buf = getattr(self, f"_counts_{i}")
                    all_counts.append(counts_buf[idx0].unsqueeze(-1))
            else:
                idx = _hash_ctx(chars, pat, i, cfg.buckets_per_table)
                emb = table(idx)
                if cfg.count_augmented:
                    counts_buf = getattr(self, f"_counts_{i}")
                    all_counts.append(counts_buf[idx].unsqueeze(-1))
            all_embs.append(emb)

        hash_feat = torch.cat(all_embs, dim=-1)

        # Depthwise causal conv
        if self.dw_conv is not None:
            h_t = hash_feat.transpose(1, 2)  # [B, C, T]
            h_t = F.pad(h_t, (self.conv_pad, 0))
            h_t = F.silu(self.dw_conv(h_t))
            hash_feat = h_t.transpose(1, 2)

        count_feat = torch.cat(all_counts, dim=-1) if all_counts else None
        return hash_feat, count_feat

    def _match_features(self, tokens):
        B, T = tokens.shape
        feats = []
        for k in self.config.match_offsets:
            if k >= T:
                feats.append(torch.zeros(B, T, device=tokens.device))
                continue
            shifted = torch.zeros_like(tokens)
            shifted[:, k:] = tokens[:, :-k]
            match = (tokens == shifted).float()
            match[:, :k] = 0.0
            feats.append(match)
        return torch.stack(feats, dim=-1) if feats else None

    def _apply_scalar_gate(self, h, e):
        """Engram-style scalar gate: alpha = sigmoid(norm(h) @ norm(e) / sqrt(d))"""
        h_n = self.gate_ln_h(h)
        e_n = self.gate_ln_e(e)
        d = h_n.shape[-1]
        alpha = torch.sigmoid((h_n * e_n).sum(dim=-1, keepdim=True) / math.sqrt(d))
        return alpha * e + (1.0 - alpha) * h

    def forward(self, chars):
        cfg = self.config
        B, T = chars.shape

        byte_emb = self.byte_embed(chars)
        parts = [byte_emb]

        # Hash features
        if cfg.pkm_mode != "replace":
            hash_feat, count_feat = self._hash_features(chars)
            # Pre-MLP scalar gate
            if cfg.scalar_gate == "pre":
                hash_feat = self._apply_scalar_gate(byte_emb, hash_feat)
            parts.append(hash_feat)
            if count_feat is not None:
                parts.append(torch.log1p(count_feat))

        # PKM features
        if cfg.pkm_mode != "none":
            for pkm in self.pkm_layers:
                parts.append(pkm(byte_emb))

        # Match features
        if cfg.match_offsets:
            mf = self._match_features(chars)
            if mf is not None:
                parts.append(mf)

        features = torch.cat(parts, dim=-1)
        h = F.silu(self.input_proj(features))

        # Post-MLP-input scalar gate
        if cfg.scalar_gate == "post":
            hash_proj = h  # h already contains hash info via input_proj
            h = self._apply_scalar_gate(F.silu(self.input_proj(torch.cat([byte_emb], dim=-1).expand_as(features))), h)

        # Long conv (parallelizable)
        if self.long_conv is not None:
            h_t = h.transpose(1, 2)
            h_t = F.pad(h_t, (self.long_conv_pad, 0))
            h = h + F.silu(self.long_conv(h_t)).transpose(1, 2)

        # Recurrence
        if cfg.scan_mode == "ema" and cfg.scan_dim > 0:
            decay = torch.sigmoid(self.scan_decay)
            inp = self.scan_in(h)
            state = torch.zeros(B, cfg.scan_dim, device=h.device)
            outs = []
            for t in range(T):
                state = decay * state + (1.0 - decay) * inp[:, t]
                outs.append(state)
            scan_out = torch.stack(outs, dim=1)
            h = h + self.scan_out(scan_out)
        elif self.scan is not None:
            h = self.scan(h)

        for block in self.mlp:
            h = block(h)

        return self.output_proj(h)
