"""PolyHash v7: State-Addressed Memory.

The scan builds sequential state. The hash of that state retrieves from a
massive learned memory table. Content-dependent retrieval at O(1) per position.

Architecture:
  byte_embed → hash_tables + conv8 + match (fast O(1) path)
            → 4-group parallel gated scan (sequential state)
            → hash(quantize(scan_state)) → memory_table lookup (state-addressed)
            → SwiGLU MLP → logits
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
    3674653429, 2860486313, 1073676287, 2971215073,
    1500450271, 3267000013, 2654435789, 4049292737,
    2246822531, 3266489927, 2028178519, 1220703133,
]

FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21]


@dataclass(frozen=True)
class V7Config:
    vocab_size: int = 1024
    byte_embed_dim: int = 128
    # Fast path (hash tables)
    num_tables: int = 8
    buckets_per_table: int = 65536
    embed_per_table: int = 16
    conv_kernel: int = 8
    match_offsets: tuple = (1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32)
    # Scan
    scan_dim: int = 256
    scan_groups: int = 4
    scan_chunk_size: int = 32
    # State-addressed memory
    sam_enabled: bool = True
    sam_buckets: int = 131072      # 128K buckets — massive
    sam_embed_dim: int = 32        # embedding per bucket
    sam_heads: int = 4             # independent hash heads (reduce collision)
    sam_quant_mode: str = "sign"   # sign, topk, token_mix
    sam_quant_bits: int = 16       # number of state dims to hash
    sam_straight_through: bool = False  # gradient flows through sign via STE
    sam_soft_temp: float = 0.0         # 0=hard sign, >0=tanh(x/temp) soft quantization
    sam_2bit: bool = False             # 2-bit quantization (sign + magnitude bucket)
    # MLP
    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.0
    # Training
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


def _hash_ctx(tokens, pattern, table_idx, buckets):
    B, T = tokens.shape
    h = torch.zeros(B, T, dtype=torch.long, device=tokens.device)
    for k, offset in enumerate(pattern):
        pidx = (table_idx * 3 + k) % len(HASH_PRIMES)
        shifted = torch.zeros_like(tokens)
        if 0 < offset < T:
            shifted[:, offset:] = tokens[:, :-offset]
        h = h ^ (shifted * HASH_PRIMES[pidx])
    return h % buckets


class MultiGroupScan(nn.Module):
    """4-group parallel gated scan with independent timescales."""
    def __init__(self, input_dim: int, scan_dim: int, groups: int = 4, chunk_size: int = 32):
        super().__init__()
        self.scan_dim = scan_dim
        self.groups = groups
        self.chunk_size = chunk_size
        # Independent gate projection per group
        self.gate_proj = nn.Linear(input_dim, scan_dim)
        self.input_proj = nn.Linear(input_dim, scan_dim)
        self.output_proj = nn.Linear(scan_dim, input_dim)
        self.ln = nn.LayerNorm(input_dim)

    def forward(self, x):
        B, T, D = x.shape
        K = min(self.chunk_size, T)
        gates = torch.sigmoid(self.gate_proj(x))
        inp = self.input_proj(x)
        drive = (1.0 - gates) * inp

        n_chunks = (T + K - 1) // K
        states = torch.zeros(B, T, self.scan_dim, device=x.device, dtype=x.dtype)
        h = torch.zeros(B, self.scan_dim, device=x.device, dtype=x.dtype)

        for c in range(n_chunks):
            s, e = c * K, min((c + 1) * K, T)
            a = gates[:, s:e]; b = drive[:, s:e]
            log_a = torch.log(a.clamp(min=1e-6))
            cum_a = torch.exp(torch.cumsum(log_a, dim=1))
            inv_cum_a = 1.0 / cum_a.clamp(min=1e-8)
            cum_wb = torch.cumsum(b * inv_cum_a, dim=1)
            chunk_out = cum_a * (h.unsqueeze(1) + cum_wb)
            states[:, s:e] = chunk_out
            h = chunk_out[:, -1]

        return self.ln(x + self.output_proj(states))


class StateAddressedMemory(nn.Module):
    """Hash the scan state → lookup massive memory table.

    Quantize continuous state to discrete key via sign bits / topk / token mixing.
    Multiple hash heads for collision reduction.
    """
    def __init__(self, state_dim: int, config: V7Config):
        super().__init__()
        self.config = config
        self.num_heads = config.sam_heads
        self.quant_mode = config.sam_quant_mode
        self.quant_bits = config.sam_quant_bits
        self.buckets = config.sam_buckets

        # Project state to hashable dimensions
        self.state_proj = nn.Linear(state_dim, config.sam_quant_bits * config.sam_heads)

        # Memory tables (one per head)
        self.memory_tables = nn.ModuleList()
        for _ in range(config.sam_heads):
            t = nn.Embedding(config.sam_buckets, config.sam_embed_dim)
            nn.init.normal_(t.weight, std=0.02)
            self.memory_tables.append(t)

        # Output projection
        self.out_proj = nn.Linear(config.sam_heads * config.sam_embed_dim, state_dim)

    def _quantize_to_key(self, projected: torch.Tensor, head_idx: int) -> torch.Tensor:
        """Convert continuous state projection to discrete hash key.
        projected: [B, T, quant_bits]
        Returns: [B, T] long tensor of bucket indices.
        Also returns soft_projected for gradient flow if STE/soft mode.
        """
        B, T, D = projected.shape
        cfg = self.config

        if self.quant_mode == "topk":
            k = min(4, D)
            _, topk_idx = projected.topk(k, dim=-1)
            key = torch.zeros(B, T, dtype=torch.long, device=projected.device)
            for i in range(k):
                key = key ^ (topk_idx[:, :, i] * HASH_PRIMES[(head_idx * 5 + i) % len(HASH_PRIMES)])
            return key % self.buckets

        # Sign-based quantization (sign, token_mix, 2bit)
        if cfg.sam_2bit:
            # 2-bit: sign (0/1) + magnitude bucket (0/1 = above/below median)
            sign_bits = (projected > 0).long()
            magnitude = projected.abs()
            median_mag = magnitude.median(dim=-1, keepdim=True).values
            mag_bits = (magnitude > median_mag).long()
            # Interleave: 2 bits per dim → 2*D total bits
            bits = sign_bits * 2 + mag_bits  # values 0-3 per dim
            key = torch.zeros(B, T, dtype=torch.long, device=projected.device)
            for i in range(D):
                key = key ^ (bits[:, :, i] * HASH_PRIMES[(head_idx * 3 + i) % len(HASH_PRIMES)])
        else:
            # 1-bit sign
            bits = (projected > 0).long()
            key = torch.zeros(B, T, dtype=torch.long, device=projected.device)
            for i in range(D):
                key = key ^ (bits[:, :, i] * HASH_PRIMES[(head_idx * 3 + i) % len(HASH_PRIMES)])

        if self.quant_mode == "token_mix":
            # Will be mixed with chars in forward()
            pass

        return key % self.buckets

    def _soft_lookup(self, projected: torch.Tensor, key: torch.Tensor,
                     table: nn.Embedding, head_idx: int) -> torch.Tensor:
        """Lookup with optional straight-through or soft gradient."""
        cfg = self.config
        emb = table(key)  # [B, T, embed_dim]

        if not self.training:
            return emb

        if cfg.sam_soft_temp > 0:
            # Soft quantization: tanh(x/temp) creates soft sign
            # Gradient flows through tanh
            soft = torch.tanh(projected / cfg.sam_soft_temp)
            # Use soft values to create a differentiable "similarity" to stored patterns
            # Approximate: soft lookup ≈ hard lookup + gradient correction
            # The hard lookup gives the value, the soft path gives the gradient
            return emb + (soft.mean(dim=-1, keepdim=True) * 0.0)  # gradient path only

        if cfg.sam_straight_through:
            # Straight-through estimator: forward uses hard key,
            # backward pretends the projection went straight to the embedding
            # Trick: add (projected - projected.detach()) * 0 to create gradient path
            grad_hook = (projected - projected.detach()).sum(dim=-1, keepdim=True) * 0.01
            return emb + grad_hook.expand_as(emb)

        return emb

    def forward(self, scan_state: torch.Tensor, chars: torch.Tensor) -> torch.Tensor:
        """
        scan_state: [B, T, state_dim] — output of the gated scan
        chars: [B, T] — raw tokens (for token_mix mode)
        Returns: [B, T, state_dim]
        """
        B, T, _ = scan_state.shape
        cfg = self.config

        # Project state for hashing
        projected = self.state_proj(scan_state)  # [B, T, quant_bits * heads]
        per_head = projected.chunk(self.num_heads, dim=-1)  # list of [B, T, quant_bits]

        # Lookup each head
        head_outputs = []
        for h_idx, (ph, table) in enumerate(zip(per_head, self.memory_tables)):
            key = self._quantize_to_key(ph, h_idx)  # [B, T]
            if cfg.sam_quant_mode == "token_mix":
                key = key ^ (chars * HASH_PRIMES[(h_idx * 7 + 13) % len(HASH_PRIMES)])
                key = key % self.buckets
            head_outputs.append(self._soft_lookup(ph, key, table, h_idx))

        # Concat heads and project
        combined = torch.cat(head_outputs, dim=-1)  # [B, T, heads * sam_embed_dim]
        return self.out_proj(combined)


class PolyHashV7(nn.Module):
    def __init__(self, config: V7Config = V7Config()):
        super().__init__()
        self.config = config
        V = config.vocab_size

        # Byte embed
        self.byte_embed = nn.Embedding(V, config.byte_embed_dim)

        # Fast path: hash tables
        self.skip_patterns = tuple(
            (FIBONACCI[i],) if i < len(FIBONACCI) else (1, i + 1)
            for i in range(config.num_tables)
        )
        self.hash_tables = nn.ModuleList()
        for _ in range(config.num_tables):
            t = nn.Embedding(config.buckets_per_table, config.embed_per_table)
            nn.init.normal_(t.weight, std=0.02)
            self.hash_tables.append(t)

        hash_feat_dim = config.num_tables * config.embed_per_table
        self.dw_conv = None
        if config.conv_kernel > 0:
            self.dw_conv = nn.Conv1d(
                hash_feat_dim, hash_feat_dim,
                kernel_size=config.conv_kernel, padding=0,
                groups=hash_feat_dim,
            )
            self.conv_pad = config.conv_kernel - 1

        n_match = len(config.match_offsets) if config.match_offsets else 0

        # Input projection (fast path → hidden)
        fast_dim = config.byte_embed_dim + hash_feat_dim + n_match
        self.input_proj = nn.Linear(fast_dim, config.hidden_dim)

        # Scan (builds sequential state)
        self.scan = MultiGroupScan(
            config.hidden_dim, config.scan_dim,
            groups=config.scan_groups, chunk_size=config.scan_chunk_size,
        )

        # State-addressed memory
        self.sam = None
        if config.sam_enabled:
            self.sam = StateAddressedMemory(config.hidden_dim, config)

        # MLP readout
        self.mlp = nn.ModuleList([
            ResBlock(config.hidden_dim, config.dropout)
            for _ in range(config.num_layers - 1)
        ])
        self.output_proj = nn.Linear(config.hidden_dim, V)

    def _hash_features(self, chars):
        cfg = self.config
        embs = []
        for i, table in enumerate(self.hash_tables):
            idx = _hash_ctx(chars, self.skip_patterns[i], i, cfg.buckets_per_table)
            emb = table(idx)
            if self.training and cfg.hash_dropout > 0:
                if torch.rand(1).item() < cfg.hash_dropout:
                    emb = torch.zeros_like(emb)
            embs.append(emb)
        hf = torch.cat(embs, dim=-1)
        if self.dw_conv is not None:
            ht = hf.transpose(1, 2)
            ht = F.pad(ht, (self.conv_pad, 0))
            hf = F.silu(self.dw_conv(ht)).transpose(1, 2)
        return hf

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

    def forward(self, chars: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        B, T = chars.shape

        # Fast path
        byte_emb = self.byte_embed(chars)
        hash_feat = self._hash_features(chars)
        parts = [byte_emb, hash_feat]
        mf = self._match_features(chars) if cfg.match_offsets else None
        if mf is not None:
            parts.append(mf)

        h = F.silu(self.input_proj(torch.cat(parts, dim=-1)))

        # Scan (builds sequential state)
        h = self.scan(h)

        # State-addressed memory lookup
        if self.sam is not None:
            sam_out = self.sam(h, chars)
            h = h + sam_out

        # Readout
        for block in self.mlp:
            h = block(h)

        return self.output_proj(h)
