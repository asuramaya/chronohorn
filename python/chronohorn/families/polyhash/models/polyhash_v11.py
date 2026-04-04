"""PolyHash v11: Million-bucket architecture.

The oracle says: exact 4-gram = 0.27 bpb. Our 65K-bucket tables = 1.64 bpb.
The gap is collision noise, not architecture. The fix: millions of buckets
with tiny embeddings instead of thousands of buckets with rich embeddings.

Variable: buckets_per_scale × embed_per_scale = constant memory budget.
More buckets = less collision = better discrimination.
Fewer dims = less expressive per bucket = needs MLP to compensate.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

HASH_PRIMES = [
    2654435761, 2246822519, 3266489917, 2028178513,
    1220703125, 1610612741, 805306457, 402653189,
    3674653429, 2860486313, 1073676287, 2971215073,
    1500450271, 3267000013, 2654435789, 4049292737,
]

SCALE_WINDOWS = [1, 2, 4, 8, 16, 32, 64, 128]


@dataclass(frozen=True)
class V11Config:
    vocab_size: int = 1024
    byte_embed_dim: int = 64
    # Hash tables — the key variable
    num_scales: int = 4
    buckets_per_scale: int = 2_000_000
    embed_per_scale: int = 2
    # Scan
    scan_dim: int = 32
    scan_chunk_size: int = 32
    # MLP
    hidden_dim: int = 256
    num_layers: int = 2
    # Local features
    conv_kernel: int = 8
    match_offsets: tuple = (1, 2, 3, 4, 5, 6, 7, 8)
    # Conditioning
    condition_long_on_short: bool = True
    condition_boundary: int = 2
    sign_bits: int = 8
    # Training
    dropout: float = 0.0
    max_seq_len: int = 512
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


class TinyScan(nn.Module):
    def __init__(self, input_dim, scan_dim, chunk_size=32):
        super().__init__()
        self.scan_dim = scan_dim
        self.chunk_size = chunk_size
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


class MillionBucketPyramid(nn.Module):
    """Hash pyramid with millions of buckets and tiny embeddings."""

    def __init__(self, config: V11Config):
        super().__init__()
        self.config = config
        ns = config.num_scales
        B = config.buckets_per_scale
        E = config.embed_per_scale

        self.tables = nn.ModuleList()
        for _ in range(ns):
            t = nn.Embedding(B, E)
            nn.init.normal_(t.weight, std=0.02)
            self.tables.append(t)

        # Register primes per scale
        for s in range(ns):
            w = SCALE_WINDOWS[s]
            primes = HASH_PRIMES[:w] if w <= len(HASH_PRIMES) else (HASH_PRIMES * (w // len(HASH_PRIMES) + 1))[:w]
            self.register_buffer(f"primes_{s}", torch.tensor(primes, dtype=torch.long))

        # Conditioning
        self.condition = config.condition_long_on_short
        self.boundary = min(config.condition_boundary, ns)
        if self.condition and self.boundary < ns:
            short_dim = self.boundary * E
            self.cond_proj = nn.Linear(short_dim, config.sign_bits, bias=False)
            self.register_buffer("cond_primes",
                torch.tensor(HASH_PRIMES[:config.sign_bits], dtype=torch.long))

    def _hash_scale(self, tokens, scale_idx):
        B_dim, T = tokens.shape
        window = SCALE_WINDOWS[scale_idx]
        primes = getattr(self, f"primes_{scale_idx}")
        h = torch.zeros(B_dim, T, dtype=torch.long, device=tokens.device)
        for i in range(window):
            offset = i + 1
            if offset >= T: continue
            shifted = torch.zeros_like(tokens)
            shifted[:, offset:] = tokens[:, :-offset]
            h = h ^ (shifted * primes[i])
        return h % self.config.buckets_per_scale

    def forward(self, tokens):
        cfg = self.config
        B_dim, T = tokens.shape
        ns = cfg.num_scales

        keys = [self._hash_scale(tokens, s) for s in range(ns)]

        # Short-range lookups
        short_embeds = []
        for s in range(self.boundary):
            emb = self.tables[s](keys[s])
            if self.training and cfg.hash_dropout > 0 and torch.rand(1).item() < cfg.hash_dropout:
                emb = torch.zeros_like(emb)
            short_embeds.append(emb)

        # Condition long-range
        if self.condition and self.boundary < ns:
            short_cat = torch.cat(short_embeds, dim=-1)
            cond_logits = self.cond_proj(short_cat)
            sign_bits = (cond_logits > 0).long()
            cond_key = torch.zeros(B_dim, T, dtype=torch.long, device=tokens.device)
            for i in range(cfg.sign_bits):
                cond_key = cond_key ^ (sign_bits[:, :, i] * self.cond_primes[i])
            for s in range(self.boundary, ns):
                keys[s] = (keys[s] ^ cond_key) % cfg.buckets_per_scale

        # Long-range lookups
        long_embeds = []
        for s in range(self.boundary, ns):
            emb = self.tables[s](keys[s])
            if self.training and cfg.hash_dropout > 0 and torch.rand(1).item() < cfg.hash_dropout:
                emb = torch.zeros_like(emb)
            long_embeds.append(emb)

        all_embeds = short_embeds + long_embeds
        return torch.cat(all_embeds, dim=-1)  # [B, T, ns * E]


class PolyHashV11(nn.Module):
    def __init__(self, config: V11Config = V11Config()):
        super().__init__()
        self.config = config
        V = config.vocab_size

        self.byte_embed = nn.Embedding(V, config.byte_embed_dim)
        self.pyramid = MillionBucketPyramid(config)

        hash_feat_dim = config.num_scales * config.embed_per_scale
        self.dw_conv = None
        if config.conv_kernel > 0:
            self.dw_conv = nn.Conv1d(hash_feat_dim, hash_feat_dim,
                kernel_size=config.conv_kernel, padding=0, groups=hash_feat_dim)
            self.conv_pad = config.conv_kernel - 1

        n_match = len(config.match_offsets) if config.match_offsets else 0
        feat_dim = config.byte_embed_dim + hash_feat_dim + n_match

        self.scan = None
        if config.scan_dim > 0:
            self.scan = TinyScan(feat_dim, config.scan_dim, config.scan_chunk_size)

        self.input_proj = nn.Linear(feat_dim, config.hidden_dim) if config.hidden_dim > 0 else None
        self.mlp = nn.ModuleList([
            ResBlock(config.hidden_dim, config.dropout)
            for _ in range(config.num_layers - 1)
        ]) if config.hidden_dim > 0 and config.num_layers > 1 else nn.ModuleList()

        out_dim = config.hidden_dim if config.hidden_dim > 0 else feat_dim
        self.output_proj = nn.Linear(out_dim, V)

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
        hash_feat = self.pyramid(chars)

        if self.dw_conv is not None:
            ht = hash_feat.transpose(1, 2)
            ht = F.pad(ht, (self.conv_pad, 0))
            hash_feat = F.silu(self.dw_conv(ht)).transpose(1, 2)

        parts = [byte_emb, hash_feat]
        mf = self._match_features(chars) if cfg.match_offsets else None
        if mf is not None:
            parts.append(mf)

        h = torch.cat(parts, dim=-1)

        if self.scan is not None:
            h = self.scan(h)

        if self.input_proj is not None:
            h = F.silu(self.input_proj(h))
            for block in self.mlp:
                h = block(h)

        return self.output_proj(h)
