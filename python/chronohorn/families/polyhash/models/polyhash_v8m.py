"""PolyHash v8m: Multi-hash Decepticon — collision reduction via redundant hashing.

Instead of 1 hash × 65K buckets per scale, use K independent hash functions
× (65K/K) buckets each. Same total params. The K embeddings per scale are
concatenated, giving the MLP K views of each context — collisions in one hash
are unlikely to collide in all K.

Also incorporates the v8 ablation finding: conditioning helps, scale attention hurts.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from chronohorn.families.polyhash.models.polyhash_v8 import (
    HASH_PRIMES,
    SCALE_WINDOWS,
    ResBlock,
    V8Config,
)

# Extra prime families for multi-hash (each hash function needs independent primes)
MULTI_HASH_SALTS = [0x0, 0xDEADBEEF, 0xCAFEBABE, 0x8BADF00D, 0xFEEDFACE, 0xC0FFEE, 0xBAADF00D, 0xDEADC0DE]


@dataclass(frozen=True)
class V8MConfig(V8Config):
    hashes_per_scale: int = 4
    scale_attention: bool = False  # ablation showed this hurts


class MultiHashPyramid(nn.Module):
    """Multi-hash pyramid: K independent hash functions per scale."""

    def __init__(self, config: V8MConfig):
        super().__init__()
        self.config = config
        ns = config.num_scales
        K = config.hashes_per_scale
        # Buckets per individual hash table (split budget)
        self.buckets_each = config.buckets_per_scale // K
        E = config.embed_per_scale

        # K tables per scale
        self.tables = nn.ModuleList()
        for s in range(ns):
            for k in range(K):
                t = nn.Embedding(self.buckets_each, E)
                nn.init.normal_(t.weight, std=0.02)
                self.tables.append(t)

        # Register primes for each (scale, hash_idx) pair
        for s in range(ns):
            w = SCALE_WINDOWS[s]
            base_primes = HASH_PRIMES[s][:w]
            for k in range(K):
                salt = MULTI_HASH_SALTS[k % len(MULTI_HASH_SALTS)]
                salted = [p ^ salt for p in base_primes]
                self.register_buffer(f"primes_{s}_{k}", torch.tensor(salted, dtype=torch.long))

        # Conditioning (from v8: short-range sign bits refine long-range keys)
        self.condition = config.condition_long_on_short
        self.boundary = config.condition_boundary
        if self.condition and self.boundary < ns:
            short_dim = config.condition_boundary * K * E  # K embeddings per short scale
            self.cond_proj = nn.Linear(short_dim, config.sign_bits, bias=False)
            self.register_buffer(
                "cond_primes",
                torch.tensor([2654435761, 2246822519, 3266489917, 2028178513,
                              1220703125, 1610612741, 805306457, 402653189][:config.sign_bits],
                             dtype=torch.long)
            )

    def _hash_one(self, tokens: torch.Tensor, scale_idx: int, hash_idx: int) -> torch.Tensor:
        B, T = tokens.shape
        window = SCALE_WINDOWS[scale_idx]
        primes = getattr(self, f"primes_{scale_idx}_{hash_idx}")
        h = torch.zeros(B, T, dtype=torch.long, device=tokens.device)
        for i in range(window):
            offset = i + 1
            if offset >= T:
                continue
            shifted = torch.zeros_like(tokens)
            shifted[:, offset:] = tokens[:, :-offset]
            h = h ^ (shifted * primes[i])
        return h % self.buckets_each

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        B, T = tokens.shape
        ns = cfg.num_scales
        K = cfg.hashes_per_scale
        E = cfg.embed_per_scale

        # Stage 1: hash all (scale, hash_idx) pairs
        all_keys = {}
        for s in range(ns):
            for k in range(K):
                all_keys[(s, k)] = self._hash_one(tokens, s, k)

        # Stage 2: lookup short-range (0..boundary-1)
        short_embeds = []  # list of [B, T, E] — K per scale
        for s in range(self.boundary):
            for k in range(K):
                table_idx = s * K + k
                emb = self.tables[table_idx](all_keys[(s, k)])
                if self.training and cfg.hash_dropout > 0 and torch.rand(1).item() < cfg.hash_dropout:
                    emb = torch.zeros_like(emb)
                short_embeds.append(emb)

        # Stage 3: condition long-range keys
        if self.condition and self.boundary < ns:
            short_cat = torch.cat(short_embeds, dim=-1)
            cond_logits = self.cond_proj(short_cat)
            sign_bits = (cond_logits > 0).long()
            cond_key = torch.zeros(B, T, dtype=torch.long, device=tokens.device)
            for i in range(cfg.sign_bits):
                cond_key = cond_key ^ (sign_bits[:, :, i] * self.cond_primes[i])
            for s in range(self.boundary, ns):
                for k in range(K):
                    all_keys[(s, k)] = (all_keys[(s, k)] ^ cond_key) % self.buckets_each

        # Stage 4: lookup long-range
        long_embeds = []
        for s in range(self.boundary, ns):
            for k in range(K):
                table_idx = s * K + k
                emb = self.tables[table_idx](all_keys[(s, k)])
                if self.training and cfg.hash_dropout > 0 and torch.rand(1).item() < cfg.hash_dropout:
                    emb = torch.zeros_like(emb)
                long_embeds.append(emb)

        # Concatenate: [B, T, ns * K * E]
        all_embeds = short_embeds + long_embeds
        return torch.cat(all_embeds, dim=-1)


class PolyHashV8M(nn.Module):
    def __init__(self, config: V8MConfig = V8MConfig()):
        super().__init__()
        self.config = config
        V = config.vocab_size

        self.byte_embed = nn.Embedding(V, config.byte_embed_dim)
        self.pyramid = MultiHashPyramid(config)

        hash_feat_dim = config.num_scales * config.hashes_per_scale * config.embed_per_scale

        self.dw_conv = None
        if config.conv_kernel > 0:
            self.dw_conv = nn.Conv1d(
                hash_feat_dim, hash_feat_dim,
                kernel_size=config.conv_kernel, padding=0,
                groups=hash_feat_dim,
            )
            self.conv_pad = config.conv_kernel - 1

        n_match = len(config.match_offsets) if config.match_offsets else 0
        feat_dim = config.byte_embed_dim + hash_feat_dim + n_match
        self.input_proj = nn.Linear(feat_dim, config.hidden_dim)

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

    def forward(self, chars: torch.Tensor) -> torch.Tensor:
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

        h = F.silu(self.input_proj(torch.cat(parts, dim=-1)))
        for block in self.mlp:
            h = block(h)
        return self.output_proj(h)
