"""PolyHash Model: polysemy-inspired hash embedding architecture.

Informed by Haber & Poesio (2024) polysemy survey findings:
  1. CORELEX insight: many small tables (underspecified type dimensions)
     beat fewer large tables — each table captures a relationship TYPE
  2. Graded similarity: soft hash lookup (lerp K=2 nearest buckets)
     produces continuous representations, not discrete sense selection
  3. Layered specialization: residual MLP lets layers specialize
     (lower=word order, upper=context-specific) like transformer layers

Architecture:
  byte -> thick_embed(256d)
  context -> soft_hash_lookup(N tables, K=2 interpolated buckets)
  features -> residual_MLP(4 layers with skip connections)
  -> logits
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple


# Large primes for hashing — need enough for many tables
HASH_PRIMES = [
    2654435761, 2246822519, 3266489917, 2028178513,
    1220703125, 1610612741, 805306457, 402653189,
    3674653429, 2860486313, 1073676287, 2971215073,
    1500450271, 3267000013, 2654435789, 4049292737,
    2246822531, 3266489927, 2028178519, 1220703133,
    1610612743, 805306459, 402653191, 3674653433,
]


@dataclass(frozen=True)
class PolyHashConfig:
    vocab_size: int = 1024
    byte_embed_dim: int = 256          # thick underspecified core
    num_tables: int = 32               # many small tables (CORELEX)
    buckets_per_table: int = 4096      # moderate buckets per table
    embed_per_table: int = 8           # small embedding per table
    hidden_dim: int = 512              # MLP hidden width
    num_layers: int = 4                # MLP depth (sweet spot from ablation)
    soft_k: int = 2                    # interpolate K nearest hash buckets
    use_residual: bool = True          # residual connections in MLP
    use_layer_norm: bool = False       # layer norm (optional, off by default)
    skip_patterns: tuple = ()          # auto-generated if empty
    max_seq_len: int = 512
    init_seed: int = 42
    dropout: float = 0.0


def _generate_skip_patterns(num_tables: int) -> tuple:
    """Generate diverse skip patterns for hash tables.

    Strategy: cover all 2-gram, 3-gram, and skip-gram patterns
    to maximize context diversity. Each pattern is a tuple of
    offsets to look back from the current position.
    """
    patterns = []

    # Bigrams: (1,), (2,), (3,), (4,)
    for offset in range(1, min(num_tables // 4 + 1, 9)):
        patterns.append((offset,))

    # Dense bigrams: (1,2), (2,3), (3,4), (1,3), (2,4), (1,4)
    pairs = [(1, 2), (2, 3), (3, 4), (1, 3), (2, 4), (1, 4),
             (1, 5), (2, 5), (3, 5), (1, 6), (2, 6), (1, 7)]
    for p in pairs:
        if len(patterns) >= num_tables:
            break
        patterns.append(p)

    # Trigrams if we still need more
    trigrams = [(1, 2, 3), (1, 2, 4), (1, 3, 5), (2, 3, 4),
                (1, 2, 5), (1, 3, 4), (2, 4, 6), (1, 4, 7)]
    for t in trigrams:
        if len(patterns) >= num_tables:
            break
        patterns.append(t)

    # Pad with increasing skip patterns if needed
    offset = 8
    while len(patterns) < num_tables:
        patterns.append((1, offset))
        offset += 1

    return tuple(patterns[:num_tables])


class ResidualBlock(nn.Module):
    """Linear + ReLU + optional residual + optional LayerNorm."""

    def __init__(self, in_dim: int, out_dim: int, residual: bool = True,
                 layer_norm: bool = False, dropout: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.residual = residual and (in_dim == out_dim)
        self.layer_norm = nn.LayerNorm(out_dim) if layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.linear(x))
        if self.dropout is not None:
            h = self.dropout(h)
        if self.residual:
            h = h + x
        if self.layer_norm is not None:
            h = self.layer_norm(h)
        return h


class PolyHashModel(nn.Module):
    """Polysemy-inspired hash embedding model.

    Key differences from HashEmbedModel:
    1. Soft hashing: interpolates K=2 nearest buckets per table
    2. Residual MLP: skip connections for layered specialization
    3. Many small tables: 32 tables x 8-dim > 4 tables x 32-dim
    4. Thick byte embedding: 256-dim underspecified core
    """

    def __init__(self, config: PolyHashConfig = PolyHashConfig()) -> None:
        super().__init__()
        self.config = config
        V = config.vocab_size

        # Generate skip patterns if not provided
        if config.skip_patterns:
            self.skip_patterns = config.skip_patterns
        else:
            self.skip_patterns = _generate_skip_patterns(config.num_tables)

        # Byte embedding — thick underspecified core
        self.byte_embed = nn.Embedding(V, config.byte_embed_dim)

        # Hash embedding tables — many small tables
        self.hash_tables = nn.ModuleList()
        for _ in range(config.num_tables):
            table = nn.Embedding(config.buckets_per_table, config.embed_per_table)
            nn.init.normal_(table.weight, std=0.02)
            self.hash_tables.append(table)

        # Feature projection: concat(byte_embed, hash_embeds) -> hidden_dim
        feature_dim = config.byte_embed_dim + config.num_tables * config.embed_per_table
        self.input_proj = nn.Linear(feature_dim, config.hidden_dim)

        # Residual MLP readout
        layers = []
        for _ in range(config.num_layers - 1):
            layers.append(ResidualBlock(
                config.hidden_dim, config.hidden_dim,
                residual=config.use_residual,
                layer_norm=config.use_layer_norm,
                dropout=config.dropout,
            ))
        self.mlp = nn.ModuleList(layers)
        self.output_proj = nn.Linear(config.hidden_dim, V)

        self._init_weights()

    def _init_weights(self) -> None:
        rng = torch.Generator().manual_seed(self.config.init_seed)
        nn.init.xavier_uniform_(self.input_proj.weight, generator=rng)
        nn.init.zeros_(self.input_proj.bias)
        for block in self.mlp:
            nn.init.xavier_uniform_(block.linear.weight, generator=rng)
            nn.init.zeros_(block.linear.bias)
        nn.init.xavier_uniform_(self.output_proj.weight, generator=rng)
        nn.init.zeros_(self.output_proj.bias)

    def _soft_hash_context(
        self, tokens: torch.Tensor, table_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute soft hash indices for interpolated lookup.

        Returns (indices_lo, indices_hi, frac) where:
          indices_lo, indices_hi: [batch, seq] bucket indices
          frac: [batch, seq] interpolation weight for indices_hi
        """
        batch, seq = tokens.shape
        cfg = self.config
        pattern = self.skip_patterns[table_idx]
        buckets = cfg.buckets_per_table

        # Compute raw hash value (before modulo)
        h = torch.zeros(batch, seq, dtype=torch.long, device=tokens.device)
        for k, offset in enumerate(pattern):
            prime_idx = (table_idx * 3 + k) % len(HASH_PRIMES)
            prime = HASH_PRIMES[prime_idx]
            shifted = torch.zeros_like(tokens)
            if offset < seq:
                shifted[:, offset:] = tokens[:, :-offset] if offset > 0 else tokens
            h = h ^ (shifted * prime)

        # Hard hash — primary bucket
        indices_lo = h % buckets

        if cfg.soft_k <= 1:
            return indices_lo, indices_lo, torch.zeros(batch, seq, device=tokens.device)

        # Second bucket: use a different prime to perturb
        secondary_prime = HASH_PRIMES[(table_idx * 3 + len(pattern)) % len(HASH_PRIMES)]
        h2 = h ^ (h * secondary_prime >> 16)
        indices_hi = h2 % buckets

        # Fractional weight from hash bits — deterministic but
        # pseudo-random, creating smooth interpolation
        frac_bits = ((h >> 3) & 0xFF).float() / 255.0
        # Scale fraction so primary bucket dominates (0.6-1.0 weight)
        frac = frac_bits * 0.4  # secondary gets 0.0 to 0.4 weight

        return indices_lo, indices_hi, frac

    def forward(self, chars: torch.Tensor) -> torch.Tensor:
        """
        chars: [batch, seq] long tensor of token ids
        Returns: [batch, seq, vocab] logits
        """
        # Byte embedding — the underspecified core
        x = self.byte_embed(chars)  # [batch, seq, byte_embed_dim]

        # Hash embeddings from each table with soft lookup
        parts = [x]
        for i, table in enumerate(self.hash_tables):
            idx_lo, idx_hi, frac = self._soft_hash_context(chars, i)
            emb_lo = table(idx_lo)   # [batch, seq, embed_per_table]
            if self.config.soft_k > 1:
                emb_hi = table(idx_hi)
                # Interpolate: (1-frac) * lo + frac * hi
                frac = frac.unsqueeze(-1)  # [batch, seq, 1]
                emb = emb_lo * (1.0 - frac) + emb_hi * frac
            else:
                emb = emb_lo
            parts.append(emb)

        # Concatenate all features
        features = torch.cat(parts, dim=-1)  # [batch, seq, feature_dim]

        # Project to hidden dim
        h = torch.relu(self.input_proj(features))

        # Residual MLP
        for block in self.mlp:
            h = block(h)

        # Output projection
        logits = self.output_proj(h)  # [batch, seq, vocab]
        return logits
