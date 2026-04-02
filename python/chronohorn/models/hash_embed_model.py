"""Hash Embedding Model: learned lookup table replaces recurrent substrate.

No recurrence. No sequential processing. Pure table lookup + MLP.
Context comes from learned hash embeddings indexed by recent tokens.

Architecture:
  byte → embed → concat(embed, hash_embeds...) → MLP → logits

Each hash table maps a context pattern (e.g., last 2 tokens) to a
learned embedding. Multiple tables with different hash functions
capture different context relationships.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Any


HASH_PRIMES = [
    2654435761, 2246822519, 3266489917, 2028178513,
    1220703125, 1610612741, 805306457, 402653189,
]


@dataclass(frozen=True)
class HashEmbedConfig:
    vocab_size: int = 1024
    byte_embed_dim: int = 64
    num_tables: int = 4           # number of independent hash tables
    buckets_per_table: int = 16384
    embed_per_table: int = 32     # embedding dim per table
    hidden_dim: int = 512         # MLP hidden
    num_layers: int = 2           # MLP depth
    context_orders: tuple = (2, 3, 2, 3)  # n-gram order per table
    skip_patterns: tuple = ((1, 2), (1, 2, 3), (1, 3), (2, 3))  # which offsets to hash
    max_seq_len: int = 512
    init_seed: int = 42


class HashEmbedModel(nn.Module):
    """Learned hash embedding model — no recurrence, pure lookup + MLP."""

    def __init__(self, config: HashEmbedConfig = HashEmbedConfig()) -> None:
        super().__init__()
        self.config = config
        V = config.vocab_size

        # Byte embedding
        self.byte_embed = nn.Embedding(V, config.byte_embed_dim)

        # Hash embedding tables
        self.hash_tables = nn.ModuleList()
        for i in range(config.num_tables):
            table = nn.Embedding(config.buckets_per_table, config.embed_per_table)
            nn.init.normal_(table.weight, std=0.02)
            self.hash_tables.append(table)

        # Store hash primes for each table
        self.register_buffer(
            "_hash_primes",
            torch.tensor(HASH_PRIMES[: config.num_tables * 3], dtype=torch.long),
        )

        # MLP readout
        feature_dim = config.byte_embed_dim + config.num_tables * config.embed_per_table
        layers = []
        in_dim = feature_dim
        for _ in range(config.num_layers - 1):
            layers.extend([nn.Linear(in_dim, config.hidden_dim), nn.ReLU()])
            in_dim = config.hidden_dim
        layers.append(nn.Linear(in_dim, V))
        self.readout = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        rng = torch.Generator().manual_seed(self.config.init_seed)
        for m in self.readout:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, generator=rng)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _hash_context(
        self, tokens: torch.Tensor, table_idx: int
    ) -> torch.Tensor:
        """Compute hash indices for a context pattern.

        tokens: [batch, seq] long tensor
        Returns: [batch, seq] long tensor of bucket indices
        """
        batch, seq = tokens.shape
        cfg = self.config
        pattern = cfg.skip_patterns[table_idx] if table_idx < len(cfg.skip_patterns) else (1, 2)
        buckets = cfg.buckets_per_table

        # For each position t, hash tokens at t-offset for each offset in pattern
        h = torch.zeros(batch, seq, dtype=torch.long, device=tokens.device)
        for k, offset in enumerate(pattern):
            prime_idx = (table_idx * 3 + k) % len(HASH_PRIMES)
            prime = HASH_PRIMES[prime_idx]
            # Shifted tokens: t-offset, clamped to 0 for early positions
            shifted = torch.zeros_like(tokens)
            if offset < seq:
                shifted[:, offset:] = tokens[:, :-offset] if offset > 0 else tokens
            h = h ^ (shifted * prime)

        return h % buckets

    def forward(self, chars: torch.Tensor) -> torch.Tensor:
        """
        chars: [batch, seq] long tensor of token ids
        Returns: [batch, seq, vocab] logits
        """
        # Byte embedding
        x = self.byte_embed(chars)  # [batch, seq, byte_embed_dim]

        # Hash embeddings from each table
        parts = [x]
        for i, table in enumerate(self.hash_tables):
            indices = self._hash_context(chars, i)  # [batch, seq]
            emb = table(indices)  # [batch, seq, embed_per_table]
            parts.append(emb)

        # Concatenate all features
        features = torch.cat(parts, dim=-1)  # [batch, seq, feature_dim]

        # MLP readout
        logits = self.readout(features)  # [batch, seq, vocab]
        return logits
