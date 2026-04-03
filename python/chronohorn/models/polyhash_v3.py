"""PolyHash v3: stacked cheap O(1) features from trading/chaos math.

Non-multiplication features computed per-position in O(1):
  - Match features: binary token[t] == token[t-k]
  - Delta features: token[t] - token[t-k], hashed
  - Local frequency: sliding window token counts
  - Bitwise fingerprints: XOR-shift approximate EMA, used as hash keys
  - Reservoir counts: count-min sketch of local context

All combined with hash tables + SwiGLU MLP.
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
    1610612743, 805306459, 402653191, 3674653433,
    2654435771, 2246822527, 3266489933, 2028178529,
    1220703137, 1610612747, 805306463, 402653197,
]

FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21]


@dataclass(frozen=True)
class PolyHashV3Config:
    vocab_size: int = 1024
    byte_embed_dim: int = 128
    # Hash tables
    num_tables: int = 8
    buckets_per_table: int = 32768
    embed_per_table: int = 16
    # MLP
    hidden_dim: int = 512
    num_layers: int = 2
    activation: str = "swiglu"
    # O(1) feature toggles
    match_offsets: tuple = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 32)
    delta_offsets: tuple = (1, 2, 3, 5, 8)
    delta_embed_dim: int = 8
    delta_buckets: int = 2048
    local_freq_windows: tuple = (4, 8, 16, 32)
    use_fingerprints: bool = True
    fp_shift_rates: tuple = (1, 2, 4)
    fp_buckets: int = 16384
    fp_embed_dim: int = 16
    use_reservoir: bool = True
    reservoir_dim: int = 64
    reservoir_window: int = 8
    # Training
    lr_hash_mult: float = 2.0
    max_seq_len: int = 512
    init_seed: int = 42
    dropout: float = 0.0
    label_smoothing: float = 0.0


class SwiGLU(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.proj = nn.Linear(hidden_dim, in_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.silu(self.w1(x)) * self.w2(x))


class ResBlock(nn.Module):
    def __init__(self, dim: int, activation: str = "swiglu",
                 dropout: float = 0.0) -> None:
        super().__init__()
        if activation == "swiglu":
            self.block = SwiGLU(dim, dim)
        elif activation == "gelu":
            self.block = nn.Sequential(nn.Linear(dim, dim), nn.GELU())
        else:
            self.block = nn.Sequential(nn.Linear(dim, dim), nn.ReLU())
        self.ln = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.block(x)
        if self.drop is not None:
            h = self.drop(h)
        return self.ln(h + x)


def _hash_context(tokens, pattern, table_idx, buckets):
    B, T = tokens.shape
    h = torch.zeros(B, T, dtype=torch.long, device=tokens.device)
    for k, offset in enumerate(pattern):
        pidx = (table_idx * 3 + k) % len(HASH_PRIMES)
        prime = HASH_PRIMES[pidx]
        shifted = torch.zeros_like(tokens)
        if offset < T:
            shifted[:, offset:] = tokens[:, :-offset] if offset > 0 else tokens
        h = h ^ (shifted * prime)
    return h % buckets


class PolyHashV3(nn.Module):
    def __init__(self, config: PolyHashV3Config = PolyHashV3Config()) -> None:
        super().__init__()
        self.config = config
        V = config.vocab_size

        # Byte embedding
        self.byte_embed = nn.Embedding(V, config.byte_embed_dim)

        # Hash tables (fibonacci offsets)
        self.hash_tables = nn.ModuleList()
        for _ in range(config.num_tables):
            t = nn.Embedding(config.buckets_per_table, config.embed_per_table)
            nn.init.normal_(t.weight, std=0.02)
            self.hash_tables.append(t)
        self.skip_patterns = tuple(
            (FIBONACCI[i],) if i < len(FIBONACCI) else (1, i + 1)
            for i in range(config.num_tables)
        )

        # Delta hash tables
        self.delta_tables = nn.ModuleList()
        if config.delta_offsets:
            for _ in range(len(config.delta_offsets)):
                t = nn.Embedding(config.delta_buckets, config.delta_embed_dim)
                nn.init.normal_(t.weight, std=0.02)
                self.delta_tables.append(t)

        # Fingerprint hash tables
        self.fp_tables = nn.ModuleList()
        if config.use_fingerprints:
            for _ in range(len(config.fp_shift_rates)):
                t = nn.Embedding(config.fp_buckets, config.fp_embed_dim)
                nn.init.normal_(t.weight, std=0.02)
                self.fp_tables.append(t)

        # Match embedding (learnable projection of binary match vector)
        n_match = len(config.match_offsets) if config.match_offsets else 0
        n_delta = len(config.delta_offsets) * config.delta_embed_dim if config.delta_offsets else 0
        n_freq = len(config.local_freq_windows) if config.local_freq_windows else 0
        n_fp = len(config.fp_shift_rates) * config.fp_embed_dim if config.use_fingerprints else 0
        n_reservoir = config.reservoir_dim if config.use_reservoir else 0

        feature_dim = (
            config.byte_embed_dim
            + config.num_tables * config.embed_per_table
            + n_match
            + n_delta
            + n_freq
            + n_fp
            + n_reservoir
        )

        # Input projection + MLP
        self.input_proj = nn.Linear(feature_dim, config.hidden_dim)
        self.mlp = nn.ModuleList([
            ResBlock(config.hidden_dim, activation=config.activation,
                     dropout=config.dropout)
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

    def _match_features(self, tokens: torch.Tensor) -> torch.Tensor:
        """Binary: does token[t] == token[t-k]? [B, T, num_offsets]"""
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
        return torch.stack(feats, dim=-1)

    def _delta_features(self, tokens: torch.Tensor) -> list[torch.Tensor]:
        """Hash of token[t] - token[t-k] into learned embedding."""
        B, T = tokens.shape
        feats = []
        for i, k in enumerate(self.config.delta_offsets):
            shifted = torch.zeros_like(tokens)
            if k < T:
                shifted[:, k:] = tokens[:, :-k]
            delta = (tokens - shifted) % self.config.vocab_size
            bucket = (delta * HASH_PRIMES[i % len(HASH_PRIMES)]) % self.config.delta_buckets
            feats.append(self.delta_tables[i](bucket))
        return feats

    def _local_freq_features(self, tokens: torch.Tensor) -> torch.Tensor:
        """Count of token[t] in sliding window of size k. [B, T, num_windows]"""
        B, T = tokens.shape
        feats = []
        for window in self.config.local_freq_windows:
            # One-hot then cumsum for sliding window count
            # Efficient: for each position, count matches in [t-window, t)
            counts = torch.zeros(B, T, device=tokens.device)
            for lag in range(1, min(window + 1, T)):
                shifted = torch.zeros_like(tokens)
                shifted[:, lag:] = tokens[:, :-lag]
                counts += (tokens == shifted).float()
            # Normalize by window size
            counts = counts / window
            feats.append(counts)
        return torch.stack(feats, dim=-1)

    def _fingerprint_features(self, tokens: torch.Tensor) -> list[torch.Tensor]:
        """Bitwise XOR-shift fingerprints as hash keys."""
        B, T = tokens.shape
        feats = []
        for fi, shift_rate in enumerate(self.config.fp_shift_rates):
            # Sequential fingerprint computation
            fp = torch.zeros(B, dtype=torch.long, device=tokens.device)
            all_fp = []
            for t in range(T):
                fp = (fp >> shift_rate) ^ (tokens[:, t] * HASH_PRIMES[fi])
                all_fp.append(fp)
            fp_seq = torch.stack(all_fp, dim=1)  # [B, T]
            bucket_idx = fp_seq % self.config.fp_buckets
            feats.append(self.fp_tables[fi](bucket_idx.abs()))
        return feats

    def _reservoir_features(self, tokens: torch.Tensor) -> torch.Tensor:
        """Count-min sketch of local context. [B, T, reservoir_dim]"""
        B, T = tokens.shape
        cfg = self.config
        sketch = torch.zeros(B, T, cfg.reservoir_dim, device=tokens.device)
        for lag in range(1, min(cfg.reservoir_window + 1, T)):
            shifted = torch.zeros_like(tokens)
            if lag < T:
                shifted[:, lag:] = tokens[:, :-lag]
            # Hash token to reservoir bin
            bin_idx = (shifted * HASH_PRIMES[lag % len(HASH_PRIMES)]) % cfg.reservoir_dim
            # Scatter add 1.0 to the bin
            sketch.scatter_add_(2, bin_idx.unsqueeze(-1).abs(), torch.ones(B, T, 1, device=tokens.device))
        # Normalize
        sketch = sketch / max(cfg.reservoir_window, 1)
        return sketch

    def forward(self, chars: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        B, T = chars.shape

        parts = []

        # 1. Byte embedding
        parts.append(self.byte_embed(chars))

        # 2. Hash tables (fibonacci)
        for i, table in enumerate(self.hash_tables):
            idx = _hash_context(chars, self.skip_patterns[i], i, cfg.buckets_per_table)
            parts.append(table(idx))

        # 3. Match features
        if cfg.match_offsets:
            parts.append(self._match_features(chars))

        # 4. Delta features
        if cfg.delta_offsets:
            parts.extend(self._delta_features(chars))

        # 5. Local frequency
        if cfg.local_freq_windows:
            parts.append(self._local_freq_features(chars))

        # 6. Fingerprints
        if cfg.use_fingerprints:
            parts.extend(self._fingerprint_features(chars))

        # 7. Reservoir
        if cfg.use_reservoir:
            parts.append(self._reservoir_features(chars))

        features = torch.cat(parts, dim=-1)
        h = F.silu(self.input_proj(features))

        for block in self.mlp:
            h = block(h)

        return self.output_proj(h)
