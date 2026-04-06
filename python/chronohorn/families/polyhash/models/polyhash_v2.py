"""PolyHash v2: extended architecture with gated mixture, FiLM, scan, pyramid, MoE.

All new features are toggled via config flags — the base PolyHash behavior is
preserved when all extensions are off.
"""
from __future__ import annotations

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
    3674653441, 2860486319, 1073676293, 2971215077,
    1500450281, 3267000017, 2654435801, 4049292743,
    2246822537, 3266489939, 2028178531, 1220703143,
    1610612753, 805306467, 402653201, 3674653447,
]


@dataclass(frozen=True)
class PolyHashV2Config:
    vocab_size: int = 1024
    byte_embed_dim: int = 128
    num_tables: int = 16
    buckets_per_table: int = 8192
    embed_per_table: int = 16
    hidden_dim: int = 512
    num_layers: int = 2
    use_residual: bool = True
    use_layer_norm: bool = True
    max_seq_len: int = 512
    init_seed: int = 42
    dropout: float = 0.0

    # --- Skip pattern geometry ---
    skip_pattern_mode: str = "default"  # default, stratified, exponential, dense_near, fibonacci
    max_offset: int = 8                 # max lookback offset for pattern generation

    # --- Gated hash mixture (Axis A) ---
    gate_mode: str = "none"    # none, static, byte, hash, deep
    gate_topk: int = 0         # 0=no topk, >0=keep top-k gates

    # --- Feature interaction (Axis E) ---
    film: bool = False         # FiLM conditioning: byte_embed modulates hash features

    # --- Collision reduction (Axis C) ---
    multi_hash: int = 1        # 1=normal, 2+=multi-hash per table
    multi_hash_pool: str = "mean"  # mean, max
    exact_unigram: bool = False    # dedicate first table to exact unigram (0 collision)

    # --- Recurrence (Axis F) ---
    scan_dim: int = 0          # 0=no scan, >0=linear scan with this state dim
    scan_mode: str = "gated"   # gated, ema

    # --- Pyramid (Axis D) ---
    pyramid_levels: int = 0    # 0=no pyramid, 2+=hierarchical hash levels
    pyramid_quant: str = "sign"  # sign, topk

    # --- MLP (Axis G) ---
    activation: str = "relu"   # relu, gelu, swiglu
    num_experts: int = 1       # 1=normal, 2+=mixture of experts

    # --- Training (Axis H) ---
    hash_dropout: float = 0.0  # randomly zero out hash tables during training
    label_smoothing: float = 0.0


def _generate_patterns(mode: str, num_tables: int, max_offset: int) -> tuple:
    """Generate skip patterns based on mode."""
    if mode == "stratified":
        local = [(i,) for i in range(1, min(max_offset + 1, num_tables // 2 + 1))]
        medium = [(i, i * 2) for i in range(2, min(max_offset + 1, 6))]
        far = [(1, i) for i in range(max_offset, max_offset * 4 + 1, max_offset)]
        patterns = local + medium + far
    elif mode == "exponential":
        # One unigram per power-of-2 offset, rest are bigram combos
        singles = [(2**i,) for i in range(min(8, num_tables // 2))]
        combos = [(1, 2**i) for i in range(1, min(8, num_tables))]
        patterns = singles + combos
    elif mode == "dense_near":
        # All offset-1 and offset-2 combos: maximize local coverage
        patterns = []
        for i in range(1, max_offset + 1):
            patterns.append((i,))
        for i in range(1, max_offset):
            for j in range(i + 1, max_offset + 1):
                patterns.append((i, j))
    elif mode == "fibonacci":
        fibs = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
        singles = [(f,) for f in fibs[:num_tables // 2]]
        combos = [(1, f) for f in fibs[1:]]
        patterns = singles + combos
    else:
        # Default: same as v1
        patterns = []
        for offset in range(1, min(num_tables // 4 + 1, 9)):
            patterns.append((offset,))
        pairs = [(1, 2), (2, 3), (3, 4), (1, 3), (2, 4), (1, 4),
                 (1, 5), (2, 5), (3, 5), (1, 6), (2, 6), (1, 7)]
        for p in pairs:
            if len(patterns) >= num_tables:
                break
            patterns.append(p)
        trigrams = [(1, 2, 3), (1, 2, 4), (1, 3, 5), (2, 3, 4)]
        for t in trigrams:
            if len(patterns) >= num_tables:
                break
            patterns.append(t)
        offset = 8
        while len(patterns) < num_tables:
            patterns.append((1, offset))
            offset += 1
    return tuple(patterns[:num_tables])


def _hash_context(tokens: torch.Tensor, pattern: tuple, table_idx: int,
                  buckets: int, hash_seed: int = 0) -> torch.Tensor:
    """Compute hash indices for a context pattern. Pure function."""
    batch, seq = tokens.shape
    h = torch.zeros(batch, seq, dtype=torch.long, device=tokens.device)
    for k, offset in enumerate(pattern):
        prime_idx = (table_idx * 3 + k + hash_seed) % len(HASH_PRIMES)
        prime = HASH_PRIMES[prime_idx]
        shifted = torch.zeros_like(tokens)
        if offset < seq:
            shifted[:, offset:] = tokens[:, :-offset] if offset > 0 else tokens
        h = h ^ (shifted * prime)
    return h % buckets


class SwiGLU(nn.Module):
    """SwiGLU activation: out = swish(W1 x) * (W2 x)."""
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(in_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.w1(x)) * self.w2(x)


class ResBlock(nn.Module):
    """Residual block with configurable activation."""
    def __init__(self, dim: int, activation: str = "relu",
                 residual: bool = True, layer_norm: bool = False,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.residual = residual
        if activation == "swiglu":
            self.act = SwiGLU(dim, dim)
        else:
            self.linear = nn.Linear(dim, dim)
            self.act_fn = F.gelu if activation == "gelu" else F.relu
        self.activation = activation
        self.ln = nn.LayerNorm(dim) if layer_norm else None
        self.drop = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "swiglu":
            h = self.act(x)
        else:
            h = self.act_fn(self.linear(x))
        if self.drop is not None:
            h = self.drop(h)
        if self.residual:
            h = h + x
        if self.ln is not None:
            h = self.ln(h)
        return h


class GatedScan(nn.Module):
    """Minimal O(n) linear scan on features."""
    def __init__(self, input_dim: int, state_dim: int, mode: str = "gated") -> None:
        super().__init__()
        self.state_dim = state_dim
        self.mode = mode
        if mode == "gated":
            self.gate_proj = nn.Linear(input_dim, state_dim)
            self.input_proj = nn.Linear(input_dim, state_dim)
            self.output_proj = nn.Linear(state_dim, input_dim)
        else:  # ema
            self.decay = nn.Parameter(torch.full((state_dim,), 0.9))
            self.input_proj = nn.Linear(input_dim, state_dim)
            self.output_proj = nn.Linear(state_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq, input_dim] -> [batch, seq, input_dim]"""
        batch, seq, _ = x.shape
        inp = self.input_proj(x)  # [B, T, state_dim]

        if self.mode == "gated":
            gate = torch.sigmoid(self.gate_proj(x))  # [B, T, state_dim]
            states = torch.zeros(batch, self.state_dim, device=x.device)
            out_states = []
            for t in range(seq):
                states = gate[:, t] * states + (1.0 - gate[:, t]) * inp[:, t]
                out_states.append(states)
            scan_out = torch.stack(out_states, dim=1)
        else:  # ema
            decay = torch.sigmoid(self.decay)
            states = torch.zeros(batch, self.state_dim, device=x.device)
            out_states = []
            for t in range(seq):
                states = decay * states + (1.0 - decay) * inp[:, t]
                out_states.append(states)
            scan_out = torch.stack(out_states, dim=1)

        return x + self.output_proj(scan_out)


class MoELayer(nn.Module):
    """Simple top-1 mixture of experts."""
    def __init__(self, dim: int, num_experts: int, activation: str = "relu",
                 layer_norm: bool = False) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.router = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([
            ResBlock(dim, activation=activation, residual=True, layer_norm=layer_norm)
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, dim]
        logits = self.router(x)  # [B, T, E]
        weights = F.softmax(logits, dim=-1)  # [B, T, E]
        # Weighted sum of expert outputs
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            out = out + weights[..., i:i+1] * expert(x)
        return out


class PolyHashV2(nn.Module):
    """PolyHash v2 with all extension axes."""

    def __init__(self, config: PolyHashV2Config = PolyHashV2Config()) -> None:
        super().__init__()
        self.config = config
        V = config.vocab_size

        # Skip patterns
        self.skip_patterns = _generate_patterns(
            config.skip_pattern_mode, config.num_tables, config.max_offset
        )

        # Byte embedding
        self.byte_embed = nn.Embedding(V, config.byte_embed_dim)

        # Hash embedding tables
        n_hash_tables = config.num_tables
        if config.exact_unigram:
            self.unigram_table = nn.Embedding(V, config.embed_per_table)
            nn.init.normal_(self.unigram_table.weight, std=0.02)
            n_hash_tables -= 1
        else:
            self.unigram_table = None

        self.hash_tables = nn.ModuleList()
        for _ in range(n_hash_tables):
            table = nn.Embedding(config.buckets_per_table, config.embed_per_table)
            nn.init.normal_(table.weight, std=0.02)
            self.hash_tables.append(table)

        # Gated mixture
        hash_feature_dim = config.num_tables * config.embed_per_table
        if config.gate_mode == "static":
            self.gate_weight = nn.Parameter(torch.ones(config.num_tables))
        elif config.gate_mode == "byte":
            self.gate_proj = nn.Linear(config.byte_embed_dim, config.num_tables)
        elif config.gate_mode == "hash":
            self.gate_proj = nn.Linear(
                config.byte_embed_dim + config.embed_per_table, config.num_tables
            )
        elif config.gate_mode == "deep":
            self.gate_net = nn.Sequential(
                nn.Linear(config.byte_embed_dim, 64),
                nn.ReLU(),
                nn.Linear(64, config.num_tables),
            )

        # FiLM conditioning
        if config.film:
            self.film_scale = nn.Linear(config.byte_embed_dim, hash_feature_dim)
            self.film_shift = nn.Linear(config.byte_embed_dim, hash_feature_dim)

        # Feature dim after all concatenation
        feature_dim = config.byte_embed_dim + hash_feature_dim

        # Input projection
        self.input_proj = nn.Linear(feature_dim, config.hidden_dim)

        # Pyramid hash tables (extra tables for hierarchical levels)
        self.pyramid_tables = nn.ModuleList()
        if config.pyramid_levels >= 2:
            for level in range(1, config.pyramid_levels):
                n_pyr = max(4, config.num_tables // 2)
                level_tables = nn.ModuleList()
                for _ in range(n_pyr):
                    t = nn.Embedding(config.buckets_per_table, config.embed_per_table)
                    nn.init.normal_(t.weight, std=0.02)
                    level_tables.append(t)
                self.pyramid_tables.append(level_tables)
            # Adjust input projection for pyramid features
            pyr_extra = sum(
                max(4, config.num_tables // 2) * config.embed_per_table
                for _ in range(1, config.pyramid_levels)
            )
            self.input_proj = nn.Linear(feature_dim + pyr_extra, config.hidden_dim)

        # Scan (O(n) recurrence)
        self.scan = None
        if config.scan_dim > 0:
            self.scan = GatedScan(config.hidden_dim, config.scan_dim, config.scan_mode)

        # MLP readout
        if config.num_experts > 1:
            self.mlp = nn.ModuleList([
                MoELayer(config.hidden_dim, config.num_experts,
                         activation=config.activation, layer_norm=config.use_layer_norm)
                for _ in range(config.num_layers - 1)
            ])
        else:
            self.mlp = nn.ModuleList([
                ResBlock(config.hidden_dim, activation=config.activation,
                         residual=config.use_residual, layer_norm=config.use_layer_norm,
                         dropout=config.dropout)
                for _ in range(config.num_layers - 1)
            ])

        self.output_proj = nn.Linear(config.hidden_dim, V)
        self._init_weights()

    def _init_weights(self) -> None:
        rng = torch.Generator().manual_seed(self.config.init_seed)
        nn.init.xavier_uniform_(self.input_proj.weight, generator=rng)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight, generator=rng)
        nn.init.zeros_(self.output_proj.bias)

    def _lookup_hash_features(self, chars: torch.Tensor) -> list[torch.Tensor]:
        """Look up all hash table features. Returns list of [B, T, E] tensors."""
        cfg = self.config
        features = []

        table_offset = 0
        if cfg.exact_unigram:
            features.append(self.unigram_table(chars))
            table_offset = 0  # hash_tables start at index 0, patterns skip first

        for i, table in enumerate(self.hash_tables):
            pat_idx = i + (1 if cfg.exact_unigram else 0)
            if pat_idx >= len(self.skip_patterns):
                pat_idx = i % len(self.skip_patterns)
            pattern = self.skip_patterns[pat_idx]

            if cfg.multi_hash > 1:
                # Multiple hashes, pool
                embeds = []
                for seed in range(cfg.multi_hash):
                    idx = _hash_context(chars, pattern, pat_idx, cfg.buckets_per_table, hash_seed=seed * 7)
                    embeds.append(table(idx))
                stacked = torch.stack(embeds, dim=0)
                if cfg.multi_hash_pool == "max":
                    emb = stacked.max(dim=0).values
                else:
                    emb = stacked.mean(dim=0)
            else:
                idx = _hash_context(chars, pattern, pat_idx, cfg.buckets_per_table)
                emb = table(idx)

            features.append(emb)
        return features

    def _apply_gate(self, byte_emb: torch.Tensor, hash_features: list[torch.Tensor]) -> torch.Tensor:
        """Apply gated mixture to hash features. Returns [B, T, num_tables * E]."""
        cfg = self.config
        stacked = torch.stack(hash_features, dim=-2)  # [B, T, num_tables, E]

        if cfg.gate_mode == "none":
            return stacked.reshape(*stacked.shape[:2], -1)

        # Compute gate weights
        if cfg.gate_mode == "static":
            weights = F.softmax(self.gate_weight, dim=0)  # [num_tables]
            weights = weights.view(1, 1, -1, 1)
        elif cfg.gate_mode == "byte":
            weights = F.softmax(self.gate_proj(byte_emb), dim=-1)  # [B, T, num_tables]
            weights = weights.unsqueeze(-1)  # [B, T, num_tables, 1]
        elif cfg.gate_mode == "hash":
            hash_mean = stacked.mean(dim=-2)  # [B, T, E]
            gate_input = torch.cat([byte_emb, hash_mean], dim=-1)
            weights = F.softmax(self.gate_proj(gate_input), dim=-1)
            weights = weights.unsqueeze(-1)
        elif cfg.gate_mode == "deep":
            weights = F.softmax(self.gate_net(byte_emb), dim=-1)
            weights = weights.unsqueeze(-1)
        else:
            return stacked.reshape(*stacked.shape[:2], -1)

        # Apply top-k sparsity
        if cfg.gate_topk > 0 and cfg.gate_mode != "static":
            w_squeezed = weights.squeeze(-1)  # [B, T, num_tables]
            topk_vals, topk_idx = w_squeezed.topk(cfg.gate_topk, dim=-1)
            mask = torch.zeros_like(w_squeezed)
            mask.scatter_(-1, topk_idx, 1.0)
            weights = (w_squeezed * mask).unsqueeze(-1)
            # Re-normalize
            weights = weights / (weights.sum(dim=-2, keepdim=True) + 1e-8)

        gated = stacked * weights  # [B, T, num_tables, E]
        return gated.reshape(*stacked.shape[:2], -1)

    def _apply_film(self, byte_emb: torch.Tensor, hash_features: torch.Tensor) -> torch.Tensor:
        """FiLM conditioning: byte_embed modulates hash features."""
        scale = torch.sigmoid(self.film_scale(byte_emb))
        shift = self.film_shift(byte_emb)
        return hash_features * scale + shift

    def _pyramid_features(self, chars: torch.Tensor,
                          base_features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Compute hierarchical pyramid hash features."""
        cfg = self.config
        all_extra = []
        prev_features = base_features

        for level_idx, level_tables in enumerate(self.pyramid_tables):
            scale = 4 * (2 ** level_idx)  # offset multiplier: 4, 8, 16...
            level_feats = []
            # Quantize previous features to produce hash keys
            stacked = torch.stack(prev_features, dim=-2)  # [B, T, N, E]
            if cfg.pyramid_quant == "sign":
                # Sign quantization: each dim -> 0 or 1
                quant = (stacked.mean(dim=-2) > 0).long()  # [B, T, E]
                # Pack binary into pseudo-token: sum of bit positions
                quant_key = (quant * torch.arange(1, quant.shape[-1] + 1, device=quant.device)).sum(dim=-1)  # [B, T]
            else:  # topk
                quant_key = stacked.mean(dim=-2).topk(3, dim=-1).indices.sum(dim=-1)

            for t_idx, table in enumerate(level_tables):
                offset = scale * (t_idx + 1)
                pattern = (offset,) if t_idx % 2 == 0 else (offset, offset * 2)
                idx = _hash_context(quant_key, pattern, t_idx + 100 * (level_idx + 1),
                                    cfg.buckets_per_table)
                level_feats.append(table(idx))
            all_extra.extend(level_feats)
            prev_features = level_feats

        return all_extra

    def forward(self, chars: torch.Tensor) -> torch.Tensor:
        """chars: [batch, seq] long -> [batch, seq, vocab] logits."""
        cfg = self.config

        # Byte embedding
        byte_emb = self.byte_embed(chars)

        # Hash features
        hash_features = self._lookup_hash_features(chars)

        # Hash dropout (training only)
        if self.training and cfg.hash_dropout > 0:
            for i in range(len(hash_features)):
                if torch.rand(1).item() < cfg.hash_dropout:
                    hash_features[i] = torch.zeros_like(hash_features[i])

        # Gate
        gated_hash = self._apply_gate(byte_emb, hash_features)

        # FiLM
        if cfg.film:
            gated_hash = self._apply_film(byte_emb, gated_hash)

        # Pyramid
        parts = [byte_emb, gated_hash]
        if cfg.pyramid_levels >= 2:
            pyr_feats = self._pyramid_features(chars, hash_features)
            parts.extend(pyr_feats)

        features = torch.cat(parts, dim=-1)

        # Input projection
        h = F.relu(self.input_proj(features))

        # Scan (O(n) recurrence, optional)
        if self.scan is not None:
            h = self.scan(h)

        # MLP
        for block in self.mlp:
            h = block(h)

        return self.output_proj(h)
