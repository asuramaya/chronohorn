"""PolyHash v5: exotic math — depth recurrence, learned FSM, Hopfield retrieval,
hyperdimensional binding, budget-filling, test-time training.

Built on v4's proven winners: conv8, SwiGLU, lr-hash-2x, fibonacci offsets.
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
class V5Config:
    vocab_size: int = 1024
    byte_embed_dim: int = 128
    num_tables: int = 8
    buckets_per_table: int = 32768
    embed_per_table: int = 16
    hidden_dim: int = 512
    num_layers: int = 2
    conv_kernel: int = 8
    max_seq_len: int = 512
    init_seed: int = 42
    dropout: float = 0.0
    # Match features (proven +0.04 bpb)
    match_offsets: tuple = (1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32)
    # --- Exotic features ---
    depth_recurrence: int = 1         # apply MLP block this many times (weight tying)
    fsm_states: int = 0              # 0=off, >0=learned finite state machine
    fsm_embed_dim: int = 32          # FSM state embedding dimension
    hopfield_steps: int = 0          # 0=off, >0=Hopfield retrieval iterations
    hopfield_patterns: int = 1024    # number of stored patterns
    hopfield_dim: int = 64           # pattern dimension
    hdc_dim: int = 0                 # 0=off, >0=hyperdimensional computing dim
    hdc_orders: int = 4              # n-gram orders for HDC encoding


class SwiGLU(nn.Module):
    def __init__(self, d: int, h: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d, h, bias=False)
        self.w2 = nn.Linear(d, h, bias=False)
        self.proj = nn.Linear(h, d, bias=False)

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
        if self.drop:
            h = self.drop(h)
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


class LearnedFSM(nn.Module):
    """Differentiable finite state machine.
    State transitions via learned embedding lookup.
    transition[state, token] -> next_state (via soft assignment)
    emission[state] -> feature vector
    """
    def __init__(self, num_states: int, vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.num_states = num_states
        # Transition: for each (state, token) -> score over next states
        # Factored: state_embed @ token_embed -> logits over states
        self.state_embed = nn.Embedding(num_states, embed_dim)
        self.token_proj = nn.Linear(vocab_size, embed_dim, bias=False)
        self.transition_head = nn.Linear(embed_dim, num_states, bias=False)
        # Emission: state -> feature
        self.emission = nn.Embedding(num_states, embed_dim)
        # Initial state distribution
        self.initial_logits = nn.Parameter(torch.zeros(num_states))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: [B, T] long -> [B, T, embed_dim] features"""
        B, T = tokens.shape
        # One-hot tokens
        tok_oh = F.one_hot(tokens, self.token_proj.in_features).float()
        tok_feat = self.token_proj(tok_oh)  # [B, T, embed_dim]

        # Soft state tracking (sequential)
        state_dist = F.softmax(self.initial_logits, dim=0)  # [num_states]
        state_dist = state_dist.unsqueeze(0).expand(B, -1)  # [B, num_states]

        outputs = []
        for t in range(T):
            # Emission from current state distribution
            # [B, num_states] @ [num_states, embed_dim] -> [B, embed_dim]
            emitted = state_dist @ self.emission.weight
            outputs.append(emitted)

            # Transition: combine state embedding with token
            # Expected state embedding: [B, embed_dim]
            state_emb = state_dist @ self.state_embed.weight
            # Combine with token feature
            combined = state_emb * tok_feat[:, t]  # [B, embed_dim]
            # Next state logits
            next_logits = self.transition_head(combined)  # [B, num_states]
            state_dist = F.softmax(next_logits, dim=-1)

        return torch.stack(outputs, dim=1)  # [B, T, embed_dim]


class HopfieldLayer(nn.Module):
    """One-step modern Hopfield retrieval.
    Stores N patterns, retrieves via one softmax step (= one attention step
    but against FIXED learned patterns, not against other sequence positions).
    O(1) per position (N patterns is fixed, not sequence-length dependent).
    """
    def __init__(self, input_dim: int, num_patterns: int, pattern_dim: int) -> None:
        super().__init__()
        self.query_proj = nn.Linear(input_dim, pattern_dim)
        # Stored patterns (keys and values)
        self.keys = nn.Parameter(torch.randn(num_patterns, pattern_dim) * 0.02)
        self.values = nn.Parameter(torch.randn(num_patterns, pattern_dim) * 0.02)
        self.out_proj = nn.Linear(pattern_dim, input_dim)
        self.scale = pattern_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, input_dim] -> [B, T, input_dim]"""
        q = self.query_proj(x)  # [B, T, pattern_dim]
        # Attention against stored patterns (NOT against other positions)
        scores = torch.matmul(q, self.keys.t()) * self.scale  # [B, T, N]
        weights = F.softmax(scores, dim=-1)
        retrieved = torch.matmul(weights, self.values)  # [B, T, pattern_dim]
        return x + self.out_proj(retrieved)


class HDCEncoder(nn.Module):
    """Hyperdimensional computing encoder.
    Encodes n-gram context via bind (XOR-like) + bundle (sum) + shift (rotate).
    Uses continuous relaxation: bind = element-wise product of random +-1 vectors,
    shift = learned circular permutation, bundle = sum + normalize.
    """
    def __init__(self, vocab_size: int, hdc_dim: int, orders: int = 4) -> None:
        super().__init__()
        self.hdc_dim = hdc_dim
        self.orders = orders
        # Random bipolar vectors for each token (fixed, not learned)
        self.register_buffer(
            "token_vectors",
            (torch.randint(0, 2, (vocab_size, hdc_dim)).float() * 2 - 1)
        )
        # Learned projection from HDC space to feature space
        self.out_proj = nn.Linear(hdc_dim, hdc_dim)

    def _shift(self, v: torch.Tensor, k: int) -> torch.Tensor:
        """Circular shift by k positions."""
        return torch.roll(v, shifts=k, dims=-1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: [B, T] -> [B, T, hdc_dim]"""
        B, T = tokens.shape
        # Get bipolar vectors for each token
        vecs = self.token_vectors[tokens]  # [B, T, hdc_dim]

        # Bundle n-gram encodings: shift^(n-1)(t_{-n}) * ... * shift^0(t_0)
        bundle = torch.zeros(B, T, self.hdc_dim, device=tokens.device)
        for order in range(1, self.orders + 1):
            # Build n-gram encoding for this order
            ngram = torch.ones(B, T, self.hdc_dim, device=tokens.device)
            for k in range(order):
                shifted_vecs = torch.zeros_like(vecs)
                offset = order - 1 - k
                if offset < T:
                    shifted_vecs[:, offset:] = vecs[:, :-offset] if offset > 0 else vecs
                # Bind = element-wise multiply (continuous analog of XOR)
                ngram = ngram * self._shift(shifted_vecs, k)
            bundle = bundle + ngram

        # Normalize
        bundle = bundle / max(self.orders, 1)
        return self.out_proj(bundle)


class PolyHashV5(nn.Module):
    def __init__(self, config: V5Config = V5Config()) -> None:
        super().__init__()
        self.config = config
        V = config.vocab_size

        self.byte_embed = nn.Embedding(V, config.byte_embed_dim)

        # Hash tables (fibonacci)
        self.skip_patterns = tuple(
            (FIBONACCI[i],) if i < len(FIBONACCI) else (1, i + 1)
            for i in range(config.num_tables)
        )
        self.hash_tables = nn.ModuleList()
        for _ in range(config.num_tables):
            t = nn.Embedding(config.buckets_per_table, config.embed_per_table)
            nn.init.normal_(t.weight, std=0.02)
            self.hash_tables.append(t)

        # Depthwise causal conv (proven winner)
        hash_feat_dim = config.num_tables * config.embed_per_table
        self.dw_conv = None
        if config.conv_kernel > 0:
            self.dw_conv = nn.Conv1d(
                hash_feat_dim, hash_feat_dim,
                kernel_size=config.conv_kernel, padding=0,
                groups=hash_feat_dim,
            )
            self.conv_pad = config.conv_kernel - 1

        # Exotic features
        self.fsm = None
        if config.fsm_states > 0:
            self.fsm = LearnedFSM(config.fsm_states, V, config.fsm_embed_dim)

        self.hopfield = None
        if config.hopfield_steps > 0:
            self.hopfield = HopfieldLayer(
                config.hidden_dim, config.hopfield_patterns, config.hopfield_dim
            )

        self.hdc = None
        if config.hdc_dim > 0:
            self.hdc = HDCEncoder(V, config.hdc_dim, config.hdc_orders)

        # Feature dim
        n_match = len(config.match_offsets) if config.match_offsets else 0
        feat_dim = config.byte_embed_dim + hash_feat_dim + n_match
        if config.fsm_states > 0:
            feat_dim += config.fsm_embed_dim
        if config.hdc_dim > 0:
            feat_dim += config.hdc_dim

        self.input_proj = nn.Linear(feat_dim, config.hidden_dim)

        # MLP (applied depth_recurrence times with weight tying)
        self.mlp = nn.ModuleList([
            ResBlock(config.hidden_dim, config.dropout)
            for _ in range(config.num_layers - 1)
        ])
        self.output_proj = nn.Linear(config.hidden_dim, V)

    def _hash_features(self, chars):
        embs = []
        for i, table in enumerate(self.hash_tables):
            idx = _hash_ctx(chars, self.skip_patterns[i], i, self.config.buckets_per_table)
            embs.append(table(idx))
        hash_feat = torch.cat(embs, dim=-1)
        if self.dw_conv is not None:
            h_t = hash_feat.transpose(1, 2)
            h_t = F.pad(h_t, (self.conv_pad, 0))
            hash_feat = F.silu(self.dw_conv(h_t)).transpose(1, 2)
        return hash_feat

    def _match_features(self, tokens):
        B, T = tokens.shape
        feats = []
        for k in self.config.match_offsets:
            if k >= T:
                feats.append(torch.zeros(B, T, device=tokens.device))
            else:
                shifted = torch.zeros_like(tokens)
                shifted[:, k:] = tokens[:, :-k]
                m = (tokens == shifted).float()
                m[:, :k] = 0.0
                feats.append(m)
        return torch.stack(feats, dim=-1) if feats else None

    def forward(self, chars: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        parts = [self.byte_embed(chars)]
        parts.append(self._hash_features(chars))

        if cfg.match_offsets:
            mf = self._match_features(chars)
            if mf is not None:
                parts.append(mf)

        if self.fsm is not None:
            parts.append(self.fsm(chars))

        if self.hdc is not None:
            parts.append(self.hdc(chars))

        h = F.silu(self.input_proj(torch.cat(parts, dim=-1)))

        # Hopfield retrieval (before MLP, enriches hidden state)
        if self.hopfield is not None:
            for _ in range(cfg.hopfield_steps):
                h = self.hopfield(h)

        # Depth recurrence: apply same MLP blocks multiple times
        for _rep in range(cfg.depth_recurrence):
            for block in self.mlp:
                h = block(h)

        return self.output_proj(h)
