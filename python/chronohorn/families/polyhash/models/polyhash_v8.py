"""PolyHash v8: Parallel Decepticon.

No sequential scan. Multi-scale rolling polynomial hashes with hierarchical
conditioning and input-dependent scale attention. All O(1) per position.

Architecture:
  byte_embed
    → 8 rolling hash tables (1,2,4,8,16,32,64,128-gram)
    → hierarchical conditioning (short-range sign bits refine long-range keys)
    → scale attention (input-dependent 8-way routing)
    → conv8 + match features
    → SwiGLU MLP → logits

The hash collision pattern is the attention mechanism:
  - Each table = one attention head with fixed positional weights (primes)
  - Contexts that hash to the same bucket share a learned embedding
  - Collision structure naturally groups contexts by next-token distribution
  - Scale attention routes between granularities input-dependently
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# Prime families — each scale gets its own prime sequence so collision
# patterns are independent across scales.
HASH_PRIMES = [
    # Scale 0 (unigram)
    [2654435761],
    # Scale 1 (bigram)
    [2654435761, 2246822519],
    # Scale 2 (4-gram)
    [2654435761, 2246822519, 3266489917, 2028178513],
    # Scale 3 (8-gram)
    [2654435761, 2246822519, 3266489917, 2028178513,
     1220703125, 1610612741, 805306457, 402653189],
    # Scale 4 (16-gram)
    [2654435761, 2246822519, 3266489917, 2028178513,
     1220703125, 1610612741, 805306457, 402653189,
     3674653429, 2860486313, 1073676287, 2971215073,
     1500450271, 3267000013, 2654435789, 4049292737],
    # Scale 5 (32-gram) — reuse with salt
    [p ^ 0xDEADBEEF for p in [
     2654435761, 2246822519, 3266489917, 2028178513,
     1220703125, 1610612741, 805306457, 402653189,
     3674653429, 2860486313, 1073676287, 2971215073,
     1500450271, 3267000013, 2654435789, 4049292737,
     2246822531, 3266489927, 2028178519, 1220703133,
     1610612759, 805306463, 402653201, 3674653441,
     2860486319, 1073676311, 2971215091, 1500450277,
     3267000023, 2654435801, 4049292751, 2246822537]],
    # Scale 6 (64-gram) — reuse with different salt
    [p ^ 0xCAFEBABE for p in [
     2654435761, 2246822519, 3266489917, 2028178513,
     1220703125, 1610612741, 805306457, 402653189] * 8],
    # Scale 7 (128-gram) — reuse with another salt
    [p ^ 0x8BADF00D for p in [
     2654435761, 2246822519, 3266489917, 2028178513,
     1220703125, 1610612741, 805306457, 402653189] * 16],
]

SCALE_WINDOWS = [1, 2, 4, 8, 16, 32, 64, 128]
FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21]


@dataclass(frozen=True)
class V8Config:
    vocab_size: int = 1024
    byte_embed_dim: int = 128
    # Hash pyramid
    num_scales: int = 8
    buckets_per_scale: int = 65536
    embed_per_scale: int = 16
    # Hierarchical conditioning
    condition_long_on_short: bool = True
    condition_boundary: int = 4       # scales 0..3 parallel, 4..7 conditioned
    sign_bits: int = 8               # bits from short-range to condition long-range
    # Scale attention
    scale_attention: bool = True
    # Local features
    conv_kernel: int = 8
    match_offsets: tuple = (1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32)
    # MLP
    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.0
    # Training
    max_seq_len: int = 512
    init_seed: int = 42
    hash_dropout: float = 0.0
    lr_hash_mult: float = 2.0


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
        if self.drop:
            h = self.drop(h)
        return self.ln(h + x)


class RollingHashPyramid(nn.Module):
    """Multi-scale rolling polynomial hash with hierarchical conditioning.

    Each scale hashes a window of tokens (1, 2, 4, ..., 128) into a bucket
    index, then looks up a learned embedding. Short-range scale embeddings
    are quantized to sign bits and XOR'd into long-range hash keys, creating
    cross-scale conditioning without sequential matmuls.
    """

    def __init__(self, config: V8Config):
        super().__init__()
        self.config = config
        ns = config.num_scales
        B = config.buckets_per_scale
        E = config.embed_per_scale

        # One embedding table per scale
        self.tables = nn.ModuleList()
        for _ in range(ns):
            t = nn.Embedding(B, E)
            nn.init.normal_(t.weight, std=0.02)
            self.tables.append(t)

        # Pre-register prime tensors for each scale
        for s in range(ns):
            w = SCALE_WINDOWS[s]
            primes = HASH_PRIMES[s][:w]
            self.register_buffer(f"primes_{s}", torch.tensor(primes, dtype=torch.long))

        self.condition = config.condition_long_on_short
        self.boundary = config.condition_boundary
        self.sign_bits = config.sign_bits

        # Conditioning projection: short-range embeddings → sign bits for hashing
        if self.condition:
            short_dim = config.condition_boundary * E
            self.cond_proj = nn.Linear(short_dim, config.sign_bits, bias=False)
            # Primes for conditioning hash
            self.register_buffer(
                "cond_primes",
                torch.tensor([2654435761, 2246822519, 3266489917, 2028178513,
                              1220703125, 1610612741, 805306457, 402653189,
                              3674653429, 2860486313, 1073676287, 2971215073,
                              1500450271, 3267000013, 2654435789, 4049292737][:config.sign_bits],
                             dtype=torch.long)
            )

    def _hash_scale(self, tokens: torch.Tensor, scale_idx: int) -> torch.Tensor:
        """Compute hash keys for one scale. tokens: [B, T] long."""
        B, T = tokens.shape
        window = SCALE_WINDOWS[scale_idx]
        primes = getattr(self, f"primes_{scale_idx}")  # [window]
        buckets = self.config.buckets_per_scale

        # Build hash: XOR of shifted tokens × primes
        h = torch.zeros(B, T, dtype=torch.long, device=tokens.device)
        for i in range(window):
            offset = i + 1  # offset from current position (causal: look back)
            if offset >= T:
                continue
            shifted = torch.zeros_like(tokens)
            shifted[:, offset:] = tokens[:, :-offset]
            h = h ^ (shifted * primes[i])

        return h % buckets

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: [B, T] long tensor of token IDs
        Returns: [B, T, num_scales * embed_per_scale]
        """
        cfg = self.config
        B, T = tokens.shape
        ns = cfg.num_scales
        E = cfg.embed_per_scale
        buckets = cfg.buckets_per_scale

        # Stage 1: compute all hash keys
        keys = [self._hash_scale(tokens, s) for s in range(ns)]

        # Stage 2: lookup short-range scales (0 .. boundary-1)
        short_embeds = []
        for s in range(self.boundary):
            emb = self.tables[s](keys[s])  # [B, T, E]
            if self.training and cfg.hash_dropout > 0:
                if torch.rand(1).item() < cfg.hash_dropout:
                    emb = torch.zeros_like(emb)
            short_embeds.append(emb)

        # Stage 3: condition long-range keys on short-range sign bits
        if self.condition and self.boundary < ns:
            # Concatenate short-range embeddings
            short_cat = torch.cat(short_embeds, dim=-1)  # [B, T, boundary * E]
            # Project to sign bits
            cond_logits = self.cond_proj(short_cat)  # [B, T, sign_bits]
            sign_bits = (cond_logits > 0).long()  # [B, T, sign_bits]

            # Hash the sign bits into a conditioning key
            cond_key = torch.zeros(B, T, dtype=torch.long, device=tokens.device)
            for i in range(self.sign_bits):
                cond_key = cond_key ^ (sign_bits[:, :, i] * self.cond_primes[i])

            # XOR conditioning key into long-range hash keys
            for s in range(self.boundary, ns):
                keys[s] = (keys[s] ^ cond_key) % buckets

        # Stage 4: lookup long-range scales (boundary .. ns-1)
        long_embeds = []
        for s in range(self.boundary, ns):
            emb = self.tables[s](keys[s])
            if self.training and cfg.hash_dropout > 0:
                if torch.rand(1).item() < cfg.hash_dropout:
                    emb = torch.zeros_like(emb)
            long_embeds.append(emb)

        # Concatenate all scale embeddings
        all_embeds = short_embeds + long_embeds  # list of [B, T, E]
        return torch.stack(all_embeds, dim=2)  # [B, T, ns, E]


class ScaleAttention(nn.Module):
    """Input-dependent routing across hash scales.

    Query from byte embedding, keys/values from scale embeddings.
    8-way attention — 2K FLOPs, trivial.
    """

    def __init__(self, byte_dim: int, scale_dim: int, num_scales: int):
        super().__init__()
        self.q_proj = nn.Linear(byte_dim, scale_dim, bias=False)
        self.k_proj = nn.Linear(scale_dim, scale_dim, bias=False)
        self.scale = scale_dim ** -0.5

    def forward(self, byte_emb: torch.Tensor, scale_embeds: torch.Tensor) -> torch.Tensor:
        """
        byte_emb: [B, T, byte_dim]
        scale_embeds: [B, T, num_scales, scale_dim]
        Returns: [B, T, scale_dim]
        """
        Q = self.q_proj(byte_emb)  # [B, T, scale_dim]
        K = self.k_proj(scale_embeds)  # [B, T, ns, scale_dim]
        V = scale_embeds  # [B, T, ns, scale_dim]

        # Attention scores: [B, T, ns]
        scores = torch.einsum("btd,btnd->btn", Q, K) * self.scale
        attn = F.softmax(scores, dim=-1)  # [B, T, ns]

        # Weighted sum: [B, T, scale_dim]
        out = torch.einsum("btn,btnd->btd", attn, V)
        return out


class PolyHashV8(nn.Module):
    def __init__(self, config: V8Config = V8Config()):
        super().__init__()
        self.config = config
        V = config.vocab_size

        # Byte embedding
        self.byte_embed = nn.Embedding(V, config.byte_embed_dim)

        # Rolling hash pyramid
        self.pyramid = RollingHashPyramid(config)

        # Scale attention
        self.scale_attn = None
        if config.scale_attention:
            self.scale_attn = ScaleAttention(
                config.byte_embed_dim,
                config.embed_per_scale,
                config.num_scales,
            )

        # Depthwise causal conv
        hash_feat_dim = config.num_scales * config.embed_per_scale
        self.dw_conv = None
        if config.conv_kernel > 0:
            # Conv on the concatenated scale embeddings
            conv_in = config.embed_per_scale if config.scale_attention else hash_feat_dim
            self.dw_conv = nn.Conv1d(
                conv_in, conv_in,
                kernel_size=config.conv_kernel, padding=0,
                groups=conv_in,
            )
            self.conv_pad = config.conv_kernel - 1

        # Match features
        n_match = len(config.match_offsets) if config.match_offsets else 0

        # Input projection
        if config.scale_attention:
            feat_dim = config.byte_embed_dim + config.embed_per_scale + n_match
        else:
            feat_dim = config.byte_embed_dim + hash_feat_dim + n_match
        self.input_proj = nn.Linear(feat_dim, config.hidden_dim)

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

    def forward(self, chars: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        B, T = chars.shape

        # Byte embedding
        byte_emb = self.byte_embed(chars)  # [B, T, byte_dim]

        # Hash pyramid: [B, T, num_scales, embed_per_scale]
        scale_embeds = self.pyramid(chars)

        # Scale attention or flatten
        if self.scale_attn is not None:
            hash_feat = self.scale_attn(byte_emb, scale_embeds)  # [B, T, embed_per_scale]
        else:
            hash_feat = scale_embeds.reshape(B, T, -1)  # [B, T, ns * E]

        # Depthwise conv
        if self.dw_conv is not None:
            ht = hash_feat.transpose(1, 2)
            ht = F.pad(ht, (self.conv_pad, 0))
            hash_feat = F.silu(self.dw_conv(ht)).transpose(1, 2)

        # Match features
        parts = [byte_emb, hash_feat]
        mf = self._match_features(chars) if cfg.match_offsets else None
        if mf is not None:
            parts.append(mf)

        # Project + MLP readout
        h = F.silu(self.input_proj(torch.cat(parts, dim=-1)))
        for block in self.mlp:
            h = block(h)

        return self.output_proj(h)
