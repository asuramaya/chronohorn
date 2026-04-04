"""PolyHash v10: Combo Decepticon — MLP-free hash-based readout.

Replaces the SwiGLU MLP with:
  1. Combo hash table: XOR all scale keys → lookup cross-interaction embedding
  2. Pairwise scale gating: scale_0 * sigmoid(scale_1) + scale_2 * sigmoid(scale_3)...
  3. Optional tiny MLP (h128) for residual interactions

The hash pyramid provides O(1) multi-scale context recognition.
The combo table provides O(1) cross-scale interaction (replaces MLP).
The pairwise gating provides O(1) nonlinearity.
A tiny scan (8-32d) provides disambiguation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from chronohorn.families.polyhash.models.polyhash_v8 import (
    V8Config,
    RollingHashPyramid,
    SwiGLU,
    ResBlock,
    SCALE_WINDOWS,
    HASH_PRIMES,
)
from chronohorn.families.polyhash.models.polyhash_v8h import TinyScan


@dataclass(frozen=True)
class V10Config(V8Config):
    # Combo table
    combo_table: bool = True
    combo_buckets: int = 65536
    combo_embed_dim: int = 16
    num_combo_tables: int = 1      # >1 = multi-combo (different key subsets)
    # Pairwise gating
    pairwise_gate: bool = True
    # MLP (set hidden_dim=0 to disable)
    hidden_dim: int = 0            # 0 = no MLP, >0 = tiny MLP
    num_layers: int = 1
    # Scan
    scan_dim: int = 32
    scan_chunk_size: int = 32
    # Scale attention (off by default — ablation showed it hurts)
    scale_attention: bool = False


class ComboTable(nn.Module):
    """Cross-scale interaction via hash-of-hashes lookup."""

    def __init__(self, num_tables: int, buckets: int, embed_dim: int, num_scales: int):
        super().__init__()
        self.num_tables = num_tables
        self.buckets = buckets
        self.tables = nn.ModuleList()
        for _ in range(num_tables):
            t = nn.Embedding(buckets, embed_dim)
            nn.init.normal_(t.weight, std=0.02)
            self.tables.append(t)

        # For multi-combo: each table uses a different subset/permutation of keys
        # Table 0: XOR all keys
        # Table 1: XOR even keys
        # Table 2: XOR odd keys
        # Table 3: XOR(key0^key1, key2^key3, key4^key5, key6^key7) — pairwise then XOR
        self.combo_primes = [1, 2654435761, 2246822519, 3266489917]

    def forward(self, scale_keys: list[torch.Tensor]) -> torch.Tensor:
        """
        scale_keys: list of [B, T] long tensors (one per scale)
        Returns: [B, T, num_tables * embed_dim]
        """
        outputs = []
        for t_idx, table in enumerate(self.tables):
            if t_idx == 0:
                # XOR all keys
                combo = scale_keys[0]
                for k in scale_keys[1:]:
                    combo = combo ^ k
            elif t_idx == 1 and len(scale_keys) >= 4:
                # XOR even-indexed keys
                combo = scale_keys[0]
                for i in range(2, len(scale_keys), 2):
                    combo = combo ^ scale_keys[i]
            elif t_idx == 2 and len(scale_keys) >= 4:
                # XOR odd-indexed keys
                combo = scale_keys[1]
                for i in range(3, len(scale_keys), 2):
                    combo = combo ^ scale_keys[i]
            elif t_idx == 3 and len(scale_keys) >= 4:
                # Pairwise XOR then combine
                pairs = []
                for i in range(0, len(scale_keys) - 1, 2):
                    pairs.append(scale_keys[i] ^ scale_keys[i + 1])
                combo = pairs[0]
                for p in pairs[1:]:
                    combo = combo ^ (p * self.combo_primes[t_idx % len(self.combo_primes)])
            else:
                # Salt and XOR all
                salt = self.combo_primes[t_idx % len(self.combo_primes)]
                combo = scale_keys[0] * salt
                for k in scale_keys[1:]:
                    combo = combo ^ (k * salt)

            outputs.append(table(combo % self.buckets))

        return torch.cat(outputs, dim=-1)


class PairwiseGate(nn.Module):
    """Element-wise gating between pairs of scale embeddings.

    scale_0 * sigmoid(scale_1) + scale_2 * sigmoid(scale_3) + ...
    Provides nonlinear cross-scale interaction at zero matmul cost.
    """

    def __init__(self, num_scales: int, embed_dim: int):
        super().__init__()
        self.num_pairs = num_scales // 2
        self.leftover = num_scales % 2
        # Learnable bias for each gate
        self.gate_bias = nn.Parameter(torch.zeros(self.num_pairs, embed_dim))

    def forward(self, scale_embeds: torch.Tensor) -> torch.Tensor:
        """
        scale_embeds: [B, T, num_scales, embed_dim]
        Returns: [B, T, (num_pairs + leftover) * embed_dim]
        """
        parts = []
        for i in range(self.num_pairs):
            val = scale_embeds[:, :, 2 * i]
            gate = scale_embeds[:, :, 2 * i + 1]
            parts.append(val * torch.sigmoid(gate + self.gate_bias[i]))
        if self.leftover:
            parts.append(scale_embeds[:, :, -1])
        return torch.cat(parts, dim=-1)


class PolyHashV10(nn.Module):
    def __init__(self, config: V10Config = V10Config()):
        super().__init__()
        self.config = config
        V = config.vocab_size

        # Byte embedding
        self.byte_embed = nn.Embedding(V, config.byte_embed_dim)

        # Hash pyramid (from v8)
        self.pyramid = RollingHashPyramid(config)

        # Combo table (cross-scale interaction)
        self.combo = None
        if config.combo_table:
            self.combo = ComboTable(
                config.num_combo_tables,
                config.combo_buckets,
                config.combo_embed_dim,
                config.num_scales,
            )

        # Pairwise gating
        self.pw_gate = None
        if config.pairwise_gate:
            self.pw_gate = PairwiseGate(config.num_scales, config.embed_per_scale)

        # Depthwise conv on hash features
        n_scales = config.num_scales
        E = config.embed_per_scale
        if config.pairwise_gate:
            gate_dim = (n_scales // 2 + n_scales % 2) * E
        else:
            gate_dim = n_scales * E
        hash_out_dim = gate_dim + (config.num_combo_tables * config.combo_embed_dim if config.combo_table else 0)

        self.dw_conv = None
        if config.conv_kernel > 0:
            self.dw_conv = nn.Conv1d(
                hash_out_dim, hash_out_dim,
                kernel_size=config.conv_kernel, padding=0,
                groups=hash_out_dim,
            )
            self.conv_pad = config.conv_kernel - 1

        # Match features
        n_match = len(config.match_offsets) if config.match_offsets else 0

        # Total feature dim
        feat_dim = config.byte_embed_dim + hash_out_dim + n_match

        # Scan (tiny)
        self.scan = None
        if config.scan_dim > 0:
            self.scan = TinyScan(feat_dim, config.scan_dim, config.scan_chunk_size)
            proj_in_dim = feat_dim
        else:
            proj_in_dim = feat_dim

        # MLP readout (optional — 0 = skip)
        self.mlp = None
        if config.hidden_dim > 0:
            self.input_proj = nn.Linear(proj_in_dim, config.hidden_dim)
            self.mlp = nn.ModuleList([
                ResBlock(config.hidden_dim, config.dropout)
                for _ in range(config.num_layers)
            ])
            self.output_proj = nn.Linear(config.hidden_dim, V)
        else:
            # Direct output projection (no MLP)
            self.output_proj = nn.Linear(proj_in_dim, V)
            self.input_proj = None

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

        # Hash pyramid: get both scale embeddings and raw keys
        scale_embeds = self.pyramid(chars)  # [B, T, num_scales, E]

        # Also need the raw hash keys for combo table
        scale_keys = [self.pyramid._hash_scale(chars, s) for s in range(cfg.num_scales)]

        # Pairwise gating or flatten
        if self.pw_gate is not None:
            hash_feat = self.pw_gate(scale_embeds)  # [B, T, pairs*E]
        else:
            hash_feat = scale_embeds.reshape(B, T, -1)  # [B, T, ns*E]

        # Combo table
        if self.combo is not None:
            combo_feat = self.combo(scale_keys)  # [B, T, combo_dim]
            hash_feat = torch.cat([hash_feat, combo_feat], dim=-1)

        # Conv
        if self.dw_conv is not None:
            ht = hash_feat.transpose(1, 2)
            ht = F.pad(ht, (self.conv_pad, 0))
            hash_feat = F.silu(self.dw_conv(ht)).transpose(1, 2)

        # Match features
        parts = [byte_emb, hash_feat]
        mf = self._match_features(chars) if cfg.match_offsets else None
        if mf is not None:
            parts.append(mf)

        h = torch.cat(parts, dim=-1)

        # Scan
        if self.scan is not None:
            h = self.scan(h)

        # Readout
        if self.mlp is not None:
            h = F.silu(self.input_proj(h))
            for block in self.mlp:
                h = block(h)

        return self.output_proj(h)
