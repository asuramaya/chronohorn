"""PolyHash v8h: Hybrid Decepticon — hash pyramid + tiny gated scan.

The hash pyramid provides O(1) multi-scale context recognition.
A small gated scan (32-128 dim) provides the adaptive gating signal
the pyramid lacks — which positions to weight more at each step.

Best of both: pyramid speed + scan's input-dependent attention.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from chronohorn.families.polyhash.models.polyhash_v8 import (
    ResBlock,
    RollingHashPyramid,
    ScaleAttention,
    V8Config,
)


@dataclass(frozen=True)
class V8HConfig(V8Config):
    """V8 config extended with a tiny scan."""
    scan_dim: int = 32
    scan_chunk_size: int = 32


class TinyScan(nn.Module):
    """Lightweight gated scan — just enough for adaptive context weighting."""

    def __init__(self, input_dim: int, scan_dim: int, chunk_size: int = 32):
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


class PolyHashV8H(nn.Module):
    """Hybrid: hash pyramid + tiny scan + scale attention + MLP."""

    def __init__(self, config: V8HConfig = V8HConfig()):
        super().__init__()
        self.config = config
        V = config.vocab_size

        self.byte_embed = nn.Embedding(V, config.byte_embed_dim)
        self.pyramid = RollingHashPyramid(config)

        self.scale_attn = None
        if config.scale_attention:
            self.scale_attn = ScaleAttention(
                config.byte_embed_dim,
                config.embed_per_scale,
                config.num_scales,
            )

        # Feature dimension after scale attention
        hash_out_dim = config.embed_per_scale if config.scale_attention else config.num_scales * config.embed_per_scale

        # Depthwise conv on hash features
        self.dw_conv = None
        if config.conv_kernel > 0:
            self.dw_conv = nn.Conv1d(
                hash_out_dim, hash_out_dim,
                kernel_size=config.conv_kernel, padding=0,
                groups=hash_out_dim,
            )
            self.conv_pad = config.conv_kernel - 1

        n_match = len(config.match_offsets) if config.match_offsets else 0

        # Input projection: byte_embed + hash_feat + match → hidden
        feat_dim = config.byte_embed_dim + hash_out_dim + n_match
        self.input_proj = nn.Linear(feat_dim, config.hidden_dim)

        # Tiny scan on the projected features
        self.scan = TinyScan(config.hidden_dim, config.scan_dim, config.scan_chunk_size)

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

        byte_emb = self.byte_embed(chars)
        scale_embeds = self.pyramid(chars)

        if self.scale_attn is not None:
            hash_feat = self.scale_attn(byte_emb, scale_embeds)
        else:
            hash_feat = scale_embeds.reshape(B, T, -1)

        if self.dw_conv is not None:
            ht = hash_feat.transpose(1, 2)
            ht = F.pad(ht, (self.conv_pad, 0))
            hash_feat = F.silu(self.dw_conv(ht)).transpose(1, 2)

        parts = [byte_emb, hash_feat]
        mf = self._match_features(chars) if cfg.match_offsets else None
        if mf is not None:
            parts.append(mf)

        h = F.silu(self.input_proj(torch.cat(parts, dim=-1)))
        h = self.scan(h)

        for block in self.mlp:
            h = block(h)

        return self.output_proj(h)
