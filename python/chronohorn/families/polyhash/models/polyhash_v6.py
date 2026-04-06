"""PolyHash v6: O(1) hash features + parallel gated scan.

v6b additions (bottleneck ablations):
  - Multi-timescale scan: groups with independent or fixed decay rates
  - Cross-dim mixing: linear mix between chunks
  - Scan-first pipeline: scan on raw bytes, hash added at readout
  - Outer-product state update (RWKV-style)
"""
from __future__ import annotations

import math
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
]

FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21]


@dataclass(frozen=True)
class V6Config:
    vocab_size: int = 1024
    byte_embed_dim: int = 128
    num_tables: int = 8
    buckets_per_table: int = 65536
    embed_per_table: int = 16
    conv_kernel: int = 8
    match_offsets: tuple = (1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32)
    scan_dim: int = 256
    scan_heads: int = 1
    scan_chunk_size: int = 32
    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.0
    hash_dropout: float = 0.0
    scan_dropout: float = 0.0
    weight_decay: float = 0.0
    max_seq_len: int = 512
    init_seed: int = 42
    # --- v6b: bottleneck ablation flags ---
    timescale_mode: str = "learned"   # learned, fixed_groups, log_spaced
    timescale_groups: int = 1         # 1=all share gate, 4/8=independent groups
    cross_dim_mix: int = 0            # 0=off, N=mix every N chunks
    scan_pipeline: str = "default"    # default, scan_first, dual_path
    outer_product_dim: int = 0        # 0=off, >0=RWKV-style v@k.T with this dim
    binding_mode: str = "outer"       # outer, additive, concat, shift, xor_soft, gate_cross


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


class ParallelGatedScan(nn.Module):
    """Chunked parallel gated scan with multi-timescale and cross-dim mixing."""

    def __init__(self, input_dim: int, scan_dim: int, config: V6Config):
        super().__init__()
        self.scan_dim = scan_dim
        self.chunk_size = config.scan_chunk_size
        self.timescale_mode = config.timescale_mode
        self.timescale_groups = config.timescale_groups
        self.cross_dim_mix = config.cross_dim_mix
        self.outer_product_dim = config.outer_product_dim

        # Gate projection
        if config.timescale_mode == "learned" and config.timescale_groups > 1:
            # Independent gate per group
            self.gate_proj = nn.Linear(input_dim, scan_dim)
        elif config.timescale_mode == "learned":
            self.gate_proj = nn.Linear(input_dim, scan_dim)
        else:
            # Fixed timescales — no gate projection needed
            self.gate_proj = None
            if config.timescale_mode == "fixed_groups":
                n_groups = max(config.timescale_groups, 4)
                dims_per = scan_dim // n_groups
                decays = []
                fixed_rates = [0.3, 0.7, 0.95, 0.99, 0.5, 0.85, 0.97, 0.999]
                for g in range(n_groups):
                    rate = fixed_rates[g % len(fixed_rates)]
                    decays.extend([rate] * dims_per)
                while len(decays) < scan_dim:
                    decays.append(0.9)
                self.register_buffer("fixed_gates", torch.tensor(decays))
            else:  # log_spaced
                decays = torch.logspace(math.log10(0.1), math.log10(0.999), scan_dim)
                self.register_buffer("fixed_gates", decays)

        self.input_proj = nn.Linear(input_dim, scan_dim)
        self.output_proj = nn.Linear(scan_dim, input_dim)
        self.ln = nn.LayerNorm(input_dim)
        self.drop = nn.Dropout(config.scan_dropout) if config.scan_dropout > 0 else None

        # Cross-dim mixing layer
        self.mix_layer = None
        if config.cross_dim_mix > 0:
            self.mix_layer = nn.Linear(scan_dim, scan_dim)

        # State binding (cross-dim interaction)
        self.op_v = None
        self.binding_mode = config.binding_mode
        if config.outer_product_dim > 0:
            od = config.outer_product_dim
            self.op_v = nn.Linear(input_dim, od)
            self.op_k = nn.Linear(input_dim, od)
            self.op_q = nn.Linear(input_dim, od)
            self.op_decay = nn.Parameter(torch.full((od,), 0.95))
            if config.binding_mode == "concat":
                # Concat state: 2*od dims, no binding operation
                self.op_out = nn.Linear(od * 2, input_dim)
            elif config.binding_mode == "gate_cross":
                # Cross-dim gated: gate[i] depends on all dims
                self.op_gate_matrix = nn.Linear(od, od)
                self.op_out = nn.Linear(od, input_dim)
            else:
                self.op_out = nn.Linear(od, input_dim)

    def _chunked_scan(self, gates, drive):
        B, T, sd = drive.shape
        K = min(self.chunk_size, T)
        n_chunks = (T + K - 1) // K
        states = torch.zeros(B, T, sd, device=drive.device, dtype=drive.dtype)
        h = torch.zeros(B, sd, device=drive.device, dtype=drive.dtype)

        for c in range(n_chunks):
            start = c * K
            end = min(start + K, T)
            a_chunk = gates[:, start:end]
            b_chunk = drive[:, start:end]

            log_a = torch.log(a_chunk.clamp(min=1e-6))
            log_cum_a = torch.cumsum(log_a, dim=1)
            cum_a = torch.exp(log_cum_a)
            inv_cum_a = 1.0 / cum_a.clamp(min=1e-8)
            weighted_b = b_chunk * inv_cum_a
            cum_wb = torch.cumsum(weighted_b, dim=1)
            chunk_states = cum_a * (h.unsqueeze(1) + cum_wb)
            states[:, start:end] = chunk_states
            h = chunk_states[:, -1]

            # Cross-dim mixing at chunk boundaries
            if self.mix_layer is not None and self.cross_dim_mix > 0:
                if (c + 1) % self.cross_dim_mix == 0:
                    h = h + F.silu(self.mix_layer(h))

        return states

    def _binding_scan(self, x):
        """Cross-dim state interaction via various binding modes."""
        B, T, _ = x.shape
        od = self.outer_product_dim
        v = self.op_v(x)  # [B, T, od]
        k = self.op_k(x)
        q = self.op_q(x)
        decay = torch.sigmoid(self.op_decay)  # [od]
        mode = self.binding_mode

        if mode == "concat":
            # No binding — concat v and k into 2*od state, readout learns the rest
            S_v = torch.zeros(B, od, device=x.device, dtype=x.dtype)
            S_k = torch.zeros(B, od, device=x.device, dtype=x.dtype)
            outs = []
            for t in range(T):
                S_v = decay * S_v + (1 - decay) * v[:, t]
                S_k = decay * S_k + (1 - decay) * k[:, t]
                outs.append(torch.cat([S_v, S_k], dim=-1))
            return self.op_out(torch.stack(outs, dim=1))

        elif mode == "additive":
            # Additive binding: S[i,j] += v[i] + k[j] instead of v[i]*k[j]
            S = torch.zeros(B, od, od, device=x.device, dtype=x.dtype)
            outs = []
            for t in range(T):
                binding = v[:, t].unsqueeze(-1) + k[:, t].unsqueeze(-2)  # [B, od, od]
                S = torch.diag_embed(decay) @ S + binding
                o = (S @ q[:, t].unsqueeze(-1)).squeeze(-1)
                outs.append(o)
            return self.op_out(torch.stack(outs, dim=1))

        elif mode == "shift":
            # Shift binding: circularly shift v by argmax(k)
            S = torch.zeros(B, od, device=x.device, dtype=x.dtype)
            outs = []
            for t in range(T):
                shift_amount = k[:, t].argmax(dim=-1)  # [B]
                shifted_v = torch.stack([
                    torch.roll(v[b, t], shifts=int(shift_amount[b].item()), dims=0)
                    for b in range(B)
                ])
                S = decay * S + (1 - decay) * shifted_v
                outs.append(S * q[:, t])  # gated readout
            return self.op_out(torch.stack(outs, dim=1))

        elif mode == "xor_soft":
            # Soft XOR: S[i^j mod od] += v_mag, using scatter_add
            S = torch.zeros(B, od, device=x.device, dtype=x.dtype)
            outs = []
            v_idx = v.argmax(dim=-1)  # [B, T]
            k_idx = k.argmax(dim=-1)  # [B, T]
            for t in range(T):
                xor_idx = (v_idx[:, t] ^ k_idx[:, t]) % od  # [B]
                v_mag = v[:, t].norm(dim=-1, keepdim=True)  # [B, 1]
                update = torch.zeros_like(S)
                update.scatter_(1, xor_idx.unsqueeze(-1), v_mag)
                S = decay * S + (1 - decay) * update
                outs.append(S)
            return self.op_out(torch.stack(outs, dim=1))

        elif mode == "gate_cross":
            # Cross-dim gating: gate[i] = sigmoid(W @ state), creates interaction
            S = torch.zeros(B, od, device=x.device, dtype=x.dtype)
            outs = []
            for t in range(T):
                cross_gate = torch.sigmoid(self.op_gate_matrix(S))  # [B, od]
                S = cross_gate * S + (1 - cross_gate) * v[:, t]
                outs.append(S * q[:, t])
            return self.op_out(torch.stack(outs, dim=1))

        else:  # "outer" — original RWKV-style
            S = torch.zeros(B, od, od, device=x.device, dtype=x.dtype)
            outs = []
            for t in range(T):
                S = torch.diag_embed(decay) @ S + v[:, t].unsqueeze(-1) @ k[:, t].unsqueeze(-2)
                o = (S @ q[:, t].unsqueeze(-1)).squeeze(-1)
                outs.append(o)
            return self.op_out(torch.stack(outs, dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        # Compute gates
        if self.gate_proj is not None:
            gates = torch.sigmoid(self.gate_proj(x))
        else:
            gates = self.fixed_gates.view(1, 1, -1).expand(B, T, -1)

        inp = self.input_proj(x)
        drive = (1.0 - gates) * inp
        states = self._chunked_scan(gates, drive)

        scan_out = self.output_proj(states)
        if self.drop:
            scan_out = self.drop(scan_out)

        # Add binding scan if enabled
        if self.op_v is not None:
            scan_out = scan_out + self._binding_scan(x)

        return self.ln(x + scan_out)


class PolyHashV6(nn.Module):
    def __init__(self, config: V6Config = V6Config()):
        super().__init__()
        self.config = config
        V = config.vocab_size

        self.byte_embed = nn.Embedding(V, config.byte_embed_dim)

        self.skip_patterns = tuple(
            (FIBONACCI[i],) if i < len(FIBONACCI) else (1, i + 1)
            for i in range(config.num_tables)
        )
        self.hash_tables = nn.ModuleList()
        for _ in range(config.num_tables):
            t = nn.Embedding(config.buckets_per_table, config.embed_per_table)
            nn.init.normal_(t.weight, std=0.02)
            self.hash_tables.append(t)

        hash_feat_dim = config.num_tables * config.embed_per_table
        self.dw_conv = None
        if config.conv_kernel > 0:
            self.dw_conv = nn.Conv1d(
                hash_feat_dim, hash_feat_dim,
                kernel_size=config.conv_kernel, padding=0,
                groups=hash_feat_dim,
            )
            self.conv_pad = config.conv_kernel - 1

        n_match = len(config.match_offsets) if config.match_offsets else 0

        # Pipeline-dependent wiring
        if config.scan_pipeline == "scan_first":
            # Scan on bytes only, hash features added at readout
            scan_input_dim = config.byte_embed_dim + n_match
            self.scan_proj = nn.Linear(scan_input_dim, config.hidden_dim)
            self.scan = ParallelGatedScan(config.hidden_dim, config.scan_dim, config) if config.scan_dim > 0 else None
            # After scan, concat hash features
            readout_input = config.hidden_dim + hash_feat_dim
            self.readout_proj = nn.Linear(readout_input, config.hidden_dim)
            self.input_proj = None
        elif config.scan_pipeline == "dual_path":
            # Two paths: scan on bytes, hash on its own, merge
            scan_input_dim = config.byte_embed_dim + n_match
            self.scan_proj = nn.Linear(scan_input_dim, config.hidden_dim)
            self.scan = ParallelGatedScan(config.hidden_dim, config.scan_dim, config) if config.scan_dim > 0 else None
            self.hash_proj = nn.Linear(hash_feat_dim, config.hidden_dim)
            self.merge_proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
            self.input_proj = None
            self.readout_proj = None
        else:
            # Default: hash → proj → scan → MLP
            feat_dim = config.byte_embed_dim + hash_feat_dim + n_match
            self.input_proj = nn.Linear(feat_dim, config.hidden_dim)
            self.scan = ParallelGatedScan(config.hidden_dim, config.scan_dim, config) if config.scan_dim > 0 else None
            self.scan_proj = None
            self.readout_proj = None

        self.mlp = nn.ModuleList([
            ResBlock(config.hidden_dim, config.dropout)
            for _ in range(config.num_layers - 1)
        ])
        self.output_proj = nn.Linear(config.hidden_dim, V)
        self._init_weights()

    def _init_weights(self):
        rng = torch.Generator().manual_seed(self.config.init_seed)
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def _hash_features(self, chars):
        cfg = self.config
        embs = []
        for i, table in enumerate(self.hash_tables):
            idx = _hash_ctx(chars, self.skip_patterns[i], i, cfg.buckets_per_table)
            emb = table(idx)
            if self.training and cfg.hash_dropout > 0:
                if torch.rand(1).item() < cfg.hash_dropout:
                    emb = torch.zeros_like(emb)
            embs.append(emb)
        hash_feat = torch.cat(embs, dim=-1)
        if self.dw_conv is not None:
            ht = hash_feat.transpose(1, 2)
            ht = F.pad(ht, (self.conv_pad, 0))
            hash_feat = F.silu(self.dw_conv(ht)).transpose(1, 2)
        return hash_feat

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
        hash_feat = self._hash_features(chars)
        match_feat = self._match_features(chars) if cfg.match_offsets else None

        if cfg.scan_pipeline == "scan_first":
            # Scan on bytes + match, then add hash at readout
            scan_parts = [byte_emb]
            if match_feat is not None: scan_parts.append(match_feat)
            h = F.silu(self.scan_proj(torch.cat(scan_parts, dim=-1)))
            if self.scan is not None:
                h = self.scan(h)
            h = F.silu(self.readout_proj(torch.cat([h, hash_feat], dim=-1)))

        elif cfg.scan_pipeline == "dual_path":
            # Scan path (bytes)
            scan_parts = [byte_emb]
            if match_feat is not None: scan_parts.append(match_feat)
            h_scan = F.silu(self.scan_proj(torch.cat(scan_parts, dim=-1)))
            if self.scan is not None:
                h_scan = self.scan(h_scan)
            # Hash path
            h_hash = F.silu(self.hash_proj(hash_feat))
            # Merge
            h = F.silu(self.merge_proj(torch.cat([h_scan, h_hash], dim=-1)))

        else:
            # Default pipeline
            parts = [byte_emb, hash_feat]
            if match_feat is not None: parts.append(match_feat)
            h = F.silu(self.input_proj(torch.cat(parts, dim=-1)))
            if self.scan is not None:
                h = self.scan(h)

        for block in self.mlp:
            h = block(h)

        return self.output_proj(h)
