from __future__ import annotations

"""Torch readout modules for Chronohorn descendant models."""

import numpy as np
import torch
from torch import nn

from typing import Any

from .common import _rng_for, _xavier_uniform


def _copy_linear_(layer: nn.Linear, weight: np.ndarray, bias: np.ndarray | None = None) -> None:
    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(weight).to(device=layer.weight.device, dtype=layer.weight.dtype))
        if layer.bias is not None:
            src = np.zeros(layer.bias.shape, dtype=np.float32) if bias is None else bias
            layer.bias.copy_(torch.from_numpy(src).to(device=layer.bias.device, dtype=layer.bias.dtype))


def _copy_embedding_(layer: nn.Embedding, weight: np.ndarray) -> None:
    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(weight).to(device=layer.weight.device, dtype=layer.weight.dtype))


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: tuple[int, ...], out_dim: int):
        super().__init__()
        layers: list[nn.Linear] = []
        prev = in_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev, hidden_dim))
            prev = hidden_dim
        self.layers = nn.ModuleList(layers)
        self.out = nn.Linear(prev, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.nn.functional.gelu(layer(x))
        return self.out(x)

    def reset_parameters_with_seed(self, seed: int, prefix: str) -> None:
        for index, layer in enumerate(self.layers):
            weight = _xavier_uniform(tuple(layer.weight.shape), _rng_for(seed, f"{prefix}.layers.{index}.weight"))
            _copy_linear_(layer, weight)
        out_weight = _xavier_uniform(tuple(self.out.weight.shape), _rng_for(seed, f"{prefix}.out.weight"))
        _copy_linear_(self.out, out_weight)


class TiedRecursiveReadout(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, depth: int):
        super().__init__()
        if depth < 1:
            raise ValueError("TiedRecursiveReadout depth must be >= 1.")
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.block = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)
        self.depth = depth
        self.depth_deltas = nn.Parameter(torch.zeros((depth, hidden_dim), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for depth_index in range(self.depth):
            h = torch.nn.functional.gelu(self.block(h + self.depth_deltas[depth_index]))
        return self.out(h)

    def reset_parameters_with_seed(self, seed: int, prefix: str) -> None:
        in_weight = _xavier_uniform(tuple(self.in_proj.weight.shape), _rng_for(seed, f"{prefix}.in_proj.weight"))
        block_weight = _xavier_uniform(tuple(self.block.weight.shape), _rng_for(seed, f"{prefix}.block.weight"))
        out_weight = _xavier_uniform(tuple(self.out.weight.shape), _rng_for(seed, f"{prefix}.out.weight"))
        _copy_linear_(self.in_proj, in_weight)
        _copy_linear_(self.block, block_weight)
        _copy_linear_(self.out, out_weight)
        with torch.no_grad():
            self.depth_deltas.zero_()


class RoutedSquaredReLUReadout(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_experts: int):
        super().__init__()
        if num_experts < 2:
            raise ValueError("RoutedSquaredReLUReadout requires at least 2 experts.")
        self.router = nn.Linear(in_dim, num_experts)
        self.num_experts = num_experts
        # Individual expert modules (kept for parameter naming / checkpoint compat)
        self.experts_in = nn.ModuleList(nn.Linear(in_dim, hidden_dim) for _ in range(num_experts))
        self.experts_out = nn.ModuleList(nn.Linear(hidden_dim, out_dim) for _ in range(num_experts))
        # Batched weight views — one large matmul instead of num_experts sequential ones.
        # Rebuilt from individual expert params before each forward via _sync_batched_weights.
        self._in_dim = in_dim
        self._hidden_dim = hidden_dim
        self._out_dim = out_dim

    def _batched_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Batched expert computation: single bmm per stage instead of a Python loop."""
        # Stack expert weights: [num_experts, hidden, in] and [num_experts, out, hidden]
        W_in = torch.stack([e.weight for e in self.experts_in])      # [E, H, I]
        b_in = torch.stack([e.bias for e in self.experts_in])        # [E, H]
        W_out = torch.stack([e.weight for e in self.experts_out])    # [E, O, H]
        b_out = torch.stack([e.bias for e in self.experts_out])      # [E, O]

        # x: [batch, seq, in_dim] -> [batch*seq, in_dim]
        shape = x.shape[:-1]
        x_flat = x.reshape(-1, self._in_dim)                        # [N, I]

        # All experts in one bmm: [E, N, I] @ [E, I, H] -> [E, N, H]
        x_exp = x_flat.unsqueeze(0).expand(self.num_experts, -1, -1)  # [E, N, I]
        hidden = torch.bmm(x_exp, W_in.transpose(1, 2))              # [E, N, H]
        hidden = hidden + b_in.unsqueeze(1)                           # [E, N, H]
        hidden = torch.relu(hidden)
        hidden = hidden * hidden                                      # squared ReLU

        # Second stage: [E, N, H] @ [E, H, O] -> [E, N, O]
        logits = torch.bmm(hidden, W_out.transpose(1, 2))            # [E, N, O]
        logits = logits + b_out.unsqueeze(1)                          # [E, N, O]

        # Route: [N, E] * [E, N, O] -> [N, O]
        route = torch.softmax(self.router(x_flat), dim=-1)            # [N, E]
        # einsum is cleaner than manual broadcast for this reduction
        out = torch.einsum("en,eno->no", route, logits.permute(1, 0, 2))
        return out.reshape(*shape, self._out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._batched_forward(x)

    def reset_parameters_with_seed(self, seed: int, prefix: str) -> None:
        router_weight = _xavier_uniform(tuple(self.router.weight.shape), _rng_for(seed, f"{prefix}.router.weight"))
        _copy_linear_(self.router, router_weight)
        for index, (expert_in, expert_out) in enumerate(zip(self.experts_in, self.experts_out)):
            in_weight = _xavier_uniform(tuple(expert_in.weight.shape), _rng_for(seed, f"{prefix}.experts_in.{index}.weight"))
            out_weight = _xavier_uniform(tuple(expert_out.weight.shape), _rng_for(seed, f"{prefix}.experts_out.{index}.weight"))
            _copy_linear_(expert_in, in_weight)
            _copy_linear_(expert_out, out_weight)


class GRUReadout(nn.Module):
    """Recurrent readout using GRU.

    Processes positions sequentially with hidden state carryover.
    Uses nn.GRU (not GRUCell) for CUDA-optimized sequence processing.
    """

    def __init__(self, in_features: int, out_features: int, config: Any) -> None:
        super().__init__()
        hidden_size = config.linear_hidden[0] if config.linear_hidden else 256
        self.gru = nn.GRU(in_features, hidden_size, batch_first=True)
        self.output_proj = nn.Linear(hidden_size, out_features)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len, in_features] -> [batch, seq_len, out_features]"""
        gru_out, _ = self.gru(x)  # [batch, seq, hidden]
        return self.output_proj(gru_out)  # [batch, seq, vocab]

    def reset_parameters_with_seed(self, seed: int, prefix: str) -> None:
        # nn.GRU has its own reset_parameters; we deterministically re-init the output proj.
        out_weight = _xavier_uniform(tuple(self.output_proj.weight.shape), _rng_for(seed, f"{prefix}.output_proj.weight"))
        _copy_linear_(self.output_proj, out_weight)
