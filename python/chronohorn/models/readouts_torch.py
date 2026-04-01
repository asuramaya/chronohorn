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
        self.experts_in = nn.ModuleList(nn.Linear(in_dim, hidden_dim) for _ in range(num_experts))
        self.experts_out = nn.ModuleList(nn.Linear(hidden_dim, out_dim) for _ in range(num_experts))
        self.num_experts = num_experts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        route = torch.softmax(self.router(x), dim=-1)
        expert_logits = []
        for expert_in, expert_out in zip(self.experts_in, self.experts_out):
            hidden = torch.relu(expert_in(x))
            hidden = hidden * hidden
            expert_logits.append(expert_out(hidden))
        stacked = torch.stack(expert_logits, dim=-2)
        return torch.sum(route.unsqueeze(-1) * stacked, dim=-2)

    def reset_parameters_with_seed(self, seed: int, prefix: str) -> None:
        router_weight = _xavier_uniform(tuple(self.router.weight.shape), _rng_for(seed, f"{prefix}.router.weight"))
        _copy_linear_(self.router, router_weight)
        for index, (expert_in, expert_out) in enumerate(zip(self.experts_in, self.experts_out)):
            in_weight = _xavier_uniform(tuple(expert_in.weight.shape), _rng_for(seed, f"{prefix}.experts_in.{index}.weight"))
            out_weight = _xavier_uniform(tuple(expert_out.weight.shape), _rng_for(seed, f"{prefix}.experts_out.{index}.weight"))
            _copy_linear_(expert_in, in_weight)
            _copy_linear_(expert_out, out_weight)


class GRUReadout(nn.Module):
    """Recurrent readout using a GRU cell.

    Processes positions sequentially, carrying hidden state forward.
    The hidden state lets the readout accumulate information across positions,
    making it a compute absorber that improves with more training.
    """

    def __init__(self, in_features: int, out_features: int, config: Any) -> None:
        super().__init__()
        hidden_size = config.linear_hidden[0] if config.linear_hidden else 256
        self.gru = nn.GRUCell(in_features, hidden_size)
        self.output_proj = nn.Linear(hidden_size, out_features)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len, in_features] -> [batch, seq_len, out_features]"""
        batch_size, seq_len, _ = x.shape
        device = x.device

        h = torch.zeros(batch_size, self.hidden_size, device=device, dtype=x.dtype)
        outputs = []
        for t in range(seq_len):
            h = self.gru(x[:, t, :], h)
            outputs.append(self.output_proj(h))
        return torch.stack(outputs, dim=1)

    def reset_parameters_with_seed(self, seed: int, prefix: str) -> None:
        # GRUCell has its own reset_parameters; we deterministically re-init the output proj.
        out_weight = _xavier_uniform(tuple(self.output_proj.weight.shape), _rng_for(seed, f"{prefix}.output_proj.weight"))
        _copy_linear_(self.output_proj, out_weight)
