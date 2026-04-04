from __future__ import annotations

"""MLX readout modules for Chronohorn descendant models."""

import mlx.core as mx
import mlx.nn as nn

from .common import _rng_for, _xavier_uniform


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: tuple[int, ...], out_dim: int):
        super().__init__()
        self.layers = []
        prev = in_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev, hidden_dim))
            prev = hidden_dim
        self.out = nn.Linear(prev, out_dim)

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = nn.gelu(layer(x))
        return self.out(x)

    def reset_parameters_with_seed(self, seed: int, prefix: str) -> None:
        for index, layer in enumerate(self.layers):
            layer.weight = mx.array(_xavier_uniform(tuple(layer.weight.shape), _rng_for(seed, f"{prefix}.layers.{index}.weight")))
            if layer.bias is not None:
                layer.bias = mx.zeros(layer.bias.shape, dtype=mx.float32)
        self.out.weight = mx.array(_xavier_uniform(tuple(self.out.weight.shape), _rng_for(seed, f"{prefix}.out.weight")))
        if self.out.bias is not None:
            self.out.bias = mx.zeros(self.out.bias.shape, dtype=mx.float32)


class TiedRecursiveReadout(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, depth: int):
        super().__init__()
        if depth < 1:
            raise ValueError("TiedRecursiveReadout depth must be >= 1.")
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.block = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)
        self.depth = depth
        self.depth_deltas = mx.zeros((depth, hidden_dim), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.in_proj(x)
        for depth_index in range(self.depth):
            h = nn.gelu(self.block(h + self.depth_deltas[depth_index]))
        return self.out(h)

    def reset_parameters_with_seed(self, seed: int, prefix: str) -> None:
        self.in_proj.weight = mx.array(_xavier_uniform(tuple(self.in_proj.weight.shape), _rng_for(seed, f"{prefix}.in_proj.weight")))
        self.block.weight = mx.array(_xavier_uniform(tuple(self.block.weight.shape), _rng_for(seed, f"{prefix}.block.weight")))
        self.out.weight = mx.array(_xavier_uniform(tuple(self.out.weight.shape), _rng_for(seed, f"{prefix}.out.weight")))
        if self.in_proj.bias is not None:
            self.in_proj.bias = mx.zeros(self.in_proj.bias.shape, dtype=mx.float32)
        if self.block.bias is not None:
            self.block.bias = mx.zeros(self.block.bias.shape, dtype=mx.float32)
        if self.out.bias is not None:
            self.out.bias = mx.zeros(self.out.bias.shape, dtype=mx.float32)
        self.depth_deltas = mx.zeros(self.depth_deltas.shape, dtype=mx.float32)


class RoutedSquaredReLUReadout(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_experts: int):
        super().__init__()
        if num_experts < 2:
            raise ValueError("RoutedSquaredReLUReadout requires at least 2 experts.")
        self.router = nn.Linear(in_dim, num_experts)
        self.experts_in = []
        self.experts_out = []
        for _ in range(num_experts):
            self.experts_in.append(nn.Linear(in_dim, hidden_dim))
            self.experts_out.append(nn.Linear(hidden_dim, out_dim))
        self.num_experts = num_experts

    def __call__(self, x: mx.array) -> mx.array:
        route = mx.softmax(self.router(x), axis=-1)
        expert_logits = []
        for expert_in, expert_out in zip(self.experts_in, self.experts_out):
            hidden = nn.relu(expert_in(x))
            hidden = hidden * hidden
            expert_logits.append(expert_out(hidden))
        stacked = mx.stack(expert_logits, axis=-2)
        return mx.sum(route[..., None] * stacked, axis=-2)

    def reset_parameters_with_seed(self, seed: int, prefix: str) -> None:
        self.router.weight = mx.array(_xavier_uniform(tuple(self.router.weight.shape), _rng_for(seed, f"{prefix}.router.weight")))
        if self.router.bias is not None:
            self.router.bias = mx.zeros(self.router.bias.shape, dtype=mx.float32)
        for index, (expert_in, expert_out) in enumerate(zip(self.experts_in, self.experts_out)):
            expert_in.weight = mx.array(
                _xavier_uniform(tuple(expert_in.weight.shape), _rng_for(seed, f"{prefix}.experts_in.{index}.weight"))
            )
            expert_out.weight = mx.array(
                _xavier_uniform(tuple(expert_out.weight.shape), _rng_for(seed, f"{prefix}.experts_out.{index}.weight"))
            )
            if expert_in.bias is not None:
                expert_in.bias = mx.zeros(expert_in.bias.shape, dtype=mx.float32)
            if expert_out.bias is not None:
                expert_out.bias = mx.zeros(expert_out.bias.shape, dtype=mx.float32)
