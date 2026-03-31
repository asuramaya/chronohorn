"""Chronohorn model-family implementations.

This package is the concrete descendant layer, not the shared OPC kernel.
Import concrete backend modules directly:

- ``chronohorn.models.causal_bank_mlx`` for MLX/Metal
- ``chronohorn.models.causal_bank_torch`` for Torch/CUDA
- ``open_predictive_coder.causal_bank`` for backend-neutral causal-bank family/config/substrate
- ``chronohorn.models.readouts_mlx`` for MLX readouts
- ``chronohorn.models.readouts_torch`` for Torch readouts
"""

from __future__ import annotations

__all__: list[str] = []
