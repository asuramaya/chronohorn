from __future__ import annotations

import importlib.metadata
from typing import Any


def _safe_call(func: Any) -> Any:
    try:
        return func()
    except Exception:
        return None


def build_backend_environment_metadata(*, backend: str, stack: Any, device: str) -> dict[str, Any]:
    if backend == "torch":
        torch = stack.torch
        metadata: dict[str, Any] = {
            "backend": "torch",
            "device": device,
            "version": getattr(torch, "__version__", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "mps_available": bool(torch.backends.mps.is_available()),
            "num_threads": int(torch.get_num_threads()),
            "num_interop_threads": int(torch.get_num_interop_threads()),
        }
        if hasattr(torch, "get_float32_matmul_precision"):
            metadata["float32_matmul_precision"] = torch.get_float32_matmul_precision()
        if hasattr(torch, "version"):
            metadata["cuda_version"] = getattr(torch.version, "cuda", None)
        cuda_backends = getattr(torch.backends, "cuda", None)
        if cuda_backends is not None and hasattr(cuda_backends, "matmul"):
            metadata["cuda_matmul_allow_tf32"] = getattr(cuda_backends.matmul, "allow_tf32", None)
        cudnn_backends = getattr(torch.backends, "cudnn", None)
        if cudnn_backends is not None:
            metadata["cudnn_enabled"] = getattr(cudnn_backends, "enabled", None)
            metadata["cudnn_allow_tf32"] = getattr(cudnn_backends, "allow_tf32", None)
            metadata["cudnn_version"] = _safe_call(cudnn_backends.version)
        if str(device).startswith("cuda") and torch.cuda.is_available():
            index = torch.cuda.current_device()
            metadata["device_index"] = int(index)
            metadata["device_name"] = torch.cuda.get_device_name(index)
            metadata["device_capability"] = list(torch.cuda.get_device_capability(index))
        return metadata
    if backend == "mlx":
        mx = stack.mx
        version = getattr(mx, "__version__", None)
        if version is None:
            try:
                version = importlib.metadata.version("mlx")
            except importlib.metadata.PackageNotFoundError:
                version = None
        return {
            "backend": "mlx",
            "device": device,
            "version": version,
        }
    raise ValueError(f"Unsupported backend environment metadata: {backend}")
