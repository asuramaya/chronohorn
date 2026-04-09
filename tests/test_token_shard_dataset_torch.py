from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import numpy as np


class _FakeTensor:
    def __init__(self, array: np.ndarray):
        self.array = array
        self.pin_calls = 0
        self.to_calls: list[tuple[str, object | None, bool]] = []

    def pin_memory(self):
        self.pin_calls += 1
        return self

    def to(self, device: str, dtype=None, non_blocking: bool = False):
        self.to_calls.append((device, dtype, non_blocking))
        return self


class _FakeTorch:
    def __init__(self):
        self.last_array: np.ndarray | None = None
        self.last_arrays: list[np.ndarray] = []
        self.long = "torch.long"

    def from_numpy(self, array: np.ndarray) -> _FakeTensor:
        self.last_array = array
        self.last_arrays.append(array)
        return _FakeTensor(array)


def _reload_module(monkeypatch, fake_torch: _FakeTorch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    sys.modules.pop("chronohorn.train.token_shard_dataset_torch", None)
    return importlib.import_module("chronohorn.train.token_shard_dataset_torch")


def test_to_torch_preserves_int32_and_pins_only_for_cuda(monkeypatch):
    fake_torch = _FakeTorch()
    module = _reload_module(monkeypatch, fake_torch)
    dataset = module.TorchTokenShardDataset(
        dataset=SimpleNamespace(),
        device="cuda:0",
        pin_memory=True,
    )
    source = np.arange(24, dtype=np.int32).reshape(4, 6)[:, ::2]

    tensor = dataset._to_torch(source)

    assert fake_torch.last_array is not None
    assert fake_torch.last_array.dtype == np.int32
    assert fake_torch.last_array.flags.c_contiguous
    assert tensor.pin_calls == 1
    assert tensor.to_calls == [("cuda:0", None, True)]


def test_to_torch_skips_pin_for_cpu(monkeypatch):
    fake_torch = _FakeTorch()
    module = _reload_module(monkeypatch, fake_torch)
    dataset = module.TorchTokenShardDataset(
        dataset=SimpleNamespace(),
        device="cpu",
        pin_memory=True,
    )
    source = np.arange(12, dtype=np.int32).reshape(3, 4)

    tensor = dataset._to_torch(source)

    assert fake_torch.last_array is not None
    assert fake_torch.last_array.dtype == np.int32
    assert tensor.pin_calls == 0
    assert tensor.to_calls == [("cpu", None, False)]


def test_batch_promotes_targets_to_long_on_device(monkeypatch):
    fake_torch = _FakeTorch()
    module = _reload_module(monkeypatch, fake_torch)
    dataset = module.TorchTokenShardDataset(
        dataset=SimpleNamespace(
            batch_numpy=lambda split, batch_size, seq_len: (
                np.arange(batch_size * seq_len, dtype=np.int32).reshape(batch_size, seq_len),
                np.arange(batch_size * seq_len, dtype=np.int32).reshape(batch_size, seq_len),
            )
        ),
        device="cuda:0",
        pin_memory=True,
    )

    x, y = dataset.batch("train", 2, 4)

    assert len(fake_torch.last_arrays) == 2
    assert fake_torch.last_arrays[0].dtype == np.int32
    assert fake_torch.last_arrays[1].dtype == np.int32
    assert x.to_calls == [("cuda:0", None, True)]
    assert y.to_calls == [("cuda:0", fake_torch.long, True)]
