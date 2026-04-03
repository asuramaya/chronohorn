from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .token_shard_dataset import TokenShardDataset, build_token_shard_dataset


@dataclass
class TorchTokenShardDataset:
    dataset: TokenShardDataset
    device: str = "cuda"
    pin_memory: bool = False

    @property
    def vocab_size(self) -> int:
        return self.dataset.vocab_size

    @property
    def train_files(self):
        return self.dataset.train_files

    @property
    def test_files(self):
        return self.dataset.test_files

    @property
    def train_token_count(self) -> int:
        return self.dataset.train_token_count

    @property
    def test_token_count(self) -> int:
        return self.dataset.test_token_count

    @property
    def test_tokens_per_byte(self) -> float | None:
        return self.dataset.test_tokens_per_byte

    @property
    def test_bytes_per_token(self) -> float | None:
        return self.dataset.test_bytes_per_token

    def sample_split_tokens_per_byte(
        self,
        split: str,
        *,
        batch_size: int,
        seq_len: int,
        batches: int,
    ) -> float | None:
        return self.dataset.sample_split_tokens_per_byte(
            split,
            batch_size=batch_size,
            seq_len=seq_len,
            batches=batches,
        )

    def sample_split_bytes_per_token(
        self,
        split: str,
        *,
        batch_size: int,
        seq_len: int,
        batches: int,
    ) -> float | None:
        return self.dataset.sample_split_bytes_per_token(
            split,
            batch_size=batch_size,
            seq_len=seq_len,
            batches=batches,
        )

    def _to_torch(self, array: np.ndarray):
        import torch

        tensor = torch.from_numpy(array.astype(np.int64, copy=False))
        if self.pin_memory:
            tensor = tensor.pin_memory()
        return tensor.to(self.device, non_blocking=self.pin_memory and self.device.startswith("cuda"))

    def batch(self, split: str, batch_size: int, seq_len: int):
        x, y = self.dataset.batch_numpy(split, batch_size, seq_len)
        return self._to_torch(x), self._to_torch(y)

    def rollout_batch(
        self,
        split: str,
        batch_size: int,
        prompt_len: int,
        rollout_len: int,
        near_boundaries: bool = False,
        boundary_band: int = 128,
    ):
        prompts, targets = self.dataset.rollout_batch_numpy(
            split=split,
            batch_size=batch_size,
            prompt_len=prompt_len,
            rollout_len=rollout_len,
            near_boundaries=near_boundaries,
            boundary_band=boundary_band,
        )
        return self._to_torch(prompts), self._to_torch(targets)


def build_token_shard_torch_dataset(
    data_root: str,
    *,
    vocab_size: int = 1024,
    device: str = "cuda",
    pin_memory: bool = False,
) -> TorchTokenShardDataset:
    dataset = build_token_shard_dataset(data_root, vocab_size=vocab_size)
    return TorchTokenShardDataset(dataset=dataset, device=device, pin_memory=pin_memory)
