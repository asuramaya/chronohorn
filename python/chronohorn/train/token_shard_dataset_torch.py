from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .token_shard_dataset import TokenShardDataset, build_token_shard_dataset


@dataclass
class TorchTokenShardDataset:
    dataset: TokenShardDataset
    device: str = "cuda"
    pin_memory: bool = False
    _stochastic_tokenizer: StochasticTokenizer | None = None

    def _cuda_transfer_enabled(self) -> bool:
        return self.pin_memory and str(self.device).startswith("cuda")

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

    def _to_torch(self, array: np.ndarray, *, dtype=None):
        import torch

        tensor = torch.from_numpy(np.ascontiguousarray(array))
        if self._cuda_transfer_enabled():
            tensor = tensor.pin_memory()
        if dtype is None:
            return tensor.to(self.device, non_blocking=self._cuda_transfer_enabled())
        return tensor.to(self.device, dtype=dtype, non_blocking=self._cuda_transfer_enabled())

    def batch(self, split: str, batch_size: int, seq_len: int):
        import torch

        if self._stochastic_tokenizer is not None and split == "train":
            x, y = self._stochastic_tokenizer.stochastic_batch(
                self.dataset, split, batch_size, seq_len
            )
            return self._to_torch(x), self._to_torch(y, dtype=torch.long)
        x, y = self.dataset.batch_numpy(split, batch_size, seq_len)
        return self._to_torch(x), self._to_torch(y, dtype=torch.long)

    def batch_stateful(self, split: str, batch_size: int, seq_len: int):
        """Per-lane contiguous batch — for persistent-substrate training."""
        import torch
        x, y = self.dataset.batch_numpy_stateful(split, batch_size, seq_len)
        return self._to_torch(x), self._to_torch(y, dtype=torch.long)

    def rollout_batch(
        self,
        split: str,
        batch_size: int,
        prompt_len: int,
        rollout_len: int,
        near_boundaries: bool = False,
        boundary_band: int = 128,
    ):
        import torch

        prompts, targets = self.dataset.rollout_batch_numpy(
            split=split,
            batch_size=batch_size,
            prompt_len=prompt_len,
            rollout_len=rollout_len,
            near_boundaries=near_boundaries,
            boundary_band=boundary_band,
        )
        return self._to_torch(prompts), self._to_torch(targets, dtype=torch.long)


class StochasticTokenizer:
    """BPE dropout wrapper for stochastic tokenization during training.

    Decodes token IDs back to text via sentencepiece, then re-encodes
    with sentencepiece sampling (nbest_size=-1, alpha=dropout_alpha).
    This produces variable tokenizations of the same text, making the
    model tokenizer-invariant.
    """

    def __init__(self, model_path: str, *, dropout_alpha: float = 0.1):
        import sentencepiece as spm
        self._sp = spm.SentencePieceProcessor(model_file=model_path)
        self._alpha = dropout_alpha

    def stochastic_encode(self, token_ids: np.ndarray, seq_len: int) -> np.ndarray:
        """Decode token_ids to text, re-encode with BPE dropout, pad/truncate to seq_len."""
        text = self._sp.decode(token_ids.tolist())
        # Re-encode with sampling (BPE dropout)
        new_ids = self._sp.encode(text, enable_sampling=True, nbest_size=-1, alpha=self._alpha)
        # Pad or truncate to seq_len + 1 (need input + target)
        arr = np.array(new_ids, dtype=np.int32)
        if len(arr) >= seq_len + 1:
            return arr[:seq_len + 1]
        # Pad with continuation from the stream
        padded = np.zeros(seq_len + 1, dtype=np.int32)
        padded[:len(arr)] = arr
        return padded

    def stochastic_batch(
        self,
        dataset: TokenShardDataset,
        split: str,
        batch_size: int,
        seq_len: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate a batch with stochastic tokenization (BPE dropout)."""
        # Get a normal batch first (slightly longer to handle re-tokenization length changes)
        stream = dataset.train_stream if split == "train" else dataset.test_stream
        x_list = []
        y_list = []
        for _ in range(batch_size):
            tokens = stream.take(seq_len * 2)  # take extra to handle expansion
            retokenized = self.stochastic_encode(tokens, seq_len)
            x_list.append(retokenized[:seq_len])
            y_list.append(retokenized[1:seq_len + 1])
        return np.stack(x_list), np.stack(y_list)


def build_token_shard_torch_dataset(
    data_root: str,
    *,
    vocab_size: int = 1024,
    device: str = "cuda",
    pin_memory: bool = False,
) -> TorchTokenShardDataset:
    dataset = build_token_shard_dataset(data_root, vocab_size=vocab_size)
    return TorchTokenShardDataset(dataset=dataset, device=device, pin_memory=pin_memory)
