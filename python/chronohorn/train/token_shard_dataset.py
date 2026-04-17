from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

try:
    import sentencepiece as spm
except ImportError:
    import sys
    print("chronohorn: sentencepiece not installed — bpb will be unavailable", file=sys.stderr)
    spm = None

if TYPE_CHECKING:
    pass


TOKEN_SHARD_MAGIC = 20240520
TOKEN_SHARD_VERSION = 1
HEADER_INTS = 256
HEADER_BYTES = HEADER_INTS * np.dtype(np.int32).itemsize


def _read_shard_header(path: Path) -> tuple[int | None, int]:
    """Return (token_count_from_magic_header, file_size). token_count is None
    when the shard lacks the magic header and must be derived from file size.
    Reads only the 1KB header — not the full shard — so inventorying 80+ shards
    during dataset construction costs kilobytes, not gigabytes."""
    file_size = path.stat().st_size
    if file_size == 0:
        raise ValueError(f"token shard is empty: {path}")
    with open(path, "rb") as fh:
        head = fh.read(HEADER_BYTES)
    if len(head) >= HEADER_BYTES:
        header = np.frombuffer(head, dtype=np.int32, count=HEADER_INTS)
        if (
            header.size >= 3
            and int(header[0]) == TOKEN_SHARD_MAGIC
            and int(header[1]) == TOKEN_SHARD_VERSION
        ):
            return int(header[2]), file_size
    return None, file_size


def _load_token_shard(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"token shard not found: {path}")
    token_count, file_size = _read_shard_header(path)
    if token_count is not None:
        # Magic header present: mmap the payload region, skip header bytes.
        payload = np.memmap(
            path, dtype=np.uint16, mode="r",
            offset=HEADER_BYTES, shape=(token_count,),
        )
    else:
        n = file_size // np.dtype(np.uint16).itemsize
        if n == 0:
            raise ValueError(f"token shard produced zero tokens: {path}")
        payload = np.memmap(path, dtype=np.uint16, mode="r", shape=(n,))
    # Cast to int32 for downstream consumers (torch.long-compatible). This
    # allocates one shard's worth of int32 heap (2x the shard's uint16 bytes)
    # but the memmap itself is file-backed and not counted against cgroup anon.
    return payload.astype(np.int32, copy=True)


def _count_shard_tokens(path: Path) -> int:
    token_count, file_size = _read_shard_header(path)
    if token_count is not None:
        return token_count
    return file_size // np.dtype(np.uint16).itemsize


class TokenStream:
    def __init__(self, pattern: str, dataset_name: str):
        self.pattern = pattern
        self.dataset_name = dataset_name
        self.files = tuple(Path(path) for path in sorted(glob.glob(pattern)))
        if not self.files:
            raise FileNotFoundError(f"No token shards matched pattern: {pattern}")
        self.file_idx = -1
        self.tokens = np.empty((0,), dtype=np.int32)
        self.pos = 0
        self.reset()

    def reset(self) -> None:
        self.file_idx = -1
        self.tokens = np.empty((0,), dtype=np.int32)
        self.pos = 0
        self.next_file()

    def next_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = _load_token_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> np.ndarray:
        chunks: list[np.ndarray] = []
        left = n
        while left > 0:
            if self.pos >= self.tokens.size:
                self.next_file()
            step = min(left, int(self.tokens.size - self.pos))
            chunks.append(self.tokens[self.pos : self.pos + step])
            self.pos += step
            left -= step
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)


class PerLaneTokenStream:
    """B independent streams, one per batch lane.

    Enables persistent-substrate training: each lane carries its own
    substrate state forward, so the state-carry is actually on contiguous
    text. Lane i starts at file index i * len(files) // n_lanes, so lanes
    are placed on distinct shards to avoid cross-lane correlation.
    """

    def __init__(self, pattern: str, dataset_name: str, n_lanes: int):
        self.pattern = pattern
        self.dataset_name = dataset_name
        self.n_lanes = int(n_lanes)
        if self.n_lanes < 1:
            raise ValueError("n_lanes must be >= 1")
        files = tuple(Path(p) for p in sorted(glob.glob(pattern)))
        if not files:
            raise FileNotFoundError(f"No token shards matched pattern: {pattern}")
        self.files = files
        # Each lane owns its own TokenStream starting on a distinct shard.
        self.lanes: list[TokenStream] = []
        n_files = len(files)
        for i in range(self.n_lanes):
            lane = TokenStream.__new__(TokenStream)
            lane.pattern = pattern
            lane.dataset_name = f"{dataset_name}[lane{i}]"
            lane.files = files
            start = (i * n_files) // self.n_lanes
            lane.file_idx = start - 1  # next_file() bumps to `start`
            lane.tokens = np.empty((0,), dtype=np.int32)
            lane.pos = 0
            lane.next_file()
            self.lanes.append(lane)

    def take(self, seq_tokens: int) -> np.ndarray:
        """Return [n_lanes, seq_tokens] int32 — each row from its own cursor."""
        out = np.empty((self.n_lanes, seq_tokens), dtype=np.int32)
        for i, lane in enumerate(self.lanes):
            out[i] = lane.take(seq_tokens)
        return out


@dataclass
class TokenShardDataset:
    train_pattern: str
    test_pattern: str
    vocab_size: int
    tokenizer: str = "sp1024"
    tokenizer_path: str | None = None

    def __post_init__(self) -> None:
        self.train_stream = TokenStream(self.train_pattern, "train")
        self.test_stream = TokenStream(self.test_pattern, "test")
        self.train_files = self.train_stream.files
        self.test_files = self.test_stream.files
        self.train_token_count = sum(_count_shard_tokens(path) for path in self.train_files)
        self.test_token_count = sum(_count_shard_tokens(path) for path in self.test_files)
        self.source_path = f"{self.train_pattern} :: {self.test_pattern}"
        # Byte-level models: each token IS a byte, so tokens_per_byte = 1.0
        if self.tokenizer == "bytes":
            self.test_tokens_per_byte = 1.0
            self.test_bytes_per_token = 1.0
        elif self.tokenizer == "sp1024":
            # Fallback constants for sp1024 on fineweb — used when sentencepiece is unavailable
            _SP1024_TOKENS_PER_BYTE = 0.41052077856755560
            _SP1024_BYTES_PER_TOKEN = 2.43593029198018840
            self.test_tokens_per_byte = _SP1024_TOKENS_PER_BYTE
            self.test_bytes_per_token = _SP1024_BYTES_PER_TOKEN
        else:
            self.test_tokens_per_byte = None
            self.test_bytes_per_token = None
        self._base_bytes_lut: np.ndarray | None = None
        self._has_leading_space_lut: np.ndarray | None = None
        self._is_boundary_token_lut: np.ndarray | None = None
        if self.tokenizer_path is not None and spm is not None:
            try:
                processor = spm.SentencePieceProcessor(model_file=str(self.tokenizer_path))
            except Exception as exc:
                import sys
                print(f"chronohorn: sentencepiece model load failed ({self.tokenizer_path}): {exc}", file=sys.stderr)
                return
            (
                self._base_bytes_lut,
                self._has_leading_space_lut,
                self._is_boundary_token_lut,
            ) = _build_sentencepiece_luts(processor, self.vocab_size)
            self.test_tokens_per_byte, self.test_bytes_per_token = _compute_tokens_per_byte(
                self.test_files,
                base_bytes_lut=self._base_bytes_lut,
                has_leading_space_lut=self._has_leading_space_lut,
                is_boundary_token_lut=self._is_boundary_token_lut,
            )

    def batch_numpy(self, split: str, batch_size: int, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
        usable = batch_size * seq_len
        if usable <= 0:
            raise ValueError("batch_size * seq_len must be positive")
        stream = self.train_stream if split == "train" else self.test_stream
        chunk = stream.take(usable + 1)
        x = chunk[:-1].reshape(batch_size, seq_len)
        y = chunk[1:].reshape(batch_size, seq_len)
        return x, y

    def batch_numpy_stateful(
        self, split: str, batch_size: int, seq_len: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-lane contiguous batch, for persistent-state training.

        Each of the `batch_size` lanes reads from its own cursor, so state
        carried across calls reflects text immediately preceding the current
        batch on that lane. First call for a (split, batch_size) allocates
        the lane streams lazily.
        """
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if split == "train":
            if getattr(self, "_train_lane_stream", None) is None or \
                    self._train_lane_stream.n_lanes != batch_size:
                self._train_lane_stream = PerLaneTokenStream(
                    self.train_pattern, "train", batch_size)
            stream = self._train_lane_stream
        else:
            if getattr(self, "_test_lane_stream", None) is None or \
                    self._test_lane_stream.n_lanes != batch_size:
                self._test_lane_stream = PerLaneTokenStream(
                    self.test_pattern, "test", batch_size)
            stream = self._test_lane_stream
        chunk = stream.take(seq_len + 1)  # [B, S+1]
        x = chunk[:, :-1]
        y = chunk[:, 1:]
        return x, y

    def batch(self, split: str, batch_size: int, seq_len: int):
        import mlx.core as mx

        x, y = self.batch_numpy(split, batch_size, seq_len)
        return mx.array(x, dtype=mx.int32), mx.array(y, dtype=mx.int32)

    def target_token_stats(
        self,
        prev_ids: np.ndarray,
        target_ids: np.ndarray,
    ) -> tuple[int, float] | None:
        if (
            self._base_bytes_lut is None
            or self._has_leading_space_lut is None
            or self._is_boundary_token_lut is None
        ):
            return None
        prev_flat = np.asarray(prev_ids, dtype=np.int64).reshape(-1)
        target_flat = np.asarray(target_ids, dtype=np.int64).reshape(-1)
        if prev_flat.size == 0 or prev_flat.size != target_flat.size:
            return None
        total_bytes = _token_pair_total_bytes(
            prev_flat,
            target_flat,
            base_bytes_lut=self._base_bytes_lut,
            has_leading_space_lut=self._has_leading_space_lut,
            is_boundary_token_lut=self._is_boundary_token_lut,
        )
        if total_bytes <= 0:
            return None
        return int(target_flat.size), float(total_bytes)

    def sample_split_token_stats(
        self,
        split: str,
        *,
        batch_size: int,
        seq_len: int,
        batches: int,
    ) -> tuple[int, float] | None:
        if batches <= 0:
            return None
        total_tokens = 0
        total_bytes = 0.0
        for _ in range(int(batches)):
            prev_ids, target_ids = self.batch_numpy(split, batch_size, seq_len)
            stats = self.target_token_stats(prev_ids, target_ids)
            if stats is None:
                return None
            total_tokens += stats[0]
            total_bytes += stats[1]
        if total_tokens <= 0 or total_bytes <= 0:
            return None
        return total_tokens, total_bytes

    def sample_split_tokens_per_byte(
        self,
        split: str,
        *,
        batch_size: int,
        seq_len: int,
        batches: int,
    ) -> float | None:
        stats = self.sample_split_token_stats(
            split,
            batch_size=batch_size,
            seq_len=seq_len,
            batches=batches,
        )
        if stats is None:
            return None
        total_tokens, total_bytes = stats
        return float(total_tokens) / float(total_bytes)

    def sample_split_bytes_per_token(
        self,
        split: str,
        *,
        batch_size: int,
        seq_len: int,
        batches: int,
    ) -> float | None:
        stats = self.sample_split_token_stats(
            split,
            batch_size=batch_size,
            seq_len=seq_len,
            batches=batches,
        )
        if stats is None:
            return None
        total_tokens, total_bytes = stats
        return float(total_bytes) / float(total_tokens)

    def rollout_batch_numpy(
        self,
        split: str,
        batch_size: int,
        prompt_len: int,
        rollout_len: int,
        near_boundaries: bool = False,
        boundary_band: int = 128,
    ) -> tuple[np.ndarray, np.ndarray]:
        stream = self.train_stream if split == "train" else self.test_stream
        max_start = stream.tokens.size - prompt_len - rollout_len
        if max_start < 0:
            raise ValueError(
                f"Dataset split '{split}' is too short for prompt_len={prompt_len} and rollout_len={rollout_len}."
            )

        starts = self._sample_rollout_starts(
            max_start=max_start,
            batch_size=batch_size,
            prompt_len=prompt_len,
            switch_points=(),
            near_boundaries=near_boundaries,
            boundary_band=boundary_band,
        )
        prompt_offsets = np.arange(prompt_len, dtype=np.int32)
        target_offsets = np.arange(prompt_len, prompt_len + rollout_len, dtype=np.int32)
        prompts = stream.tokens[starts[:, None] + prompt_offsets[None, :]]
        targets = stream.tokens[starts[:, None] + target_offsets[None, :]]
        return prompts, targets

    def rollout_batch(
        self,
        split: str,
        batch_size: int,
        prompt_len: int,
        rollout_len: int,
        near_boundaries: bool = False,
        boundary_band: int = 128,
    ):
        import mlx.core as mx

        prompts, targets = self.rollout_batch_numpy(
            split=split,
            batch_size=batch_size,
            prompt_len=prompt_len,
            rollout_len=rollout_len,
            near_boundaries=near_boundaries,
            boundary_band=boundary_band,
        )
        return mx.array(prompts), mx.array(targets)

    @staticmethod
    def _sample_rollout_starts(
        max_start: int,
        batch_size: int,
        prompt_len: int,
        switch_points: tuple[int, ...],
        near_boundaries: bool,
        boundary_band: int,
    ) -> np.ndarray:
        if not near_boundaries or not switch_points:
            return np.random.randint(0, max_start + 1, size=batch_size)

        candidates = []
        for boundary in switch_points:
            start_lo = max(0, boundary - prompt_len - boundary_band)
            start_hi = min(max_start, boundary - prompt_len)
            if start_hi < start_lo:
                continue
            candidates.append(np.arange(start_lo, start_hi + 1, dtype=np.int32))

        if not candidates:
            return np.random.randint(0, max_start + 1, size=batch_size)

        pool = np.unique(np.concatenate(candidates))
        replace = len(pool) < batch_size
        return np.random.choice(pool, size=batch_size, replace=replace)


def _find_tokenizer(data_root: Path, vocab_size: int) -> str | None:
    """Search for the sentencepiece tokenizer model in likely locations."""
    model_name = f"fineweb_{vocab_size}_bpe.model"
    candidates = [
        data_root.parents[1] / "tokenizers" / model_name,           # /data/tokenizers/
        data_root.parent / "tokenizers" / model_name,                # /data/chronohorn/tokenizers/
        data_root / ".." / "tokenizers" / model_name,                # relative ../tokenizers/
        Path("/data/chronohorn/tokenizers") / model_name,            # absolute fallback
        Path("data/tokenizers") / model_name,                        # repo-relative
    ]
    for path in candidates:
        resolved = path.resolve()
        if resolved.exists():
            return str(resolved)
    return None


def build_token_shard_dataset(data_root: str | Path, vocab_size: int = 1024) -> TokenShardDataset:
    root = Path(data_root).expanduser()
    is_byte_level = vocab_size == 256
    tokenizer_path = None if is_byte_level else _find_tokenizer(root, vocab_size)
    return TokenShardDataset(
        train_pattern=str(root / "fineweb_train_*.bin"),
        test_pattern=str(root / "fineweb_val_*.bin"),
        vocab_size=vocab_size,
        tokenizer="bytes" if is_byte_level else "sp1024",
        tokenizer_path=tokenizer_path,
    )


def _build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_lut = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_lut = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_lut = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_lut[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_lut[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_lut[token_id] = True
            piece = piece[1:]
        base_bytes_lut[token_id] = len(piece.encode("utf-8"))
    return base_bytes_lut, has_leading_space_lut, is_boundary_token_lut


def _token_pair_total_bytes(
    prev_ids: np.ndarray,
    tgt_ids: np.ndarray,
    *,
    base_bytes_lut: np.ndarray,
    has_leading_space_lut: np.ndarray,
    is_boundary_token_lut: np.ndarray,
) -> float:
    bytes_np = base_bytes_lut[tgt_ids].astype(np.int16, copy=True)
    bytes_np += (
        has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
    ).astype(np.int16, copy=False)
    return float(bytes_np.astype(np.float64).sum())


def _compute_tokens_per_byte(
    shard_paths: tuple[Path, ...],
    *,
    base_bytes_lut: np.ndarray,
    has_leading_space_lut: np.ndarray,
    is_boundary_token_lut: np.ndarray,
) -> tuple[float, float]:
    tokens = np.ascontiguousarray(
        np.concatenate([_load_token_shard(path) for path in shard_paths], axis=0)
    )
    prev_ids = tokens[:-1]
    tgt_ids = tokens[1:]
    total_tokens = float(tgt_ids.size)
    if total_tokens == 0:
        raise ValueError("cannot compute tokens_per_byte from empty shard set")
    total_bytes = _token_pair_total_bytes(
        prev_ids,
        tgt_ids,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
    )
    if total_bytes <= 0:
        raise ValueError(f"total_bytes={total_bytes} — sentencepiece LUTs may be corrupt")
    tokens_per_byte = total_tokens / total_bytes
    return tokens_per_byte, total_bytes / total_tokens
