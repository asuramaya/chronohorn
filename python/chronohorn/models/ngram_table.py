"""N-gram lookup table for trust-routing prediction.

The table is built offline from training data and packed into the artifact.
For testing, a synthetic table can be built from a small sample.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Any


class NgramTable:
    """Fast n-gram lookup table."""

    def __init__(self, vocab_size: int = 1024, max_order: int = 4, bucket_count: int = 8192) -> None:
        self.vocab_size = vocab_size
        self.max_order = max_order
        self.bucket_count = bucket_count

        # Unigram: [vocab]
        self.unigram = np.zeros(vocab_size, dtype=np.float32)
        # Bigram: [vocab, vocab]
        self.bigram = np.zeros((vocab_size, vocab_size), dtype=np.float32)
        # Trigram+: hashed [bucket_count, vocab]
        self.trigram = np.zeros((bucket_count, vocab_size), dtype=np.float32)

        self._total = 0

    def build_from_tokens(self, tokens: np.ndarray) -> None:
        """Build table from a token sequence (1D array of ints)."""
        tokens = tokens.astype(np.int64)
        # Filter out-of-vocab tokens
        mask = (tokens >= 0) & (tokens < self.vocab_size)
        if not mask.all():
            tokens = tokens[mask]
        n = len(tokens)
        self._total = n

        # Unigram (vectorized)
        np.add.at(self.unigram, tokens, 1)

        # Bigram (vectorized)
        np.add.at(self.bigram, (tokens[:-1], tokens[1:]), 1)

        # Trigram (hashed, vectorized)
        if n >= 3:
            ctx0 = tokens[:-2]
            ctx1 = tokens[1:-1]
            target = tokens[2:]
            h = (ctx0 * 2654435761 + ctx1 * 2246822519) % self.bucket_count
            np.add.at(self.trigram, (h, target), 1)

    def lookup_probs(self, context: np.ndarray) -> tuple[np.ndarray, float]:
        """Look up probability distribution given context bytes.

        Returns (probs, confidence) where confidence is the total count
        for this context (0 = never seen, high = reliable).
        """
        v = self.vocab_size

        # Start with unigram
        probs = self.unigram.copy()
        confidence = self._total

        if len(context) >= 1:
            prev = int(context[-1])
            row = self.bigram[prev]
            row_total = row.sum()
            if row_total > 0:
                probs = row
                confidence = row_total

        if len(context) >= 2:
            h = (int(context[-2]) * 2654435761 + int(context[-1]) * 2246822519) % self.bucket_count
            row = self.trigram[h]
            row_total = row.sum()
            if row_total > 0:
                probs = row
                confidence = row_total

        # Normalize
        total = probs.sum()
        if total > 0:
            probs = probs / total
        else:
            probs = np.ones(v, dtype=np.float32) / v

        return probs, confidence

    def save(self, path: str) -> None:
        np.savez_compressed(path, unigram=self.unigram, bigram=self.bigram,
                           trigram=self.trigram, total=self._total)

    @classmethod
    def load(cls, path: str) -> "NgramTable":
        data = np.load(path)
        table = cls(vocab_size=len(data["unigram"]), bucket_count=len(data["trigram"]))
        table.unigram = data["unigram"]
        table.bigram = data["bigram"]
        table.trigram = data["trigram"]
        table._total = float(data["total"])
        return table
