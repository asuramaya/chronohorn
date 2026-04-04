"""Tests for bpb calibration — ensure the unit systems are correct and never confused."""
from __future__ import annotations

import math

import pytest


# Two different ratios — the source of the 2.44 vs 1.235 confusion:
#
# TEXT bytes per token (sentencepiece): ~2.436
#   Each sp1024 token covers ~2.4 bytes of original text.
#   Measured by ctrl-hemi-patch4 pilot: tokens_per_byte = 0.4105
#   This is what bpb needs: bpb = bpt / text_bytes_per_token
#
# SHARD bytes per token (uint16 file): ~1.235
#   Each token is stored as 2 bytes in the shard file, but 10B text bytes
#   became 8.1B tokens, so 10B/8.1B = 1.235 shard bytes per token.
#   This is NOT what bpb needs — it measures disk, not text.

TEXT_BYTES_PER_TOKEN = 2.436     # from sentencepiece on FineWeb sp1024
SHARD_BYTES_PER_TOKEN = 1.235   # from 10B file bytes / 8.1B tokens


def test_text_bytes_per_token_is_near_2_44():
    """The sentencepiece-measured ratio is ~2.44, not ~1.24."""
    assert abs(TEXT_BYTES_PER_TOKEN - 2.44) < 0.01
    assert abs(TEXT_BYTES_PER_TOKEN - 1.235) > 1.0  # NOT the shard ratio


def test_bpb_formula_uses_text_not_shard():
    """bpb = bpt / text_bytes_per_token. Using shard ratio gives 2x error."""
    bpt = 4.3  # typical polyhash bits per token
    bpb_correct = bpt / TEXT_BYTES_PER_TOKEN
    bpb_wrong = bpt / SHARD_BYTES_PER_TOKEN
    assert abs(bpb_correct - 1.76) < 0.1   # correct: ~1.76 bpb
    assert abs(bpb_wrong - 3.48) < 0.1     # wrong: ~3.48 bpb (2x too high)


def test_uniform_prediction_bpb():
    """A model predicting uniformly over 1024 tokens: bpb = 8 / ~2.44 ≈ 4.1."""
    loss = math.log(1024)
    bpt = loss / math.log(2)  # 10.0
    bpb = bpt / TEXT_BYTES_PER_TOKEN
    assert abs(bpt - 10.0) < 0.001
    assert abs(bpb - 10.0 / 2.436) < 0.01


def test_causal_bank_results_use_own_measurement():
    """Causal-bank results measure tokens_per_byte from sentencepiece directly."""
    # The ctrl-hemi-patch4 pilot measured:
    tokens_per_byte = 0.4105
    measured_text_bpt = 1.0 / tokens_per_byte  # 2.436
    assert abs(measured_text_bpt - TEXT_BYTES_PER_TOKEN) < 0.01
