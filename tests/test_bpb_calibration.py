"""Tests for bpb calibration — ensure the tokenizer constant is correct and consistent."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

# The ground truth: 10B bytes of fineweb text tokenized into ~8.1B sp1024 tokens
EXPECTED_TOTAL_TOKENS = 8_100_041_472
EXPECTED_TOTAL_BYTES = 10_000_000_000
CORRECT_BYTES_PER_TOKEN = EXPECTED_TOTAL_BYTES / EXPECTED_TOTAL_TOKENS  # ~1.2346


def test_bytes_per_token_is_not_2_44():
    """The old hardcoded 2.44 was wrong. Verify the correct constant."""
    assert abs(CORRECT_BYTES_PER_TOKEN - 1.2346) < 0.001
    assert abs(CORRECT_BYTES_PER_TOKEN - 2.44) > 1.0  # nowhere near 2.44


def test_bpb_from_bpt_formula():
    """bpb = bits_per_token / bytes_per_token. Verify the math."""
    # A model with cross-entropy loss of ln(256) predicts uniformly over bytes
    # That's 8 bits per token = 8 bpt
    loss = math.log(256)
    bpt = loss / math.log(2)  # 8.0
    bpb = bpt / CORRECT_BYTES_PER_TOKEN
    assert abs(bpt - 8.0) < 0.001
    assert abs(bpb - 8.0 / 1.2346) < 0.01


def test_all_polyhash_results_use_correct_bpb(tmp_path):
    """Every polyhash result JSON should have bpb = bpt / 1.2346."""
    results_dir = Path("out/results")
    if not results_dir.exists():
        pytest.skip("no results directory")

    wrong = []
    for f in sorted(results_dir.glob("v*.json")):
        with open(f) as fh:
            d = json.load(fh)
        bpt = d["model"].get("test_bits_per_token")
        bpb = d["model"].get("test_bpb")
        if bpt is None or bpb is None:
            continue
        expected_bpb = bpt / CORRECT_BYTES_PER_TOKEN
        if abs(bpb - expected_bpb) > 0.001:
            wrong.append(f"{f.stem}: bpb={bpb:.4f} expected={expected_bpb:.4f}")

    assert wrong == [], f"Miscalibrated results:\n" + "\n".join(wrong)


def test_causal_bank_results_not_recalibrated():
    """Causal-bank results use their own evaluation — should NOT use sp1024 conversion."""
    results_dir = Path("out/results")
    if not results_dir.exists():
        pytest.skip("no results directory")

    for f in sorted(results_dir.glob("sub*.json")):
        with open(f) as fh:
            d = json.load(fh)
        unit = d["model"].get("_bpb_unit", "")
        assert unit == "causal-bank-native", (
            f"{f.stem} has unit={unit!r}, should be 'causal-bank-native'"
        )
