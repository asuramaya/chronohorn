from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass


def test_causal_bank_benchmark_parser_does_not_require_data_root():
    from chronohorn.families.causal_bank.training.benchmark_causal_bank_torch import build_parser

    parser = build_parser()
    args = parser.parse_args(["--seed", "42", "--json", "out/bench.json"])

    assert args.data_root == ""
    assert args.warmup_steps == 10
    assert args.measure_train_steps == 50
    assert args.measure_eval_batches == 50


def test_causal_bank_benchmark_eval_summary():
    from chronohorn.families.causal_bank.training.benchmark_causal_bank_torch import (
        _summarize_eval_benchmark,
    )

    summary = _summarize_eval_benchmark(
        {
            "tokens_per_step": 4096,
            "forward_total_flops_per_token": 2_000_000_000.0,
        },
        eval_batches=20,
        elapsed_sec=4.0,
    )

    assert summary["tokens_completed"] == 81_920
    assert summary["tokens_per_second"] == 20_480.0
    assert summary["estimated_forward_tflops"] == 40.96
    assert summary["seconds_per_batch"] == 0.2


def test_causal_bank_benchmark_payload_includes_train_and_eval():
    from chronohorn.families.causal_bank.training.benchmark_causal_bank_torch import (
        _build_benchmark_payload,
    )

    @dataclass
    class _Train:
        seq_len: int
        batch_size: int
        steps: int
        learning_rate: float
        weight_decay: float

    @dataclass
    class _Runtime:
        train: _Train

    runtime = _Runtime(train=_Train(seq_len=256, batch_size=16, steps=1000, learning_rate=0.001, weight_decay=1e-5))
    args = Namespace(
        json="out/bench.json",
        warmup_steps=10,
        warmup_eval_batches=None,
        measure_train_steps=50,
        measure_eval_batches=40,
        torch_compile=False,
        variant="base",
        scale=8.0,
        seed=42,
    )
    config = Namespace(
        linear_modes=2048,
        local_window=8,
        share_embedding=False,
        linear_impl="kernel",
        linear_readout_kind="mlp",
        linear_readout_depth=1,
        linear_readout_num_experts=4,
        readout_bands=4,
        linear_half_life_min=1.5,
        linear_half_life_max=16.0,
        oscillatory_frac=0.0,
        oscillatory_schedule="logspace",
        input_proj_scheme="random",
        substrate_mode="frozen",
        state_dim=32,
        state_impl="scan",
        num_heads=8,
        num_hemispheres=1,
        block_mixing_ratio=0.25,
        patch_size=1,
        patch_causal_decoder="none",
        static_bank_gate=False,
        bank_gate_span=0.5,
        embedding_dim=256,
        linear_hidden=(1024,),
        local_hidden=(1024,),
        local_scale=0.25,
        mix_mode="additive",
    )

    payload = _build_benchmark_payload(
        args=args,
        runtime=runtime,
        config=config,
        device="cuda",
        backend_environment={"cuda": True},
        param_count=7_000_000,
        performance_estimate={"tokens_per_step": 4096},
        train_summary={"tokens_per_second": 20_000.0},
        eval_summary={"tokens_per_second": 25_000.0},
        peak_memory_mb=1234.5,
    )

    assert payload["kind"] == "benchmark"
    assert payload["benchmark"]["train"]["tokens_per_second"] == 20_000.0
    assert payload["benchmark"]["eval"]["tokens_per_second"] == 25_000.0
    assert payload["model"]["state_impl"] == "scan"
    assert payload["model"]["readout_bands"] == 4
