#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np

from chronohorn.engine.backend_metadata import build_backend_environment_metadata
from chronohorn.engine.budgets import DEFAULT_GOLF_V1_BUDGET
from chronohorn.engine.forecasting import build_result_forecast
from chronohorn.engine.optimizer_policy import (
    build_adamw_kwargs,
    build_adamw_policy_defaults,
    build_train_policy_metadata,
)
from chronohorn.engine.performance import (
    bits_per_token_from_loss,
    format_observed_training_performance,
    summarize_observed_training_performance,
)
from chronohorn.engine.probes import (
    format_probe_plan,
    probe_entry_by_step,
    resolve_probe_plan,
)
from chronohorn.engine.signatures import summarize_named_arrays
from chronohorn.families.causal_bank import CAUSAL_BANK_TRAINING_ADAPTER
from chronohorn.families.causal_bank.training.causal_bank_training_primitives import (
    build_causal_bank_training_runtime,
)
from chronohorn.families.causal_bank.training.causal_bank_training_stack import load_training_backend_stack
from chronohorn.families.causal_bank.training.causal_bank_training_support import (
    build_compute_accounting_inputs,
    build_probe_compute_accounting_inputs,
)
from chronohorn.service_log import configure_service_log, service_log


def _patch_cross_entropy(
    logits,
    y,
    *,
    reduction: str = "mean",
    ignore_index: int = -100,
):
    """Cross-entropy that accepts [B, T, V] or [B, T, N, V] logits.

    For 4-d logits, builds N-shifted targets with ignore_index at the tail of
    each shifted copy so out-of-bounds positions contribute no loss.
    """
    import torch
    import torch.nn.functional as F
    if logits.dim() == 3:
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1),
            reduction=reduction,
            ignore_index=ignore_index,
        )
    if logits.dim() != 4:
        raise ValueError(f"logits must be 3-d or 4-d, got {logits.dim()}-d")
    b, t, n, v = logits.shape
    if y.shape != (b, t):
        raise ValueError(f"y shape {tuple(y.shape)} must be (B,T)=({b},{t})")
    targets = torch.full((b, t, n), ignore_index, dtype=y.dtype, device=y.device)
    targets[:, :, 0] = y
    for i in range(1, n):
        if i < t:
            targets[:, : t - i, i] = y[:, i:]
    return F.cross_entropy(
        logits.reshape(-1, v),
        targets.reshape(-1),
        reduction=reduction,
        ignore_index=ignore_index,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chronohorn Torch/CUDA causal-bank trainer on token shards."
    )
    CAUSAL_BANK_TRAINING_ADAPTER.add_training_arguments(parser, backend="torch")
    return parser


def seed_everything(seed: int) -> None:
    import random

    stack = load_training_backend_stack("torch")
    torch = stack.torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(raw: str | None) -> str:
    stack = load_training_backend_stack("torch")
    torch = stack.torch
    if raw:
        return raw
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def config_for_variant(args: argparse.Namespace, seq_len: int, stack):
    return CAUSAL_BANK_TRAINING_ADAPTER.build_variant_config(
        args,
        ConfigClass=stack.ConfigClass,
        scale_config=stack.scale_config,
        seq_len=seq_len,
        vocab_size=args.vocab_size,
    )


def build_runtime(args: argparse.Namespace, stack):
    return build_causal_bank_training_runtime(
        args,
        RuntimeConfig=stack.RuntimeConfig,
        train_config_for_profile=stack.train_config_for_profile,
    )


def assert_safe_readout_budget(
    args: argparse.Namespace,
    config,
    *,
    baseline_linear_hidden: tuple[int, ...],
    out_dim: int,
) -> None:
    CAUSAL_BANK_TRAINING_ADAPTER.validate_config(
        args,
        config,
        baseline_linear_hidden=baseline_linear_hidden,
        out_dim=out_dim,
    )


def evaluate(model, dataset, train_config, split: str, *, eval_batches: int | None = None, amp_dtype=None, seq_len: int | None = None, batch_size: int | None = None) -> float:  # noqa: S307
    stack = load_training_backend_stack("torch")
    F = stack.functional
    batches = train_config.eval_batches if eval_batches is None else eval_batches
    eff_seq_len = train_config.seq_len if seq_len is None else seq_len
    eff_batch_size = train_config.batch_size if batch_size is None else batch_size
    # CUDA graphs / reduce-overhead compile reuses the training forward's output
    # tensor buffer across steps. The probe path reads that tensor later with
    # reduction='none' + reshape which triggers the overwrite detector. Mark the
    # step boundary so the compiler materializes fresh outputs for this eval.
    # Safe no-op under default compile / eager.
    _t = stack.torch
    if _t.cuda.is_available() and hasattr(_t, "compiler") and \
            hasattr(_t.compiler, "cudagraph_mark_step_begin"):
        try:
            _t.compiler.cudagraph_mark_step_begin()
        except Exception:
            pass
    was_training = model.training
    model.eval()
    # Reset stream so every probe measures the same data slice
    inner = dataset.dataset if hasattr(dataset, "dataset") else dataset
    stream = inner.test_stream if split == "test" else inner.train_stream
    stream.reset()
    total_loss = None
    total_tokens = 0
    use_amp = amp_dtype is not None
    with stack.torch.inference_mode():
        for _ in range(batches):
            x, y = dataset.batch(split, eff_batch_size, eff_seq_len)
            with stack.torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                logits = model(x)
                # For patch-at-readout (logits.dim()==4) report bpb on head 0
                # only — that's the next-byte prediction, comparable to baseline.
                # Auxiliary heads are training signal, not the eval target.
                if logits.dim() == 4:
                    logits = logits[:, :, 0, :]
                n_tokens = y.numel()
                loss = _patch_cross_entropy(logits, y, reduction="sum")
            total_loss = loss.float() if total_loss is None else total_loss + loss.float()
            total_tokens += n_tokens
    if was_training:
        model.train()
    if total_tokens <= 0 or total_loss is None:
        return float("inf")
    return float((total_loss / total_tokens).item())


def run_bridge(args: argparse.Namespace) -> dict[str, object]:
    stack = load_training_backend_stack("torch")
    torch = stack.torch
    F = stack.functional
    CausalBankModel = stack.ModelClass
    build_token_shard_torch_dataset = stack.build_token_shard_torch_dataset
    configure_service_log(Path(args.json).parent / "chronohorn.service.jsonl")
    log_component = "train.causal_bank.torch"

    runtime = build_runtime(args, stack)
    device = choose_device(args.device)

    # CUDA performance defaults
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Auto batch size: scale to fill ~60% of VRAM if user didn't override.
        # Skip when curriculum is set — the curriculum explicitly prescribes
        # batch/seq per phase, and running auto-batch against args.seq_len (the
        # initial phase) produces a huge batch that blows up when later phases
        # have larger seq_len. Trust the curriculum spec.
        curriculum_set = bool(getattr(args, "curriculum", None))
        if (args.batch_size <= 16
                and not getattr(args, "_batch_size_explicit", False)
                and not curriculum_set):
            try:
                gpu_mem_mb = torch.cuda.get_device_properties(0).total_mem // (1024 * 1024)
                # Rough heuristic: ~4KB per token at s8/v1024, scales with model
                # size and vocab.  Byte-level (v256) uses smaller embeddings and
                # output projections, so memory per token is proportionally lower.
                scale_factor = max(args.scale / 8.0, 1.0)
                vocab_factor = max(args.vocab_size / 1024, 0.25)
                # Scan-based substrates (gated_delta, learned_recurrence) hold
                # per-position state for backward, so memory grows super-linearly
                # with seq_len — at seq>=1024 this dominates the embedding budget.
                substrate_mode = getattr(args, "substrate_mode", "frozen")
                state_impl = getattr(args, "state_impl", "scan")
                uses_scan_state = (
                    substrate_mode in ("gated_delta", "learned_recurrence")
                    and state_impl == "scan"
                )
                scan_factor = max(1.0, (args.seq_len / 512.0) ** 2) if uses_scan_state else 1.0
                safe_mb = int(gpu_mem_mb * 0.6)
                mb_per_token = 0.004 * scale_factor * vocab_factor * scan_factor
                max_batch = int(safe_mb / (args.seq_len * mb_per_token))
                max_batch = max(max_batch, args.batch_size)
                # Round down to power of 2 for matmul alignment
                auto_batch = 1
                while auto_batch * 2 <= max_batch:
                    auto_batch *= 2
                if auto_batch > args.batch_size:
                    service_log(log_component, "auto batch size",
                                original=args.batch_size, auto=auto_batch,
                                gpu_mem_mb=gpu_mem_mb)
                    args.batch_size = auto_batch
                    runtime.train.batch_size = auto_batch
            except Exception:
                pass  # fallback to user-specified batch size

    seed_everything(args.seed)
    dataset = build_token_shard_torch_dataset(
        args.data_root,
        vocab_size=args.vocab_size,
        device=device,
        pin_memory=device.startswith("cuda"),
    )
    # Enable stochastic tokenization (BPE dropout) if requested
    if getattr(args, "stochastic_tokenization", False):
        from chronohorn.train.token_shard_dataset_torch import StochasticTokenizer
        tokenizer_path = dataset.dataset.tokenizer_path
        if tokenizer_path:
            alpha = getattr(args, "stochastic_alpha", 0.1)
            dataset._stochastic_tokenizer = StochasticTokenizer(tokenizer_path, dropout_alpha=alpha)
            import sys
            print(f"chronohorn: stochastic tokenization enabled (alpha={alpha})", file=sys.stderr)
    # When using curriculum, max_seq_len must cover the largest phase.
    _config_seq_len = runtime.train.seq_len
    if hasattr(args, "curriculum") and args.curriculum:
        import json as _json_mod
        try:
            _phases = _json_mod.loads(args.curriculum)
            _config_seq_len = max(p.get("seq_len", _config_seq_len) for p in _phases)
        except Exception:
            pass
    config, baseline_linear_hidden = config_for_variant(args, _config_seq_len, stack)
    assert_safe_readout_budget(
        args,
        config,
        baseline_linear_hidden=baseline_linear_hidden,
        out_dim=dataset.vocab_size,
    )
    model = CausalBankModel(vocab_size=dataset.vocab_size, config=config).to(device)
    # Init signature: hash of the model state dict before any optimizer step.
    # Same seed across variants can yield different inits when added modules
    # consume RNG draws. Heinrich measured ~0.004 bpb of init noise from this
    # in session 11; logging it makes cross-variant comparisons honest.
    import hashlib
    _init_hasher = hashlib.sha256()
    for _name in sorted(model.state_dict().keys()):
        _t = model.state_dict()[_name]
        _init_hasher.update(_name.encode("utf-8"))
        _init_hasher.update(_t.detach().cpu().contiguous().numpy().tobytes())
    init_signature_sha256 = _init_hasher.hexdigest()
    service_log(log_component, "init signature computed",
                init_signature_sha256=init_signature_sha256[:16],
                seed=args.seed, n_params=int(sum(p.numel() for p in model.parameters())))
    optimizer_model = model
    # Inject ngram table for trust-routing mode (decepticons doesn't import chronohorn)
    if getattr(config, "trust_routing", False) and getattr(config, "table_path", ""):
        import pathlib

        from chronohorn.families.polyhash.models.ngram_table import NgramTable
        table_path = config.table_path
        if pathlib.Path(table_path).exists():
            model.set_ngram_table(NgramTable.load(table_path))
        else:
            model.set_ngram_table(NgramTable(vocab_size=dataset.vocab_size))
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if param_count > args.max_params and not args.unsafe_large_model:
        raise ValueError(
            f"Refusing model with {param_count:,} trainable params > max_params={args.max_params:,}. "
            "Use --unsafe-large-model or raise --max-params to override."
        )
    # Auto torch.compile: enable for runs >= 5000 steps on CUDA where compile
    # overhead (~60s) amortizes.  Skip for learned_recurrence — the complex
    # parallel scan with torch.complex triggers CUDA device-side asserts
    # during autotune.  Frozen/learnable_decays substrates and rotation scans
    # (lasso/SO5) use roll+where and compile fine.
    use_compile = args.torch_compile
    has_complex_scan = getattr(config, "substrate_mode", "frozen") == "learned_recurrence"
    # Escape hatch: CHRONOHORN_DISABLE_AUTO_COMPILE=1 suppresses the auto-enable.
    # Needed when Dynamo tracing over a complex scan (e.g. lasso rotation at
    # seq>=4096) balloons host RSS past the pod's memory cgroup limit.
    import os as _os
    auto_compile_disabled = _os.environ.get("CHRONOHORN_DISABLE_AUTO_COMPILE") == "1"
    if (not use_compile and not auto_compile_disabled and device.startswith("cuda")
            and args.steps >= 5000 and not has_complex_scan):
        use_compile = True
        service_log(log_component, "auto torch.compile enabled", steps=args.steps)
    if use_compile:
        # default mode: kernel fusion only, no CUDA graphs.
        # Historical note: reduce-overhead and max-autotune used to break on
        # scan tensor aliasing from torch.where. The 2026-04-17 fast-scan
        # rewrite removed torch.where entirely (F.pad + identity element),
        # so reduce-overhead is now safe on the adaptive / complex-rotation
        # paths. Opt in via CHRONOHORN_COMPILE_MODE=reduce-overhead for
        # additional ~1.5-2× from CUDA graph capture.
        import os as _os_mode
        _compile_mode = _os_mode.environ.get("CHRONOHORN_COMPILE_MODE", "default")
        if _compile_mode == "default":
            model = torch.compile(model)
        else:
            model = torch.compile(model, mode=_compile_mode)
            service_log(log_component, "torch.compile mode override", mode=_compile_mode)
    # Mixed precision: fp16 uses tensor cores (8x matmul throughput on A4000).
    # Scans stay fp32 via torch.amp.custom_fwd in decepticons model code.
    use_amp = getattr(args, "mixed_precision", "off") == "fp16"
    if not use_amp and device.startswith("cuda") and args.steps >= 5000:
        use_amp = True
        service_log(log_component, "auto mixed precision fp16 enabled", steps=args.steps)
    amp_dtype = torch.float16 if use_amp else None
    grad_scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    if use_amp:
        service_log(log_component, "AMP enabled", dtype="fp16")

    initial_trainable_state = {
        name: param.detach().cpu().to(dtype=torch.float32).numpy()
        for name, param in optimizer_model.named_parameters()
        if param.requires_grad
    }
    init_report = summarize_named_arrays(initial_trainable_state)
    use_fused = device.startswith("cuda")
    optimizer_kwargs = build_adamw_kwargs(
        backend="torch",
        learning_rate=runtime.train.learning_rate,
        weight_decay=runtime.train.weight_decay,
        device=device,
        fused=use_fused,
    )
    optimizer_policy_defaults = build_adamw_policy_defaults(
        backend="torch",
        learning_rate=runtime.train.learning_rate,
        weight_decay=runtime.train.weight_decay,
        device=device,
        fused=use_fused,
    )
    optimizer_params = (
        optimizer_model.param_groups(runtime.train.learning_rate)
        if hasattr(optimizer_model, "param_groups")
        else optimizer_model.parameters()
    )
    optimizer = torch.optim.AdamW(optimizer_params, **optimizer_kwargs)
    _resume_step = 0
    if getattr(args, "resume", None):
        _resume_state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(_resume_state["model"])
        optimizer.load_state_dict(_resume_state["optimizer"])
        _resume_step = _resume_state.get("step", 0)
        if _resume_state.get("rng_cpu") is not None:
            torch.random.set_rng_state(_resume_state["rng_cpu"])
        if _resume_state.get("rng_cuda") is not None and device.startswith("cuda"):
            torch.cuda.set_rng_state(_resume_state["rng_cuda"])
        service_log(log_component, "resumed from checkpoint", resume_path=args.resume, resume_step=_resume_step)
    lr_schedule_name = str(getattr(args, "lr_schedule", "none") or "none")
    lr_warmup_steps = max(int(getattr(args, "lr_warmup_steps", 0) or 0), 0)
    lr_min_factor = float(getattr(args, "lr_min_factor", 0.1))
    if not 0.0 <= lr_min_factor <= 1.0:
        raise ValueError(f"--lr-min-factor must be in [0, 1], got {lr_min_factor}")
    scheduler = None
    if lr_schedule_name == "cosine":
        total_steps = max(int(runtime.train.steps), 1)
        warmup_steps = min(lr_warmup_steps, max(total_steps - 1, 0))

        def _lr_multiplier(epoch: int) -> float:
            step = epoch + 1
            if warmup_steps > 0 and step <= warmup_steps:
                return max(step / warmup_steps, 1e-8)
            if total_steps <= warmup_steps:
                return 1.0
            progress = min(max(step - warmup_steps, 0), total_steps - warmup_steps) / max(total_steps - warmup_steps, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return lr_min_factor + (1.0 - lr_min_factor) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_multiplier)
    backend_environment = build_backend_environment_metadata(
        backend="torch",
        stack=stack,
        device=device,
    )
    train_policy = build_train_policy_metadata(
        backend="torch",
        device=device,
        dtype_policy="fp32",
        optimizer_name=stack.optimizer_name,
        optimizer_impl=stack.optimizer_impl,
        optimizer_like=optimizer,
        learning_rate=runtime.train.learning_rate,
        weight_decay=runtime.train.weight_decay,
        grad_clip=runtime.train.grad_clip,
        compile_train_step=False,
        compile_eval=False,
        torch_compile=bool(args.torch_compile),
        init_policy="chronohorn_v1",
        init_seed=config.init_seed,
        explicit_defaults=optimizer_policy_defaults,
    )
    train_policy["lr_schedule"] = {
        "name": lr_schedule_name,
        "warmup_steps": lr_warmup_steps,
        "min_factor": lr_min_factor,
    }
    performance_estimate = CAUSAL_BANK_TRAINING_ADAPTER.estimate_training_performance(
        config=config,
        vocab_size=dataset.vocab_size,
        batch_size=runtime.train.batch_size,
        seq_len=runtime.train.seq_len,
        trainable_param_count=param_count,
    )

    effective_final_eval_batches = (
        runtime.train.eval_batches if args.final_eval_batches is None else args.final_eval_batches
    )
    probe_plan = resolve_probe_plan(
        max_step=runtime.train.steps,
        raw_steps=args.probe_steps,
        policy=args.probe_policy,
        default_eval_batches=runtime.train.eval_batches,
        standard_eval_batches=(
            args.probe_standard_eval_batches
            if args.probe_standard_eval_batches is not None
            else args.probe_eval_batches
        ),
        micro_eval_batches=args.probe_micro_eval_batches,
        promotion_eval_batches=args.probe_promotion_eval_batches,
        final_eval_batches=effective_final_eval_batches,
        geometric_start_step=args.probe_geometric_start,
        geometric_ratio=args.probe_geometric_ratio,
        micro_cutoff_step=args.probe_micro_cutoff_step,
        promotion_count=args.probe_promotion_count,
    )
    probe_steps = [int(step) for step in probe_plan.get("steps", [])]
    probe_step_set = set(probe_steps)
    probe_history: list[dict[str, float | int | None]] = []
    effective_probe_eval_batches = int(probe_plan.get("eval_batches", {}).get("standard") or runtime.train.eval_batches)
    performance_log: list[dict[str, float | int | None]] = []
    recent_losses: deque[torch.Tensor] = deque(maxlen=runtime.train.log_every)
    best_loss = torch.full((), float("inf"), device=device)
    start = time.time()
    last_log_time = start
    last_log_step = 0
    cumulative_probe_tflops_est = 0.0
    cumulative_probe_elapsed_sec = 0.0
    last_log_probe_tflops_est = 0.0
    last_log_probe_elapsed_sec = 0.0

    # Substrate training hints — the code tells the operator what the mode needs
    from decepticons.causal_bank import substrate_training_hints
    hints = substrate_training_hints(config)
    for warning in hints.get("warnings", []):
        service_log(log_component, warning, level="warning")

    service_log(
        log_component,
        "trainer started",
        data_root=args.data_root,
        device=device,
        seed=args.seed,
        steps=runtime.train.steps,
        seq_len=runtime.train.seq_len,
        batch_size=runtime.train.batch_size,
        learning_rate=runtime.train.learning_rate,
        train_tokens=dataset.train_token_count,
        val_tokens=dataset.test_token_count,
        variant=args.variant,
        scale=args.scale,
        linear_modes=config.linear_modes,
        linear_readout=f"{config.linear_readout_kind}:{config.linear_readout_depth}",
        linear_hidden=list(config.linear_hidden),
        local_window=config.local_window,
        osc_schedule=config.oscillatory_schedule,
        static_bank_gate=config.static_bank_gate,
        params=param_count,
        lr_schedule=lr_schedule_name,
        lr_warmup_steps=lr_warmup_steps,
        lr_min_factor=lr_min_factor,
    )
    if probe_steps:
        service_log(log_component, "probe plan", plan=format_probe_plan(probe_plan))
    if scheduler is not None or lr_warmup_steps > 0:
        service_log(
            log_component,
            "lr schedule",
            schedule=lr_schedule_name,
            warmup_steps=lr_warmup_steps,
            min_factor=lr_min_factor,
        )

    # Optional CUDA profiling: writes Chrome trace for the first N steps
    _profiler = None
    if getattr(args, "profile_cuda", 0) > 0 and device.startswith("cuda"):
        profile_dir = Path("out/profile")
        profile_dir.mkdir(parents=True, exist_ok=True)
        _profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=args.profile_cuda, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profile_dir)),
            record_shapes=True,
            with_stack=True,
        )
        _profiler.__enter__()
        service_log(log_component, "cuda profiling enabled", profile_steps=args.profile_cuda, profile_dir=str(profile_dir))

    _balance_coeff = getattr(args, "balance_coeff", 0.0)
    if _balance_coeff > 0:
        service_log(log_component, "expert load-balancing enabled", balance_coeff=_balance_coeff)

    # Enforce substrate warmup: freeze substrate params for N steps, then unfreeze
    _warmup_steps = hints.get("warmup_steps", 0)
    _substrate_params = []
    if _warmup_steps > 0:
        from decepticons.causal_bank import learnable_substrate_keys
        _substrate_keys = learnable_substrate_keys(config)
        for pname, param in model.named_parameters():
            if any(k in pname for k in _substrate_keys):
                _substrate_params.append((pname, param))
                param.requires_grad = False
        if _substrate_params:
            service_log(log_component, "substrate warmup started", params=len(_substrate_params), warmup_steps=_warmup_steps)

    model.train()
    if _resume_step > 0:
        # Fast-forward data stream past already-trained steps
        for _ in range(_resume_step):
            dataset.batch("train", runtime.train.batch_size, runtime.train.seq_len)
        # Fast-forward scheduler
        if scheduler is not None:
            for _ in range(_resume_step):
                scheduler.step()
        service_log(log_component, "data stream fast-forwarded", skipped_steps=_resume_step)
    # Parse curriculum if provided: [{"steps":2000,"seq_len":64,"batch_size":128},...]
    _curriculum = None
    _curriculum_phase = 0
    if hasattr(args, "curriculum") and args.curriculum:
        import json as _json
        _curriculum = _json.loads(args.curriculum)
        _curriculum_boundaries = []
        _cum = 0
        for phase in _curriculum:
            _cum += phase["steps"]
            _curriculum_boundaries.append(_cum)
        service_log(log_component, "curriculum enabled", phases=len(_curriculum), boundaries=_curriculum_boundaries)
    _cur_batch_size = runtime.train.batch_size
    _cur_seq_len = runtime.train.seq_len
    # Persistent-state bookkeeping (truncated-BPTT substrate carry-over).
    # None until first forward; tuple of (re, im) [B, modes] after that.
    _persistent_state: tuple[torch.Tensor, torch.Tensor] | None = None
    _persistent_batch_size: int | None = None
    # Compile the persistent-path forward so it matches regular forward's
    # throughput (otherwise the state-carry path bypasses torch.compile and
    # loses the ~2× speedup from fused scan levels).
    _persistent_fws_fn = None
    if bool(getattr(args, "persistent_state", False)) and use_compile:
        _unwrapped_for_fws = model._orig_mod if hasattr(model, "_orig_mod") else model
        if hasattr(_unwrapped_for_fws, "forward_with_state"):
            _persistent_fws_fn = torch.compile(_unwrapped_for_fws.forward_with_state)
            service_log(log_component, "persistent forward_with_state compiled")

    for step in range(_resume_step + 1, runtime.train.steps + 1):
        # Curriculum: update seq_len and batch_size at phase boundaries
        if _curriculum is not None:
            _cum = 0
            for _pi, _phase in enumerate(_curriculum):
                _cum += _phase["steps"]
                if step <= _cum:
                    new_bs = _phase["batch_size"]
                    new_sl = _phase["seq_len"]
                    if new_bs != _cur_batch_size or new_sl != _cur_seq_len:
                        _cur_batch_size = new_bs
                        _cur_seq_len = new_sl
                        service_log(log_component, "curriculum phase change",
                                    phase=_pi, seq_len=new_sl, batch_size=new_bs, step=step)
                    break
        # Unfreeze substrate params after warmup
        if step == _warmup_steps + 1 and _substrate_params:
            for _pname, param in _substrate_params:
                param.requires_grad = True
            service_log(log_component, "substrate warmup complete", params=len(_substrate_params), step=step)
        # Persistent-substrate training: per-lane contiguous stream, carry
        # (detached) state across batches. Only valid for adaptive substrate.
        _persistent = bool(getattr(args, "persistent_state", False))
        if _persistent:
            if _cur_batch_size != _persistent_batch_size:
                _persistent_batch_size = _cur_batch_size
                _persistent_state = None  # reset on batch size change
            x, y = dataset.batch_stateful("train", _cur_batch_size, _cur_seq_len)
        else:
            x, y = dataset.batch("train", _cur_batch_size, _cur_seq_len)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            if _persistent:
                if _persistent_fws_fn is not None:
                    # compiled path
                    logits, _final_state = _persistent_fws_fn(
                        x, initial_state=_persistent_state,
                    )
                else:
                    unwrapped = model._orig_mod if hasattr(model, "_orig_mod") else model
                    logits, _final_state = unwrapped.forward_with_state(
                        x, initial_state=_persistent_state,
                    )
                # Detach for next step (truncated BPTT: values carry, gradient does not)
                _persistent_state = (
                    _final_state[0].detach(), _final_state[1].detach(),
                )
            else:
                logits = model(x)
            _loss_ce = _patch_cross_entropy(logits, y)
            loss = _loss_ce
            if hasattr(model, "substrate_regularization"):
                loss = loss + model.substrate_regularization(step=step)
            _loss_balance = torch.tensor(0.0, device=loss.device)
            if _balance_coeff > 0 and hasattr(model, "linear_readout") and hasattr(model.linear_readout, "balance_loss"):
                _loss_balance = _loss_balance + model.linear_readout.balance_loss()
            if _balance_coeff > 0 and hasattr(model, "band_balance_loss"):
                _loss_balance = _loss_balance + model.band_balance_loss()
            if _balance_coeff > 0:
                loss = loss + _balance_coeff * _loss_balance
        if grad_scaler is not None:
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            _grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), runtime.train.grad_clip).item())
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            _grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), runtime.train.grad_clip).item())
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if _profiler is not None:
            _profiler.step()

        # Periodic checkpoint save — Heinrich trajectory forensics.
        # Triggered when --save-checkpoint-every N > 0 and step % N == 0.
        # Writes {json_stem}_step{step}.checkpoint.pt beside the result JSON.
        _ckpt_every = int(getattr(args, "save_checkpoint_every", 0) or 0)
        if _ckpt_every > 0 and step > 0 and step % _ckpt_every == 0:
            _periodic_path = (
                Path(args.json).parent
                / f"{Path(args.json).stem}_step{step}.checkpoint.pt"
            )
            _unwrapped = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save(_unwrapped.state_dict(), _periodic_path)
            service_log(log_component, "periodic checkpoint saved",
                        step=step, path=str(_periodic_path))

        loss_detached = loss.detach()
        recent_losses.append(loss_detached)
        best_loss = torch.minimum(best_loss, loss_detached)

        if step in probe_step_set:
            probe_entry = probe_entry_by_step(probe_plan, step) or {}
            row_probe_eval_batches = int(probe_entry.get("eval_batches") or effective_probe_eval_batches)
            probe_started = time.time()
            probe_loss = evaluate(
                model,
                dataset,
                runtime.train,
                args.probe_split,
                eval_batches=row_probe_eval_batches,
                amp_dtype=amp_dtype,
                seq_len=_cur_seq_len,
                batch_size=_cur_batch_size,
            )
            probe_elapsed_sec = time.time() - probe_started
            probe_bpt = bits_per_token_from_loss(probe_loss)
            tokens_per_byte = dataset.test_tokens_per_byte if args.probe_split == "test" else None
            probe_bpb = probe_bpt * tokens_per_byte if tokens_per_byte is not None else None
            recent_train_loss = (
                float(torch.stack(tuple(recent_losses)).mean().item())
                if recent_losses
                else float("nan")
            )
            probe_row = {
                "step": step,
                "tier": probe_entry.get("tier"),
                "split": args.probe_split,
                "eval_batches": row_probe_eval_batches,
                "eval_loss": probe_loss,
                "bits_per_token": probe_bpt,
                "bpb": probe_bpb,
                "recent_train_loss": recent_train_loss,
                "elapsed_sec": probe_elapsed_sec,
                # Cumulative wall-clock seconds from start of training. Enables
                # real ETA computation (rate = step / train_elapsed_sec) rather
                # than per-probe duration which doesn't reflect training progress.
                "train_elapsed_sec": time.time() - start,
            }
            # --- Full telemetry block ---
            _telemetry = {}

            # Training dynamics
            _telemetry["grad_norm"] = round(_grad_norm, 6)
            _telemetry["loss_ce"] = round(float(_loss_ce.detach().item()), 6)
            _telemetry["loss_balance"] = round(float(_loss_balance.detach().item()), 6) if _balance_coeff > 0 else 0.0
            _telemetry["weight_norm"] = round(float(sum(p.detach().norm().item() for p in model.parameters() if p.requires_grad)), 4)
            _telemetry["lr_current"] = round(float(optimizer.param_groups[0]["lr"]), 8)

            # GPU memory
            if device.startswith("cuda"):
                _telemetry["gpu_mem_current_mb"] = round(torch.cuda.memory_allocated(device) / 1e6, 1)
                _telemetry["gpu_mem_peak_mb"] = round(torch.cuda.max_memory_allocated(device) / 1e6, 1)

            # Routing telemetry (top-level experts or band experts)
            if hasattr(model, "linear_readout") and hasattr(model.linear_readout, "_last_route") and model.linear_readout._last_route is not None:
                _route = model.linear_readout._last_route.detach()
                _assignments = _route.argmax(dim=-1)
                _n_experts = _route.shape[-1]
                _fracs = [float((_assignments == e).float().mean()) for e in range(_n_experts)]
                _telemetry["routing_max_frac"] = round(max(_fracs), 4)
                _telemetry["routing_fracs"] = [round(f, 4) for f in _fracs]
                _telemetry["routing_balance_loss"] = round(float(model.linear_readout.balance_loss()), 4)

            # Per-band routing (if bands with experts)
            if hasattr(model, "_band_readouts"):
                from decepticons.models.readouts_torch import RoutedSquaredReLUReadout
                _band_routing = []
                for _bi, _br in enumerate(model._band_readouts):
                    if isinstance(_br, RoutedSquaredReLUReadout) and _br._last_route is not None:
                        _br_route = _br._last_route.detach()
                        _br_assign = _br_route.argmax(dim=-1)
                        _br_fracs = [float((_br_assign == e).float().mean()) for e in range(_br_route.shape[-1])]
                        _band_routing.append({"band": _bi, "fracs": [round(f, 4) for f in _br_fracs], "max_frac": round(max(_br_fracs), 4)})
                if _band_routing:
                    _telemetry["band_routing"] = _band_routing

            # Sticky register telemetry
            if getattr(model, "_sticky_registers", 0) > 0:
                _sw = torch.sigmoid(model._sticky_write_gate(model._embed_linear(x))).detach()
                _telemetry["sticky_write_rate_mean"] = round(float(_sw.mean().item()), 6)
                _telemetry["sticky_write_rate_max"] = round(float(_sw.max().item()), 6)
                _telemetry["sticky_write_rate_std"] = round(float(_sw.std().item()), 6)

            # Positional loss (first 64 vs last 64 tokens).
            # Skipped under CHRONOHORN_COMPILE_MODE=reduce-overhead: the logits
            # tensor lives in the compiled forward's CUDA-graph memory pool and
            # any cross-step access (even through .clone(), which still reads
            # from the pooled source) trips the cudagraph overwrite guard. This
            # telemetry is purely diagnostic; drop it for speed-stack runs.
            import os as _os_telemetry
            _skip_pos_telemetry = _os_telemetry.environ.get(
                "CHRONOHORN_COMPILE_MODE", "default"
            ) == "reduce-overhead"
            if not _skip_pos_telemetry:
                with torch.inference_mode():
                    _pos_logits = logits.detach().clone()
                    _pos_y = y.detach()
                    if _pos_logits.dim() == 4:
                        # Per-position eval not yet wired for patch-at-readout; collapse to head 0.
                        _pos_logits = _pos_logits[:, :, 0, :]
                    _per_pos_loss = F.cross_entropy(_pos_logits.reshape(-1, _pos_logits.shape[-1]), _pos_y.reshape(-1), reduction="none").reshape(_pos_y.shape)
                    _telemetry["loss_first_64"] = round(float(_per_pos_loss[:, :64].mean().item()), 6)
                    _telemetry["loss_last_64"] = round(float(_per_pos_loss[:, -64:].mean().item()), 6) if _pos_y.shape[1] > 64 else None

            probe_row["telemetry"] = _telemetry
            # Backward compat: keep top-level routing fields
            if "routing_max_frac" in _telemetry:
                probe_row["routing_max_frac"] = _telemetry["routing_max_frac"]
                probe_row["routing_fracs"] = _telemetry["routing_fracs"]
                probe_row["routing_balance_loss"] = _telemetry["routing_balance_loss"]
            probe_row["compute"] = build_probe_compute_accounting_inputs(
                performance_estimate,
                [probe_row],
                split=args.probe_split,
                eval_batches=row_probe_eval_batches,
            )["per_probe"][0]
            cumulative_probe_tflops_est += float(probe_row["compute"].get("eval_tflops_est") or 0.0)
            cumulative_probe_elapsed_sec += float(probe_elapsed_sec)
            probe_history.append(probe_row)
            # Write incremental probe for live ingestion
            _probe_path = Path(args.json).parent / f"{Path(args.json).stem}.probes.jsonl"
            with _probe_path.open("a") as _pf:
                _pf.write(json.dumps({"step": step, "bpb": probe_bpb, "loss": probe_loss, "elapsed_sec": probe_elapsed_sec, "eval_batches": row_probe_eval_batches}) + "\n")
            service_log(
                log_component,
                "probe",
                step=step,
                split=args.probe_split,
                eval_loss=round(probe_loss, 6),
                bits_per_token=round(probe_bpt, 6),
                bpb=None if probe_bpb is None else round(probe_bpb, 6),
                eval_batches=row_probe_eval_batches,
            )
            # Diagnostics are expensive and should be opt-in on throughput-focused runs.
            if args.probe_diagnostics and row_probe_eval_batches >= 8:
                try:
                    from decepticons.models.diagnostics import diagnose
                    diag_tokens = torch.randint(0, dataset.vocab_size, (2, 64), device=device)
                    diag = diagnose(model, diag_tokens, vocab_size=dataset.vocab_size)
                    probe_row["diagnostics"] = {
                        "modes_alive_pct": diag["summary"]["modes_alive_pct"],
                        "dominant_timescale": diag["summary"]["dominant_timescale"],
                        "findings": diag.get("findings", []),
                    }
                    if diag.get("phase"):
                        probe_row["diagnostics"]["phase_mismatch"] = diag["phase"].get("mismatch_by_band")
                    if diag.get("readout_selectivity"):
                        probe_row["diagnostics"]["readout_by_timescale"] = diag["readout_selectivity"].get("by_timescale")
                    for finding in diag.get("findings", []):
                        service_log(log_component, "diagnostic finding", step=step, finding=finding)
                except Exception as diag_exc:
                    service_log(log_component, "diagnostics skipped", level="warning", step=step, error=str(diag_exc))

        if step % runtime.train.log_every == 0:
            recent = float(torch.stack(tuple(recent_losses)).mean().item()) if recent_losses else float("nan")
            best = float(best_loss.item())
            now = time.time()
            elapsed = now - start
            interval_steps = step - last_log_step
            interval_elapsed = now - last_log_time
            perf_summary = summarize_observed_training_performance(
                performance_estimate,
                steps_completed=step,
                elapsed_sec=elapsed,
                interval_steps=interval_steps,
                interval_elapsed_sec=interval_elapsed,
                probe_tflops_consumed_est=cumulative_probe_tflops_est,
                probe_elapsed_sec=cumulative_probe_elapsed_sec,
                interval_probe_tflops_est=cumulative_probe_tflops_est - last_log_probe_tflops_est,
                interval_probe_elapsed_sec=cumulative_probe_elapsed_sec - last_log_probe_elapsed_sec,
            )
            performance_log.append({"step": step, **perf_summary})
            service_log(
                log_component,
                "training progress",
                step=step,
                loss=round(recent, 6),
                best=round(best, 6),
                summary=format_observed_training_performance(perf_summary),
                tokens_per_second=perf_summary.get("tokens_per_second"),
                estimated_sustained_tflops=perf_summary.get("estimated_sustained_tflops"),
            )
            last_log_time = now
            last_log_step = step
            last_log_probe_tflops_est = cumulative_probe_tflops_est
            last_log_probe_elapsed_sec = cumulative_probe_elapsed_sec

    elapsed = time.time() - start
    performance_summary = summarize_observed_training_performance(
        performance_estimate,
        steps_completed=runtime.train.steps,
        elapsed_sec=elapsed,
        probe_tflops_consumed_est=cumulative_probe_tflops_est,
        probe_elapsed_sec=cumulative_probe_elapsed_sec,
    )
    # Final eval uses the LAST curriculum phase (so a byte-curriculum ending at
    # seq=4096 is evaluated at seq=4096, not at the phase-0 args.seq_len).
    train_eval = evaluate(
        model, dataset, runtime.train, "train",
        eval_batches=args.final_eval_batches, amp_dtype=amp_dtype,
        seq_len=_cur_seq_len, batch_size=_cur_batch_size,
    )
    replay_fixture = CAUSAL_BANK_TRAINING_ADAPTER.build_replay_fixture(
        dataset,
        split="test",
        sequence_length=runtime.train.seq_len,
    )
    was_training = model.training
    model.eval()
    with torch.no_grad():
        fixture_x = torch.tensor([replay_fixture["input_token_ids"]], dtype=torch.long, device=device)
        fixture_logits = model(fixture_x).detach().cpu().numpy()
    if was_training:
        model.train()
    replay_fixture = CAUSAL_BANK_TRAINING_ADAPTER.attach_replay_reference(
        replay_fixture,
        fixture_logits,
    )
    test_eval = evaluate(
        model, dataset, runtime.train, "test",
        eval_batches=args.final_eval_batches, amp_dtype=amp_dtype,
        seq_len=_cur_seq_len, batch_size=_cur_batch_size,
    )
    train_bpt = bits_per_token_from_loss(train_eval)
    test_bpt = bits_per_token_from_loss(test_eval)
    test_bpb = test_bpt * dataset.test_tokens_per_byte if dataset.test_tokens_per_byte is not None else None
    params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    payload_bytes_est = float(
        sum(param.numel() * param.element_size() for param in model.parameters() if param.requires_grad)
    )
    result = {
        "title": "causal-bank torch trainer",
        "config": asdict(runtime),
        "dataset": {
            "source_path": dataset.dataset.source_path,
            "tokenizer": dataset.dataset.tokenizer,
            "tokenizer_path": dataset.dataset.tokenizer_path,
            "train_token_count": int(dataset.train_token_count),
            "test_token_count": int(dataset.test_token_count),
            "test_tokens_per_byte": dataset.test_tokens_per_byte,
            "test_bytes_per_token": dataset.test_bytes_per_token,
        },
        "training": {
            "backend": "torch",
            "device": device,
            "backend_environment": backend_environment,
            "dtype_policy": "fp16_mixed" if use_amp else "fp32",
            "compile_train_step": False,
            "compile_eval": False,
            "torch_compile": args.torch_compile,
            "compile": train_policy["compile"],
            "optimizer": train_policy["optimizer"],
            "train_policy_version": train_policy["version"],
            "train_policy": train_policy,
            "probe_diagnostics": bool(args.probe_diagnostics),
            "init_policy": "chronohorn_v1",
            "init_seed": config.init_seed,
            "initial_trainable_signature": init_report,
            "probe_policy": probe_plan.get("policy"),
            "probe_steps": probe_steps,
            "probe_plan": probe_plan,
            "probe_split": args.probe_split,
            "probe_eval_batches": effective_probe_eval_batches,
            "final_eval_batches": effective_final_eval_batches,
            "performance_estimate": performance_estimate,
            "performance": performance_summary,
            "performance_log": performance_log,
            "probes": probe_history,
            "replay_fixture": replay_fixture,
            "compute_accounting_inputs": build_compute_accounting_inputs(
                performance_estimate,
                train_steps_completed=runtime.train.steps,
                train_elapsed_sec=elapsed,
                probe_rows=probe_history,
                probe_split=args.probe_split,
                probe_eval_batches=effective_probe_eval_batches,
                final_eval_batches=effective_final_eval_batches,
                final_eval_splits=2,
                replay_tokens=len(replay_fixture.get("input_token_ids", [])),
                performance_summary=performance_summary,
            ),
        },
        "model": {
            "preset": "causal_bank_torch",
            "variant": args.variant,
            "scale": args.scale,
            "params": params,
            "seed": args.seed,
            "init_signature_sha256": init_signature_sha256,
            "linear_modes": config.linear_modes,
            "local_window": config.local_window,
            "share_embedding": config.share_embedding,
            "linear_impl": config.linear_impl,
            "linear_readout_kind": config.linear_readout_kind,
            "linear_readout_depth": config.linear_readout_depth,
            "linear_readout_num_experts": config.linear_readout_num_experts,
            "balance_coeff": _balance_coeff,
            "readout_bands": config.readout_bands,
            "band_experts": list(config.band_experts) if config.band_experts else [],
            "magnitude_normalize": config.magnitude_normalize,
            "overwrite_gate": config.overwrite_gate,
            "mode_selector": config.mode_selector,
            "temporal_attention": config.temporal_attention,
            "temporal_snapshot_interval": config.temporal_snapshot_interval,
            "temporal_attention_heads": config.temporal_attention_heads,
            "temporal_attention_head_dim": config.temporal_attention_head_dim,
            "tied_readout_normalize": config.tied_readout_normalize,
            "complex_rotation": config.complex_rotation,
            "lasso_rotation": config.lasso_rotation,
            "linear_hidden_match": args.linear_hidden_match,
            "linear_half_life_min": config.linear_half_life_min,
            "linear_half_life_max": config.linear_half_life_max,
            "oscillatory_frac": config.oscillatory_frac,
            "oscillatory_schedule": config.oscillatory_schedule,
            "oscillatory_period_min": config.oscillatory_period_min,
            "oscillatory_period_max": config.oscillatory_period_max,
            "input_proj_scheme": config.input_proj_scheme,
            "substrate_mode": config.substrate_mode,
            "state_dim": config.state_dim,
            "state_impl": getattr(config, "state_impl", "scan"),
            "num_heads": config.num_heads,
            "block_mixing_ratio": config.block_mixing_ratio,
            "static_bank_gate": config.static_bank_gate,
            "bank_gate_span": config.bank_gate_span,
            "train_eval_loss": train_eval,
            "test_eval_loss": test_eval,
            "train_bits_per_token": train_bpt,
            "test_bits_per_token": test_bpt,
            "test_bpb": test_bpb,
            "overfit_pct": (test_eval / train_eval - 1.0) * 100.0,
            "train_time_sec": elapsed,
            "learning_rate": runtime.train.learning_rate,
            "payload_bytes_est": payload_bytes_est,
            "payload_mb_est": payload_bytes_est / (1024.0 * 1024.0),
            "embedding_dim": config.embedding_dim,
            "linear_hidden": list(config.linear_hidden),
            "local_hidden": list(config.local_hidden),
            "local_scale": config.local_scale,
            "mix_mode": config.mix_mode,
            "init_policy": "chronohorn_v1",
            "init_seed": config.init_seed,
            "initial_trainable_signature": init_report,
        },
    }
    service_log(
        log_component,
        "training complete",
        test_eval_loss=round(test_eval, 6),
        test_bits_per_token=round(test_bpt, 6),
        test_bpb=None if test_bpb is None else round(test_bpb, 6),
        tokens_per_second=performance_summary.get("tokens_per_second"),
        estimated_sustained_tflops=performance_summary.get("estimated_sustained_tflops"),
    )

    output_path = Path(args.json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.export_dir:
        learned_state = {
            name: param.detach().cpu()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        for name, buffer in model.named_buffers():
            learned_state[name] = buffer.detach().cpu()
        export_dir = CAUSAL_BANK_TRAINING_ADAPTER.write_export_bundle(
            export_root=args.export_dir,
            summary_path=output_path,
            variant=args.variant,
            scale=args.scale,
            readout_kind=config.linear_readout_kind,
            seed=args.seed,
            tokenizer_id=dataset.dataset.tokenizer,
            data_root=args.data_root,
            config=config,
            learned_state=learned_state,
            train_step=runtime.train.steps,
            train_wallclock_s=elapsed,
            sequence_length=runtime.train.seq_len,
            vocab_size=dataset.vocab_size,
            profile=args.profile,
            backend="torch",
            train_policy=train_policy,
            replay_fixture=replay_fixture,
        )
        result["model"]["export_dir"] = str(export_dir)
        result["model"]["export_manifest_path"] = str(export_dir / "manifest.json")
    if getattr(args, "save_checkpoint", False):
        # Inference checkpoint (for decepticons.loader / heinrich)
        ckpt_path = output_path.with_suffix(".checkpoint.pt")
        _model_state = model.state_dict()
        torch.save(_model_state, ckpt_path)
        result["model"]["checkpoint_path"] = str(ckpt_path)
        # Full training state (for --resume) — reuses model state, no duplicate
        train_state_path = output_path.with_suffix(".training_state.pt")
        torch.save({
            "model": _model_state,
            "optimizer": optimizer.state_dict(),
            "step": runtime.train.steps,
            "rng_cpu": torch.random.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state() if device.startswith("cuda") else None,
        }, train_state_path)
        del _model_state
        result["model"]["training_state_path"] = str(train_state_path)
        service_log(log_component, "checkpoint saved", checkpoint_path=str(ckpt_path), training_state_path=str(train_state_path))
    result["forecast"] = build_result_forecast(result, budget=DEFAULT_GOLF_V1_BUDGET)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    service_log(log_component, "json summary written", output_path=str(output_path))
    if args.export_dir:
        service_log(log_component, "export bundle written", export_dir=result["model"]["export_dir"])
    return result


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_bridge(args)


if __name__ == "__main__":
    main()
