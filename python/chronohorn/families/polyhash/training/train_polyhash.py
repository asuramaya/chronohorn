#!/usr/bin/env python3
"""Unified PolyHash trainer -- one script for all architecture versions.

Supports: v6, v7, v8, v8h, v8m, v10, v11

Usage:
    python scripts/train_polyhash.py --arch v6 --data-root /data --json out.json
    python scripts/train_polyhash.py --arch v11 --data-root /data --json out.json --buckets-per-scale 2000000

Architecture-specific flags are auto-generated from each config dataclass.
Any flag that matches a config field will be forwarded to the config constructor.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import glob as _glob
import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Version registry: arch_name -> (module_path, model_class, config_class)
# ---------------------------------------------------------------------------
VERSIONS: dict[str, tuple[str, str, str]] = {
    "v6":  ("chronohorn.families.polyhash.models.polyhash_v6",  "PolyHashV6",  "V6Config"),
    "v7":  ("chronohorn.families.polyhash.models.polyhash_v7",  "PolyHashV7",  "V7Config"),
    "v8":  ("chronohorn.families.polyhash.models.polyhash_v8",  "PolyHashV8",  "V8Config"),
    "v8h": ("chronohorn.families.polyhash.models.polyhash_v8h", "PolyHashV8H", "V8HConfig"),
    "v8m": ("chronohorn.families.polyhash.models.polyhash_v8m", "PolyHashV8M", "V8MConfig"),
    "v10": ("chronohorn.families.polyhash.models.polyhash_v10", "PolyHashV10", "V10Config"),
    "v11": ("chronohorn.families.polyhash.models.polyhash_v11", "PolyHashV11", "V11Config"),
    "v12": ("chronohorn.families.polyhash.models.polyhash_v12", "PolyHashV12", "V12Config"),
}

# Fields that live in the trainer, not the model config -- skip when
# auto-generating argparse flags from config dataclass fields.
TRAINER_FIELDS = {"max_seq_len", "init_seed"}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

class ShardedDataset:
    """Lazy-loading sharded token dataset (uint16 .bin files)."""

    def __init__(self, dr: str, sl: int = 512):
        self.paths = sorted(_glob.glob(f"{dr}/fineweb_train_*.bin"))
        if not self.paths:
            raise FileNotFoundError(dr)
        self.sl = sl
        self._s: np.ndarray | None = None
        self._load(0)

    def _load(self, i: int) -> None:
        t = np.fromfile(self.paths[i], dtype=np.uint16).astype(np.int64)
        self._s = t[t < 1024]

    def sample(self, bs: int) -> np.ndarray:
        if np.random.randint(100) == 0 or self._s is None:
            self._load(np.random.randint(len(self.paths)))
        n = len(self._s) - self.sl
        ix = np.random.randint(0, n, size=bs)
        return np.stack([self._s[i : i + self.sl] for i in ix])

    @property
    def num_shards(self) -> int:
        return len(self.paths)


def load_val(dr: str) -> np.ndarray:
    t = np.fromfile(f"{dr}/fineweb_val_000000.bin", dtype=np.uint16).astype(np.int64)
    return t[t < 1024]


# ---------------------------------------------------------------------------
# Config auto-builder
# ---------------------------------------------------------------------------

_TYPE_MAP: dict[str, type] = {
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "tuple": str,  # parsed as comma-separated string, converted later
}


def _python_type(annotation: str | type) -> type:
    """Map a dataclass field's type annotation to a Python type for argparse."""
    s = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", str(annotation))
    return _TYPE_MAP.get(s, str)


def _flag_name(field_name: str) -> str:
    """Convert dataclass field name to argparse flag: embed_per_table -> --embed-per-table."""
    return "--" + field_name.replace("_", "-")


def add_config_flags(parser: argparse.ArgumentParser, config_cls: type) -> None:
    """Dynamically add argparse flags for every field in a frozen dataclass config."""
    existing = set()
    for action in parser._actions:
        existing.update(action.option_strings)

    for f in dataclasses.fields(config_cls):
        if f.name in TRAINER_FIELDS:
            continue
        flag = _flag_name(f.name)
        if flag in existing:
            continue
        py_type = _python_type(f.type)
        default = f.default if f.default is not dataclasses.MISSING else None
        if py_type is bool:
            # Bools: use --flag / --no-flag style
            no_flag = f"--no-{f.name.replace('_', '-')}"
            if default is True:
                if no_flag not in existing:
                    parser.add_argument(no_flag, dest=f.name,
                                        action="store_false", default=True)
            else:
                parser.add_argument(flag, dest=f.name,
                                    action="store_true", default=False)
        else:
            parser.add_argument(flag, dest=f.name, type=py_type, default=default)


def build_config(config_cls: type, ns: argparse.Namespace, seq_len: int, seed: int) -> Any:
    """Build a frozen config dataclass from parsed argparse namespace."""
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(config_cls):
        if f.name == "max_seq_len":
            kwargs["max_seq_len"] = seq_len
        elif f.name == "init_seed":
            kwargs["init_seed"] = seed
        elif hasattr(ns, f.name):
            val = getattr(ns, f.name)
            # Handle tuple fields stored as comma-separated strings
            if f.type == "tuple" or f.type is tuple:
                if isinstance(val, str):
                    val = tuple(int(x) for x in val.split(",")) if val else ()
                elif val is None:
                    val = f.default if f.default is not dataclasses.MISSING else ()
            if val is not None:
                kwargs[f.name] = val
            elif f.default is not dataclasses.MISSING:
                kwargs[f.name] = f.default
        elif f.default is not dataclasses.MISSING:
            kwargs[f.name] = f.default
    return config_cls(**kwargs)


# ---------------------------------------------------------------------------
# Hash parameter detection
# ---------------------------------------------------------------------------

def find_hash_params(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    """Find hash/table/embed parameters (excluding byte_embed) for per-group LR."""
    return [
        p
        for n, p in model.named_parameters()
        if any(k in n for k in ("table", "hash", "embed")) and "byte" not in n
    ]


# ---------------------------------------------------------------------------
# Probe schedule
# ---------------------------------------------------------------------------

def build_probe_schedule(max_steps: int) -> set[int]:
    """Powers of 2 starting at 100, always includes max_steps."""
    ps: set[int] = set()
    s = 100
    while s <= max_steps:
        ps.add(s)
        s = int(s * 2)
    ps.add(max_steps)
    return ps


# ---------------------------------------------------------------------------
# Muon optimizer — Newton-Schulz orthogonalized momentum (2x efficiency vs AdamW)
# ---------------------------------------------------------------------------

class Muon(torch.optim.Optimizer):
    """Simplified Muon: SGD momentum + Newton-Schulz orthogonalization.

    For 2D+ weight matrices, the momentum update is orthogonalized via
    3 iterations of Newton-Schulz, which approximates the matrix square root
    inverse. This decorrelates gradient directions and improves conditioning.
    1D params (biases, norms, embeddings) fall back to standard AdamW.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, ns_iters=3,
                 adamw_lr=3e-4, adamw_betas=(0.9, 0.95), weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, ns_iters=ns_iters,
                        adamw_lr=adamw_lr, adamw_betas=adamw_betas,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

    @staticmethod
    def _newton_schulz(M, iters=3):
        """Approximate M @ (M^T M)^{-1/2} via Newton-Schulz iteration."""
        a, b, c = (3.4445, -4.7750, 2.0315)  # coefficients for cubic NS
        X = M.float()
        X = X / (X.norm() + 1e-7)
        for _ in range(iters):
            A = X @ X.T
            X = a * X + b * (A @ X) + c * (A @ (A @ X))
        return X.to(M.dtype)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            mom = group["momentum"]
            ns = group["ns_iters"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["momentum_buffer"] = torch.zeros_like(p)
                state["step"] += 1

                buf = state["momentum_buffer"]
                buf.mul_(mom).add_(g)

                if p.dim() >= 2 and p.shape[0] >= 8 and p.shape[1] >= 8 and p.shape[0] <= 4096:
                    # Orthogonalize for matrices (skip huge ones like hash tables)
                    update = self._newton_schulz(buf.view(p.shape[0], -1), ns)
                    update = update.view_as(p)
                    if wd > 0:
                        p.add_(p, alpha=-wd * lr)
                    p.add_(update, alpha=-lr)
                else:
                    # AdamW fallback for 1D params (biases, norms, embeddings)
                    if wd > 0:
                        p.add_(p, alpha=-wd * group["adamw_lr"])
                    p.add_(buf, alpha=-group["adamw_lr"])


# ---------------------------------------------------------------------------
# Sanity checks (run automatically after model build)
# ---------------------------------------------------------------------------

def _sanity_warnings(config, params, hash_param_count, steps, batch_size, seq_len, model=None):
    """Print warnings about obviously bad configurations. Runs automatically."""
    warnings = []

    # Check table/neural ratio
    neural = params - hash_param_count
    if hash_param_count > 0 and neural > 0:
        ratio = hash_param_count / neural
        if ratio > 10:
            warnings.append(f"WARNING: {ratio:.0f}:1 table-to-neural ratio. Tables have {hash_param_count:,} params but neural network only {neural:,}. The MLP may be too small to decode the table embeddings.")

    # Check visits per bucket — extract actual bucket count and embed dim from config
    tokens_per_epoch = steps * batch_size * seq_len
    bucket_count = getattr(config, 'hash_buckets', None) or getattr(config, 'buckets_per_table', None) or getattr(config, 'buckets_per_scale', None)
    embed_dim = getattr(config, 'hash_embed_dim', None) or getattr(config, 'embed_per_table', None) or getattr(config, 'embed_per_scale', None)
    if bucket_count and bucket_count > 1000:
        visits = tokens_per_epoch / bucket_count
        if visits < 50:
            warnings.append(f"WARNING: ~{visits:.0f} visits per bucket at {steps:,} steps. Each of {bucket_count:,} buckets sees <50 gradient updates. Consider fewer buckets or more steps.")
        if embed_dim and embed_dim <= 2 and bucket_count > 100_000:
            warnings.append(f"WARNING: {embed_dim}d embeddings with {bucket_count:,} buckets. Sub-4d embeddings lack expressiveness for the MLP to decode. Prior experiments (v11-1d-4M, v11-2d-2M) showed this doesn't work.")

    # Check total size
    mb = params * 6 / 8 / 1024 / 1024
    if mb > 16:
        warnings.append(f"WARNING: {mb:.1f} MB exceeds 16 MB competition limit.")

    # Causality check: detect cross-attention without causal masking
    if model is not None:
        try:
            import inspect
            for mod_name, module in model.named_modules():
                cls_name = type(module).__name__.lower()
                if any(k in cls_name for k in ("isab", "crossattention", "setattention", "inducing")):
                    src = inspect.getsource(type(module).forward)
                    if "triu" not in src and "causal_mask" not in src:
                        warnings.append(
                            f"CAUSALITY VIOLATION: '{mod_name}' ({type(module).__name__}) uses "
                            f"cross-attention without causal masking. Leaks future tokens. "
                            f"Results will be INVALID for autoregressive prediction."
                        )
        except Exception:
            pass

    # Causality violations are HARD BLOCKS, not warnings
    causal_violations = [w for w in warnings if "CAUSALITY" in w]
    other_warnings = [w for w in warnings if "CAUSALITY" not in w]

    for w in other_warnings:
        print(f"\n  \u26a0 {w}", file=sys.stderr)
    for w in causal_violations:
        print(f"\n  \u274c {w}", file=sys.stderr)

    if causal_violations:
        raise RuntimeError(
            f"BLOCKED: {len(causal_violations)} causality violation(s). "
            f"Fix the model or disable the component."
        )
    if other_warnings:
        print(f"\n  ({len(other_warnings)} warnings \u2014 training will proceed anyway)\n", file=sys.stderr)


# ---------------------------------------------------------------------------
# G1: Pareto dominance check — warns if this config is already dominated
# ---------------------------------------------------------------------------

def _pareto_check(params, config, arch, steps, batch_size, seq_len):
    """Check if this config is already Pareto-dominated by an existing result."""
    try:
        sys_path = str(Path(__file__).resolve().parent.parent / "python")
        if sys_path not in sys.path:
            sys.path.insert(0, sys_path)
        from chronohorn.db import ChronohornDB
        db_path = Path("out/chronohorn.db")
        if not db_path.exists():
            return
        db = ChronohornDB.open_read_only(str(db_path))
        frontier = db.frontier(20)
        if not frontier:
            db.close()
            return
        # Estimate this model's tok/s from similar-sized models
        similar = db.query(
            "SELECT tok_s, bpb FROM results WHERE ABS(params - ?) < ? AND tok_s > 0 ORDER BY bpb LIMIT 3",
            (params, params * 0.3)
        )
        est_tok_s = sum(r["tok_s"] for r in similar) / len(similar) if similar else 0

        # Check if any frontier result dominates (better bpb AND faster)
        for f in frontier:
            f_tok = f.get("tok_s") or 0
            if f.get("bpb") and f_tok > 0 and est_tok_s > 0:
                if f["bpb"] < 2.0 and f_tok >= est_tok_s * 0.8:  # within 20% speed
                    print(f"\n  \u26a0 PARETO WARNING: {f['name']} already achieves {f['bpb']:.4f} bpb at {f_tok:,.0f} tok/s", file=sys.stderr)
                    print(f"    Your estimated speed: ~{est_tok_s:,.0f} tok/s. You'd need <{f['bpb']:.4f} bpb to be useful.", file=sys.stderr)
                    break
        db.close()
    except Exception:
        pass  # DB not available — skip check


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    # --- Phase 1: parse --arch to know which config to introspect -----------
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--arch", required=True, choices=list(VERSIONS.keys()))
    pre_args, remaining = pre.parse_known_args(argv)

    arch = pre_args.arch
    mod_path, model_name, config_name = VERSIONS[arch]

    # Ensure the python/ directory is on the import path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

    mod = importlib.import_module(mod_path)
    ModelClass = getattr(mod, model_name)
    ConfigClass = getattr(mod, config_name)

    # --- Phase 2: full argument parsing ------------------------------------
    p = argparse.ArgumentParser(
        description=f"Unified PolyHash trainer (arch={arch})",
        parents=[pre],
    )
    # Trainer flags (shared across all architectures)
    p.add_argument("--data-root", required=True)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--json", required=True)
    # LR / schedule
    p.add_argument("--lr-hash-mult", type=float, default=2.0)
    p.add_argument("--cosine-decay", action="store_true")
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--min-lr-ratio", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--fp16", action="store_true", help="Mixed precision training (2x speed on GPU)")
    p.add_argument("--save-model", default=None, help="Save model checkpoint to this path after training")
    p.add_argument("--ema", type=float, default=0, help="EMA decay (e.g. 0.999). 0=disabled.")
    p.add_argument("--depth-recurrence", type=int, default=1, help="Repeat MLP blocks N times (weight-tied)")
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "muon"], help="Optimizer")
    p.add_argument("--compile", action="store_true", help="torch.compile the model (reduce-overhead or max-autotune)")
    p.add_argument("--profile-cuda", type=int, default=0, metavar="N",
                    help="Profile first N steps, write Chrome trace to out/profile/")
    p.add_argument("--auto-batch", action="store_true",
                    help="Scale batch size to fill GPU memory (keeps tokens/step constant via gradient accumulation)")

    # Auto-add architecture-specific flags from config dataclass
    add_config_flags(p, ConfigClass)

    args = p.parse_args(["--arch", arch] + remaining)

    # --- Setup -------------------------------------------------------------
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    is_cuda = device.startswith("cuda") or device == "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # CUDA performance defaults
    if is_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    config = build_config(ConfigClass, args, seq_len=args.seq_len, seed=args.seed)
    model = ModelClass(config).to(device)
    params = sum(pp.numel() for pp in model.parameters())

    # Hash params summary
    hash_p = find_hash_params(model)
    hash_param_count = sum(pp.numel() for pp in hash_p)
    other_param_count = params - hash_param_count

    print(f"[{arch}] Model: {params:,} params ({params * 6 / 8 / 1024 / 1024:.2f} MB int6)")
    print(f"  Hash/table: {hash_param_count:,} ({hash_param_count * 6 / 8 / 1024 / 1024:.1f} MB)")
    print(f"  Neural:     {other_param_count:,} ({other_param_count * 6 / 8 / 1024 / 1024:.1f} MB)")

    _sanity_warnings(config, params, hash_param_count, args.steps, args.batch_size, args.seq_len, model=model)

    _pareto_check(params, config, arch, args.steps, args.batch_size, args.seq_len)

    # G3: Similar-config detection
    try:
        _sys_path = str(Path(__file__).resolve().parent.parent / "python")
        if _sys_path not in sys.path:
            sys.path.insert(0, _sys_path)
        from chronohorn.db import ChronohornDB
        _db_path = Path("out/chronohorn.db")
        if _db_path.exists():
            _db = ChronohornDB.open_read_only(str(_db_path))
            _arch_cfg = {}
            for _f in dataclasses.fields(config):
                _v = getattr(config, _f.name)
                if isinstance(_v, tuple):
                    _v = list(_v)
                _arch_cfg[_f.name] = _v
            similar = _db.find_similar(_arch_cfg, threshold=0.2)
            if similar:
                print(f"\n  \u26a0 SIMILAR CONFIGS already tested:", file=sys.stderr)
                for s in similar[:3]:
                    print(f"    {s['name']:30s} bpb={s['bpb']:.4f} ({s['match_pct']}% match)", file=sys.stderr)
            _db.close()
    except Exception:
        pass

    # --- Data --------------------------------------------------------------
    td = ShardedDataset(args.data_root, sl=args.seq_len)
    vt = load_val(args.data_root)
    print(f"Train: {td.num_shards} shards, Val: {len(vt) / 1e6:.1f}M tokens")

    # Measure bytes-per-token from the dataset itself
    # For sp1024 on fineweb: ~1.23 bytes/token (NOT 2.44)
    _total_file_bytes = sum(Path(p).stat().st_size for p in td.paths)
    _total_tokens = _total_file_bytes // 2  # uint16
    _total_text_bytes = _total_file_bytes  # raw file = uint16 tokens, 2 bytes storage per token
    # The competition metric: bits per byte of ORIGINAL text
    # For byte-level vocab (256 or patch), each token IS a byte: bytes_per_token = 1
    # For sp1024, we need the actual ratio from the tokenizer
    # Best estimate: total_raw_bytes / total_tokens from competition spec
    # fineweb10B = 10B bytes of text, tokenized into ~8B sp1024 tokens
    bytes_per_token = patch_size if patch_size > 1 else 10_000_000_000 / _total_tokens
    print(f"  bytes_per_token: {bytes_per_token:.4f}")

    # --- Auto-batch: scale batch size to fill GPU memory ---------------------
    effective_batch = args.batch_size
    grad_accum_steps = 1
    if args.auto_batch and is_cuda:
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_mem / 1024**3
        # Heuristic: ~1GB per 64 batch at seq_len=512 for 11M param model
        model_mb = params * 4 / 1024 / 1024  # fp32
        tokens_per_batch = args.batch_size * args.seq_len
        max_batch = max(args.batch_size, int(gpu_mem_gb * 64 / max(model_mb / 40, 1)))
        # Round down to multiple of original batch size
        max_batch = (max_batch // args.batch_size) * args.batch_size
        if max_batch > args.batch_size:
            effective_batch = max_batch
            grad_accum_steps = max_batch // args.batch_size
            print(f"  Auto-batch: {args.batch_size} x {grad_accum_steps} accumulation = {effective_batch} effective batch")
            print(f"    GPU: {gpu_mem_gb:.1f} GB, model: {model_mb:.0f} MB fp32")

    # --- torch.compile --------------------------------------------------------
    _compiled = False
    if args.compile:
        compile_mode = "max-autotune" if is_cuda else "reduce-overhead"
        try:
            compiled_model = torch.compile(model, mode=compile_mode)
            # Force compilation with a dummy forward to catch CC/Triton errors early
            with torch.no_grad():
                _dummy = compiled_model(torch.randint(0, 1024, (1, 32), device=device))
                del _dummy
            model = compiled_model
            _compiled = True
            print(f"  torch.compile: mode={compile_mode}")
        except Exception as e:
            print(f"  torch.compile failed ({type(e).__name__}: {e}), running without compilation")

    # --- Depth recurrence: repeat MLP blocks (weight-tied) ------------------
    if args.depth_recurrence > 1 and hasattr(model, 'mlp') and len(model.mlp) > 0:
        original_forward = model.forward
        recurrence = args.depth_recurrence
        def _recurrent_forward(chars):
            # Run the original forward but repeat MLP blocks
            # Monkey-patch: save original mlp list, repeat it, run, restore
            orig_mlp = list(model.mlp)
            model.mlp = torch.nn.ModuleList(orig_mlp * recurrence)
            out = original_forward(chars)
            model.mlp = torch.nn.ModuleList(orig_mlp)
            return out
        model.forward = _recurrent_forward
        print(f"  Depth recurrence: {len(model.mlp)} unique blocks × {recurrence} = {len(model.mlp) * recurrence} effective layers")

    # --- Optimizer -----------------------------------------------------------
    hids = set(id(pp) for pp in hash_p)
    other_p = [pp for pp in model.parameters() if id(pp) not in hids]

    adamw_kwargs = {"fused": True} if is_cuda else {}
    if args.optimizer == "muon":
        opt = Muon(
            [{"params": hash_p, "lr": args.lr * args.lr_hash_mult, "weight_decay": 0},
             {"params": other_p, "lr": args.lr, "weight_decay": args.weight_decay}],
            lr=args.lr, momentum=0.95,
            adamw_lr=args.lr * args.lr_hash_mult,  # fallback LR for 1D params
        )
        print(f"  Optimizer: Muon (NS-orthogonalized momentum)")
    elif args.lr_hash_mult != 1.0 and hash_p:
        opt = torch.optim.AdamW([
            {"params": hash_p, "lr": args.lr * args.lr_hash_mult, "weight_decay": 0},
            {"params": other_p, "lr": args.lr, "weight_decay": args.weight_decay},
        ], **adamw_kwargs)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay, **adamw_kwargs)
    if is_cuda and args.optimizer != "muon":
        print(f"  Optimizer: AdamW (fused=True, TF32=True)")

    # --- QAT setup -------------------------------------------------------------
    _qat_handles = []
    if config.qat_bits > 0:
        from chronohorn.families.polyhash.models.polyhash_v12 import apply_qat
        _qat_handles = apply_qat(model, bits=config.qat_bits, hash_only=config.qat_hash_only)
        scope = "hash tables only" if config.qat_hash_only else "all weights"
        print(f"  QAT: int{config.qat_bits} STE ({scope})")

    # --- EMA setup -----------------------------------------------------------
    ema_state = None
    if args.ema > 0:
        ema_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  EMA: decay={args.ema}")

    # --- Scheduler ---------------------------------------------------------
    sched = None
    if args.cosine_decay:
        min_r = args.min_lr_ratio

        def lr_fn(step: int) -> float:
            if step < args.warmup_steps:
                return step / max(args.warmup_steps, 1)
            prog = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
            return min_r + (1.0 - min_r) * 0.5 * (1.0 + math.cos(math.pi * prog))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)

    # --- Mixed precision setup ------------------------------------------------
    use_amp = args.fp16 and device != "cpu"
    from contextlib import nullcontext
    if use_amp:
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=True)
            amp_ctx = lambda: torch.amp.autocast("cuda")
        except (TypeError, AttributeError):
            # Fallback for older PyTorch
            scaler = torch.cuda.amp.GradScaler(enabled=True)
            amp_ctx = torch.cuda.amp.autocast
        print(f"  fp16 mixed precision: ON")
    else:
        scaler = None
        amp_ctx = nullcontext

    # --- CUDA profiler setup ------------------------------------------------
    _profiler = None
    if args.profile_cuda > 0 and is_cuda:
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
        print(f"  CUDA profiling: first {args.profile_cuda} steps -> {profile_dir}/")

    # --- Loss computation (byte or patch mode) --------------------------------
    patch_size = getattr(config, 'patch_size', 1)
    V = config.vocab_size

    def compute_loss(logits, batch):
        if patch_size > 1:
            # logits: [B, T_patch, P, V], batch: [B, L] raw bytes
            P = patch_size
            B, L = batch.shape
            T_patch = L // P
            patches = batch[:, :T_patch * P].reshape(B, T_patch, P)
            # Predict next patch: position t predicts patch t+1
            pred = logits[:, :-1]       # [B, T_patch-1, P, V]
            tgt = patches[:, 1:]        # [B, T_patch-1, P]
            return F.cross_entropy(pred.reshape(-1, V), tgt.reshape(-1))
        else:
            return F.cross_entropy(logits[:, :-1].reshape(-1, V), batch[:, 1:].reshape(-1))

    # --- Training loop -----------------------------------------------------
    probe_steps = build_probe_schedule(args.steps)
    probes: list[dict] = []
    t0 = time.time()
    best_bpb = float("inf")
    diverge_count = 0
    early_stopped = False
    steps_completed = 0

    for step in range(1, args.steps + 1):
        # Gradient accumulation for auto-batch scaling
        opt.zero_grad()
        accum_loss = 0.0
        for _accum in range(grad_accum_steps):
            b = torch.from_numpy(td.sample(args.batch_size)).to(device)
            with amp_ctx():
                if patch_size > 1:
                    P = patch_size
                    patches = b[:, :b.shape[1] // P * P].reshape(b.shape[0], -1, P)
                    logits = model(b, target_patches=patches)
                else:
                    logits = model(b)
                loss = compute_loss(logits, b)
                if grad_accum_steps > 1:
                    loss = loss / grad_accum_steps

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accum_loss += loss.item() * (grad_accum_steps if grad_accum_steps > 1 else 1)

        # NaN/inf guard
        if math.isnan(accum_loss) or math.isinf(accum_loss):
            print(f"NaN/Inf loss at step {step}. Stopping.", file=sys.stderr)
            early_stopped = True
            break

        if scaler:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        if sched:
            sched.step()
        if _profiler is not None:
            _profiler.step()
        steps_completed = step

        # EMA update
        if ema_state is not None:
            decay = args.ema
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    ema_state[k].mul_(decay).add_(v, alpha=1 - decay)

        if step % 500 == 0:
            el = time.time() - t0
            ts = step * effective_batch * args.seq_len / el
            print(f"  {step:>6d} | loss {accum_loss:.4f} | {ts:.0f} tok/s | {el:.0f}s")

        if step in probe_steps:
            # Swap to EMA weights for eval
            if ema_state is not None:
                saved_state = {k: v.clone() for k, v in model.state_dict().items()}
                model.load_state_dict(ema_state)
            model.eval()
            with torch.no_grad():
                vi = torch.randint(0, len(vt) - args.seq_len, (128,))
                vb = torch.stack(
                    [torch.from_numpy(vt[i : i + args.seq_len].copy()) for i in vi]
                ).to(device)
                if patch_size > 1:
                    P = patch_size
                    patches = vb[:, :vb.shape[1] // P * P].reshape(vb.shape[0], -1, P)
                    vlogits = model(vb, target_patches=patches)
                else:
                    vlogits = model(vb)
                vl = compute_loss(vlogits, vb).item()
            bpt = vl / math.log(2)
            # For byte-level prediction (patch mode), bpb = bpt directly
            # For token-level (1024 vocab), bpb = bpt / tokens_per_byte
            bpb = bpt if patch_size > 1 else bpt / bytes_per_token
            probes.append({"step": step, "bpb": bpb, "bpt": bpt, "loss": vl})
            print(f"  PROBE {step}: bpt={bpt:.4f} bpb~{bpb:.4f}")
            # Restore training weights after EMA eval
            if ema_state is not None:
                model.load_state_dict(saved_state)
            model.train()

            # --- Early stopping on divergence (W6) -------------------------
            if bpb < best_bpb:
                best_bpb = bpb
                diverge_count = 0
            elif bpb > best_bpb * 1.5:
                diverge_count += 1
                if diverge_count >= 2:
                    print(
                        f"DIVERGED at step {step}: bpb={bpb:.4f} > 1.5 x best={best_bpb:.4f}",
                        file=sys.stderr,
                    )
                    early_stopped = True
                    break
            else:
                diverge_count = 0

    # --- Cleanup profiler -----------------------------------------------------
    if _profiler is not None:
        _profiler.__exit__(None, None, None)
        print(f"  CUDA profile saved to out/profile/")

    # --- Final evaluation (use EMA weights if available) ------------------
    el = time.time() - t0
    ts = steps_completed * effective_batch * args.seq_len / el
    if ema_state is not None:
        model.load_state_dict(ema_state)
    model.eval()
    with torch.no_grad():
        tl, nc = 0.0, 0
        for i in range(0, min(len(vt) - args.seq_len, 2000 * args.seq_len), args.seq_len):
            vb = (
                torch.from_numpy(vt[i : i + args.seq_len].copy())
                .unsqueeze(0)
                .to(device)
            )
            if patch_size > 1:
                P = patch_size
                patches = vb[:, :vb.shape[1] // P * P].reshape(vb.shape[0], -1, P)
                vlogits = model(vb, target_patches=patches)
            else:
                vlogits = model(vb)
            tl += compute_loss(vlogits, vb).item()
            nc += 1
        al = tl / max(nc, 1)
    fbpt = al / math.log(2)
    fbpb = fbpt if patch_size > 1 else fbpt / bytes_per_token
    print(f"\nFINAL: bpt={fbpt:.4f} bpb~{fbpb:.4f} | {params:,} params | {el:.0f}s | {ts:.0f} tok/s")
    if early_stopped:
        print(f"  (early stopped at step {steps_completed})")

    # --- Architecture config snapshot (W4) ---------------------------------
    arch_config: dict[str, Any] = {}
    for f in dataclasses.fields(config):
        val = getattr(config, f.name)
        # Ensure JSON-serializable (tuples -> lists)
        if isinstance(val, tuple):
            val = list(val)
        arch_config[f.name] = val
    arch_config["version"] = arch

    # --- Result JSON -------------------------------------------------------
    result: dict[str, Any] = {
        "model": {
            "test_bpb": fbpb,
            "test_bits_per_token": fbpt,
            "test_eval_loss": al,
            "params": params,
            "architecture": f"polyhash_{arch}",
        },
        "config": {
            "train": {
                "steps": args.steps,
                "seq_len": args.seq_len,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "lr_hash_mult": args.lr_hash_mult,
                "cosine_decay": args.cosine_decay,
                "warmup_steps": args.warmup_steps,
                "weight_decay": args.weight_decay,
            },
            "architecture": arch_config,
        },
        "training": {
            "performance": {
                "tokens_per_second": ts,
                "elapsed_sec": el,
                "steps_completed": steps_completed,
            },
            "probes": probes,
            "early_stopped": early_stopped,
        },
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(result, indent=2))
    print(f"Saved to {args.json}")

    # Save model checkpoint if requested
    if args.save_model:
        ckpt_path = Path(args.save_model)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": arch_config,
            "arch": arch,
            "step": steps_completed,
            "bpb": fbpb,
        }, str(ckpt_path))
        print(f"Model saved to {ckpt_path}")


if __name__ == "__main__":
    main()
