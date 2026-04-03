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
    "v6":  ("chronohorn.models.polyhash_v6",  "PolyHashV6",  "V6Config"),
    "v7":  ("chronohorn.models.polyhash_v7",  "PolyHashV7",  "V7Config"),
    "v8":  ("chronohorn.models.polyhash_v8",  "PolyHashV8",  "V8Config"),
    "v8h": ("chronohorn.models.polyhash_v8h", "PolyHashV8H", "V8HConfig"),
    "v8m": ("chronohorn.models.polyhash_v8m", "PolyHashV8M", "V8MConfig"),
    "v10": ("chronohorn.models.polyhash_v10", "PolyHashV10", "V10Config"),
    "v11": ("chronohorn.models.polyhash_v11", "PolyHashV11", "V11Config"),
    "v12": ("chronohorn.models.polyhash_v12", "PolyHashV12", "V12Config"),
}

# Fields that live in the trainer, not the model config -- skip when
# auto-generating argparse flags from config dataclass fields.
TRAINER_FIELDS = {"vocab_size", "max_seq_len", "init_seed"}

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
        if f.name == "vocab_size":
            kwargs["vocab_size"] = 1024
        elif f.name == "max_seq_len":
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

    for w in warnings:
        print(f"\n  \u26a0 {w}", file=sys.stderr)
    if warnings:
        print(f"\n  ({len(warnings)} warnings \u2014 training will proceed anyway)\n", file=sys.stderr)


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

    # Auto-add architecture-specific flags from config dataclass
    add_config_flags(p, ConfigClass)

    args = p.parse_args(["--arch", arch] + remaining)

    # --- Setup -------------------------------------------------------------
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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

    # --- Optimizer (per-group LR for hash tables) --------------------------
    hids = set(id(pp) for pp in hash_p)
    other_p = [pp for pp in model.parameters() if id(pp) not in hids]
    if args.lr_hash_mult != 1.0 and hash_p:
        opt = torch.optim.AdamW([
            {"params": hash_p, "lr": args.lr * args.lr_hash_mult, "weight_decay": 0},
            {"params": other_p, "lr": args.lr, "weight_decay": args.weight_decay},
        ])
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None
    from contextlib import nullcontext
    amp_ctx = torch.cuda.amp.autocast if use_amp else nullcontext
    if use_amp:
        print(f"  fp16 mixed precision: ON")

    # --- Training loop -----------------------------------------------------
    probe_steps = build_probe_schedule(args.steps)
    probes: list[dict] = []
    t0 = time.time()
    best_bpb = float("inf")
    diverge_count = 0
    early_stopped = False
    steps_completed = 0

    for step in range(1, args.steps + 1):
        b = torch.from_numpy(td.sample(args.batch_size)).to(device)
        with amp_ctx():
            logits = model(b)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, 1024), b[:, 1:].reshape(-1))

        # NaN/inf guard — stop immediately, don't waste GPU time
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"NaN/Inf loss at step {step}. Stopping.", file=sys.stderr)
            early_stopped = True
            break

        opt.zero_grad()
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        if sched:
            sched.step()
        steps_completed = step

        if step % 500 == 0:
            el = time.time() - t0
            ts = step * args.batch_size * args.seq_len / el
            print(f"  {step:>6d} | loss {loss.item():.4f} | {ts:.0f} tok/s | {el:.0f}s")

        if step in probe_steps:
            model.eval()
            with torch.no_grad():
                vi = torch.randint(0, len(vt) - args.seq_len, (128,))
                vb = torch.stack(
                    [torch.from_numpy(vt[i : i + args.seq_len].copy()) for i in vi]
                ).to(device)
                vl = F.cross_entropy(
                    model(vb)[:, :-1].reshape(-1, 1024), vb[:, 1:].reshape(-1)
                ).item()
            bpt = vl / math.log(2)
            bpb = bpt / 2.44
            probes.append({"step": step, "bpb": bpb, "bpt": bpt, "loss": vl})
            print(f"  PROBE {step}: bpt={bpt:.4f} bpb~{bpb:.4f}")
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

    # --- Final evaluation --------------------------------------------------
    el = time.time() - t0
    ts = steps_completed * args.batch_size * args.seq_len / el
    model.eval()
    with torch.no_grad():
        tl, nc = 0.0, 0
        for i in range(0, min(len(vt) - args.seq_len, 2000 * args.seq_len), args.seq_len):
            vb = (
                torch.from_numpy(vt[i : i + args.seq_len].copy())
                .unsqueeze(0)
                .to(device)
            )
            tl += F.cross_entropy(
                model(vb)[:, :-1].reshape(-1, 1024), vb[:, 1:].reshape(-1)
            ).item()
            nc += 1
        al = tl / max(nc, 1)
    fbpt = al / math.log(2)
    fbpb = fbpt / 2.44
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


if __name__ == "__main__":
    main()
