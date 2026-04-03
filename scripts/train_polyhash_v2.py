#!/usr/bin/env python3
"""Train a PolyHash v2 model on the parameter-golf FineWeb data."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import glob as _glob
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
from chronohorn.models.polyhash_v2 import PolyHashV2, PolyHashV2Config


class ShardedDataset:
    def __init__(self, data_root: str, seq_len: int = 512) -> None:
        self.shard_paths = sorted(_glob.glob(f"{data_root}/fineweb_train_*.bin"))
        if not self.shard_paths:
            raise FileNotFoundError(f"No shards in {data_root}")
        self.seq_len = seq_len
        self._current_shard = None
        self._current_idx = -1
        self._load_shard(0)

    def _load_shard(self, idx: int) -> None:
        tokens = np.fromfile(self.shard_paths[idx], dtype=np.uint16).astype(np.int64)
        self._current_shard = tokens[tokens < 1024]
        self._current_idx = idx

    def sample_batch(self, batch_size: int) -> np.ndarray:
        if np.random.randint(100) == 0 or self._current_shard is None:
            self._load_shard(np.random.randint(len(self.shard_paths)))
        shard = self._current_shard
        n = len(shard) - self.seq_len
        idx = np.random.randint(0, n, size=batch_size)
        return np.stack([shard[i:i+self.seq_len] for i in idx])

    @property
    def num_shards(self) -> int:
        return len(self.shard_paths)


def load_val(data_root: str) -> np.ndarray:
    tokens = np.fromfile(f"{data_root}/fineweb_val_000000.bin", dtype=np.uint16).astype(np.int64)
    return tokens[tokens < 1024]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--json", required=True)
    # Architecture
    p.add_argument("--num-tables", type=int, default=16)
    p.add_argument("--buckets", type=int, default=8192)
    p.add_argument("--embed-dim", type=int, default=16)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--mlp-layers", type=int, default=2)
    p.add_argument("--byte-embed", type=int, default=128)
    # Extensions
    p.add_argument("--skip-mode", default="default")
    p.add_argument("--max-offset", type=int, default=8)
    p.add_argument("--gate-mode", default="none")
    p.add_argument("--gate-topk", type=int, default=0)
    p.add_argument("--film", action="store_true")
    p.add_argument("--multi-hash", type=int, default=1)
    p.add_argument("--multi-hash-pool", default="mean")
    p.add_argument("--exact-unigram", action="store_true")
    p.add_argument("--scan-dim", type=int, default=0)
    p.add_argument("--scan-mode", default="gated")
    p.add_argument("--pyramid-levels", type=int, default=0)
    p.add_argument("--pyramid-quant", default="sign")
    p.add_argument("--activation", default="relu")
    p.add_argument("--num-experts", type=int, default=1)
    p.add_argument("--hash-dropout", type=float, default=0.0)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--no-residual", action="store_true")
    p.add_argument("--no-layer-norm", action="store_true")
    p.add_argument("--cosine-decay", action="store_true")
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--lr-hash-mult", type=float, default=1.0)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    print(f"Loading data from {args.data_root}...")
    train_data = ShardedDataset(args.data_root, seq_len=args.seq_len)
    val_tokens = load_val(args.data_root)
    print(f"Train: {train_data.num_shards} shards, Val: {len(val_tokens)/1e6:.1f}M tokens")

    config = PolyHashV2Config(
        vocab_size=1024, byte_embed_dim=args.byte_embed,
        num_tables=args.num_tables, buckets_per_table=args.buckets,
        embed_per_table=args.embed_dim, hidden_dim=args.hidden,
        num_layers=args.mlp_layers,
        use_residual=not args.no_residual,
        use_layer_norm=not args.no_layer_norm,
        max_seq_len=args.seq_len, init_seed=args.seed,
        dropout=args.dropout,
        skip_pattern_mode=args.skip_mode, max_offset=args.max_offset,
        gate_mode=args.gate_mode, gate_topk=args.gate_topk,
        film=args.film,
        multi_hash=args.multi_hash, multi_hash_pool=args.multi_hash_pool,
        exact_unigram=args.exact_unigram,
        scan_dim=args.scan_dim, scan_mode=args.scan_mode,
        pyramid_levels=args.pyramid_levels, pyramid_quant=args.pyramid_quant,
        activation=args.activation, num_experts=args.num_experts,
        hash_dropout=args.hash_dropout, label_smoothing=args.label_smoothing,
    )
    model = PolyHashV2(config).to(device)
    params = sum(pp.numel() for pp in model.parameters())
    print(f"Model: {params:,} params ({params*6/8/1024/1024:.2f} MB int6)")

    # Optimizer with optional per-group LR
    if args.lr_hash_mult != 1.0:
        hash_params = list(model.hash_tables.parameters())
        if model.unigram_table is not None:
            hash_params += list(model.unigram_table.parameters())
        for t in model.pyramid_tables:
            hash_params += list(t.parameters())
        hash_ids = set(id(pp) for pp in hash_params)
        other_params = [pp for pp in model.parameters() if id(pp) not in hash_ids]
        optimizer = torch.optim.Adam([
            {"params": hash_params, "lr": args.lr * args.lr_hash_mult},
            {"params": other_params, "lr": args.lr},
        ])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = None
    if args.cosine_decay:
        def lr_lambda(step):
            if step < args.warmup_steps:
                return step / max(args.warmup_steps, 1)
            progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    probes = []
    t0 = time.time()
    probe_steps = set()
    s = 50
    while s <= args.steps:
        probe_steps.add(s)
        s = int(s * 2)
    probe_steps.add(args.steps)

    for step in range(1, args.steps + 1):
        batch_np = train_data.sample_batch(args.batch_size)
        batch = torch.from_numpy(batch_np).to(device)
        logits = model(batch)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, 1024),
            batch[:, 1:].reshape(-1),
            label_smoothing=args.label_smoothing,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if step % 100 == 0:
            elapsed = time.time() - t0
            toks_s = step * args.batch_size * args.seq_len / elapsed
            print(f"  {step:>6d} | loss {loss.item():.4f} | {toks_s:.0f} tok/s | {elapsed:.0f}s")

        if step in probe_steps:
            model.eval()
            with torch.no_grad():
                vi = torch.randint(0, len(val_tokens) - args.seq_len, (16,))
                vbatch = torch.stack([
                    torch.from_numpy(val_tokens[i:i+args.seq_len].copy())
                    for i in vi
                ]).to(device)
                vloss = torch.nn.functional.cross_entropy(
                    model(vbatch)[:, :-1, :].reshape(-1, 1024),
                    vbatch[:, 1:].reshape(-1),
                ).item()
            bpt = vloss / math.log(2)
            bpb = bpt / 2.44
            probes.append({"step": step, "bpb": bpb, "bpt": bpt, "loss": vloss})
            print(f"  PROBE {step}: bpt={bpt:.4f} bpb~{bpb:.4f}")
            model.train()

    elapsed = time.time() - t0
    toks_s = args.steps * args.batch_size * args.seq_len / elapsed

    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        n_chunks = 0
        for i in range(0, min(len(val_tokens) - args.seq_len, 100 * args.seq_len), args.seq_len):
            vb = torch.from_numpy(val_tokens[i:i+args.seq_len].copy()).unsqueeze(0).to(device)
            total_loss += torch.nn.functional.cross_entropy(
                model(vb)[:, :-1, :].reshape(-1, 1024),
                vb[:, 1:].reshape(-1),
            ).item()
            n_chunks += 1
        avg_loss = total_loss / max(n_chunks, 1)

    final_bpt = avg_loss / math.log(2)
    final_bpb = final_bpt / 2.44
    print(f"\nFINAL: bpt={final_bpt:.4f} bpb~{final_bpb:.4f} | {params:,} params | {elapsed:.0f}s")

    result = {
        "model": {
            "test_bpb": final_bpb, "test_bits_per_token": final_bpt,
            "test_eval_loss": avg_loss, "params": params,
            "architecture": "polyhash_v2",
        },
        "config": {"train": {
            "steps": args.steps, "seq_len": args.seq_len,
            "batch_size": args.batch_size, "learning_rate": args.lr,
        }},
        "training": {
            "performance": {"tokens_per_second": toks_s, "elapsed_sec": elapsed,
                            "steps_completed": args.steps},
            "probes": probes,
        },
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(result, indent=2))
    print(f"Saved to {args.json}")


if __name__ == "__main__":
    main()
