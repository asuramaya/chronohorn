#!/usr/bin/env python3
"""Train PolyHash v3 -- stacked O(1) features from trading/chaos math."""
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
from chronohorn.families.polyhash.models.polyhash_v3 import PolyHashV3, PolyHashV3Config


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
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--json", required=True)
    p.add_argument("--num-tables", type=int, default=8)
    p.add_argument("--buckets", type=int, default=32768)
    p.add_argument("--embed-dim", type=int, default=16)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--mlp-layers", type=int, default=2)
    p.add_argument("--byte-embed", type=int, default=128)
    p.add_argument("--activation", default="swiglu")
    p.add_argument("--no-match", action="store_true")
    p.add_argument("--no-delta", action="store_true")
    p.add_argument("--no-freq", action="store_true")
    p.add_argument("--no-fingerprints", action="store_true")
    p.add_argument("--no-reservoir", action="store_true")
    p.add_argument("--match-offsets", default="1,2,3,4,5,6,7,8,10,12,16,20,24,32")
    p.add_argument("--delta-offsets", default="1,2,3,5,8")
    p.add_argument("--freq-windows", default="4,8,16,32")
    p.add_argument("--fp-shifts", default="1,2,4")
    p.add_argument("--fp-buckets", type=int, default=16384)
    p.add_argument("--fp-embed", type=int, default=16)
    p.add_argument("--reservoir-dim", type=int, default=64)
    p.add_argument("--reservoir-window", type=int, default=8)
    p.add_argument("--delta-buckets", type=int, default=2048)
    p.add_argument("--delta-embed", type=int, default=8)
    p.add_argument("--lr-hash-mult", type=float, default=2.0)
    p.add_argument("--cosine-decay", action="store_true")
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--dropout", type=float, default=0.0)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    match_offsets = tuple(int(x) for x in args.match_offsets.split(",")) if not args.no_match else ()
    delta_offsets = tuple(int(x) for x in args.delta_offsets.split(",")) if not args.no_delta else ()
    freq_windows = tuple(int(x) for x in args.freq_windows.split(",")) if not args.no_freq else ()
    fp_shifts = tuple(int(x) for x in args.fp_shifts.split(",")) if not args.no_fingerprints else ()

    config = PolyHashV3Config(
        vocab_size=1024, byte_embed_dim=args.byte_embed,
        num_tables=args.num_tables, buckets_per_table=args.buckets,
        embed_per_table=args.embed_dim, hidden_dim=args.hidden,
        num_layers=args.mlp_layers, activation=args.activation,
        match_offsets=match_offsets, delta_offsets=delta_offsets,
        delta_embed_dim=args.delta_embed, delta_buckets=args.delta_buckets,
        local_freq_windows=freq_windows,
        use_fingerprints=not args.no_fingerprints,
        fp_shift_rates=fp_shifts, fp_buckets=args.fp_buckets, fp_embed_dim=args.fp_embed,
        use_reservoir=not args.no_reservoir,
        reservoir_dim=args.reservoir_dim, reservoir_window=args.reservoir_window,
        lr_hash_mult=args.lr_hash_mult,
        max_seq_len=args.seq_len, init_seed=args.seed,
        dropout=args.dropout, label_smoothing=args.label_smoothing,
    )
    model = PolyHashV3(config).to(device)
    params = sum(pp.numel() for pp in model.parameters())
    print(f"Model: {params:,} params ({params*6/8/1024/1024:.2f} MB int6)")

    train_data = ShardedDataset(args.data_root, seq_len=args.seq_len)
    val_tokens = load_val(args.data_root)
    print(f"Train: {train_data.num_shards} shards, Val: {len(val_tokens)/1e6:.1f}M tokens")

    hash_params = []
    for t in model.hash_tables:
        hash_params += list(t.parameters())
    for t in model.delta_tables:
        hash_params += list(t.parameters())
    for t in model.fp_tables:
        hash_params += list(t.parameters())
    hash_ids = set(id(pp) for pp in hash_params)
    other_params = [pp for pp in model.parameters() if id(pp) not in hash_ids]

    if args.lr_hash_mult != 1.0 and hash_params:
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
            logits[:, :-1, :].reshape(-1, 1024), batch[:, 1:].reshape(-1),
            label_smoothing=args.label_smoothing)
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
                vbatch = torch.stack([torch.from_numpy(val_tokens[i:i+args.seq_len].copy()) for i in vi]).to(device)
                vloss = torch.nn.functional.cross_entropy(
                    model(vbatch)[:, :-1, :].reshape(-1, 1024), vbatch[:, 1:].reshape(-1)).item()
            bpt = vloss / math.log(2)
            bpb = bpt / 2.44
            probes.append({"step": step, "bpb": bpb, "bpt": bpt, "loss": vloss})
            print(f"  PROBE {step}: bpt={bpt:.4f} bpb~{bpb:.4f}")
            model.train()

    elapsed = time.time() - t0
    toks_s = args.steps * args.batch_size * args.seq_len / elapsed
    model.eval()
    with torch.no_grad():
        total_loss, n_chunks = 0.0, 0
        for i in range(0, min(len(val_tokens) - args.seq_len, 100 * args.seq_len), args.seq_len):
            vb = torch.from_numpy(val_tokens[i:i+args.seq_len].copy()).unsqueeze(0).to(device)
            total_loss += torch.nn.functional.cross_entropy(
                model(vb)[:, :-1, :].reshape(-1, 1024), vb[:, 1:].reshape(-1)).item()
            n_chunks += 1
        avg_loss = total_loss / max(n_chunks, 1)
    final_bpt = avg_loss / math.log(2)
    final_bpb = final_bpt / 2.44
    print(f"\nFINAL: bpt={final_bpt:.4f} bpb~{final_bpb:.4f} | {params:,} params | {elapsed:.0f}s")

    result = {
        "model": {"test_bpb": final_bpb, "test_bits_per_token": final_bpt,
                   "test_eval_loss": avg_loss, "params": params, "architecture": "polyhash_v3"},
        "config": {"train": {"steps": args.steps, "seq_len": args.seq_len,
                              "batch_size": args.batch_size, "learning_rate": args.lr}},
        "training": {"performance": {"tokens_per_second": toks_s, "elapsed_sec": elapsed,
                                      "steps_completed": args.steps}, "probes": probes},
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(result, indent=2))
    print(f"Saved to {args.json}")


if __name__ == "__main__":
    main()
