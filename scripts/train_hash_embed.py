#!/usr/bin/env python3
"""Train a hash embedding model on the parameter-golf FineWeb data."""
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
from chronohorn.models.hash_embed_model import HashEmbedModel, HashEmbedConfig


def load_shards(data_root: str) -> np.ndarray:
    shards = sorted(_glob.glob(f"{data_root}/fineweb_train_*.bin"))
    if not shards:
        raise FileNotFoundError(f"No shards in {data_root}")
    arrays = [np.fromfile(s, dtype=np.uint16) for s in shards]
    tokens = np.concatenate(arrays)
    return tokens[tokens < 1024].astype(np.int64)


def load_val(data_root: str) -> np.ndarray:
    tokens = np.fromfile(f"{data_root}/fineweb_val_000000.bin", dtype=np.uint16).astype(np.int64)
    return tokens[tokens < 1024]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-tables", type=int, default=4)
    parser.add_argument("--buckets", type=int, default=16384)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--mlp-layers", type=int, default=2)
    parser.add_argument("--byte-embed", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    print(f"Loading data from {args.data_root}...")
    train_tokens = load_shards(args.data_root)
    val_tokens = load_val(args.data_root)
    print(f"Train: {len(train_tokens)/1e9:.2f}B tokens, Val: {len(val_tokens)/1e6:.1f}M tokens")

    skip_patterns = [(1, 2), (1, 2, 3), (1, 3), (2, 3), (1, 4), (2, 4), (1, 2, 4), (3, 4)]
    config = HashEmbedConfig(
        vocab_size=1024, byte_embed_dim=args.byte_embed,
        num_tables=args.num_tables, buckets_per_table=args.buckets,
        embed_per_table=args.embed_dim, hidden_dim=args.hidden,
        num_layers=args.mlp_layers,
        skip_patterns=tuple(skip_patterns[:args.num_tables]),
        max_seq_len=args.seq_len, init_seed=args.seed,
    )
    model = HashEmbedModel(config).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {params:,} params ({params*6/8/1024/1024:.2f} MB int6)")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    probes = []
    t0 = time.time()

    probe_steps = set()
    s = 50
    while s <= args.steps:
        probe_steps.add(s)
        s = int(s * 2)
    probe_steps.add(args.steps)

    for step in range(1, args.steps + 1):
        idx = torch.randint(0, len(train_tokens) - args.seq_len, (args.batch_size,))
        batch = torch.stack([torch.from_numpy(train_tokens[i:i+args.seq_len].copy()) for i in idx]).to(device)
        logits = model(batch)
        loss = torch.nn.functional.cross_entropy(logits[:, :-1, :].reshape(-1, 1024), batch[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            elapsed = time.time() - t0
            toks_s = step * args.batch_size * args.seq_len / elapsed
            print(f"  {step:>6d} | loss {loss.item():.4f} | {toks_s:.0f} tok/s | {elapsed:.0f}s")

        if step in probe_steps:
            model.eval_mode = True
            with torch.no_grad():
                vi = torch.randint(0, len(val_tokens) - args.seq_len, (16,))
                vbatch = torch.stack([torch.from_numpy(val_tokens[i:i+args.seq_len].copy()) for i in vi]).to(device)
                vloss = torch.nn.functional.cross_entropy(model(vbatch)[:, :-1, :].reshape(-1, 1024), vbatch[:, 1:].reshape(-1)).item()
            bpt = vloss / math.log(2)
            bpb = bpt / 2.44
            probes.append({"step": step, "bpb": bpb, "bpt": bpt, "loss": vloss})
            print(f"  PROBE {step}: bpt={bpt:.4f} bpb~{bpb:.4f}")
            model.eval_mode = False

    elapsed = time.time() - t0
    toks_s = args.steps * args.batch_size * args.seq_len / elapsed

    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        n_chunks = 0
        for i in range(0, min(len(val_tokens) - args.seq_len, 100 * args.seq_len), args.seq_len):
            vb = torch.from_numpy(val_tokens[i:i+args.seq_len].copy()).unsqueeze(0).to(device)
            total_loss += torch.nn.functional.cross_entropy(model(vb)[:, :-1, :].reshape(-1, 1024), vb[:, 1:].reshape(-1)).item()
            n_chunks += 1
        avg_loss = total_loss / max(n_chunks, 1)

    final_bpt = avg_loss / math.log(2)
    final_bpb = final_bpt / 2.44
    print(f"\nFINAL: bpt={final_bpt:.4f} bpb~{final_bpb:.4f} | {params:,} params | {elapsed:.0f}s")

    result = {
        "model": {"test_bpb": final_bpb, "test_bits_per_token": final_bpt, "test_eval_loss": avg_loss,
                   "params": params, "architecture": "hash_embed", "num_tables": args.num_tables,
                   "buckets_per_table": args.buckets, "embed_per_table": args.embed_dim, "hidden_dim": args.hidden},
        "config": {"train": {"steps": args.steps, "seq_len": args.seq_len, "batch_size": args.batch_size, "learning_rate": args.lr}},
        "training": {"performance": {"tokens_per_second": toks_s, "elapsed_sec": elapsed, "steps_completed": args.steps}, "probes": probes},
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(result, indent=2))
    print(f"Saved to {args.json}")


if __name__ == "__main__":
    main()
