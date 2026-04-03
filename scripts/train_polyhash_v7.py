#!/usr/bin/env python3
"""Train PolyHash v7 — State-Addressed Memory."""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import glob as _glob
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
from chronohorn.models.polyhash_v7 import PolyHashV7, V7Config

class ShardedDataset:
    def __init__(self, dr, sl=512):
        self.paths = sorted(_glob.glob(f"{dr}/fineweb_train_*.bin"))
        if not self.paths: raise FileNotFoundError(dr)
        self.sl = sl; self._s = None; self._load(0)
    def _load(self, i):
        t = np.fromfile(self.paths[i], dtype=np.uint16).astype(np.int64)
        self._s = t[t < 1024]
    def sample(self, bs):
        if np.random.randint(100) == 0 or self._s is None:
            self._load(np.random.randint(len(self.paths)))
        n = len(self._s) - self.sl
        ix = np.random.randint(0, n, size=bs)
        return np.stack([self._s[i:i+self.sl] for i in ix])
    @property
    def num_shards(self): return len(self.paths)

def load_val(dr):
    t = np.fromfile(f"{dr}/fineweb_val_000000.bin", dtype=np.uint16).astype(np.int64)
    return t[t < 1024]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--steps", type=int, default=3200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--json", required=True)
    # Architecture
    p.add_argument("--num-tables", type=int, default=8)
    p.add_argument("--buckets", type=int, default=65536)
    p.add_argument("--embed-dim", type=int, default=16)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--mlp-layers", type=int, default=2)
    p.add_argument("--byte-embed", type=int, default=128)
    p.add_argument("--conv-kernel", type=int, default=8)
    p.add_argument("--match-offsets", default="1,2,3,4,5,6,7,8,12,16,24,32")
    p.add_argument("--scan-dim", type=int, default=256)
    p.add_argument("--scan-groups", type=int, default=4)
    p.add_argument("--scan-chunk", type=int, default=32)
    # State-addressed memory
    p.add_argument("--no-sam", action="store_true")
    p.add_argument("--sam-buckets", type=int, default=131072)
    p.add_argument("--sam-embed", type=int, default=32)
    p.add_argument("--sam-heads", type=int, default=4)
    p.add_argument("--sam-quant", default="sign")
    p.add_argument("--sam-bits", type=int, default=16)
    p.add_argument("--sam-ste", action="store_true")
    p.add_argument("--sam-soft-temp", type=float, default=0.0)
    p.add_argument("--sam-2bit", action="store_true")
    # Training
    p.add_argument("--lr-hash-mult", type=float, default=2.0)
    p.add_argument("--cosine-decay", action="store_true")
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--hash-dropout", type=float, default=0.0)
    p.add_argument("--min-lr-ratio", type=float, default=0.01)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    mo = tuple(int(x) for x in args.match_offsets.split(",")) if args.match_offsets else ()

    config = V7Config(
        vocab_size=1024, byte_embed_dim=args.byte_embed,
        num_tables=args.num_tables, buckets_per_table=args.buckets,
        embed_per_table=args.embed_dim, hidden_dim=args.hidden,
        num_layers=args.mlp_layers, conv_kernel=args.conv_kernel,
        match_offsets=mo, scan_dim=args.scan_dim, scan_groups=args.scan_groups,
        scan_chunk_size=args.scan_chunk,
        sam_enabled=not args.no_sam, sam_buckets=args.sam_buckets,
        sam_embed_dim=args.sam_embed, sam_heads=args.sam_heads,
        sam_quant_mode=args.sam_quant, sam_quant_bits=args.sam_bits,
        sam_straight_through=args.sam_ste, sam_soft_temp=args.sam_soft_temp, sam_2bit=args.sam_2bit,
        dropout=args.dropout, max_seq_len=args.seq_len, init_seed=args.seed,
        hash_dropout=args.hash_dropout,
    )
    model = PolyHashV7(config).to(device)
    params = sum(pp.numel() for pp in model.parameters())
    print(f"Model: {params:,} params ({params*6/8/1024/1024:.2f} MB int6)")

    td = ShardedDataset(args.data_root, sl=args.seq_len)
    vt = load_val(args.data_root)
    print(f"Train: {td.num_shards} shards, Val: {len(vt)/1e6:.1f}M tokens")

    # Per-group LR for all hash/memory tables
    hash_p = list(model.hash_tables.parameters())
    if model.sam is not None:
        hash_p += list(model.sam.memory_tables.parameters())
    hids = set(id(pp) for pp in hash_p)
    other_p = [pp for pp in model.parameters() if id(pp) not in hids]
    if args.lr_hash_mult != 1.0 and hash_p:
        opt = torch.optim.AdamW([
            {"params": hash_p, "lr": args.lr * args.lr_hash_mult, "weight_decay": 0},
            {"params": other_p, "lr": args.lr, "weight_decay": args.weight_decay},
        ])
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    sched = None
    if args.cosine_decay:
        mr = args.min_lr_ratio
        def lr_fn(step):
            if step < args.warmup_steps: return step / max(args.warmup_steps, 1)
            prog = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
            return mr + (1.0 - mr) * 0.5 * (1.0 + math.cos(math.pi * prog))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)

    probes = []; t0 = time.time(); ps = set(); s = 100
    while s <= args.steps: ps.add(s); s = int(s * 2)
    ps.add(args.steps)

    for step in range(1, args.steps + 1):
        b = torch.from_numpy(td.sample(args.batch_size)).to(device)
        logits = model(b)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, 1024), b[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if sched: sched.step()
        if step % 500 == 0:
            el = time.time() - t0; ts = step * args.batch_size * args.seq_len / el
            print(f"  {step:>6d} | loss {loss.item():.4f} | {ts:.0f} tok/s | {el:.0f}s")
        if step in ps:
            model.eval()
            with torch.no_grad():
                vi = torch.randint(0, len(vt) - args.seq_len, (32,))
                vb = torch.stack([torch.from_numpy(vt[i:i+args.seq_len].copy()) for i in vi]).to(device)
                vl = F.cross_entropy(model(vb)[:, :-1].reshape(-1, 1024), vb[:, 1:].reshape(-1)).item()
            bpt = vl / math.log(2); bpb = bpt / 2.44
            probes.append({"step": step, "bpb": bpb, "bpt": bpt, "loss": vl})
            print(f"  PROBE {step}: bpt={bpt:.4f} bpb~{bpb:.4f}")
            model.train()

    el = time.time() - t0; ts = args.steps * args.batch_size * args.seq_len / el
    model.eval()
    with torch.no_grad():
        tl, nc = 0.0, 0
        for i in range(0, min(len(vt) - args.seq_len, 200 * args.seq_len), args.seq_len):
            vb = torch.from_numpy(vt[i:i+args.seq_len].copy()).unsqueeze(0).to(device)
            tl += F.cross_entropy(model(vb)[:, :-1].reshape(-1, 1024), vb[:, 1:].reshape(-1)).item(); nc += 1
        al = tl / max(nc, 1)
    fbpt = al / math.log(2); fbpb = fbpt / 2.44
    print(f"\nFINAL: bpt={fbpt:.4f} bpb~{fbpb:.4f} | {params:,} params | {el:.0f}s | {ts:.0f} tok/s")
    result = {"model": {"test_bpb": fbpb, "test_bits_per_token": fbpt, "test_eval_loss": al,
        "params": params, "architecture": "polyhash_v7"},
        "config": {"train": {"steps": args.steps, "seq_len": args.seq_len,
            "batch_size": args.batch_size, "learning_rate": args.lr}},
        "training": {"performance": {"tokens_per_second": ts, "elapsed_sec": el,
            "steps_completed": args.steps}, "probes": probes}}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(result, indent=2))
    print(f"Saved to {args.json}")

if __name__ == "__main__": main()
