#!/usr/bin/env python3
"""Build an n-gram table from all training shards."""
import numpy as np
import sys
import glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
from chronohorn.families.polyhash.models.ngram_table import NgramTable

data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/roots/fineweb10B_sp1024"
output = sys.argv[2] if len(sys.argv) > 2 else "out/ngram_table_8B.npz"

# Find all training shards
shards = sorted(glob.glob(f"{data_dir}/fineweb_train_*.bin"))
if not shards:
    print(f"No shards found in {data_dir}")
    sys.exit(1)

print(f"Found {len(shards)} training shards in {data_dir}")

table = NgramTable(vocab_size=1024, max_order=4, bucket_count=16384)
total_tokens = 0

for i, shard_path in enumerate(shards):
    tokens = np.fromfile(shard_path, dtype=np.uint16)
    print(f"  [{i+1}/{len(shards)}] {Path(shard_path).name}: {len(tokens):,} tokens")
    table.build_from_tokens(tokens)
    total_tokens += len(tokens)

print(f"\nTotal: {total_tokens:,} tokens ({total_tokens/1e9:.1f}B)")
print(f"Unigram non-zero: {(table.unigram > 0).sum()}")
print(f"Bigram non-zero: {(table.bigram > 0).sum()}")
print(f"Trigram non-zero: {(table.trigram > 0).sum()}")

Path(output).parent.mkdir(parents=True, exist_ok=True)
table.save(output)
size_mb = Path(output).stat().st_size / 1024 / 1024
print(f"Saved to {output} ({size_mb:.1f} MB)")
