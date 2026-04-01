#!/usr/bin/env python3
"""Build an n-gram table from training shards."""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
from chronohorn.models.ngram_table import NgramTable

data_path = sys.argv[1] if len(sys.argv) > 1 else "data/roots/fineweb10B_sp1024/fineweb_train_000000.bin"
output = sys.argv[2] if len(sys.argv) > 2 else "out/ngram_table_4gram.npz"

print(f"Loading {data_path}...")
tokens = np.fromfile(data_path, dtype=np.uint16)
print(f"Loaded {len(tokens):,} tokens")

table = NgramTable(vocab_size=1024, max_order=4, bucket_count=8192)
print("Building table...")
table.build_from_tokens(tokens)
print(f"Unigram non-zero: {(table.unigram > 0).sum()}")
print(f"Bigram non-zero: {(table.bigram > 0).sum()}")
print(f"Trigram non-zero: {(table.trigram > 0).sum()}")

Path(output).parent.mkdir(parents=True, exist_ok=True)
table.save(output)
size_mb = Path(output).stat().st_size / 1024 / 1024
print(f"Saved to {output} ({size_mb:.1f} MB)")
