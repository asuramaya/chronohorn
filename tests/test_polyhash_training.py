from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from chronohorn.families.polyhash.training.train_polyhash import ShardedDataset, load_val


def test_sharded_dataset_filters_by_configured_vocab_size(tmp_path):
    tokens = np.array([0, 7, 15, 16, 31], dtype=np.uint16)
    (tmp_path / "fineweb_train_000000.bin").write_bytes(tokens.tobytes())
    (tmp_path / "fineweb_val_000000.bin").write_bytes(tokens.tobytes())

    dataset = ShardedDataset(str(tmp_path), sl=2, vocab_size=16)
    val = load_val(str(tmp_path), vocab_size=16)

    assert dataset._s is not None
    assert dataset._s.tolist() == [0, 7, 15]
    assert val.tolist() == [0, 7, 15]
