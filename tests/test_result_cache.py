from __future__ import annotations

import json
from pathlib import Path

from chronohorn.pipeline import _local_result_path, _try_local_result


def test_local_result_path():
    path = _local_result_path("ex-a-s12-mlp")
    assert path.name == "ex-a-s12-mlp.json"
    assert "out/results" in str(path)


def test_try_local_result_returns_payload(tmp_path: Path):
    local = tmp_path / "test-job.json"
    local.write_text(json.dumps({"model": {"test_bpb": 2.05}}))
    payload = _try_local_result("test-job", result_dir=tmp_path)
    assert payload is not None
    assert payload["model"]["test_bpb"] == 2.05


def test_try_local_result_returns_none_when_missing(tmp_path: Path):
    payload = _try_local_result("nonexistent", result_dir=tmp_path)
    assert payload is None
