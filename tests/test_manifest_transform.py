from __future__ import annotations

from chronohorn.fleet.manifest_transform import filter_manifest, mutate_manifest


def test_filter_by_name_glob():
    rows = [
        {"name": "ex-a-s12-mlp", "steps": 1000},
        {"name": "ex-a-s15-mlp", "steps": 1000},
        {"name": "ex-b-mincorr", "steps": 1000},
        {"name": "ex-j-s17-allknobs", "steps": 1000},
    ]
    result = filter_manifest(rows, name_pattern="ex-a-*")
    assert len(result) == 2
    assert all(r["name"].startswith("ex-a-") for r in result)


def test_filter_by_name_glob_j_group():
    rows = [
        {"name": "ex-a-s12-mlp", "steps": 1000},
        {"name": "ex-j-s17-allknobs", "steps": 1000},
        {"name": "ex-j-s12-e2-mincorr", "steps": 1000},
    ]
    result = filter_manifest(rows, name_pattern="ex-j-*")
    assert len(result) == 2


def test_mutate_steps():
    rows = [{"name": "job1", "steps": 1000, "seq_len": 256, "batch_size": 16,
             "command": "--steps 1000 --json /run/results/job1.json"}]
    result = mutate_manifest(rows, steps=5200)
    assert result[0]["steps"] == 5200
    assert "--steps 5200" in result[0]["command"]


def test_mutate_seed():
    rows = [{"name": "job1", "seed": 42,
             "command": "--seed 42 --json /run/results/job1.json"}]
    result = mutate_manifest(rows, seed=43)
    assert result[0]["seed"] == 43
    assert "--seed 43" in result[0]["command"]
    assert result[0]["name"] == "job1-seed43"
