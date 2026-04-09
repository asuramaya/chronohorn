from __future__ import annotations


def test_normalize_hosts_respects_explicit_empty_list():
    from chronohorn.fleet.hosts import normalize_hosts

    assert normalize_hosts([]) == []


def test_job_matches_manifest_by_path_or_basename():
    from chronohorn.fleet.hosts import _job_matches_manifest

    job = {"manifest": "/tmp/runs/sample.jsonl"}

    assert _job_matches_manifest(job, "/tmp/runs/sample.jsonl") is True
    assert _job_matches_manifest(job, "sample.jsonl") is True
    assert _job_matches_manifest(job, "other.jsonl") is False
