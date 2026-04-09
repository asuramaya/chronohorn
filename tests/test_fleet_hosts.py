from __future__ import annotations


def test_normalize_hosts_respects_explicit_empty_list():
    from chronohorn.fleet.hosts import normalize_hosts

    assert normalize_hosts([]) == []
