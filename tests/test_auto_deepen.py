from __future__ import annotations

from chronohorn.fleet.auto_deepen import next_step_target, should_deepen


def test_should_deepen_alive_slope():
    probes = [{"step": 400, "bpb": 2.2}, {"step": 800, "bpb": 2.1}, {"step": 1000, "bpb": 2.05}]
    assert should_deepen(probes, current_steps=1000, max_steps=10000) is True

def test_should_not_deepen_flat_slope():
    probes = [{"step": 400, "bpb": 2.05}, {"step": 800, "bpb": 2.05}, {"step": 1000, "bpb": 2.05}]
    assert should_deepen(probes, current_steps=1000, max_steps=10000) is False

def test_should_not_deepen_at_max():
    probes = [{"step": 8000, "bpb": 1.90}, {"step": 10000, "bpb": 1.85}]
    assert should_deepen(probes, current_steps=10000, max_steps=10000) is False

def test_should_not_deepen_insufficient_probes():
    probes = [{"step": 1000, "bpb": 2.05}]
    assert should_deepen(probes, current_steps=1000, max_steps=10000) is False

def test_next_step_target_progression():
    assert next_step_target(1000) == 5000
    assert next_step_target(5000) == 10000
    assert next_step_target(10000) == 10000  # already at max
    assert next_step_target(500) == 1000
    assert next_step_target(2000) == 5000
