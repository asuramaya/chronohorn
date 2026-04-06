"""Tests for engine/saturation.py — asymptote fitting and reliability flags."""
from __future__ import annotations


def test_analyze_saturation_basic():
    from chronohorn.engine.saturation import analyze_saturation

    probes = [
        {"step": 100, "bpb": 2.5},
        {"step": 200, "bpb": 2.3},
        {"step": 400, "bpb": 2.1},
        {"step": 800, "bpb": 1.95},
        {"step": 1600, "bpb": 1.85},
        {"step": 3200, "bpb": 1.80},
    ]
    result = analyze_saturation(probes)
    assert "asymptote" in result
    assert result["asymptote"] > 0
    assert result["asymptote"] < 2.5  # must be below starting bpb


def test_analyze_saturation_too_few_probes():
    from chronohorn.engine.saturation import analyze_saturation

    result = analyze_saturation([{"step": 100, "bpb": 2.5}])
    # With too few probes, should report insufficient_data
    assert result.get("status") == "insufficient_data" or result.get("reliable") is False


def test_analyze_saturation_flat_curve():
    from chronohorn.engine.saturation import analyze_saturation

    probes = [{"step": s, "bpb": 1.80} for s in range(100, 1100, 100)]
    result = analyze_saturation(probes)
    # Flat curve should have asymptote near 1.80
    if result.get("asymptote"):
        assert abs(result["asymptote"] - 1.80) < 0.1


def test_format_saturation_summary():
    from chronohorn.engine.saturation import analyze_saturation, format_saturation_summary

    probes = [
        {"step": s, "bpb": 2.5 - s * 0.0001}
        for s in range(100, 5100, 100)
    ]
    # format_saturation_summary takes the analysis dict, not raw probes
    analysis = analyze_saturation(probes)
    text = format_saturation_summary(analysis)
    assert isinstance(text, str)
    assert len(text) > 0
