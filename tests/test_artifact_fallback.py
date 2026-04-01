from __future__ import annotations

from chronohorn.fleet.forecast_results import _resolve_artifact_viable


def test_resolve_uses_forecast_when_available():
    assert _resolve_artifact_viable(
        forecast_artifact_viable=True,
        manifest_artifact_mb_est=None,
        artifact_limit_mb=16.0,
    ) is True


def test_resolve_uses_forecast_false():
    assert _resolve_artifact_viable(
        forecast_artifact_viable=False,
        manifest_artifact_mb_est=10.0,
        artifact_limit_mb=16.0,
    ) is False


def test_resolve_falls_back_to_manifest_under():
    assert _resolve_artifact_viable(
        forecast_artifact_viable=None,
        manifest_artifact_mb_est=10.0,
        artifact_limit_mb=16.0,
    ) is True


def test_resolve_falls_back_to_manifest_over():
    assert _resolve_artifact_viable(
        forecast_artifact_viable=None,
        manifest_artifact_mb_est=20.0,
        artifact_limit_mb=16.0,
    ) is False


def test_resolve_unknown_when_no_data():
    assert _resolve_artifact_viable(
        forecast_artifact_viable=None,
        manifest_artifact_mb_est=None,
        artifact_limit_mb=16.0,
    ) is None
