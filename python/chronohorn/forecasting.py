"""Compatibility wrapper for the promoted engine forecasting surface."""

from __future__ import annotations

from chronohorn.engine.forecasting import (
    FORECAST_VERSION,
    MAX_LOG_EXTRAPOLATION_MULTIPLE,
    MAX_OBSERVED_IMPROVEMENT_MULTIPLE,
    build_result_forecast,
)
from chronohorn.engine.budgets import (
    DEFAULT_COMPETITION_BUDGETS,
    DEFAULT_GOLF_V1_BUDGET,
    resolve_competition_budget,
)
from chronohorn.engine.results import load_result_json

__all__ = [
    "FORECAST_VERSION",
    "MAX_LOG_EXTRAPOLATION_MULTIPLE",
    "MAX_OBSERVED_IMPROVEMENT_MULTIPLE",
    "DEFAULT_COMPETITION_BUDGETS",
    "DEFAULT_GOLF_V1_BUDGET",
    "build_result_forecast",
    "load_result_json",
    "resolve_competition_budget",
]
