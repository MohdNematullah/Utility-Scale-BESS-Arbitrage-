"""
forecasting package

 Forecasting Module

Provides forecasting models, recursive forecasting,
training pipeline, and evaluation metrics.
"""

from .baselines import (
    PersistenceForecaster,
    SeasonalNaiveForecaster,
    MovingAverageForecaster,
)

from .models import (
    XGBoostForecaster,
    RandomForestForecaster,
)

__all__ = [
    "PersistenceForecaster",
    "SeasonalNaiveForecaster",
    "MovingAverageForecaster",
    "XGBoostForecaster",
    "RandomForestForecaster",
]