
"""
baselines.py
============

Benchmark forecasting models for .

These models provide reference performance
before using machine learning models.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd


# ==========================================================
# Base Forecaster
# ==========================================================

@dataclass(slots=True)
class ForecastResult:
    model_name: str
    horizon: int
    prediction: np.ndarray


class BaseForecaster:

    def fit(self, series: pd.Series):
        self.series = series.astype(float)
        return self

    def predict(self, horizon: int):
        raise NotImplementedError


# ==========================================================
# Persistence Forecast
# ==========================================================

class PersistenceForecaster(BaseForecaster):
    """
    Predict future values equal to the latest observed price.
    """

    def predict(self, horizon: int = 24):

        value = float(self.series.iloc[-1])

        prediction = np.repeat(value, horizon)

        return ForecastResult(
            model_name="Persistence",
            horizon=horizon,
            prediction=prediction,
        )


# ==========================================================
# Seasonal Naive Forecast
# ==========================================================

class SeasonalNaiveForecaster(BaseForecaster):
    """
    Forecast using values from previous day.
    """

    def predict(
        self,
        horizon: int = 24,
        season_length: int = 24,
    ):

        history = self.series.iloc[-season_length:]

        repeats = int(np.ceil(horizon / season_length))

        prediction = np.tile(history.values, repeats)[:horizon]

        return ForecastResult(
            model_name="SeasonalNaive",
            horizon=horizon,
            prediction=prediction,
        )


# ==========================================================
# Moving Average Forecast
# ==========================================================

class MovingAverageForecaster(BaseForecaster):
    """
    Forecast using recent rolling average.
    """

    def predict(
        self,
        horizon: int = 24,
        window: int = 24,
    ):

        avg = self.series.tail(window).mean()

        prediction = np.repeat(avg, horizon)

        return ForecastResult(
            model_name="MovingAverage",
            horizon=horizon,
            prediction=prediction,
        )