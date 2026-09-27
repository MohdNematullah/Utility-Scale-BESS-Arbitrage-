"""
forecasting/recursive_engine.py
===============================

Recursive Multi-Step Forecasting Engine

Implements recursive 24h / 48h forecasting without future-data leakage
by directly reusing FeatureEngineer rules to guarantee exact alignment
between training and inference features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from features.engineering import FeatureEngineer


@dataclass(slots=True)
class ForecastHorizon:
    model_name: str
    horizon: int
    dataframe: pd.DataFrame


class RecursiveForecastEngine:
    def __init__(
        self,
        model,
        timezone: str = "UTC",
    ):
        self.model = model
        self.engineer = FeatureEngineer()
        self.timezone = timezone

    def calendar_features(
        self,
        timestamp: pd.Timestamp,
    ) -> dict:
        return {
            "hour": timestamp.hour,
            "day_of_week": timestamp.dayofweek,
            "day_of_month": timestamp.day,
            "day_of_year": timestamp.dayofyear,
            "week_of_year": int(timestamp.isocalendar().week),
            "month": timestamp.month,
            "quarter": timestamp.quarter,
            "is_weekend": int(timestamp.dayofweek >= 5),
            "hour_sin": np.sin(2 * np.pi * timestamp.hour / 24),
            "hour_cos": np.cos(2 * np.pi * timestamp.hour / 24),
            "weekday_sin": np.sin(2 * np.pi * timestamp.dayofweek / 7),
            "weekday_cos": np.cos(2 * np.pi * timestamp.dayofweek / 7),
            "month_sin": np.sin(2 * np.pi * timestamp.month / 12),
            "month_cos": np.cos(2 * np.pi * timestamp.month / 12),
        }

    def next_timestamp(
        self,
        current_timestamp: pd.Timestamp,
    ) -> pd.Timestamp:
        return current_timestamp + pd.Timedelta(hours=1)

    def future_index(
        self,
        start_timestamp: pd.Timestamp,
        horizon: int,
    ) -> pd.DatetimeIndex:
        return pd.date_range(
            start=start_timestamp + pd.Timedelta(hours=1),
            periods=horizon,
            freq="h",
            tz=self.timezone,
        )

    def build_recursive_row(
        self,
        history: pd.DataFrame,
        timestamp: pd.Timestamp,
    ) -> pd.DataFrame:
        """Use the training feature rules for the next forecast hour."""
        if not isinstance(history.index, pd.DatetimeIndex):
            raise ValueError("History must use a DatetimeIndex.")

        # Keep only the prices available before this forecast hour
        prices = history[["price"]].copy().sort_index()
        self.validate_history(prices)

        next_time = self.next_timestamp(prices.index[-1])
        if timestamp != next_time:
            raise ValueError(
                "Forecast timestamp must follow the history by one hour."
            )

        # Add the forecast hour without supplying its unknown price
        next_row = pd.DataFrame(
            {"price": [np.nan]},
            index=pd.DatetimeIndex([timestamp]),
        )
        frame = pd.concat([prices, next_row])
        frame.index.name = "timestamp"

        # Apply the exact feature calculations used during training
        frame = self.engineer.add_calendar_features(frame)
        frame = self.engineer.add_cyclical_features(frame)
        frame = self.engineer.add_lag_features(frame)
        frame = self.engineer.add_rolling_features(frame)
        frame = self.engineer.add_price_dynamics(frame)

        # Return only the inputs for the forecast hour
        row = frame.iloc[[-1]].drop(columns=["price"])
        if not np.isfinite(row.to_numpy(dtype=float)).all():
            raise ValueError("Forecast features contain invalid values.")

        return row

    def append_prediction(
        self,
        history: pd.DataFrame,
        timestamp: pd.Timestamp,
        prediction: float,
    ) -> pd.DataFrame:
        new_row = pd.DataFrame(
            {"price": prediction},
            index=[timestamp],
        )
        new_row.index.name = "timestamp"
        return pd.concat([history, new_row])

    def initialize_history(
        self,
        feature_dataframe: pd.DataFrame,
        window: int = 168,
    ) -> pd.DataFrame:
        """Keep only historical prices needed for recursive feature generation."""
        history = feature_dataframe[["price"]].tail(window).copy()
        history.index.name = "timestamp"
        return history

    def validate_history(
        self,
        history: pd.DataFrame,
        minimum_window: int = 168,
    ) -> None:
        if len(history) < minimum_window:
            raise ValueError(
                "History buffer too small for recursive forecasting."
            )
        if history["price"].isna().any():
            raise ValueError(
                "History contains missing prices."
            )

    def predict_one_step(
        self,
        feature_row: pd.DataFrame,
    ) -> float:
        """
        Predict one hour ahead.
        Ensures recursive features are aligned with features used during model training.
        """
        expected_features = self.model.features

        # Align columns
        for column in expected_features:
            if column not in feature_row.columns:
                feature_row[column] = 0.0

        feature_row = feature_row[expected_features]
        prediction = self.model.predict(feature_row).prediction

        return float(prediction[0])

    def recursive_forecast(
        self,
        feature_dataframe: pd.DataFrame,
        horizon: int = 24,
    ) -> ForecastHorizon:
        history = feature_dataframe.copy()
        if "price" not in history.columns:
            raise ValueError("Feature dataframe must contain 'price' column.")

        history = history.sort_index()
        predictions = []
        current_history = history.copy()

        for step in range(horizon):
            next_time = self.next_timestamp(current_history.index.max())

            feature_row = self.build_recursive_row(
                current_history,
                next_time,
            )

            prediction = float(self.predict_one_step(feature_row))

            predictions.append(
                {
                    "timestamp": next_time,
                    "prediction": prediction,
                    "step": step + 1,
                    "model": self.model.__class__.__name__,
                }
            )

            # Append predicted price into history buffer
            new_row = pd.DataFrame(
                {"price": prediction},
                index=[next_time],
            )
            new_row.index.name = "timestamp"

            current_history = pd.concat([current_history, new_row], axis=0)

        forecast_df = pd.DataFrame(predictions)

        return ForecastHorizon(
            model_name=self.model.__class__.__name__,
            horizon=horizon,
            dataframe=forecast_df,
        )

    def forecast(
        self,
        feature_dataframe: pd.DataFrame,
        horizon: int = 24,
    ) -> pd.DataFrame:
        """Compatibility wrapper returning only the forecast dataframe."""
        return self.recursive_forecast(
            feature_dataframe=feature_dataframe,
            horizon=horizon,
        ).dataframe

    def forecast_dataframe(
        self,
        feature_dataframe: pd.DataFrame,
        horizon: int = 24,
    ) -> pd.DataFrame:
        """Compatibility wrapper used by optimization modules."""
        return self.forecast(
            feature_dataframe=feature_dataframe,
            horizon=horizon,
        )

    def recursive_forecast_with_actuals(
        self,
        train_features: pd.DataFrame,
        actual_future: pd.DataFrame,
        horizon: int = 24,
    ) -> ForecastHorizon:
        forecast = self.recursive_forecast(
            train_features,
            horizon=horizon,
        )

        actual = (
            actual_future[["price"]]
            .head(horizon)
            .reset_index()
            .rename(columns={"price": "actual"})
        )

        merged = forecast.dataframe.merge(
            actual,
            on="timestamp",
            how="left",
            validate="one_to_one",
        )

        merged = merged.dropna(subset=["actual"]).copy()
        merged["error"] = merged["prediction"] - merged["actual"]
        merged["absolute_error"] = merged["error"].abs()

        forecast.dataframe = merged
        return forecast

    def multi_horizon_forecast(
        self,
        feature_dataframe: pd.DataFrame,
        horizons: tuple[int, ...] = (24, 48),
    ) -> dict[int, ForecastHorizon]:
        forecasts = {}
        for horizon in horizons:
            forecasts[horizon] = self.recursive_forecast(
                feature_dataframe,
                horizon=horizon,
            )
        return forecasts

    def forecast_summary(self, forecast: ForecastHorizon) -> dict:
        df = forecast.dataframe
        return {
            "model": forecast.model_name,
            "horizon": forecast.horizon,
            "rows": len(df),
            "forecast_start": str(df.timestamp.min()),
            "forecast_end": str(df.timestamp.max()),
            "minimum_prediction": round(df.prediction.min(), 3),
            "maximum_prediction": round(df.prediction.max(), 3),
            "average_prediction": round(df.prediction.mean(), 3),
        }