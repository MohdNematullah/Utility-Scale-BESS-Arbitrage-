
"""
recursive_engine.py
===================

Recursive Multi-Step Forecasting Engine
for .

Implements recursive 24h / 48h forecasting without
future-data leakage.

Compatible with:
    â€¢ features/engineering.py
    â€¢ forecasting/models.py
    â€¢ forecasting/trainer.py
    â€¢ backtesting/rolling_runner.py
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from features.engineering import FeatureEngineer


# ==========================================================
# Forecast Result
# ==========================================================

@dataclass(slots=True)
class ForecastHorizon:
    model_name: str
    horizon: int
    dataframe: pd.DataFrame


# ==========================================================
# Recursive Forecast Engine
# ==========================================================

class RecursiveForecastEngine:

    def __init__(
        self,
        model,
        timezone: str = "UTC",
    ):
        self.model = model
        self.engineer = FeatureEngineer()
        self.timezone = timezone

    # ------------------------------------------------------
    # Calendar Features
    # ------------------------------------------------------
    def calendar_features(
        self,
        timestamp: pd.Timestamp,
    ) -> dict:

        return {
            "hour": timestamp.hour,
            "day_of_week": timestamp.dayofweek,
            "day_of_month": timestamp.day,
            "day_of_year": timestamp.dayofyear,
            "week_of_year": timestamp.isocalendar().week,
            "month": timestamp.month,
            "quarter": timestamp.quarter,
            "is_weekend": int(timestamp.dayofweek >= 5),

            "hour_sin":
                np.sin(2 * np.pi * timestamp.hour / 24),

            "hour_cos":
                np.cos(2 * np.pi * timestamp.hour / 24),

            "weekday_sin":
                np.sin(2 * np.pi * timestamp.dayofweek / 7),

            "weekday_cos":
                np.cos(2 * np.pi * timestamp.dayofweek / 7),

            "month_sin":
                np.sin(2 * np.pi * timestamp.month / 12),

            "month_cos":
                np.cos(2 * np.pi * timestamp.month / 12),
        }

    # ------------------------------------------------------
    # Create Next Timestamp
    # ------------------------------------------------------
    def next_timestamp(
        self,
        current_timestamp: pd.Timestamp,
    ) -> pd.Timestamp:

        return current_timestamp + pd.Timedelta(hours=1)

    # ------------------------------------------------------
    # Create Future Calendar Index
    # ------------------------------------------------------
    def future_index(
        self,
        start_timestamp: pd.Timestamp,
        horizon: int,
    ):

        return pd.date_range(
            start=start_timestamp + pd.Timedelta(hours=1),
            periods=horizon,
            freq="h",
            tz=self.timezone,
        )

    # ------------------------------------------------------
    # Build Recursive Feature Row
    # ------------------------------------------------------
    def build_recursive_row(
        self,
        history: pd.DataFrame,
        timestamp: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Create one feature row using ONLY information
        available before timestamp.
        """

        price_history = history["price"]

        row = self.calendar_features(timestamp)

        # -------------------------------
        # Lag Features
        # -------------------------------
        lag_list = [
            1, 2, 3, 6, 12,
            24, 48, 72, 96, 168,
        ]

        history_length = len(price_history)

        for lag in lag_list:

            if history_length >= lag:
                row[f"price_lag_{lag}"] = float(price_history.iloc[-lag])

            else:
                # Earliest available value avoids IndexError.
                row[f"price_lag_{lag}"] = float(price_history.iloc[0])

        # --------------------------------------------------
        # Rolling window statistics
        # MUST match FeatureEngineer.transform()
        # --------------------------------------------------
        rolling_windows = [
            3,
            6,
            12,
            24,
            48,
            72,
            168,
        ]

        for window in rolling_windows:
            values = price_history.tail(min(window, len(price_history)))

            row[f"price_mean_{window}"] = values.mean()
            row[f"price_std_{window}"] = values.std(ddof=0)
            row[f"price_min_{window}"] = values.min()
            row[f"price_max_{window}"] = values.max()
        # -------------------------------
        # Price Change Features
        # -------------------------------
        last_price = float(price_history.iloc[-1])

        prev_1h = float(price_history.iloc[-2]) if len(price_history) >= 2 else last_price
        prev_6h = float(price_history.iloc[-6]) if len(price_history) >= 6 else last_price
        prev_24h = float(price_history.iloc[-24]) if len(price_history) >= 24 else last_price

        row["price_change_1h"] = last_price - prev_1h
        row["price_change_24h"] = last_price - prev_24h
        row["price_momentum_6h"] = last_price - prev_6h

        window24 = price_history.tail(min(24, len(price_history)))
        row["price_volatility_24h"] = float(window24.std(ddof=0))

        dataframe = pd.DataFrame(
            row,
            index=[timestamp],
        )

        dataframe.index.name = "timestamp"

        return dataframe

    # ------------------------------------------------------
    # Update History with Prediction
    # ------------------------------------------------------
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

        history = pd.concat([history, new_row])

        return history

    # ------------------------------------------------------
    # Initial History Buffer
    # ------------------------------------------------------
    def initialize_history(
        self,
        feature_dataframe: pd.DataFrame,
        window: int = 168,
    ) -> pd.DataFrame:
        """
        Keep only historical prices needed
        for recursive feature generation.
        """

        history = feature_dataframe[
            ["price"]
        ].tail(window).copy()

        history.index.name = "timestamp"

        return history

    # ------------------------------------------------------
    # Validate History
    # ------------------------------------------------------
    def validate_history(
        self,
        history: pd.DataFrame,
        minimum_window: int = 168,
    ):

        if len(history) < minimum_window:
            raise ValueError(
                "History buffer too small for recursive forecasting."
            )

        if history["price"].isna().any():
            raise ValueError(
                "History contains missing prices."
            )

    # ------------------------------------------------------
    # Prediction Wrapper
    # ------------------------------------------------------
    def predict_one_step(
            self,
            feature_row: pd.DataFrame,
    ) -> float:
        """
        Predict one hour ahead.

        Ensures recursive features are aligned with the
        features used during model training.
        """

        # Feature names learned during training
        expected_features = self.model.features

        # Add any missing columns with zeros
        for column in expected_features:
            if column not in feature_row.columns:
                feature_row[column] = 0.0

        # Remove unexpected columns
        feature_row = feature_row[expected_features]

        prediction = self.model.predict(feature_row).prediction

        return float(prediction[0])

    def recursive_forecast(
            self,
            feature_dataframe,
            horizon: int = 24,
    ):
        """
        Generates a recursive multi-step forecast.

        Parameters
        ----------
        feature_dataframe : pd.DataFrame
            Engineered feature dataframe including historical prices.

        horizon : int
            Forecast horizon in hours.

        Returns
        -------
        ForecastHorizon
        """

        import pandas as pd

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

            # append predicted price into history
            new_row = feature_row.copy()
            new_row.index = [next_time]
            new_row["price"] = prediction

            current_history = pd.concat(
                [current_history, new_row],
                axis=0,
            )

        forecast_df = pd.DataFrame(predictions)

        return ForecastHorizon(
            model_name=self.model.__class__.__name__,
            horizon=horizon,
            dataframe=forecast_df,
        )

    def forecast(self, feature_dataframe, horizon=24):
        """
        Compatibility wrapper for older modules.
        Returns only the forecast dataframe.
        """
        return self.recursive_forecast(
            feature_dataframe=feature_dataframe,
            horizon=horizon,
        ).dataframe

    def forecast_dataframe(self, feature_dataframe, horizon=24):
        """
        Compatibility wrapper used by optimization module.
        """
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
        """
        Generate recursive forecast and align it with
        actual future prices using timestamps.
        """

        forecast = self.recursive_forecast(
            train_features,
            horizon=horizon,
        )

        # --------------------------------------------
        # Prepare actual observations
        # --------------------------------------------
        actual = (
            actual_future[["price"]]
            .head(horizon)
            .reset_index()
            .rename(columns={"price": "actual"})
        )

        # --------------------------------------------
        # Merge on timestamp
        # --------------------------------------------
        merged = forecast.dataframe.merge(
            actual,
            on="timestamp",
            how="left",
            validate="one_to_one",
        )

        # Keep only rows with actual prices
        merged = merged.dropna(subset=["actual"]).copy()

        merged["error"] = (
                merged["prediction"] - merged["actual"]
        )

        merged["absolute_error"] = (
            merged["error"].abs()
        )

        forecast.dataframe = merged

        return forecast

    def multi_horizon_forecast(
            self,
            feature_dataframe: pd.DataFrame,
            horizons=(24, 48),
    ):
        """
        Produce forecasts for multiple horizons.
        """

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