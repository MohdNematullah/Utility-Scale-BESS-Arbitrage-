
"""
engineering.py
==============

feature engineering for electricity price forecasting.

 ---------------

Creates leakage-free features for:
- Multi-step forecasting (24h / 48h)
- Rolling-horizon optimisation
- Battery arbitrage experiments

Feature Groups
--------------
1. Calendar Features
2. Cyclical Time Features
3. Lag Features
4. Rolling Statistics
5. Price Dynamics
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd


# ==========================================================
# Configuration
# ==========================================================

@dataclass(slots=True)
class FeatureConfig:
    lag_hours: tuple = (1, 2, 3, 6, 12, 24, 48, 72, 168)
    rolling_windows: tuple = (3, 6, 12, 24, 48, 168)
    target_column: str = "price"


# ==========================================================
# Feature Engineering Class
# ==========================================================

class FeatureEngineer:

    def __init__(self, config: FeatureConfig = FeatureConfig()):
        self.config = config

    # ------------------------------------------------------
    # Main Pipeline
    # ------------------------------------------------------
    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Create complete feature matrix.

        Input
        -----
        Index must be timestamp.

        Returns
        -------
        Feature dataframe.
        """

        df = dataframe.copy()

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                "DataFrame index must be DatetimeIndex."
            )

        # Calendar
        df = self.add_calendar_features(df)

        # Cyclical encoding
        df = self.add_cyclical_features(df)

        # Lag features
        df = self.add_lag_features(df)

        # Rolling statistics
        df = self.add_rolling_features(df)

        # Price dynamics
        df = self.add_price_dynamics(df)

        # Remove rows containing NaN caused by lags.
        df = df.dropna()

        return df

    # ------------------------------------------------------
    # Calendar Features
    # ------------------------------------------------------
    def add_calendar_features(self, df):

        index = df.index

        df["hour"] = index.hour
        df["day_of_week"] = index.dayofweek
        df["day_of_month"] = index.day
        df["day_of_year"] = index.dayofyear
        df["week_of_year"] = index.isocalendar().week.astype(int)
        df["month"] = index.month
        df["quarter"] = index.quarter
        df["is_weekend"] = (index.dayofweek >= 5).astype(int)

        return df

    # ------------------------------------------------------
    # Cyclical Features
    # ------------------------------------------------------
    def add_cyclical_features(self, df):

        df["hour_sin"] = np.sin(
            2 * np.pi * df["hour"] / 24
        )
        df["hour_cos"] = np.cos(
            2 * np.pi * df["hour"] / 24
        )

        df["weekday_sin"] = np.sin(
            2 * np.pi * df["day_of_week"] / 7
        )
        df["weekday_cos"] = np.cos(
            2 * np.pi * df["day_of_week"] / 7
        )

        df["month_sin"] = np.sin(
            2 * np.pi * df["month"] / 12
        )
        df["month_cos"] = np.cos(
            2 * np.pi * df["month"] / 12
        )

        return df

    # ------------------------------------------------------
    # Lag Features
    # ------------------------------------------------------
    def add_lag_features(self, df):

        target = self.config.target_column

        for lag in self.config.lag_hours:
            df[f"{target}_lag_{lag}"] = (
                df[target].shift(lag)
            )

        return df

    # ------------------------------------------------------
    # Rolling Statistics
    # ------------------------------------------------------
    def add_rolling_features(self, df):

        target = self.config.target_column

        for window in self.config.rolling_windows:

            shifted = df[target].shift(1)

            rolling = shifted.rolling(window)

            df[f"{target}_mean_{window}"] = rolling.mean()
            df[f"{target}_std_{window}"] = rolling.std()
            df[f"{target}_min_{window}"] = rolling.min()
            df[f"{target}_max_{window}"] = rolling.max()

        return df

    # ------------------------------------------------------
    # Price Dynamics
    # ------------------------------------------------------
    def add_price_dynamics(self, df):

        target = self.config.target_column

        df["price_change_1h"] = (
            df[target].shift(1)
            - df[target].shift(2)
        )

        df["price_change_24h"] = (
            df[target].shift(1)
            - df[target].shift(25)
        )

        df["price_momentum_6h"] = (
            df[target].shift(1)
            - df[target].shift(7)
        )

        df["price_volatility_24h"] = (
            df[target]
            .shift(1)
            .rolling(24)
            .std()
        )

        return df

    # ------------------------------------------------------
    # Feature Names
    # ------------------------------------------------------
    def feature_columns(self, dataframe):

        target = self.config.target_column

        return [
            col
            for col in dataframe.columns
            if col != target
        ]