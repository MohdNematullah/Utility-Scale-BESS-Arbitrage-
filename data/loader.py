"""
loader.py
==========

Electricity market data loader for .

Purpose
-------
Loads hourly electricity market datasets into a clean,
continuous time-series DataFrame for forecasting,
battery ageing, and rolling-horizon optimisation.

Features
--------
- CSV and DataFrame loading
- UTC timestamp parsing
- Timezone conversion
- Column normalization
- Duplicate removal
- Hourly continuity enforcement
- Time interpolation for small gaps
- Dataset metadata
- Chronological train/test split
"""

from pathlib import Path
from dataclasses import dataclass

import pandas as pd


# ==========================================================
# Dataset Metadata
# ==========================================================

@dataclass(slots=True)
class DatasetMetadata:
    """Summary information about a loaded dataset."""

    rows: int
    start_timestamp: str
    end_timestamp: str
    timezone: str
    missing_hours: int
    duplicate_timestamps: int
    missing_prices: int


# ==========================================================
# Market Data Loader
# ==========================================================

class MarketDataLoader:
    """
    Loads and prepares hourly electricity price datasets.

    Output DataFrame:
        Index  : timestamp (timezone-aware DatetimeIndex)
        Column : price (float64)
    """

    def __init__(
        self,
        timestamp_column: str = "timestamp",
        price_column: str = "price",
        timezone: str = "UTC",
    ):
        self.timestamp_column = timestamp_column.lower()
        self.price_column = price_column.lower()
        self.timezone = timezone

    # ------------------------------------------------------
    # Load CSV
    # ------------------------------------------------------
    def load_csv(self, file_path: str | Path) -> pd.DataFrame:
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset not found: {file_path}")

        dataframe = pd.read_csv(file_path)

        return self.prepare_dataframe(dataframe)

    # ------------------------------------------------------
    # Load Existing DataFrame
    # ------------------------------------------------------
    def load_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        return self.prepare_dataframe(dataframe.copy())

    # ------------------------------------------------------
    # Prepare DataFrame
    # ------------------------------------------------------
    def prepare_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize dataset."""

        # Normalize column names
        dataframe.columns = [
            col.lower().strip().replace(" ", "_")
            for col in dataframe.columns
        ]

        # Required columns
        required = {self.timestamp_column, self.price_column}
        missing = required.difference(dataframe.columns)

        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        # Parse timestamps
        dataframe[self.timestamp_column] = pd.to_datetime(
            dataframe[self.timestamp_column],
            utc=True,
            errors="raise",
        ).dt.tz_convert(self.timezone)

        # Convert price column
        dataframe[self.price_column] = pd.to_numeric(
            dataframe[self.price_column],
            errors="coerce",
        )

        dataframe = dataframe.dropna(subset=[self.price_column])

        # Sort chronologically
        dataframe = dataframe.sort_values(self.timestamp_column)

        # Remove duplicate timestamps
        dataframe = dataframe.drop_duplicates(
            subset=self.timestamp_column,
            keep="first",
        )

        # Timestamp index
        dataframe = dataframe.set_index(self.timestamp_column)
        dataframe.index.name = "timestamp"

        # Ensure hourly continuity
        hourly_index = pd.date_range(
            start=dataframe.index.min(),
            end=dataframe.index.max(),
            freq="h",
            tz=self.timezone,
        )

        dataframe = dataframe.reindex(hourly_index)
        dataframe.index.name = "timestamp"

        # Fill small missing gaps
        dataframe[self.price_column] = (
            dataframe[self.price_column]
            .interpolate(method="time")
            .ffill()
            .bfill()
            .astype("float64")
        )

        return dataframe

    # ------------------------------------------------------
    # Dataset Metadata
    # ------------------------------------------------------
    def metadata(self, dataframe: pd.DataFrame) -> DatasetMetadata:
        expected = pd.date_range(
            start=dataframe.index.min(),
            end=dataframe.index.max(),
            freq="h",
            tz=self.timezone,
        )

        return DatasetMetadata(
            rows=int(len(dataframe)),
            start_timestamp=str(dataframe.index.min()),
            end_timestamp=str(dataframe.index.max()),
            timezone=self.timezone,
            missing_hours=int(
                len(expected.difference(dataframe.index))
            ),
            duplicate_timestamps=int(
                dataframe.index.duplicated().sum()
            ),
            missing_prices=int(
                dataframe[self.price_column].isna().sum()
            ),
        )

    # ------------------------------------------------------
    # Chronological Train/Test Split
    # ------------------------------------------------------
    def train_test_split(
        self,
        dataframe: pd.DataFrame,
        train_ratio: float = 0.80,
    ):
        """
        Chronological split (never random).

        Returns
        -------
        train_df, test_df
        """

        if not (0 < train_ratio < 1):
            raise ValueError("train_ratio must be between 0 and 1.")

        split = int(len(dataframe) * train_ratio)

        train = dataframe.iloc[:split].copy()
        test = dataframe.iloc[split:].copy()

        return train, test