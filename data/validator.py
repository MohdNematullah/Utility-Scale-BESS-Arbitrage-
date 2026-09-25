"""
validator.py

Validates hourly electricity price datasets before forecasting.
"""

from dataclasses import dataclass
import pandas as pd


@dataclass(slots=True)
class ValidationReport:
    rows: int
    missing_values: int
    duplicate_timestamps: int
    negative_prices: int
    start_timestamp: str
    end_timestamp: str
    passed: bool


class MarketDataValidator:

    REQUIRED_COLUMNS = ["timestamp", "price"]

    def validate(self, dataframe: pd.DataFrame) -> ValidationReport:

        df = dataframe.copy()

        # ---------- Required Columns ----------
        missing_columns = [
            col for col in self.REQUIRED_COLUMNS
            if col not in df.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Missing columns: {missing_columns}"
            )

        # ---------- Timestamp ----------
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        duplicates = df["timestamp"].duplicated().sum()

        missing = df["price"].isna().sum()

        negatives = (df["price"] < 0).sum()

        # ---------- Hourly Continuity ----------
        expected = pd.date_range(
            start=df["timestamp"].min(),
            end=df["timestamp"].max(),
            freq="h",
            tz="UTC",
        )

        if len(expected) != len(df):
            raise ValueError(
                "Dataset has missing hourly timestamps."
            )

        # Convert NumPy boolean to native Python bool
        passed = bool(
            duplicates == 0
            and missing == 0
        )

        return ValidationReport(
            rows=len(df),
            missing_values=int(missing),
            duplicate_timestamps=int(duplicates),
            negative_prices=int(negatives),
            start_timestamp=str(df["timestamp"].min()),
            end_timestamp=str(df["timestamp"].max()),
            passed=bool(passed),
        )