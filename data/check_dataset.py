"""
check_dataset.py

Dataset diagnostics for .

Purpose
-------
Checks the ERCOT hourly dataset before forecasting
and rolling-horizon optimisation.
"""

from pathlib import Path
import pandas as pd

DATASET = Path("data/raw/ercot_prices.csv")

if not DATASET.exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET}")

# Load dataset
df = pd.read_csv(DATASET)

df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

# Sort chronologically
df = df.sort_values("timestamp").reset_index(drop=True)

# Basic diagnostics
duplicates = df["timestamp"].duplicated().sum()
missing_prices = df["price"].isna().sum()
negative_prices = (df["price"] < 0).sum()

expected_hours = pd.date_range(
    start=df["timestamp"].min(),
    end=df["timestamp"].max(),
    freq="h",
    tz="UTC",
)

missing_hours = len(expected_hours.difference(df["timestamp"]))

# Summary
print("=" * 65)
print(" DATASET QUALITY REPORT")
print("=" * 65)
print(f"Rows                  : {len(df)}")
print(f"Start                 : {df.timestamp.min()}")
print(f"End                   : {df.timestamp.max()}")
print(f"Missing Hours         : {missing_hours}")
print(f"Duplicate Timestamps  : {duplicates}")
print(f"Missing Prices        : {missing_prices}")
print(f"Negative Prices       : {negative_prices}")
print("-" * 65)
print(f"Minimum Price ($)     : {df.price.min():8.2f}")
print(f"Maximum Price ($)     : {df.price.max():8.2f}")
print(f"Average Price ($)     : {df.price.mean():8.2f}")
print(f"Median Price ($)      : {df.price.median():8.2f}")
print(f"Std. Deviation ($)    : {df.price.std():8.2f}")
print("=" * 65)

# Percentiles
print("\nPrice Percentiles")
print(df["price"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]))

# Monthly statistics
monthly = (
    df.assign(month=df["timestamp"].dt.month_name())
      .groupby("month")["price"]
      .agg(["mean", "max", "min", "std"])
)

print("\nMonthly Price Statistics")
print(monthly.round(2))

# Save monthly summary for tables
monthly.to_csv("data/raw/monthly_price_statistics.csv")

print("\nSaved monthly statistics:")
print("data/raw/monthly_price_statistics.csv")