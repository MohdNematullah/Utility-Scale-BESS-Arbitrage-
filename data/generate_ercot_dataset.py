"""
generate_ercot_dataset.py
=========================

Generates a realistic ERCOT-style hourly electricity price dataset
for the complete year 2025.

Purpose
-------
Used for developing and testing forecasting,
battery ageing, and rolling-horizon arbitrage models
when an official historical dataset is unavailable.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ==========================================================
# Configuration
# ==========================================================

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "ercot_prices.csv"

YEAR = 2025
SEED = 42

np.random.seed(SEED)


# ==========================================================
# Generate Hourly Timeline
# ==========================================================

timestamps = pd.date_range(
    start=f"{YEAR}-01-01 00:00:00",
    end=f"{YEAR}-12-31 23:00:00",
    freq="h",
    tz="UTC",
)

hours = timestamps.hour.to_numpy()
day_of_year = timestamps.dayofyear.to_numpy()
weekday = timestamps.dayofweek.to_numpy()


# ==========================================================
# ERCOT-style Market Price Components
# ==========================================================

BASE_PRICE = 35.0

daily_cycle = (
    10 * np.sin((hours - 14) * 2 * np.pi / 24)
)

seasonal_cycle = (
    6 * np.sin((day_of_year - 180) * 2 * np.pi / 365)
)

weekend_discount = np.where(
    weekday >= 5,
    -4,
    0,
)

market_noise = np.random.normal(
    loc=0,
    scale=2.5,
    size=len(timestamps),
)

price = (
    BASE_PRICE
    + daily_cycle
    + seasonal_cycle
    + weekend_discount
    + market_noise
).astype(float)


# ==========================================================
# Positive Price Spikes (High Demand Events)
# ==========================================================

positive_spikes = np.random.choice(
    len(price),
    size=260,
    replace=False,
)

price[positive_spikes] += np.random.uniform(
    30,
    120,
    size=len(positive_spikes),
)


# ==========================================================
# Negative Prices (Wind / Renewable Surplus)
# ==========================================================

negative_events = np.random.choice(
    len(price),
    size=120,
    replace=False,
)

price[negative_events] -= np.random.uniform(
    15,
    40,
    size=len(negative_events),
)

price = np.round(price, 2)


# ==========================================================
# Create Dataset
# ==========================================================

dataset = pd.DataFrame(
    {
        "timestamp": timestamps,
        "price": price,
    }
)

dataset.to_csv(OUTPUT_FILE, index=False)


# ==========================================================
# Dataset Summary
# ==========================================================

print("=" * 68)
print(" SYNTHETIC ERCOT DATASET GENERATED")
print("=" * 68)
print(f"Dataset Year           : {YEAR}")
print(f"Rows                   : {len(dataset)}")
print(f"Start Timestamp        : {dataset.timestamp.min()}")
print(f"End Timestamp          : {dataset.timestamp.max()}")
print(f"Minimum Price ($/MWh)  : {dataset.price.min():.2f}")
print(f"Maximum Price ($/MWh)  : {dataset.price.max():.2f}")
print(f"Average Price ($/MWh)  : {dataset.price.mean():.2f}")
print(f"Negative Price Hours   : {(dataset.price < 0).sum()}")
print(f"Output File            : {OUTPUT_FILE}")
print("=" * 68)

print("\nFirst Five Rows")
print(dataset.head())

print("\nLast Five Rows")
print(dataset.tail())