"""
backtesting/check_engine.py

Verification script for Rolling Horizon Backtesting Engine.
"""
import pandas as pd
from pathlib import Path

from data.loader import MarketDataLoader
from features.engineering import FeatureEngineer
from forecasting.models import XGBoostForecaster

from backtesting.engine import RollingBacktestEngine

print("=" * 70)
print("ROLLING HORIZON BACKTEST CHECK")
print("=" * 70)

loader = MarketDataLoader()
engineer = FeatureEngineer()

market = loader.load_csv("data/raw/ercot_prices.csv")
features = engineer.transform(market)

# Ensure recursive forecasting uses timestamp index.
if "timestamp" in features.columns:
    features["timestamp"] = pd.to_datetime(
        features["timestamp"],
        utc=True,
    )
    features = (
        features
        .sort_values("timestamp")
        .set_index("timestamp")
    )

features.index = pd.DatetimeIndex(features.index)

model_path = Path("forecasting/saved_models/xgboost_dayahead.pkl")

if not model_path.exists():
    raise FileNotFoundError(
        "Run python -m forecasting.check_trainer first."
    )

model = XGBoostForecaster()
model.load(model_path)

print("Forecast model loaded ✓")

engine = RollingBacktestEngine(
    forecaster=model,
    feature_dataframe=features,
)

result = engine.run()

print("Rolling horizon completed ✓")

print("-" * 70)
print("SUMMARY")
print("-" * 70)

for key, value in result.summary.items():
    print(f"{key:<25}: {value}")

print("-" * 70)
print("Dispatch Preview")
print(result.dispatch_history.head())

print("-" * 70)
print("Degradation Preview")
print(result.degradation_history.head())

files = engine.export(result)

print("-" * 70)
print("Saved Files")

for name, file in files.items():
    print(f"{name:<15}: {file}")

print("=" * 70)
print("Rolling backtesting verified successfully ✓")
print("=" * 70)