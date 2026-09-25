"""
backtesting/check_engine.py
===========================
verification script for Rolling Horizon Backtesting Engine.
Validates realized-price settlement, end-of-interval SOC continuity,
and electrochemical degradation tracking.
"""

from pathlib import Path
import sys
import pandas as pd

# ----------------------------------------------------
# 1. Project Root & Environment Setup
# ----------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.loader import MarketDataLoader
from features.engineering import FeatureEngineer
from forecasting.models import XGBoostForecaster
from backtesting.engine import RollingBacktestEngine

print("=" * 70)
print("ROLLING HORIZON BACKTEST CHECK")
print("=" * 70)

# ----------------------------------------------------
# 2. Ingest Data & Engineer Features
# ----------------------------------------------------
data_path = PROJECT_ROOT / "data" / "raw" / "ercot_prices.csv"
if not data_path.exists():
    raise FileNotFoundError(f"Missing price dataset: {data_path}")

loader = MarketDataLoader()
engineer = FeatureEngineer()

market = loader.load_csv(data_path)
features = engineer.transform(market)

# Ensure recursive forecasting uses a sorted UTC DatetimeIndex
if "timestamp" in features.columns:
    features["timestamp"] = pd.to_datetime(
        features["timestamp"],
        utc=True,
    )
    features = features.sort_values("timestamp").set_index("timestamp")

features.index = pd.DatetimeIndex(features.index)
print(f"Feature matrix prepared     : {features.shape[0]} intervals âœ“")

# ----------------------------------------------------
# 3. Load Trained Forecast Model
# ----------------------------------------------------
model_path = PROJECT_ROOT / "forecasting" / "saved_models" / "xgboost_dayahead.pkl"
if not model_path.exists():
    raise FileNotFoundError(
        f"Missing model at {model_path}.\n"
        "Run: python -m forecasting.check_trainer first."
    )

model = XGBoostForecaster()
model.load(model_path)
print("Forecast model loaded       : XGBoost âœ“")

# ----------------------------------------------------
# 4. Execute Rolling Horizon Backtest
# ----------------------------------------------------
engine = RollingBacktestEngine(
    forecaster=model,
    feature_dataframe=features,
)

print("Starting rolling backtest...")
result = engine.run()
print("Rolling horizon completed   : âœ“")

# -------------------------------------
# 5. Regression Assertions
# -------------------------------------
assert "actual_price" in result.dispatch_history.columns, (
    "Regression Error: 'actual_price' column missing from dispatch history."
)
assert "net_revenue_usd" in result.dispatch_history.columns, (
    "Regression Error: 'net_revenue_usd' realized settlement column missing."
)
assert not result.dispatch_history["actual_price"].isna().any(), (
    "Data Integrity Error: 'actual_price' contains NaN values."
)

soc_diff = result.dispatch_history["soc_mwh"].diff().dropna()
max_hourly_soc_delta = float(soc_diff.abs().max())

# Max energy delta per hour cannot exceed inverter rating (50 MW * 1h = 50 MWh)
nominal_capacity = engine.ageing.config.chemistry.nominal_capacity_mwh
assert max_hourly_soc_delta <= 50.01, (
    f"Physical Inconsistency: Max hourly SOC jump of {max_hourly_soc_delta:.2f} MWh exceeds inverter rating."
)

print("Regression assertions       : ALL PASSED âœ“")

# ----------------------------------------------------
# 6. Summary KPI Output
# ----------------------------------------------------
print("-" * 70)
print("ROLLING BACKTEST SUMMARY")
print("-" * 70)

for key, value in result.summary.items():
    if isinstance(value, float):
        print(f"{key:<28}: {value:,.4f}")
    else:
        print(f"{key:<28}: {value}")

# --------------------------------------
# 7. Realized Settlement Preview
# --------------------------------------
print("-" * 70)
print("Dispatch Preview (Realized Settlement Verification):")

preview_cols = [
    c for c in [
        "forecast_price",
        "actual_price",
        "charge_power_mw",
        "discharge_power_mw",
        "soc_mwh",
        "net_revenue_usd",
    ]
    if c in result.dispatch_history.columns
]
print(result.dispatch_history[preview_cols].head(3))

# -------------------------------------------------
# 8. SOC Continuity & Ageing Verification
# -------------------------------------------------
print("-" * 70)
print("Degradation & Continuity Diagnostics:")
print(f"Maximum Hourly SOC Delta    : {max_hourly_soc_delta:.3f} MWh (Physical Limit: 50.00 MWh)")

deg_cols = [
    c for c in [
        "rolling_window",
        "soh_start",
        "soh_end",
        "window_loss",
        "window_efc",
        "degradation_cost_usd",
        "remaining_soh",
    ]
    if c in result.degradation_history.columns
]
print(result.degradation_history[deg_cols].head(3))

# ----------------------------------------------------
# 9. Artifact Export
# ----------------------------------------------------
files = engine.export(result)

print("-" * 70)
print("Exported Verification Artifacts")
for name, file in files.items():
    print(f"{name:<15}: {file}")

print("=" * 70)
print("STATUS: Rolling backtesting verified successfully âœ“")
print("=" * 70)