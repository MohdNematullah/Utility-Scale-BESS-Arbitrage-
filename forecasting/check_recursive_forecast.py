
"""
check_recursive_forecast.py

Runs recursive 24h and 48h forecasts.
"""

from data.loader import MarketDataLoader
from features.engineering import FeatureEngineer
from forecasting.trainer import ForecastTrainer
from forecasting.recursive_engine import RecursiveForecastEngine

loader = MarketDataLoader()
engineer = FeatureEngineer()
trainer = ForecastTrainer()

df = loader.load_csv("data/raw/ercot_prices.csv")
features = engineer.transform(df)

split = trainer.chronological_split(features)

model = trainer.train(
    split.train,
    model_type="xgboost",
)

engine = RecursiveForecastEngine(model)

# -------------------------------------------------------
# 24-hour Forecast
# -------------------------------------------------------

forecast24 = engine.recursive_forecast(
    split.train,
    horizon=24,
)

print("=" * 70)
print("24-HOUR RECURSIVE FORECAST")
print("=" * 70)

print(forecast24.dataframe.head())

summary24 = engine.forecast_summary(forecast24)

print(summary24)

forecast24.dataframe.to_csv(
    "forecasting/forecast_24h.csv",
    index=False,
)

# -------------------------------------------------------
# 48-hour Forecast
# -------------------------------------------------------

forecast48 = engine.recursive_forecast(
    split.train,
    horizon=48,
)

summary48 = engine.forecast_summary(forecast48)

forecast48.dataframe.to_csv(
    "forecasting/forecast_48h.csv",
    index=False,
)

print("\n48-HOUR SUMMARY")
print(summary48)

print("\nSaved:")
print("forecasting/forecast_24h.csv")
print("forecasting/forecast_48h.csv")