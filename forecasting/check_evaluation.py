
"""
check_evaluation.py

Evaluates recursive forecasts against
actual prices.
"""
import pandas as pd
from data.loader import MarketDataLoader
from features.engineering import FeatureEngineer
from forecasting.trainer import ForecastTrainer
from forecasting.recursive_engine import RecursiveForecastEngine
from forecasting.evaluation import ForecastEvaluator

loader = MarketDataLoader()
engineer = FeatureEngineer()
trainer = ForecastTrainer()
evaluator = ForecastEvaluator()

df = loader.load_csv("data/raw/ercot_prices.csv")
features = engineer.transform(df)

split = trainer.chronological_split(features)

model = trainer.train(
    split.train,
    model_type="xgboost",
)

engine = RecursiveForecastEngine(model)


# =====================================================
# Use the last 168 hours before the test period
# =====================================================

evaluation_input = features.loc[
    : split.test.index[0] - pd.Timedelta(hours=1)
]

forecast = engine.recursive_forecast_with_actuals(
    evaluation_input,
    split.test,
    horizon=24,
)

report = evaluator.evaluate(
    forecast.dataframe,
    model_name="XGBoost Recursive",
    horizon=24,
)

print("=" * 70)
print("GLOBAL FORECAST REPORT")
print("=" * 70)

print(report)

print("\nHourly Metrics")
print("-" * 70)

hourly = evaluator.horizon_metrics(
    forecast.dataframe,
)

print(hourly.head())

print("\nDaily Metrics")
print("-" * 70)

daily = evaluator.daily_summary(
    forecast.dataframe,
)

print(daily.head())

evaluator.export(
    forecast.dataframe
)

print("\nSaved to forecasting/results/")