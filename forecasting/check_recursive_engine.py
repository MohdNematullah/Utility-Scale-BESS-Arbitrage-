
"""
check_recursive_engine.py

Verifies RecursiveForecastEngine.
"""

from data.loader import MarketDataLoader
from features.engineering import FeatureEngineer
from forecasting.trainer import ForecastTrainer
from forecasting.recursive_engine import RecursiveForecastEngine

loader = MarketDataLoader()
engineer = FeatureEngineer()
trainer = ForecastTrainer()

df = loader.load_csv("data/raw/ercot_prices.csv")
feature_df = engineer.transform(df)

split = trainer.chronological_split(feature_df)

model = trainer.train(
    split.train,
    model_type="xgboost",
)

engine = RecursiveForecastEngine(model)

history = engine.initialize_history(split.train)

engine.validate_history(history)

print("=" * 70)
print("RECURSIVE ENGINE CHECK")
print("=" * 70)

print("History Length :", len(history))
print("Start          :", history.index.min())
print("End            :", history.index.max())

next_ts = engine.next_timestamp(
    history.index.max()
)

print("Next Timestamp :", next_ts)

feature_row = engine.build_recursive_row(
    history,
    next_ts,
)

print("\nFeature Row Shape :", feature_row.shape)

prediction = engine.predict_one_step(feature_row)

print("Predicted Price   :", round(prediction, 3))

future = engine.future_index(
    history.index.max(),
    horizon=24,
)

print("\n24-Hour Forecast Index Preview")
print(future[:5])

print("\nRecursive Engine Core Passed.")