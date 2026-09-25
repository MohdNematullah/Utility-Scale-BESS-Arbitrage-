
"""
check_models.py

Verifies forecasting models
"""

from data.loader import MarketDataLoader
from features.engineering import FeatureEngineer

from forecasting.baselines import (
    PersistenceForecaster,
    SeasonalNaiveForecaster,
    MovingAverageForecaster,
)

from forecasting.models import (
    XGBoostForecaster,
    RandomForestForecaster,
)

loader = MarketDataLoader()

df = loader.load_csv("data/raw/ercot_prices.csv")

engineer = FeatureEngineer()

feature_df = engineer.transform(df)

train, test = loader.train_test_split(feature_df)

print("=" * 70)
print("FORECASTING MODEL CHECK")
print("=" * 70)

series = df["price"]

# Baselines
for model in [
    PersistenceForecaster(),
    SeasonalNaiveForecaster(),
    MovingAverageForecaster(),
]:

    model.fit(series)

    pred = model.predict(24)

    print(model.__class__.__name__, pred.prediction[:5])

# ML Models
for model in [
    RandomForestForecaster(),
    XGBoostForecaster(),
]:

    model.fit(train)

    pred = model.predict(test.head(5))

    print(model.__class__.__name__)
    print(pred.prediction)
    print("-" * 50)