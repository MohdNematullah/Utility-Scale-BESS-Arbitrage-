
import pandas as pd

from data.loader import MarketDataLoader
from features.engineering import FeatureEngineer


def test_feature_engineering():

    loader = MarketDataLoader()

    df = loader.load_csv("data/raw/ercot_prices.csv")

    engineer = FeatureEngineer()

    features = engineer.transform(df)

    assert len(features) < len(df)
    assert "price_lag_24" in features.columns
    assert "price_mean_24" in features.columns
    assert "hour_sin" in features.columns
    assert "price_volatility_24h" in features.columns

    assert features.isna().sum().sum() == 0