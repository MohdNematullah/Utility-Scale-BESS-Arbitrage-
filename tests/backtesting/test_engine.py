"""
Unit tests for Rolling Horizon Backtesting Engine.
"""

from pathlib import Path

import pytest

from data.loader import MarketDataLoader
from features.engineering import FeatureEngineer
from forecasting.models import XGBoostForecaster

from backtesting.engine import RollingBacktestEngine


def prepare():

    loader = MarketDataLoader()
    engineer = FeatureEngineer()

    market = loader.load_csv("data/raw/ercot_prices.csv")
    features = engineer.transform(market)

    model = XGBoostForecaster()
    model.load(Path("forecasting/saved_models/xgboost_dayahead.pkl"))

    engine = RollingBacktestEngine(
        forecaster=model,
        feature_dataframe=features,
    )

    return engine.run(), engine


def test_dispatch_created():

    result, _ = prepare()

    assert len(result.dispatch_history) > 0


def test_degradation_created():

    result, _ = prepare()

    assert len(result.degradation_history) > 0


def test_summary_keys():

    result, _ = prepare()

    assert "gross_revenue_usd" in result.summary
    assert "net_revenue_usd" in result.summary


def test_soh_decreases():

    result, _ = prepare()

    assert result.summary["final_soh"] <= 1.0


def test_net_revenue():

    result, _ = prepare()

    assert (
        result.summary["gross_revenue_usd"]
        >= result.summary["net_revenue_usd"]
    )


def test_export(tmp_path):

    result, engine = prepare()

    files = engine.export(result)

    assert files["dispatch"].exists()
    assert files["degradation"].exists()
    assert files["summary"].exists()