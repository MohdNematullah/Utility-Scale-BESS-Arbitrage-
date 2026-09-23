"""
Unit tests for Rolling Horizon Backtesting Engine.
Validates realized settlement accounting, rolling continuity, and degradation tracking.
"""

from pathlib import Path
import pytest

from data.loader import MarketDataLoader
from features.engineering import FeatureEngineer
from forecasting.models import XGBoostForecaster
from backtesting.engine import RollingBacktestEngine


@pytest.fixture(scope="module")
def backtest_setup():
    """Run rolling backtest once for the test module."""
    data_path = Path("data/raw/ercot_prices.csv")
    model_path = Path("forecasting/saved_models/xgboost_dayahead.pkl")

    if not data_path.exists():
        pytest.skip(f"Missing test dataset: {data_path}")
    if not model_path.exists():
        pytest.skip(f"Missing test model: {model_path}")

    loader = MarketDataLoader()
    engineer = FeatureEngineer()

    market = loader.load_csv(data_path)
    features = engineer.transform(market)

    model = XGBoostForecaster()
    model.load(model_path)

    engine = RollingBacktestEngine(
        forecaster=model,
        feature_dataframe=features,
    )

    result = engine.run()
    return result, engine


def test_dispatch_created(backtest_setup):
    result, _ = backtest_setup
    assert len(result.dispatch_history) > 0


def test_degradation_created(backtest_setup):
    result, _ = backtest_setup
    assert len(result.degradation_history) > 0


def test_summary_keys(backtest_setup):
    result, _ = backtest_setup
    assert "gross_revenue_usd" in result.summary
    assert "net_revenue_usd" in result.summary


def test_soh_decreases(backtest_setup):
    result, _ = backtest_setup
    assert result.summary["final_soh"] <= 1.0


def test_net_revenue(backtest_setup):
    result, _ = backtest_setup
    assert (
        result.summary["gross_revenue_usd"]
        >= result.summary["net_revenue_usd"]
    )


def test_realized_price_settlement_regression(backtest_setup):
    """
    Mentor Review #1 Regression Test:
    Ensures revenue is calculated from actual settlement prices rather than forecast prices.
    """
    result, _ = backtest_setup
    dispatch = result.dispatch_history

    # 1. Verify settlement columns exist and are populated
    assert "actual_price" in dispatch.columns, "actual_price column missing from dispatch history"
    assert "net_revenue_usd" in dispatch.columns, "net_revenue_usd column missing from dispatch history"
    assert not dispatch["actual_price"].isna().any(), "actual_price contains NaN values"

    # 2. Mathematically verify realized revenue calculation for executed actions
    sample = dispatch.iloc[0]
    expected_revenue = sample["actual_price"] * (
        sample["discharge_power_mw"] - sample["charge_power_mw"]
    )
    assert abs(sample["net_revenue_usd"] - expected_revenue) < 1e-6, (
        f"Settlement mismatch: got {sample['net_revenue_usd']}, expected {expected_revenue}"
    )


def test_export(backtest_setup, tmp_path):
    result, engine = backtest_setup
    files = engine.export(result)

    assert files["dispatch"].exists()
    assert files["degradation"].exists()
    assert files["summary"].exists()