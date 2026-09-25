"""
Unit tests for Battery Optimization Results Processor.
Validates result extraction, initial SOC semantics, and realized settlement export.
"""

from pathlib import Path
import pytest
import pyomo.environ as pyo

from data.loader import MarketDataLoader
from features.engineering import FeatureEngineer
from forecasting.models import XGBoostForecaster
from forecasting.recursive_engine import RecursiveForecastEngine
from optimization.model_builder import BatteryOptimizationModelBuilder
from optimization.solver import BatteryOptimizationSolver
from optimization.results import BatteryResultsProcessor


@pytest.fixture(scope="module")
def optimization_setup():
    """Build and solve optimization model once for the test module."""
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

    forecaster = XGBoostForecaster()
    forecaster.load(model_path)

    engine = RecursiveForecastEngine(forecaster)
    forecast = engine.forecast(feature_dataframe=features, horizon=24)

    builder = BatteryOptimizationModelBuilder()
    model = builder.build(forecast)

    solver = BatteryOptimizationSolver()
    solver.solve(model)

    processor = BatteryResultsProcessor()
    return model, processor


def test_dispatch_schedule(optimization_setup):
    model, processor = optimization_setup
    dispatch = processor.dispatch_schedule(model)

    assert len(dispatch) == 24
    assert "soc_mwh" in dispatch.columns
    assert "hourly_revenue_$" in dispatch.columns

    # Verify settlement price column is present when actual prices are supplied
    if "actual_price" in model.forecast_dataframe.columns:
        assert "actual_price" in dispatch.columns


def test_revenue_breakdown(optimization_setup):
    model, processor = optimization_setup
    dispatch = processor.dispatch_schedule(model)
    revenue = processor.revenue_breakdown(dispatch)

    assert "net_profit_$" in revenue.columns
    assert "charging_cost_$" in revenue.columns
    assert "discharging_income_$" in revenue.columns


def test_summary(optimization_setup):
    """
    Regression Test:
    Ensures soc_initial records the pre-dispatch baseline state from model.initial_soc.
    """
    model, processor = optimization_setup
    dispatch = processor.dispatch_schedule(model)
    summary = processor.battery_summary(model, dispatch)

    assert summary.horizon_hours == 24
    assert summary.soc_initial == float(pyo.value(model.initial_soc)), (
        f"Initial SOC mismatch: summary shows {summary.soc_initial}, "
        f"model parameter is {float(pyo.value(model.initial_soc))}"
    )


def test_export(optimization_setup):
    model, processor = optimization_setup
    summary = processor.export(model)

    assert summary.total_charge_energy >= 0
    assert summary.total_discharge_energy >= 0