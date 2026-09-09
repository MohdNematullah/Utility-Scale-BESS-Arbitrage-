from pathlib import Path

from data.loader import MarketDataLoader
from features.engineering import FeatureEngineer

from forecasting.models import XGBoostForecaster
from forecasting.recursive_engine import RecursiveForecastEngine

from optimization.model_builder import BatteryOptimizationModelBuilder
from optimization.solver import BatteryOptimizationSolver
from optimization.results import BatteryResultsProcessor


def prepare():

    loader = MarketDataLoader()
    engineer = FeatureEngineer()

    market = loader.load_csv("data/raw/ercot_prices.csv")
    features = engineer.transform(market)

    model_path = Path("forecasting/saved_models/xgboost_dayahead.pkl")

    if not model_path.exists():
        raise FileNotFoundError(
            "Run python -m forecasting.check_trainer first."
        )

    forecaster = XGBoostForecaster()
    forecaster.load(model_path)

    engine = RecursiveForecastEngine(forecaster)

    forecast = engine.forecast(
        feature_dataframe=features,
        horizon=24,
    )
    builder = BatteryOptimizationModelBuilder()
    model = builder.build(forecast)

    solver = BatteryOptimizationSolver()
    solver.solve(model)

    processor = BatteryResultsProcessor()

    return model, processor


def test_dispatch_schedule():

    model, processor = prepare()

    dispatch = processor.dispatch_schedule(model)

    assert len(dispatch) == 24
    assert "soc_mwh" in dispatch.columns


def test_revenue_breakdown():

    model, processor = prepare()

    dispatch = processor.dispatch_schedule(model)

    revenue = processor.revenue_breakdown(dispatch)

    assert "net_profit_$" in revenue.columns


def test_summary():

    model, processor = prepare()

    dispatch = processor.dispatch_schedule(model)

    summary = processor.battery_summary(model, dispatch)

    assert summary.horizon_hours == 24


def test_export():

    model, processor = prepare()

    summary = processor.export(model)

    assert summary.total_charge_energy >= 0