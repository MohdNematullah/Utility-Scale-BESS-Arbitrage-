
from data.loader import MarketDataLoader
from features.engineering import FeatureEngineer

from forecasting.trainer import ForecastTrainer
from forecasting.recursive_engine import RecursiveForecastEngine


def prepare():

    loader = MarketDataLoader()
    engineer = FeatureEngineer()

    df = loader.load_csv(
        "data/raw/ercot_prices.csv"
    )

    features = engineer.transform(df)

    trainer = ForecastTrainer()

    split = trainer.chronological_split(features)

    model = trainer.train(
        split.train,
        model_type="xgboost",
    )

    engine = RecursiveForecastEngine(model)

    return engine, split


def test_recursive_forecast_24():

    engine, split = prepare()

    forecast = engine.recursive_forecast(
        split.train,
        horizon=24,
    )

    assert len(forecast.dataframe) == 24
    assert forecast.horizon == 24
    assert forecast.dataframe.prediction.notna().all()


def test_recursive_forecast_48():

    engine, split = prepare()

    forecast = engine.recursive_forecast(
        split.train,
        horizon=48,
    )

    assert len(forecast.dataframe) == 48
    assert forecast.horizon == 48


def test_forecast_summary():

    engine, split = prepare()

    forecast = engine.recursive_forecast(
        split.train,
        horizon=24,
    )

    summary = engine.forecast_summary(forecast)

    assert summary["rows"] == 24
    assert summary["horizon"] == 24


def test_multi_horizon():

    engine, split = prepare()

    forecasts = engine.multi_horizon_forecast(
        split.train,
        horizons=(24, 48),
    )

    assert 24 in forecasts
    assert 48 in forecasts