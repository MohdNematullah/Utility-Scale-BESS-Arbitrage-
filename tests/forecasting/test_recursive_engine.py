
from data.loader import MarketDataLoader
from features.engineering import FeatureEngineer

from forecasting.trainer import ForecastTrainer
from forecasting.recursive_engine import (
    RecursiveForecastEngine,
)


def prepare():

    loader = MarketDataLoader()
    engineer = FeatureEngineer()

    df = loader.load_csv(
        "data/raw/ercot_prices.csv"
    )

    feature_df = engineer.transform(df)

    trainer = ForecastTrainer()

    split = trainer.chronological_split(feature_df)

    model = trainer.train(
        split.train,
        model_type="xgboost",
    )

    engine = RecursiveForecastEngine(model)

    history = engine.initialize_history(split.train)

    return engine, history


def test_history():

    engine, history = prepare()

    engine.validate_history(history)

    assert len(history) == 168


def test_calendar_features():

    engine, history = prepare()

    timestamp = engine.next_timestamp(
        history.index.max()
    )

    row = engine.calendar_features(timestamp)

    assert row["hour"] == timestamp.hour
    assert row["month"] == timestamp.month


def test_build_recursive_row():

    engine, history = prepare()

    timestamp = engine.next_timestamp(
        history.index.max()
    )

    row = engine.build_recursive_row(
        history,
        timestamp,
    )

    # Expected rolling windows
    for window in [3, 6, 12, 24, 48, 72, 168]:

        assert f"price_mean_{window}" in row.columns
        assert f"price_std_{window}" in row.columns
        assert f"price_min_{window}" in row.columns
        assert f"price_max_{window}" in row.columns

    # Expected lag features
    for lag in [1, 2, 3, 6, 12, 24, 48, 72, 96, 168]:
        assert f"price_lag_{lag}" in row.columns


def test_predict_one_step():

    engine, history = prepare()

    timestamp = engine.next_timestamp(
        history.index.max()
    )

    row = engine.build_recursive_row(
        history,
        timestamp,
    )

    prediction = engine.predict_one_step(row)

    assert isinstance(prediction, float)