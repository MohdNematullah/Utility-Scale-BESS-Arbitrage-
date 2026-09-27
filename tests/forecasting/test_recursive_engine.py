from data.loader import MarketDataLoader
from features.engineering import FeatureEngineer
from forecasting.trainer import ForecastTrainer
from forecasting.recursive_engine import RecursiveForecastEngine


def prepare():
    loader = MarketDataLoader()
    engineer = FeatureEngineer()

    df = loader.load_csv("data/raw/ercot_prices.csv")
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
    timestamp = engine.next_timestamp(history.index.max())
    row = engine.calendar_features(timestamp)

    assert row["hour"] == timestamp.hour
    assert row["month"] == timestamp.month


def test_build_recursive_row():
    engine, history = prepare()
    timestamp = engine.next_timestamp(history.index.max())

    row = engine.build_recursive_row(
        history,
        timestamp,
    )

    # Verify all rolling window columns configured in FeatureConfig
    for window in engine.engineer.config.rolling_windows:
        assert f"price_mean_{window}" in row.columns
        assert f"price_std_{window}" in row.columns
        assert f"price_min_{window}" in row.columns
        assert f"price_max_{window}" in row.columns

    # Verify all lag features configured in FeatureConfig
    for lag in engine.engineer.config.lag_hours:
        assert f"price_lag_{lag}" in row.columns

    # Verify dynamic features
    assert "price_change_1h" in row.columns
    assert "price_change_24h" in row.columns
    assert "price_momentum_6h" in row.columns
    assert "price_volatility_24h" in row.columns


def test_predict_one_step():
    engine, history = prepare()
    timestamp = engine.next_timestamp(history.index.max())

    row = engine.build_recursive_row(
        history,
        timestamp,
    )

    prediction = engine.predict_one_step(row)
    assert isinstance(prediction, float)