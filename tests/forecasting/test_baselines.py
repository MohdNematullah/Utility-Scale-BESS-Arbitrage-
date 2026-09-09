
import numpy as np

from data.loader import MarketDataLoader
from forecasting.baselines import (
    PersistenceForecaster,
    SeasonalNaiveForecaster,
    MovingAverageForecaster,
)


def load_series():
    loader = MarketDataLoader()
    df = loader.load_csv("data/raw/ercot_prices.csv")
    return df["price"]


def test_persistence():

    series = load_series()

    model = PersistenceForecaster().fit(series)

    forecast = model.predict(24)

    assert len(forecast.prediction) == 24
    assert np.all(
        forecast.prediction == series.iloc[-1]
    )


def test_seasonal_naive():

    series = load_series()

    model = SeasonalNaiveForecaster().fit(series)

    forecast = model.predict(24)

    assert len(forecast.prediction) == 24

    assert np.allclose(
        forecast.prediction,
        series.iloc[-24:].values,
    )


def test_moving_average():

    series = load_series()

    model = MovingAverageForecaster().fit(series)

    forecast = model.predict(24)

    expected = series.tail(24).mean()

    assert len(forecast.prediction) == 24

    assert np.allclose(
        forecast.prediction,
        expected,
    )