import pandas as pd

from data.loader import MarketDataLoader


def test_loader():

    loader = MarketDataLoader()

    df = loader.load_csv("data/raw/ercot_prices.csv")

    assert len(df) == 8760
    assert df.index.is_monotonic_increasing
    assert df.index.name == "timestamp"
    assert "price" in df.columns
    assert pd.api.types.is_float_dtype(df["price"])


def test_metadata():

    loader = MarketDataLoader()

    df = loader.load_csv("data/raw/ercot_prices.csv")

    meta = loader.metadata(df)

    assert meta.rows == 8760
    assert meta.missing_hours == 0
    assert meta.timezone == "UTC"