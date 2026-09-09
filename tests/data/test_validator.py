import pandas as pd
from data.validator import MarketDataValidator


def test_validator():

    df = pd.read_csv("data/raw/ercot_prices.csv")

    validator = MarketDataValidator()

    report = validator.validate(df)

    assert report.rows == 8760
    assert report.missing_values == 0
    assert report.duplicate_timestamps == 0
    assert report.passed is True