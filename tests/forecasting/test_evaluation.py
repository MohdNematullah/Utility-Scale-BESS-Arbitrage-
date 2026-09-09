
from forecasting.evaluation import ForecastEvaluator
import pandas as pd


def sample_dataframe():

    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2025-01-01",
                periods=24,
                freq="h",
                tz="UTC",
            ),
            "prediction": [10 + i for i in range(24)],
            "actual": [11 + i for i in range(24)],
            "step": list(range(1, 25)),
        }
    )


def test_metrics():

    evaluator = ForecastEvaluator()

    df = sample_dataframe()

    report = evaluator.evaluate(df)

    assert report.mae == 1.0
    assert report.rmse == 1.0
    assert report.smape > 0


def test_hourly_metrics():

    evaluator = ForecastEvaluator()

    metrics = evaluator.horizon_metrics(
        sample_dataframe()
    )

    assert len(metrics) == 24


def test_daily_metrics():

    evaluator = ForecastEvaluator()

    summary = evaluator.daily_summary(
        sample_dataframe()
    )

    assert len(summary) == 1


def test_relative_mae():

    evaluator = ForecastEvaluator()

    value = evaluator.relative_mae(
        sample_dataframe()
    )

    assert value >= 0