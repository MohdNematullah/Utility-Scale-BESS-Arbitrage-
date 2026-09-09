"""
tests/backtesting/test_forecast_realism.py
=========================================

Unit Test Suite for Forecast Realism & Accuracy Engine (Part 9.1).
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from backtesting.forecast_realism import (
    ForecastMetricsSummary,
    ForecastRealismEngine,
)


@pytest.fixture
def synthetic_price_series() -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(42)
    actual = 30.0 + 10.0 * np.sin(np.linspace(0, 4 * np.pi, 200))
    forecast = actual + np.random.normal(loc=0.5, scale=2.0, size=200)
    return actual, forecast


@pytest.fixture
def synthetic_dispatch_dataframe(synthetic_price_series) -> pd.DataFrame:
    actual, forecast = synthetic_price_series
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=len(actual), freq="h", tz="UTC"),
        "actual_price": actual,
        "forecast_price": forecast,
        "net_revenue_usd": [150.0] * len(actual),
    })


class TestForecastRealismMetrics:
    def test_mae_and_rmse_relationships(self, synthetic_price_series):
        actual, forecast = synthetic_price_series
        mae = ForecastRealismEngine.mean_absolute_error(actual, forecast)
        rmse = ForecastRealismEngine.root_mean_squared_error(actual, forecast)

        assert mae > 0.0
        assert rmse >= mae

    def test_perfect_accuracy_edge_case(self):
        actual = np.array([25.0, 30.0, 45.0, 10.0])
        forecast = np.array([25.0, 30.0, 45.0, 10.0])

        assert ForecastRealismEngine.mean_absolute_error(actual, forecast) == 0.0
        assert ForecastRealismEngine.root_mean_squared_error(actual, forecast) == 0.0
        assert ForecastRealismEngine.coefficient_of_determination(actual, forecast) == 1.0
        assert ForecastRealismEngine.mean_bias_error(actual, forecast) == 0.0
        assert ForecastRealismEngine.directional_accuracy(actual, forecast) == 100.0

    def test_zero_denominator_mape_handling(self):
        actual = np.array([0.0, 0.0, 0.0])
        forecast = np.array([5.0, 5.0, 5.0])
        mape = ForecastRealismEngine.mean_absolute_percentage_error(actual, forecast)
        assert mape == 0.0

    def test_directional_accuracy_bounds(self, synthetic_price_series):
        actual, forecast = synthetic_price_series
        da = ForecastRealismEngine.directional_accuracy(actual, forecast)
        assert 0.0 <= da <= 100.0

    def test_residual_higher_moments(self, synthetic_price_series):
        actual, forecast = synthetic_price_series
        residuals = forecast - actual
        std_dev, skewness, kurtosis = ForecastRealismEngine.residual_higher_moments(residuals)

        assert std_dev > 0.0
        assert np.isfinite(skewness)
        assert np.isfinite(kurtosis)

    def test_full_evaluation_summary_structure(self, synthetic_price_series):
        actual, forecast = synthetic_price_series
        engine = ForecastRealismEngine(output_directory="backtesting/results/test_forecast")
        summary = engine.evaluate_metrics(
            y_true=actual,
            y_pred=forecast,
            gross_revenue_usd=80000.0,
            perfect_foresight_revenue_usd=100000.0,
        )

        assert isinstance(summary, ForecastMetricsSummary)
        assert summary.value_capture_ratio_pct == 80.0
        assert summary.perfect_foresight_gap_usd == 20000.0
        assert summary.sample_count == len(actual)


class TestForecastRealismPipeline:
    def test_evaluate_and_export_artifacts(self, synthetic_dispatch_dataframe, tmp_path):
        engine = ForecastRealismEngine(output_directory=tmp_path)
        metrics, artifacts = engine.evaluate_and_export(
            dispatch_df=synthetic_dispatch_dataframe,
            gross_revenue_usd=90000.0,
            perfect_foresight_revenue_usd=100000.0,
        )

        assert artifacts.summary_csv.exists()
        assert artifacts.residuals_csv.exists()
        assert artifacts.summary_json.exists()

        with open(artifacts.summary_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "mae" in data
        assert "rmse" in data
        assert "r2_score" in data

        figures = list(artifacts.figures_directory.glob("*.png"))
        assert len(figures) == 6
        figure_names = {f.name for f in figures}
        assert figure_names == {
            "forecast_vs_actual.png",
            "residual_distribution.png",
            "residual_qqplot.png",
            "forecast_horizon_heatmap.png",
            "forecast_scatter.png",
            "rolling_rmse.png",
        }

    def test_missing_column_raises_keyerror(self, tmp_path):
        engine = ForecastRealismEngine(output_directory=tmp_path)
        invalid_df = pd.DataFrame({"forecast_price": [10.0, 20.0]})
        with pytest.raises(KeyError, match="must contain 'forecast_price'"):
            engine.evaluate_and_export(invalid_df)