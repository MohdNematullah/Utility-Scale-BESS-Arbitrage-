"""
tests/visualization/test_forecast_figures.py
============================================

Unit Test Suite for Forecast Figures Module (Part 10.2).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from visualization.forecast_figures import (
    ForecastFigureArtifacts,
    ForecastFigureGenerator,
)


@pytest.fixture
def synthetic_forecast_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 240
    actual = 30.0 + 10.0 * np.sin(np.linspace(0, 4 * np.pi, n))
    forecast = actual + np.random.normal(0, 2.0, n)
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC"),
        "actual_price": actual,
        "forecast_price": forecast,
    })


class TestForecastFiguresPipeline:
    def test_complete_forecast_figures_generation(self, synthetic_forecast_df, tmp_path):
        gen = ForecastFigureGenerator(output_directory=tmp_path, formats=("png", "pdf", "svg"), dpi=150)
        artifacts = gen.generate_all(synthetic_forecast_df)

        assert isinstance(artifacts, ForecastFigureArtifacts)

        groups = [
            artifacts.forecast_vs_actual,
            artifacts.residual_histogram,
            artifacts.residual_timeseries,
            artifacts.forecast_horizon_accuracy,
            artifacts.forecast_scatter,
            artifacts.forecast_heatmap,
        ]

        assert len(groups) == 6

        for g in groups:
            assert set(g.keys()) == {"png", "pdf", "svg"}
            for p in g.values():
                assert p.exists()
                assert p.stat().st_size > 500

    def test_individual_figures_isolated(self, synthetic_forecast_df, tmp_path):
        gen = ForecastFigureGenerator(output_directory=tmp_path, formats=["png"], dpi=150)
        y_t = synthetic_forecast_df["actual_price"].values
        y_p = synthetic_forecast_df["forecast_price"].values

        p_scatter = gen.plot_forecast_scatter(y_t, y_p)
        assert "png" in p_scatter
        assert p_scatter["png"].exists()

        p_dist = gen.plot_residual_distribution(y_p - y_t)
        assert "png" in p_dist
        assert p_dist["png"].exists()

    def test_missing_column_raises_keyerror(self, tmp_path):
        gen = ForecastFigureGenerator(output_directory=tmp_path)
        invalid_df = pd.DataFrame({"only_price": [1.0, 2.0]})
        with pytest.raises(KeyError, match="must contain actual_price and forecast_price"):
            gen.generate_all(invalid_df)