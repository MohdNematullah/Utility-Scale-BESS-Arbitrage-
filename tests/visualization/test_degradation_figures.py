"""
tests/visualization/test_degradation_figures.py
==============================================

Unit Test Suite for Degradation Figures Module (Part 10.4).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from visualization.degradation_figures import (
    DegradationFigureArtifacts,
    DegradationFigureGenerator,
)


@pytest.fixture
def synthetic_degradation_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 60
    fade_per_day = 0.0188 / 365.0
    soh_vals = 1.0 - np.cumsum(np.full(n, fade_per_day))

    return pd.DataFrame({
        "rolling_window": range(1, n + 1),
        "soh_end": soh_vals,
        "calendar_loss": np.full(n, fade_per_day * 0.4),
        "cycle_loss": np.full(n, fade_per_day * 0.6),
        "window_efc": np.full(n, 0.54),
        "cumulative_efc": np.cumsum(np.full(n, 0.54)),
        "degradation_cost_usd": np.full(n, 725.0),
        "cumulative_degradation_cost_usd": np.cumsum(np.full(n, 725.0)),
    })


class TestDegradationFiguresPipeline:
    def test_complete_degradation_figures_generation(self, synthetic_degradation_df, tmp_path):
        gen = DegradationFigureGenerator(output_directory=tmp_path, formats=("png", "pdf", "svg"), dpi=150)
        artifacts = gen.generate_all(synthetic_degradation_df)

        assert isinstance(artifacts, DegradationFigureArtifacts)

        groups = [
            artifacts.soh_curve,
            artifacts.capacity_fade,
            artifacts.calendar_cycle_loss,
            artifacts.rainflow_histogram,
            artifacts.efc_curve,
            artifacts.degradation_cost_curve,
        ]

        assert len(groups) == 6

        for g in groups:
            assert set(g.keys()) == {"png", "pdf", "svg"}
            for p in g.values():
                assert p.exists()
                assert p.stat().st_size > 500

    def test_isolated_figures(self, synthetic_degradation_df, tmp_path):
        gen = DegradationFigureGenerator(output_directory=tmp_path, formats=["png"], dpi=150)

        p_soh = gen.plot_soh_curve(synthetic_degradation_df)
        assert "png" in p_soh
        assert p_soh["png"].exists()

        p_rain = gen.plot_rainflow_histogram(np.array([10.0, 20.0, 50.0, 80.0, 90.0]))
        assert "png" in p_rain
        assert p_rain["png"].exists()

        p_cost = gen.plot_degradation_cost_curve(synthetic_degradation_df)
        assert "png" in p_cost
        assert p_cost["png"].exists()