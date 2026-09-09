"""
tests/visualization/test_dispatch_figures.py
============================================

Unit Test Suite for Dispatch Figures Module (Part 10.3).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from visualization.dispatch_figures import (
    DispatchFigureArtifacts,
    DispatchFigureGenerator,
)


@pytest.fixture
def synthetic_dispatch_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 240
    prices = 35.0 + 15.0 * np.sin(np.linspace(0, 10 * 2 * np.pi, n))
    chg = np.array([40.0 if (i % 24) in [1, 2] else 0.0 for i in range(n)])
    dis = np.array([35.0 if (i % 24) in [17, 18] else 0.0 for i in range(n)])
    soc = np.clip(50.0 + np.cumsum(chg - dis) * 0.1, 5.0, 95.0) / 100.0

    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC"),
        "actual_price": prices,
        "charge_power_mw": chg,
        "discharge_power_mw": dis,
        "soc": soc,
    })


class TestDispatchFiguresPipeline:
    def test_complete_dispatch_figures_generation(self, synthetic_dispatch_df, tmp_path):
        gen = DispatchFigureGenerator(output_directory=tmp_path, formats=("png", "pdf", "svg"), dpi=150)
        artifacts = gen.generate_all(synthetic_dispatch_df)

        assert isinstance(artifacts, DispatchFigureArtifacts)

        groups = [
            artifacts.price_power_dispatch,
            artifacts.state_of_charge,
            artifacts.dispatch_heatmap,
            artifacts.daily_throughput,
            artifacts.rolling_window_timeline,
            artifacts.soc_density,
            artifacts.price_spread_capture,
        ]

        assert len(groups) == 7

        for g in groups:
            assert set(g.keys()) == {"png", "pdf", "svg"}
            for p in g.values():
                assert p.exists()
                assert p.stat().st_size > 500

    def test_isolated_figures(self, synthetic_dispatch_df, tmp_path):
        gen = DispatchFigureGenerator(output_directory=tmp_path, formats=["png"], dpi=150)

        p_soc = gen.plot_state_of_charge(synthetic_dispatch_df)
        assert "png" in p_soc
        assert p_soc["png"].exists()

        p_timeline = gen.plot_rolling_window_timeline()
        assert "png" in p_timeline
        assert p_timeline["png"].exists()

        p_spread = gen.plot_price_spread_capture(synthetic_dispatch_df)
        assert "png" in p_spread
        assert p_spread["png"].exists()