"""
tests/visualization/test_risk_figures.py
========================================

Unit Test Suite for Risk Figures Module (Part 10.6).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from visualization.risk_figures import (
    RiskFigureArtifacts,
    RiskFigureGenerator,
)


@pytest.fixture
def synthetic_risk_data() -> np.ndarray:
    np.random.seed(42)
    n_days = 90
    base = 2500.0
    noise = np.random.normal(0, 450.0, n_days)
    dips = np.where(np.arange(n_days) % 12 == 0, -800.0, 0.0)
    return np.asarray(base + noise + dips, dtype=float)


class TestRiskFiguresPipeline:
    def test_complete_risk_figures_generation(self, synthetic_risk_data, tmp_path):
        dummy_df = pd.DataFrame({"net_revenue_usd": [100.0] * 24})
        gen = RiskFigureGenerator(output_directory=tmp_path, formats=("png", "pdf", "svg"), dpi=150)
        artifacts = gen.generate_all(
            dispatch_df=dummy_df,
            daily_pnl=synthetic_risk_data,
        )

        assert isinstance(artifacts, RiskFigureArtifacts)

        groups = [
            artifacts.daily_profit_dist,
            artifacts.drawdown_curve,
            artifacts.var_cvar_tail,
            artifacts.rolling_volatility,
            artifacts.return_distribution,
        ]

        assert len(groups) == 5

        for g in groups:
            assert set(g.keys()) == {"png", "pdf", "svg"}
            for p in g.values():
                assert p.exists()
                assert p.stat().st_size > 500

    def test_isolated_risk_figures(self, synthetic_risk_data, tmp_path):
        gen = RiskFigureGenerator(output_directory=tmp_path, formats=["png"], dpi=150)

        p_drawdown = gen.plot_drawdown_curve(synthetic_risk_data)
        assert "png" in p_drawdown
        assert p_drawdown["png"].exists()

        p_tail = gen.plot_var_cvar_tail(synthetic_risk_data)
        assert "png" in p_tail
        assert p_tail["png"].exists()

        p_moments = gen.plot_return_distribution(synthetic_risk_data)
        assert "png" in p_moments
        assert p_moments["png"].exists()