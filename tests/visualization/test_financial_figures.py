"""
tests/visualization/test_financial_figures.py
=============================================

Unit Test Suite for Financial Figures Module (Part 10.5).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from visualization.financial_figures import (
    FinancialFigureArtifacts,
    FinancialFigureGenerator,
)


@pytest.fixture
def synthetic_finance_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    np.random.seed(42)
    n = 240
    prices = 35.0 + 15.0 * np.sin(np.linspace(0, 10 * 2 * np.pi, n))
    chg = np.array([40.0 if (i % 24) in [1, 2] else 0.0 for i in range(n)])
    dis = np.array([35.0 if (i % 24) in [17, 18] else 0.0 for i in range(n)])

    dispatch_df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC"),
        "actual_price": prices,
        "charge_power_mw": chg,
        "discharge_power_mw": dis,
        "net_revenue_usd": (dis - chg) * prices,
    })

    degradation_df = pd.DataFrame({
        "rolling_window": range(1, 11),
        "degradation_cost_usd": [700.0] * 10,
    })

    summary_dict = {
        "gross_revenue_usd": 100000.0,
        "degradation_cost_usd": 20000.0,
        "fixed_om_cost_usd": 25000.0,
        "variable_om_cost_usd": 2000.0,
        "net_operating_profit_usd": 53000.0,
    }

    return dispatch_df, degradation_df, summary_dict


class TestFinancialFiguresPipeline:
    def test_complete_financial_figures_generation(self, synthetic_finance_data, tmp_path):
        dispatch_df, degradation_df, summary_dict = synthetic_finance_data
        gen = FinancialFigureGenerator(output_directory=tmp_path, formats=("png", "pdf", "svg"), dpi=150)
        artifacts = gen.generate_all(
            dispatch_df=dispatch_df,
            degradation_df=degradation_df,
            summary_dict=summary_dict,
        )

        assert isinstance(artifacts, FinancialFigureArtifacts)

        groups = [
            artifacts.cumulative_revenue,
            artifacts.daily_revenue,
            artifacts.revenue_waterfall,
            artifacts.revenue_distribution,
            artifacts.monthly_revenue,
            artifacts.lifetime_value_breakdown,
        ]

        assert len(groups) == 6

        for g in groups:
            assert set(g.keys()) == {"png", "pdf", "svg"}
            for p in g.values():
                assert p.exists()
                assert p.stat().st_size > 500

    def test_isolated_figures(self, synthetic_finance_data, tmp_path):
        dispatch_df, degradation_df, summary_dict = synthetic_finance_data
        gen = FinancialFigureGenerator(output_directory=tmp_path, formats=["png"], dpi=150)

        p_wf = gen.plot_revenue_waterfall(summary_dict)
        assert "png" in p_wf
        assert p_wf["png"].exists()

        p_life = gen.plot_lifetime_value_breakdown(annual_net_ebitda=500000.0)
        assert "png" in p_life
        assert p_life["png"].exists()