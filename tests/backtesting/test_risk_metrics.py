"""
tests/backtesting/test_risk_metrics.py
=====================================

Unit Test Suite for Risk & Downside Volatility Engine (Part 9.3).
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from backtesting.risk_metrics import RiskMetricsEngine, RiskMetricsSummary


@pytest.fixture
def synthetic_risk_dispatch() -> pd.DataFrame:
    n_days = 90
    n_hours = n_days * 24
    np.random.seed(42)

    # Alternate positive arbitrage days with occasional loss/spike days
    daily_profiles = []
    for d in range(n_days):
        base_profit = 2500.0 if d % 7 != 0 else -600.0  # Periodic dip
        noise = np.random.normal(0, 300.0)
        daily_profiles.append((base_profit + noise) / 24.0)

    hourly_net = np.repeat(daily_profiles, 24)

    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n_hours, freq="h", tz="UTC"),
        "net_revenue_usd": hourly_net,
    })


class TestRiskMetricsMath:
    def test_var_and_cvar_mathematical_ordering(self, synthetic_risk_dispatch):
        engine = RiskMetricsEngine(output_directory="backtesting/results/test_risk")
        summary, _ = engine.evaluate(synthetic_risk_dispatch)

        assert summary.historical_var_99_usd >= summary.historical_var_95_usd
        assert summary.cvar_95_usd >= summary.historical_var_95_usd
        assert summary.cvar_99_usd >= summary.historical_var_99_usd

    def test_annualized_volatility_scaling(self, synthetic_risk_dispatch):
        engine = RiskMetricsEngine(output_directory="backtesting/results/test_risk")
        summary, _ = engine.evaluate(synthetic_risk_dispatch)

        expected_ann_vol = summary.daily_volatility_usd * np.sqrt(365.0)
        assert np.isclose(summary.annualized_volatility_usd, expected_ann_vol, atol=0.1)

    def test_drawdown_depth_and_duration_bounds(self, synthetic_risk_dispatch):
        engine = RiskMetricsEngine(output_directory="backtesting/results/test_risk")
        summary, daily_df = engine.evaluate(synthetic_risk_dispatch)

        assert summary.max_drawdown_usd >= 0.0
        assert 0 <= summary.max_drawdown_duration_days <= summary.total_evaluated_days
        assert daily_df["drawdown_usd"].max() == summary.max_drawdown_usd

    def test_sortino_penalizes_downside_only(self):
        # Strictly positive portfolio: downside deviation must be zero
        strictly_positive = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=24 * 10, freq="h"),
            "net_revenue_usd": [100.0] * (24 * 10),
        })
        engine = RiskMetricsEngine(output_directory="backtesting/results/test_risk")
        summary, _ = engine.evaluate(strictly_positive)

        assert summary.downside_deviation_usd == 0.0
        assert summary.max_drawdown_usd == 0.0
        assert summary.sortino_ratio == 99.99


class TestRiskMetricsPipeline:
    def test_evaluate_and_export_artifacts(self, synthetic_risk_dispatch, tmp_path):
        engine = RiskMetricsEngine(output_directory=tmp_path)
        summary, artifacts = engine.evaluate_and_export(synthetic_risk_dispatch)

        assert artifacts.summary_csv.exists()
        assert artifacts.daily_risk_csv.exists()
        assert artifacts.summary_json.exists()

        with open(artifacts.summary_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "sharpe_ratio" in data
        assert "max_drawdown_usd" in data
        assert "cvar_95_usd" in data

        figures = list(artifacts.figures_directory.glob("*.png"))
        assert len(figures) == 5
        figure_names = {f.name for f in figures}
        assert figure_names == {
            "cumulative_drawdown_underwater.png",
            "tail_risk_var_cvar_dist.png",
            "rolling_risk_ratios.png",
            "monthly_risk_return_scatter.png",
            "drawdown_duration_profile.png",
        }