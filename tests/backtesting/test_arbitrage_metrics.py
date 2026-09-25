"""
tests/backtesting/test_arbitrage_metrics.py
==========================================

Unit Test Suite for Arbitrage Metrics & Cycle Economics Module (Part 9.2).
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from backtesting.arbitrage_metrics import ArbitrageKPIs, ArbitrageMetricsEngine


@pytest.fixture
def synthetic_arbitrage_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    n_hours = 168  # 7 days
    prices = [20.0, 22.0, 50.0, 60.0] * 42

    chg = [50.0 if p <= 22.0 else 0.0 for p in prices]
    dis = [45.0 if p >= 50.0 else 0.0 for p in prices]

    dispatch_df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n_hours, freq="h", tz="UTC"),
        "actual_price": prices,
        "charge_power_mw": chg,
        "discharge_power_mw": dis,
        "net_revenue_usd": (np.array(dis) - np.array(chg)) * np.array(prices),
    })

    degradation_df = pd.DataFrame({
        "rolling_window": range(1, 8),
        "window_efc": [3.8] * 7,
        "degradation_cost_usd": [500.0] * 7,
    })

    return dispatch_df, degradation_df


class TestArbitrageMetricsMath:
    def test_revenue_waterfall_conservation(self, synthetic_arbitrage_data):
        dispatch_df, degradation_df = synthetic_arbitrage_data
        engine = ArbitrageMetricsEngine(output_directory="backtesting/results/test_arb")
        kpis, _, _ = engine.evaluate(
            dispatch_df=dispatch_df,
            degradation_df=degradation_df,
            nominal_capacity_mwh=100.0,
            rated_power_mw=50.0,
        )

        assert np.isclose(kpis.gross_revenue_usd - kpis.degradation_cost_usd, kpis.net_revenue_usd, atol=0.01)
        expected_ebitda = kpis.net_revenue_usd - (kpis.fixed_om_cost_usd + kpis.variable_om_cost_usd)
        assert np.isclose(expected_ebitda, kpis.net_operating_profit_usd, atol=0.01)

    def test_unit_normalized_economics(self, synthetic_arbitrage_data):
        dispatch_df, degradation_df = synthetic_arbitrage_data
        engine = ArbitrageMetricsEngine(output_directory="backtesting/results/test_arb")
        kpis, _, _ = engine.evaluate(
            dispatch_df=dispatch_df,
            degradation_df=degradation_df,
            nominal_capacity_mwh=100.0,
            rated_power_mw=50.0,
        )

        assert kpis.gross_revenue_per_mwh_throughput > 0.0
        assert kpis.net_revenue_per_mwh_throughput < kpis.gross_revenue_per_mwh_throughput
        assert kpis.gross_revenue_per_efc > kpis.net_revenue_per_efc
        assert np.isclose(
            kpis.gross_revenue_per_efc - kpis.degradation_cost_per_efc,
            kpis.net_revenue_per_efc,
            atol=0.05,
        )

    def test_spread_capture_logic(self, synthetic_arbitrage_data):
        dispatch_df, degradation_df = synthetic_arbitrage_data
        engine = ArbitrageMetricsEngine(output_directory="backtesting/results/test_arb")
        kpis, _, _ = engine.evaluate(
            dispatch_df=dispatch_df,
            degradation_df=degradation_df,
            nominal_capacity_mwh=100.0,
            rated_power_mw=50.0,
        )

        assert kpis.avg_discharge_price_usd_per_mwh > kpis.avg_charge_price_usd_per_mwh
        assert kpis.realized_spread_usd_per_mwh > 0.0
        assert 0.0 < kpis.spread_capture_ratio_pct <= 100.0

    def test_duty_cycle_conservation(self, synthetic_arbitrage_data):
        dispatch_df, degradation_df = synthetic_arbitrage_data
        engine = ArbitrageMetricsEngine(output_directory="backtesting/results/test_arb")
        kpis, _, _ = engine.evaluate(
            dispatch_df=dispatch_df,
            degradation_df=degradation_df,
            nominal_capacity_mwh=100.0,
            rated_power_mw=50.0,
        )

        total_hours = kpis.charging_hours + kpis.discharging_hours + kpis.idle_hours
        assert total_hours == len(dispatch_df)
        assert 0.0 <= kpis.idle_fraction_pct <= 100.0

    def test_zero_discharge_safety(self):
        n_hours = 24
        dispatch_df = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=n_hours, freq="h"),
            "actual_price": [20.0] * n_hours,
            "charge_power_mw": [0.0] * n_hours,
            "discharge_power_mw": [0.0] * n_hours,
        })
        engine = ArbitrageMetricsEngine(output_directory="backtesting/results/test_arb")
        kpis, _, _ = engine.evaluate(dispatch_df)

        assert kpis.gross_revenue_usd == 0.0
        assert kpis.round_trip_efficiency_pct == 0.0
        assert kpis.realized_spread_usd_per_mwh == 0.0
        assert kpis.idle_hours == 24


class TestArbitrageMetricsPipeline:
    def test_evaluate_and_export_artifacts(self, synthetic_arbitrage_data, tmp_path):
        dispatch_df, degradation_df = synthetic_arbitrage_data
        engine = ArbitrageMetricsEngine(output_directory=tmp_path)
        kpis, artifacts = engine.evaluate_and_export(
            dispatch_df=dispatch_df,
            degradation_df=degradation_df,
            nominal_capacity_mwh=100.0,
            rated_power_mw=50.0,
        )

        assert artifacts.kpis_csv.exists()
        assert artifacts.monthly_csv.exists()
        assert artifacts.daily_csv.exists()
        assert artifacts.kpis_json.exists()

        with open(artifacts.kpis_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "gross_revenue_usd" in data
        assert "net_operating_profit_usd" in data
        assert "realized_spread_usd_per_mwh" in data

        figures = list(artifacts.figures_directory.glob("*.png"))
        assert len(figures) == 5
        figure_names = {f.name for f in figures}
        assert figure_names == {
            "arbitrage_waterfall_breakdown.png",
            "monthly_spread_vs_revenue.png",
            "cycle_economics_scatter.png",
            "daily_spread_capture_distribution.png",
            "dispatch_duration_curve.png",
        }