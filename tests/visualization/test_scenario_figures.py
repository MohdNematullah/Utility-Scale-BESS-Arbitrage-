"""
tests/visualization/test_scenario_figures.py
============================================

Unit Test Suite for Scenario & Sensitivity Figures Module (Part 10.7).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from visualization.scenario_figures import (
    ScenarioFigureArtifacts,
    ScenarioFigureGenerator,
)


@pytest.fixture
def synthetic_scenarios_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"scenario_name": "chem_nmc_baseline", "category": "Battery_Chemistry", "net_revenue_usd": 848333.0, "final_soh": 0.9812, "composite_score": 0.72},
        {"scenario_name": "chem_lfp_stationary", "category": "Battery_Chemistry", "net_revenue_usd": 932814.0, "final_soh": 0.9891, "composite_score": 0.81},
        {"scenario_name": "chem_lto_heavy_cycle", "category": "Battery_Chemistry", "net_revenue_usd": 994495.0, "final_soh": 0.9951, "composite_score": 0.88},
        {"scenario_name": "horizon_12h", "category": "Forecast_Horizon", "net_revenue_usd": 680384.0, "final_soh": 0.9830, "composite_score": 0.61},
        {"scenario_name": "horizon_24h", "category": "Forecast_Horizon", "net_revenue_usd": 795400.0, "final_soh": 0.9820, "composite_score": 0.68},
        {"scenario_name": "horizon_48h", "category": "Forecast_Horizon", "net_revenue_usd": 848333.0, "final_soh": 0.9812, "composite_score": 0.72},
        {"scenario_name": "horizon_72h", "category": "Forecast_Horizon", "net_revenue_usd": 901735.0, "final_soh": 0.9805, "composite_score": 0.76},
        {"scenario_name": "size_25mw_50mwh", "category": "System_Sizing", "net_revenue_usd": 424150.0, "final_soh": 0.9812, "composite_score": 0.45},
        {"scenario_name": "size_100mw_200mwh", "category": "System_Sizing", "net_revenue_usd": 1696600.0, "final_soh": 0.9812, "composite_score": 0.95},
    ])


class TestScenarioFiguresPipeline:
    def test_complete_scenario_figures_generation(self, synthetic_scenarios_df, tmp_path):
        gen = ScenarioFigureGenerator(output_directory=tmp_path, formats=("png", "pdf", "svg"), dpi=150)
        artifacts = gen.generate_all(synthetic_scenarios_df)

        assert isinstance(artifacts, ScenarioFigureArtifacts)

        groups = [
            artifacts.pareto_frontier,
            artifacts.tornado_sensitivity,
            artifacts.scenario_ranking,
            artifacts.scenario_heatmap,
            artifacts.horizon_comparison,
            artifacts.chemistry_comparison,
            artifacts.temperature_sensitivity,
            artifacts.efficiency_sensitivity,
        ]

        assert len(groups) == 8

        for g in groups:
            assert set(g.keys()) == {"png", "pdf", "svg"}
            for p in g.values():
                assert p.exists()
                assert p.stat().st_size > 500

    def test_isolated_horizon_comparison(self, tmp_path):
        gen = ScenarioFigureGenerator(output_directory=tmp_path, formats=["png"], dpi=150)
        benchmarks = [
            {"model": "Persistence", "net_revenue": 680000.0, "degradation": 220000.0, "vcr": 39.0},
            {"model": "Rolling 24h", "net_revenue": 795000.0, "degradation": 240000.0, "vcr": 46.0},
            {"model": "Rolling 48h", "net_revenue": 848000.0, "degradation": 253000.0, "vcr": 55.0},
            {"model": "Perfect Foresight", "net_revenue": 1739000.0, "degradation": 310000.0, "vcr": 100.0},
        ]
        p_res = gen.plot_horizon_comparison(benchmark_records=benchmarks)
        assert "png" in p_res
        assert p_res["png"].exists()

    def test_isolated_tornado_sensitivity(self, tmp_path):
        gen = ScenarioFigureGenerator(output_directory=tmp_path, formats=["png"], dpi=150)
        records = [
            {"parameter": "Sizing", "low_revenue_usd": 400000.0, "high_revenue_usd": 1600000.0, "swing_usd": 1200000.0},
            {"parameter": "Efficiency", "low_revenue_usd": 700000.0, "high_revenue_usd": 900000.0, "swing_usd": 200000.0},
        ]
        p_res = gen.plot_tornado_sensitivity(tornado_records=records)
        assert "png" in p_res
        assert p_res["png"].exists()