"""
tests/backtesting/test_experiment_framework.py
=============================================

Master Unit Test Suite for  Research Experiment Framework:
- Part 8.5.1: Master Orchestration Utilities & Provenance Metadata
- Part 8.5.2: Predefined 28 Research Scenarios Library
- Part 8.5.3: Comparison Engine, Composite Ranking & Pareto Frontier
- Part 8.5.4: Dashboard Data Aggregator & Financial Waterfall Payloads
- Part 8.5.5: Unified CLI Parser & Command Dispatching

Compatible with:
- Python 3.14+
- pytest 9.1+
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import openpyxl
import numpy as np
import pandas as pd
import pytest

from backtesting.cli import main as cli_main
from backtesting.comparison import ScenarioComparisonEngine
from backtesting.dashboard_data import DashboardDataBuilder
from backtesting.experiment_runner import (
    ExperimentConfig,
    ExperimentMetadata,
    ExperimentRunner,
    MasterRegistryRecord,
    get_git_commit_hash,
    set_deterministic_seed,
)
from backtesting.scenarios import (
    ScenarioCategory,
    build_scenario_configs,
    get_scenario,
    get_scenarios_by_category,
    list_scenarios,
)


# ============================================================================
# Shared Fixtures
# ============================================================================

@pytest.fixture
def synthetic_registry_df() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "experiment_id": "EXP_DOMINATED_A",
            "experiment_name": "test_exp",
            "scenario_name": "low_rev_low_soh",
            "category": "Forecast_Horizon",
            "forecast_horizon": 12,
            "gross_revenue_usd": 900000.0,
            "degradation_cost_usd": 200000.0,
            "net_revenue_usd": 700000.0,
            "net_profit_usd": 400000.0,
            "final_soh": 0.9700,
            "cumulative_efc": 160.0,
            "profit_factor": 3.8,
            "sharpe_ratio": 1.5,
            "status": "Completed",
            "runtime_seconds": 10.0,
            "timestamp": "2026-09-06T12:00:00Z",
        },
        {
            "experiment_id": "EXP_PARETO_PROFIT",
            "experiment_name": "test_exp",
            "scenario_name": "high_rev_mid_soh",
            "category": "Forecast_Horizon",
            "forecast_horizon": 48,
            "gross_revenue_usd": 1400000.0,
            "degradation_cost_usd": 280000.0,
            "net_revenue_usd": 1120000.0,
            "net_profit_usd": 820000.0,
            "final_soh": 0.9810,
            "cumulative_efc": 195.0,
            "profit_factor": 5.4,
            "sharpe_ratio": 2.8,
            "status": "Completed",
            "runtime_seconds": 12.5,
            "timestamp": "2026-09-06T12:00:00Z",
        },
        {
            "experiment_id": "EXP_PARETO_HEALTH",
            "experiment_name": "test_exp",
            "scenario_name": "mid_rev_high_soh",
            "category": "Battery_Chemistry",
            "forecast_horizon": 48,
            "gross_revenue_usd": 1050000.0,
            "degradation_cost_usd": 60000.0,
            "net_revenue_usd": 990000.0,
            "net_profit_usd": 690000.0,
            "final_soh": 0.9950,
            "cumulative_efc": 110.0,
            "profit_factor": 6.8,
            "sharpe_ratio": 3.2,
            "status": "Completed",
            "runtime_seconds": 11.2,
            "timestamp": "2026-09-06T12:00:00Z",
        },
        {
            "experiment_id": "EXP_DOMINATED_B",
            "experiment_name": "test_exp",
            "scenario_name": "low_rev_mid_soh",
            "category": "Thermal_Sensitivity",
            "forecast_horizon": 48,
            "gross_revenue_usd": 850000.0,
            "degradation_cost_usd": 100000.0,
            "net_revenue_usd": 750000.0,
            "net_profit_usd": 450000.0,
            "final_soh": 0.9850,
            "cumulative_efc": 140.0,
            "profit_factor": 4.1,
            "sharpe_ratio": 1.9,
            "status": "Completed",
            "runtime_seconds": 9.8,
            "timestamp": "2026-09-06T12:00:00Z",
        },
    ])


@pytest.fixture
def synthetic_backtest_telemetry() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    n_hours = 168
    dispatch_df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n_hours, freq="h", tz="UTC"),
        "forecast_price": 40.0 + 20.0 * np.sin(np.linspace(0, 14 * np.pi, n_hours)),
        "actual_price": 38.5 + 21.0 * np.sin(np.linspace(0, 14 * np.pi, n_hours)),
        "charge_power_mw": [50.0 if (i % 24) in [2, 3] else 0.0 for i in range(n_hours)],
        "discharge_power_mw": [50.0 if (i % 24) in [18, 19] else 0.0 for i in range(n_hours)],
        "soc_fraction": 0.50,
        "net_revenue_usd": 175.0,
    })

    degradation_df = pd.DataFrame({
        "rolling_window": range(1, 8),
        "remaining_soh": [1.0 - 0.00015 * i for i in range(1, 8)],
        "calendar_loss": [0.0001] * 7,
        "cycle_loss": [0.00005] * 7,
        "cumulative_efc": [3.5 * i for i in range(1, 8)],
        "degradation_cost_usd": [650.0] * 7,
    })

    summary_dict = {
        "gross_revenue_usd": 1102091.72,
        "degradation_cost_usd": 253758.72,
        "net_revenue_usd": 848333.00,
        "fixed_om_cost_usd": 359589.04,
        "variable_om_cost_usd": 19017.97,
        "total_operating_cost_usd": 378607.01,
        "net_operating_profit_usd": 469725.99,
        "profit_factor": 5.103,
        "sharpe_ratio": 2.45,
        "final_soh": 0.9812,
        "capacity_fade": 0.0188,
        "equivalent_full_cycles": 190.17,
        "charging_hours": 264,
        "discharging_hours": 612,
        "idle_hours": 7524,
        "average_soc": 0.613,
    }

    return dispatch_df, degradation_df, summary_dict


# ============================================================================
# 1. Scenarios Library Tests (Part 8.5.2)
# ============================================================================

class TestScenariosLibrary:
    def test_total_predefined_scenarios_count(self):
        scenarios = list_scenarios()
        assert len(scenarios) == 28

    def test_all_seven_categories_represented(self):
        categories = {s.category for s in list_scenarios()}
        expected_categories = {
            ScenarioCategory.HORIZON,
            ScenarioCategory.MODEL,
            ScenarioCategory.CHEMISTRY,
            ScenarioCategory.TEMPERATURE,
            ScenarioCategory.SIZING,
            ScenarioCategory.EFFICIENCY,
            ScenarioCategory.DEGRADATION,
        }
        assert categories == expected_categories

    def test_scenario_retrieval_case_insensitivity(self):
        by_upper = get_scenario("SCN_CHEM_LFP")
        by_lower = get_scenario("scn_chem_lfp")
        by_name = get_scenario("chem_lfp_stationary")
        assert by_upper.scenario_id == by_lower.scenario_id == by_name.scenario_id == "SCN_CHEM_LFP"

    def test_scenario_retrieval_invalid_id_raises_keyerror(self):
        with pytest.raises(KeyError, match="Scenario 'INVALID_SCENARIO_XYZ' not found"):
            get_scenario("INVALID_SCENARIO_XYZ")

    def test_category_filtering(self):
        chem_scenarios = get_scenarios_by_category(ScenarioCategory.CHEMISTRY)
        assert len(chem_scenarios) == 3
        assert {s.scenario_id for s in chem_scenarios} == {
            "SCN_CHEM_NMC",
            "SCN_CHEM_LFP",
            "SCN_CHEM_LTO",
        }

    def test_lfp_chemistry_modifiers(self):
        _, bat_cfg = get_scenario("SCN_CHEM_LFP").instantiate_configs()
        assert bat_cfg.chemistry.chemistry == "Lithium-Ion LFP"
        assert bat_cfg.chemistry.nominal_cycle_life == 7000
        assert bat_cfg.ageing.calendar_loss_per_year == 0.010
        assert bat_cfg.ageing.cycle_loss_per_efc == 0.000028
        assert bat_cfg.replacement.replacement_cost_per_mwh == 130000.0

    def test_lto_chemistry_modifiers(self):
        _, bat_cfg = get_scenario("SCN_CHEM_LTO").instantiate_configs()
        assert "LTO" in bat_cfg.chemistry.chemistry
        assert bat_cfg.chemistry.nominal_cycle_life == 15000
        assert bat_cfg.ageing.calendar_loss_per_year == 0.005
        assert bat_cfg.replacement.replacement_cost_per_mwh == 260000.0
        assert bat_cfg.chemistry.round_trip_efficiency == 0.88

    def test_thermal_stress_modifier(self):
        _, bat_cfg = get_scenario("SCN_TEMP_45C").instantiate_configs()
        assert bat_cfg.ageing.reference_temperature_c == 45.0

    def test_system_sizing_modifier(self):
        _, bat_cfg = get_scenario("SCN_SIZE_100MW_200MWH").instantiate_configs()
        assert bat_cfg.chemistry.nominal_capacity_mwh == 200.0
        assert bat_cfg.chemistry.max_charge_power_mw == 100.0
        assert bat_cfg.chemistry.max_discharge_power_mw == 100.0

    def test_efficiency_sensitivity_modifier(self):
        _, bat_cfg = get_scenario("SCN_EFF_95PCT").instantiate_configs()
        assert bat_cfg.chemistry.round_trip_efficiency == 0.95
        assert np.isclose(bat_cfg.chemistry.charge_efficiency, np.sqrt(0.95))

    def test_build_all_scenario_configs(self):
        configs = build_scenario_configs(base_experiment_name="batch_verification")
        assert len(configs) == 28
        for exp_cfg, bat_cfg in configs:
            assert exp_cfg.experiment_name == "batch_verification"
            assert isinstance(exp_cfg, ExperimentConfig)
            assert bat_cfg.chemistry.nominal_capacity_mwh > 0.0


# ============================================================================
# 2. Comparison & Pareto Engine Tests (Part 8.5.3)
# ============================================================================

class TestComparisonEngine:
    def test_composite_ranking_weights(self, synthetic_registry_df, tmp_path):
        engine = ScenarioComparisonEngine(output_directory=tmp_path)
        ranked = engine.rank_scenarios(synthetic_registry_df)

        assert len(ranked) == 4
        assert "composite_score" in ranked.columns
        assert "rank_overall" in ranked.columns
        assert ranked["rank_overall"].tolist() == [1, 2, 3, 4]
        assert ranked["composite_score"].max() <= 100.0
        assert ranked["composite_score"].min() >= 0.0

    def test_pareto_frontier_math(self, synthetic_registry_df, tmp_path):
        engine = ScenarioComparisonEngine(output_directory=tmp_path)
        pareto_df = engine.compute_pareto_frontier(synthetic_registry_df)

        optimal_ids = set(pareto_df[pareto_df["is_pareto_optimal"]]["experiment_id"])

        assert optimal_ids == {"EXP_PARETO_PROFIT", "EXP_PARETO_HEALTH"}

        dom_a = pareto_df.loc[pareto_df["experiment_id"] == "EXP_DOMINATED_A", "is_pareto_optimal"].iloc[0]
        assert not dom_a

        dom_b = pareto_df.loc[pareto_df["experiment_id"] == "EXP_DOMINATED_B", "is_pareto_optimal"].iloc[0]
        assert not dom_b

    def test_sensitivity_tables_generation(self, synthetic_registry_df, tmp_path):
        engine = ScenarioComparisonEngine(output_directory=tmp_path)
        tables = engine.build_sensitivity_tables(synthetic_registry_df)

        assert "Forecast_Horizon" in tables
        assert "Battery_Chemistry" in tables
        assert "Thermal_Sensitivity" in tables

        horizon_tab = tables["Forecast_Horizon"]
        assert "marginal_revenue_delta" in horizon_tab.columns
        assert len(horizon_tab) == 2

    def test_full_evaluation_and_file_exports(self, synthetic_registry_df, tmp_path):
        engine = ScenarioComparisonEngine(output_directory=tmp_path)
        artifacts = engine.evaluate_and_export(synthetic_registry_df)

        assert artifacts.ranking_csv.exists()
        assert artifacts.pareto_csv.exists()
        assert artifacts.summary_json.exists()
        assert artifacts.sensitivity_excel.exists()

        wb = openpyxl.load_workbook(artifacts.sensitivity_excel, read_only=True)
        expected_sheets = {"Master_Ranking", "Pareto_Frontier", "Forecast_Horizon"}
        assert expected_sheets.issubset(set(wb.sheetnames))
        wb.close()

        with open(artifacts.summary_json, "r", encoding="utf-8") as f:
            summary = json.load(f)
        assert summary["total_scenarios_evaluated"] == 4
        assert summary["pareto_optimal_count"] == 2

        figures = list(artifacts.figures_directory.glob("*.png"))
        assert len(figures) == 3
        figure_names = {f.name for f in figures}
        assert figure_names == {
            "pareto_frontier.png",
            "scenario_ranking_waterfall.png",
            "sensitivity_panels.png",
        }


# ============================================================================
# 3. Dashboard Data Builder Tests (Part 8.5.4)
# ============================================================================

class TestDashboardDataBuilder:
    def test_kpi_payload_structure(self, synthetic_backtest_telemetry, tmp_path):
        _, _, summary_dict = synthetic_backtest_telemetry
        builder = DashboardDataBuilder(output_directory=tmp_path)
        kpi_file = builder.build_kpis(summary_dict)

        assert kpi_file.exists()
        with open(kpi_file, "r", encoding="utf-8") as f:
            kpis = json.load(f)

        assert "financial" in kpis
        assert "battery_health" in kpis
        assert "operations" in kpis
        assert kpis["financial"]["net_revenue_usd"] == 848333.00
        assert kpis["financial"]["net_profit_usd"] == 469725.99
        assert kpis["battery_health"]["final_soh"] == 0.9812

    def test_financial_waterfall_step_arithmetic(self, synthetic_backtest_telemetry, tmp_path):
        _, _, summary_dict = synthetic_backtest_telemetry
        builder = DashboardDataBuilder(output_directory=tmp_path)
        wf_file = builder.build_financial_waterfall(summary_dict)

        assert wf_file.exists()
        with open(wf_file, "r", encoding="utf-8") as f:
            wf = json.load(f)

        assert len(wf) == 5
        gross = wf[0]["delta"]
        wear = wf[1]["delta"]
        fixed_om = wf[2]["delta"]
        var_om = wf[3]["delta"]
        ebitda = wf[4]["delta"]

        assert gross == 1102091.72
        assert wear == -253758.72
        assert fixed_om == -359589.04
        assert var_om == -19017.97
        assert np.isclose(gross + wear + fixed_om + var_om, ebitda, atol=0.05)

    def test_dispatch_and_residuals_feeds(self, synthetic_backtest_telemetry, tmp_path):
        dispatch_df, _, _ = synthetic_backtest_telemetry
        builder = DashboardDataBuilder(output_directory=tmp_path)

        disp_file = builder.build_dispatch_timeseries(dispatch_df)
        resid_file = builder.build_forecast_residuals(dispatch_df)

        assert disp_file.exists()
        assert resid_file.exists()

        df_disp = pd.read_csv(disp_file)
        assert len(df_disp) == 168
        assert "net_power_mw" in df_disp.columns
        assert "cumulative_revenue_usd" in df_disp.columns

        df_resid = pd.read_csv(resid_file)
        assert len(df_resid) == 168
        assert "residual_usd" in df_resid.columns
        assert "hour_of_day" in df_resid.columns

    def test_export_all_dashboard_artifacts(self, synthetic_backtest_telemetry, tmp_path):
        dispatch_df, degradation_df, summary_dict = synthetic_backtest_telemetry
        builder = DashboardDataBuilder(output_directory=tmp_path)
        datasets = builder.export_all(dispatch_df, degradation_df, summary_dict)

        assert datasets.kpis_json.exists()
        assert datasets.dispatch_timeseries.exists()
        assert datasets.forecast_residuals.exists()
        assert datasets.soh_evolution.exists()
        assert datasets.financial_waterfall.exists()
        assert datasets.scenario_matrix.exists()


# ============================================================================
# 4. Orchestration & Provenance Tests (Part 8.5.1)
# ============================================================================

class TestExperimentRunnerUtilities:
    def test_deterministic_seed_consistency(self):
        set_deterministic_seed(1234)
        val_a = np.random.rand(5)
        set_deterministic_seed(1234)
        val_b = np.random.rand(5)
        np.testing.assert_array_equal(val_a, val_b)

    def test_git_commit_hash_format(self):
        commit = get_git_commit_hash()
        assert isinstance(commit, str)
        assert len(commit) >= 7

    def test_master_registry_record_creation(self, tmp_path):
        reg_file = tmp_path / "test_registry.csv"
        runner = ExperimentRunner(registry_path=reg_file)

        assert reg_file.exists()
        df_init = pd.read_csv(reg_file)
        assert "experiment_id" in df_init.columns

        record = MasterRegistryRecord(
            experiment_id="TEST_EXP_001",
            experiment_name="unit_test",
            scenario_name="baseline",
            category="Baseline",
            forecast_horizon=48,
            gross_revenue_usd=1000.0,
            degradation_cost_usd=200.0,
            net_revenue_usd=800.0,
            net_profit_usd=600.0,
            final_soh=0.985,
            cumulative_efc=15.0,
            profit_factor=4.5,
            sharpe_ratio=2.2,
            status="Completed",
            runtime_seconds=5.4,
            timestamp="2026-09-06T12:00:00Z",
        )

        runner._update_registry(record)
        df_updated = pd.read_csv(reg_file)
        assert len(df_updated) == 1
        assert df_updated.iloc[0]["experiment_id"] == "TEST_EXP_001"

        record.net_revenue_usd = 850.0
        runner._update_registry(record)
        df_reupdated = pd.read_csv(reg_file)
        assert len(df_reupdated) == 1
        assert df_reupdated.iloc[0]["net_revenue_usd"] == 850.0


# ============================================================================
# 5. Command-Line Interface Tests (Part 8.5.5)
# ============================================================================

class TestCLIParser:
    def test_cli_list_command(self, capsys):
        test_args = ["cli.py", "list", "--category", "Battery_Chemistry"]
        with patch.object(sys, "argv", test_args):
            cli_main()
        captured = capsys.readouterr()
        assert "SCN_CHEM_NMC" in captured.out
        assert "SCN_CHEM_LFP" in captured.out
        assert "SCN_CHEM_LTO" in captured.out

    def test_cli_compare_command(self, synthetic_registry_df, tmp_path):
        registry_file = tmp_path / "test_reg.csv"
        synthetic_registry_df.to_csv(registry_file, index=False)
        output_dir = tmp_path / "cli_comparison_out"

        test_args = [
            "cli.py",
            "compare",
            "--registry-file",
            str(registry_file),
            "--output-dir",
            str(output_dir),
        ]
        with patch.object(sys, "argv", test_args):
            cli_main()

        assert (output_dir / "scenario_ranking.csv").exists()
        assert (output_dir / "pareto_optimal_scenarios.csv").exists()
        assert (output_dir / "sensitivity_analysis.xlsx").exists()
        assert (output_dir / "comparison_summary.json").exists()