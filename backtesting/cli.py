"""
backtesting/cli.py
==================

Unified Command-Line Interface for  Research Pipeline (Part 8.5.5)

Supported Subcommands:
- list               : Lists all 28 predefined research scenarios.
- run <id>           : Executes a specific scenario by ID.
- run-batch          : Executes a category or batch of scenarios (supports parallel workers).
- compare            : Evaluates, ranks, and plots cross-scenario Pareto frontiers.
- export-dashboard   : Builds standardized datasets for Streamlit/Plotly/Power BI.
- resume             : Resumes an interrupted scenario from its checkpoint file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from backtesting.comparison import ScenarioComparisonEngine
from backtesting.dashboard_data import DashboardDataBuilder
from backtesting.experiment_runner import ExperimentRunner, get_git_commit_hash
from backtesting.scenarios import (
    ScenarioCategory,
    build_scenario_configs,
    get_scenario,
    get_scenarios_by_category,
    list_scenarios,
)
from data.loader import MarketDataLoader
from features.engineering import FeatureEngineer
from forecasting.models import XGBoostForecaster


def _load_runtime_environment():
    """Initializes and returns market data, engineered features, and forecasting model."""
    loader = MarketDataLoader()
    engineer = FeatureEngineer()

    raw_path = Path("data/raw/ercot_prices.csv")
    if not raw_path.exists():
        raise FileNotFoundError(f"Market dataset not found at {raw_path}")

    market_df = loader.load_csv(raw_path)
    features_df = engineer.transform(market_df)

    if "timestamp" in features_df.columns:
        features_df["timestamp"] = pd.to_datetime(features_df["timestamp"], utc=True)
        features_df = features_df.sort_values("timestamp").set_index("timestamp")
    features_df.index = pd.DatetimeIndex(features_df.index)

    model_path = Path("forecasting/saved_models/xgboost_dayahead.pkl")
    if not model_path.exists():
        raise FileNotFoundError(f"Forecasting model not found at {model_path}. Run training first.")

    forecaster = XGBoostForecaster()
    forecaster.load(model_path)
    return forecaster, features_df


def handle_list(args):
    scenarios = list_scenarios()
    print("\n" + "=" * 80)
    print(f" PREDEFINED RESEARCH SCENARIOS CATALOG ({len(scenarios)} TOTAL)")
    print("=" * 80)
    print(f"{'Scenario ID':<30} {'Category':<24} {'Scenario Name'}")
    print("-" * 80)
    for s in scenarios:
        if args.category and s.category.value.lower() != args.category.lower():
            continue
        print(f"{s.scenario_id:<30} {s.category.value:<24} {s.scenario_name}")
    print("=" * 80 + "\n")


def handle_run(args):
    forecaster, features_df = _load_runtime_environment()
    scenario_def = get_scenario(args.scenario_id)
    exp_cfg, bat_cfg = scenario_def.instantiate_configs(base_exp_name=args.experiment_name)

    if args.force:
        exp_cfg.force_rerun = True

    runner = ExperimentRunner()
    result = runner.run_experiment(exp_cfg, forecaster, features_df, bat_cfg)

    print("\n" + "=" * 70)
    print(f"SCENARIO EXECUTION COMPLETED: {scenario_def.scenario_id}")
    print("=" * 70)
    for k, v in result.metrics.summary.items():
        print(f"  {k:<35}: {v}")
    print(f"\nArtifacts preserved in: {result.config.experiment_directory}\n")


def handle_run_batch(args):
    forecaster, features_df = _load_runtime_environment()

    if args.category:
        scenarios_subset = get_scenarios_by_category(args.category)
        configs = [s.instantiate_configs(base_exp_name=args.experiment_name) for s in scenarios_subset]
    else:
        configs = build_scenario_configs(base_experiment_name=args.experiment_name)

    runner = ExperimentRunner()

    if args.workers > 1:
        # Multiprocessing parallel batch execution
        forecaster_path = Path("forecasting/saved_models/xgboost_dayahead.pkl")
        features_path = Path("data/raw/ercot_prices.csv")
        runner.run_batch_parallel(
            scenarios=configs,
            forecaster_path=forecaster_path,
            features_path=features_path,
            max_workers=args.workers,
        )
    else:
        runner.run_batch_sequential(configs, forecaster, features_df)

    print("\nBatch execution complete. Triggering scenario comparison engine...")
    comp_engine = ScenarioComparisonEngine()
    comp_engine.evaluate_and_export(runner.registry_path)


def handle_compare(args):
    registry_file = Path(args.registry_file)
    comp_engine = ScenarioComparisonEngine(output_directory=args.output_dir)
    artifacts = comp_engine.evaluate_and_export(registry_file)

    print("\n" + "=" * 70)
    print("SCENARIO COMPARISON COMPLETED")
    print("=" * 70)
    print(f"Ranking CSV       : {artifacts.ranking_csv}")
    print(f"Pareto CSV        : {artifacts.pareto_csv}")
    print(f"Sensitivity Excel : {artifacts.sensitivity_excel}")
    print(f"Summary JSON      : {artifacts.summary_json}")
    print(f"Figures Saved In  : {artifacts.figures_directory}\n")


def handle_export_dashboard(args):
    dispatch_path = Path("backtesting/results/dispatch_history.csv")
    degradation_path = Path("backtesting/results/degradation_history.csv")
    summary_path = Path("backtesting/results/rolling_summary.csv")
    ranking_path = Path("backtesting/results/comparison/scenario_ranking.csv")

    if not (dispatch_path.exists() and degradation_path.exists() and summary_path.exists()):
        raise FileNotFoundError("Run a backtest experiment first before exporting dashboard data.")

    dispatch_df = pd.read_csv(dispatch_path)
    degradation_df = pd.read_csv(degradation_path)
    summary_dict = pd.read_csv(summary_path).iloc[0].to_dict()
    ranking_df = pd.read_csv(ranking_path) if ranking_path.exists() else None

    builder = DashboardDataBuilder(output_directory=args.output_dir)
    datasets = builder.export_all(dispatch_df, degradation_df, summary_dict, ranking_df)

    print("\n" + "=" * 70)
    print("DASHBOARD DATASETS EXPORTED")
    print("=" * 70)
    print(f"Directory: {builder.output_dir}")
    print(f"  â€¢ {datasets.kpis_json.name}")
    print(f"  â€¢ {datasets.dispatch_timeseries.name}")
    print(f"  â€¢ {datasets.forecast_residuals.name}")
    print(f"  â€¢ {datasets.soh_evolution.name}")
    print(f"  â€¢ {datasets.financial_waterfall.name}")
    print(f"  â€¢ {datasets.scenario_matrix.name}\n")


def handle_resume(args):
    exp_dir = Path("backtesting/results/experiments") / args.experiment_name / args.scenario_name
    checkpoint_file = exp_dir / "checkpoint.json"

    if not checkpoint_file.exists():
        print(f"No checkpoint file found at {checkpoint_file}. Starting fresh execution...")

    forecaster, features_df = _load_runtime_environment()
    scenario_def = get_scenario(args.scenario_name)
    exp_cfg, bat_cfg = scenario_def.instantiate_configs(base_exp_name=args.experiment_name)
    exp_cfg.force_rerun = False

    runner = ExperimentRunner()
    result = runner.run_experiment(exp_cfg, forecaster, features_df, bat_cfg)
    print(f"Scenario successfully resumed/completed. Final SOH: {result.metrics.summary.get('final_soh')}")


def main():
    parser = argparse.ArgumentParser(
        prog="python -m backtesting.cli",
        description=" Research Backtesting & Scenario Experiment Orchestrator",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. List Command
    p_list = subparsers.add_parser("list", help="List predefined research scenarios.")
    p_list.add_argument("--category", type=str, default=None, help="Filter scenarios by category.")
    p_list.set_defaults(func=handle_list)

    # 2. Run Command
    p_run = subparsers.add_parser("run", help="Run a specific scenario by ID.")
    p_run.add_argument("scenario_id", type=str, help="Scenario ID (e.g., SCN_CHEM_LFP).")
    p_run.add_argument("--experiment-name", type=str, default="cli_experiment", help="Experiment name container.")
    p_run.add_argument("--force", action="store_true", help="Force re-run even if checkpoint exists.")
    p_run.set_defaults(func=handle_run)

    # 3. Run-Batch Command
    p_batch = subparsers.add_parser("run-batch", help="Run a batch of scenarios.")
    p_batch.add_argument("--category", type=str, default=None, help="Specific category to run.")
    p_batch.add_argument("--workers", type=int, default=1, help="Number of parallel worker processes.")
    p_batch.add_argument("--experiment-name", type=str, default="_batch", help="Experiment name container.")
    p_batch.set_defaults(func=handle_run_batch)

    # 4. Compare Command
    p_comp = subparsers.add_parser("compare", help="Compare and rank all scenarios.")
    p_comp.add_argument("--registry-file", type=str, default="backtesting/results/experiments_registry.csv")
    p_comp.add_argument("--output-dir", type=str, default="backtesting/results/comparison")
    p_comp.set_defaults(func=handle_compare)

    # 5. Export-Dashboard Command
    p_dash = subparsers.add_parser("export-dashboard", help="Build datasets for visualization dashboards.")
    p_dash.add_argument("--output-dir", type=str, default="backtesting/results/dashboard")
    p_dash.set_defaults(func=handle_export_dashboard)

    # 6. Resume Command
    p_res = subparsers.add_parser("resume", help="Resume an interrupted experiment.")
    p_res.add_argument("scenario_name", type=str, help="Scenario name to resume.")
    p_res.add_argument("--experiment-name", type=str, default="cli_experiment")
    p_res.set_defaults(func=handle_resume)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()