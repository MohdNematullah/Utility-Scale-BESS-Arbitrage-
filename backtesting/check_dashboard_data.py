"""
backtesting/check_dashboard_data.py
===================================

Verification script for Dashboard Data Builder (Part 8.5.4).
"""

import json
from pathlib import Path
import pandas as pd

from backtesting.dashboard_data import DashboardDataBuilder

LINE = "=" * 75

print(LINE)
print(" DASHBOARD DATA BUILDER CHECK")
print(LINE)

dispatch_path = Path("backtesting/results/dispatch_history.csv")
degradation_path = Path("backtesting/results/degradation_history.csv")
summary_path = Path("backtesting/results/rolling_summary.csv")
metrics_path = Path("backtesting/results/metrics_summary.csv")
ranking_path = Path("backtesting/results/comparison/scenario_ranking.csv")

if not (dispatch_path.exists() and degradation_path.exists()):
    raise FileNotFoundError("Run 'python -m backtesting.check_engine' first.")

dispatch_df = pd.read_csv(dispatch_path)
degradation_df = pd.read_csv(degradation_path)

if metrics_path.exists():
    metrics_df = pd.read_csv(metrics_path)
    summary_dict = dict(zip(metrics_df["metric"], metrics_df["value"]))
elif summary_path.exists():
    summary_dict = pd.read_csv(summary_path).iloc[0].to_dict()
else:
    raise FileNotFoundError("Neither metrics_summary.csv nor rolling_summary.csv found.")

ranking_df = pd.read_csv(ranking_path) if ranking_path.exists() else None

builder = DashboardDataBuilder(output_directory="backtesting/results/dashboard")
datasets = builder.export_all(dispatch_df, degradation_df, summary_dict, ranking_df)

print("EXPORTED DASHBOARD ARTIFACTS")
print("-" * 75)
print(f"Overview KPIs JSON     : {datasets.kpis_json.exists()} ({datasets.kpis_json})")
print(f"Dispatch Timeseries    : {datasets.dispatch_timeseries.exists()} ({datasets.dispatch_timeseries})")
print(f"Forecast Residuals     : {datasets.forecast_residuals.exists()} ({datasets.forecast_residuals})")
print(f"SOH Evolution Feed     : {datasets.soh_evolution.exists()} ({datasets.soh_evolution})")
print(f"Financial Waterfall    : {datasets.financial_waterfall.exists()} ({datasets.financial_waterfall})")
print(f"Scenario Matrix Feed   : {datasets.scenario_matrix.exists()} ({datasets.scenario_matrix})")

print("\n" + "-" * 75)
print("FINANCIAL WATERFALL PREVIEW")
print("-" * 75)
with open(datasets.financial_waterfall, "r", encoding="utf-8") as f:
    wf = json.load(f)
for step in wf:
    print(f"  {step['label']:<32} : {step['delta']:>12,.2f} USD ({step['type']})")

print("\n" + LINE)
print("Dashboard dataset builder verified successfully ✓")
print(LINE)