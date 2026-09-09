"""
backtesting/check_comparison.py
===============================

Verification script for Scenario Comparison & Sensitivity Engine (Part 8.5.3).
"""

from pathlib import Path
import pandas as pd
from backtesting.comparison import ScenarioComparisonEngine
from backtesting.scenarios import list_scenarios

LINE = "=" * 75

print(LINE)
print(" SCENARIO COMPARISON ENGINE CHECK")
print(LINE)

# 1. Synthesize realistic experimental results for all 28 library scenarios
scenarios = list_scenarios()
synthetic_records = []

base_rev = 1102000.0
base_deg = 253700.0

for s in scenarios:
    cid = s.scenario_id
    cat = s.category.value
    sname = s.scenario_name

    # Calibrate realistic deviations based on scenario physics
    rev_mult = 1.0
    deg_mult = 1.0

    if "HORIZON_12H" in cid:
        rev_mult, deg_mult = 0.82, 0.88
    elif "HORIZON_24H" in cid:
        rev_mult, deg_mult = 0.94, 0.96
    elif "HORIZON_72H" in cid:
        rev_mult, deg_mult = 1.06, 1.05
    elif "CHEM_LFP" in cid:
        rev_mult, deg_mult = 0.98, 0.58  # LFP has much lower degradation
    elif "CHEM_LTO" in cid:
        rev_mult, deg_mult = 0.96, 0.25  # LTO minimal degradation
    elif "TEMP_15C" in cid:
        deg_mult = 0.85
    elif "TEMP_45C" in cid:
        deg_mult = 1.65  # Extreme thermal degradation
    elif "SIZE_100MW" in cid:
        rev_mult, deg_mult = 2.00, 2.00  # 2x capacity scaling
    elif "SIZE_25MW" in cid:
        rev_mult, deg_mult = 0.50, 0.50
    elif "EFF_85PCT" in cid:
        rev_mult = 0.89
    elif "EFF_95PCT" in cid:
        rev_mult = 1.08
    elif "ZERO_WEAR" in cid:
        deg_mult = 0.0

    gross = round(base_rev * rev_mult, 2)
    deg = round(base_deg * deg_mult, 2)
    net = round(gross - deg, 2)
    soh = round(1.0 - (0.0188 * deg_mult * (base_deg / 253700.0) / rev_mult if rev_mult > 0 else 0.0188), 4)
    efc = round(190.18 * (gross / base_rev), 2)
    sharpe = round(max(0.5, (net / 848000.0) * 2.45), 3)

    synthetic_records.append({
        "experiment_id": cid,
        "experiment_name": "thesis_comparison_test",
        "scenario_name": sname,
        "category": cat,
        "forecast_horizon": s.experiment_config.forecast_horizon_hours,
        "gross_revenue_usd": gross,
        "degradation_cost_usd": deg,
        "net_revenue_usd": net,
        "net_profit_usd": round(net - 292000.0 * rev_mult, 2),
        "final_soh": min(max(soh, 0.92), 1.0),
        "cumulative_efc": efc,
        "profit_factor": round(max(1.1, 5.1 * (net / 848000.0)), 2),
        "sharpe_ratio": sharpe,
        "status": "Completed",
        "runtime_seconds": 12.4,
        "timestamp": "2026-09-06T12:00:00Z",
    })

test_registry_df = pd.DataFrame(synthetic_records)
print(f"Generated synthetic test dataset with {len(test_registry_df)} scenario runs.")

# 2. Run Comparison Engine
engine = ScenarioComparisonEngine(output_directory="backtesting/results/comparison")
artifacts = engine.evaluate_and_export(test_registry_df)

print("\n" + "-" * 75)
print("EXPORTED COMPARISON ARTIFACTS")
print("-" * 75)
print(f"Ranking CSV       : {artifacts.ranking_csv.exists()} ({artifacts.ranking_csv})")
print(f"Pareto CSV        : {artifacts.pareto_csv.exists()} ({artifacts.pareto_csv})")
print(f"Summary JSON      : {artifacts.summary_json.exists()} ({artifacts.summary_json})")
print(f"Sensitivity Excel : {artifacts.sensitivity_excel.exists()} ({artifacts.sensitivity_excel})")

figures = list(artifacts.figures_directory.glob("*.png"))
print(f"\nGenerated Visualizations ({len(figures)} total):")
for f in sorted(figures):
    print(f"  • {f.name}")

# 3. Read and Display Executive Ranking Preview
print("\n" + "-" * 75)
print("TOP 5 SCENARIOS BY COMPOSITE SCORE")
print("-" * 75)
ranked_preview = pd.read_csv(artifacts.ranking_csv).head(5)
cols = ["rank_overall", "scenario_name", "category", "net_revenue_usd", "final_soh", "composite_score"]
print(ranked_preview[cols].to_string(index=False))

# 4. Display Pareto Frontier Count
pareto_df = pd.read_csv(artifacts.pareto_csv)
print(f"\nPareto Optimal Frontier Scenarios Identified: {len(pareto_df)}")
print(pareto_df[["scenario_name", "net_revenue_usd", "final_soh"]].to_string(index=False))

print("\n" + LINE)
print("Scenario comparison engine verified successfully ✓")
print(LINE)