"""
backtesting/check_sensitivity_analysis.py
=========================================

Verification and sanity-check script for Sensitivity Analysis Engine (Part 9.4).
"""

from pathlib import Path
import pandas as pd

from backtesting.scenarios import list_scenarios
from backtesting.sensitivity_analysis import SensitivityAnalysisEngine

LINE = "=" * 75

print(LINE)
print(" SENSITIVITY ANALYSIS & ELASTICITY CHECK (PART 9.4)")
print(LINE)

registry_file = Path("backtesting/results/comparison/scenario_ranking.csv")

if registry_file.exists():
    print("Loading scenario evaluation registry from disk...")
    scenarios_df = pd.read_csv(registry_file)
else:
    print("Synthesizing 28-scenario experimental catalog dataset for verification...")
    scenarios = list_scenarios()
    synthetic_records = []
    base_rev = 1102000.0
    base_deg = 253700.0

    for s in scenarios:
        cid = s.scenario_id
        cat = s.category.value
        sname = s.scenario_name

        rev_mult = 1.0
        deg_mult = 1.0

        if "HORIZON_12H" in cid:
            rev_mult, deg_mult = 0.82, 0.88
        elif "HORIZON_24H" in cid:
            rev_mult, deg_mult = 0.94, 0.96
        elif "HORIZON_72H" in cid:
            rev_mult, deg_mult = 1.06, 1.05
        elif "CHEM_LFP" in cid:
            rev_mult, deg_mult = 0.98, 0.58
        elif "CHEM_LTO" in cid:
            rev_mult, deg_mult = 0.96, 0.25
        elif "TEMP_15C" in cid:
            deg_mult = 0.85
        elif "TEMP_45C" in cid:
            deg_mult = 1.65
        elif "SIZE_100MW" in cid:
            rev_mult, deg_mult = 2.00, 2.00
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
        soh = round(1.0 - (0.0188 * deg_mult / rev_mult if rev_mult > 0 else 0.0188), 4)

        synthetic_records.append({
            "experiment_id": cid,
            "scenario_name": sname,
            "category": cat,
            "forecast_horizon": s.experiment_config.forecast_horizon_hours,
            "gross_revenue_usd": gross,
            "degradation_cost_usd": deg,
            "net_revenue_usd": net,
            "final_soh": min(max(soh, 0.92), 1.0),
            "cumulative_efc": round(190.18 * (gross / base_rev), 2),
            "sharpe_ratio": round(max(0.5, (net / 848000.0) * 2.45), 3),
        })
    scenarios_df = pd.DataFrame(synthetic_records)

engine = SensitivityAnalysisEngine(output_directory="backtesting/results/sensitivity_analysis")
elasticities, tornado_records, artifacts = engine.evaluate_and_export(
    scenarios_df=scenarios_df,
    baseline_revenue_usd=848333.00,
)

print("-" * 75)
print("TORNADO SENSITIVITY SPECTRUM (PARAMETRIC VALUE SWING)")
print("-" * 75)
for t in tornado_records:
    print(f"  Rank {t.sensitivity_rank}: {t.parameter:<28} | Swing: ${t.swing_usd:>10,.2f} | Range: [${t.low_revenue_usd:>10,.2f} -> ${t.high_revenue_usd:>10,.2f}]")

print("-" * 75)
print("ELASTICITY SPOTLIGHT (POINT & ARC SENSITIVITIES)")
print("-" * 75)
for e in elasticities[:6]:
    print(f"  {e.parameter_dimension:<22} | {e.parameter_variation:<16} | Arc Elasticity: {e.arc_elasticity:>7.3f} | Marginal Rate: ${e.marginal_rate_usd:,.2f}/unit")

assert len(tornado_records) > 0, "Tornado records must not be empty."
assert len(elasticities) > 0, "Elasticity matrix must not be empty."
assert tornado_records[0].swing_usd >= tornado_records[-1].swing_usd, "Tornado records must be sorted by swing."

print("-" * 75)
print("EXPORTED SENSITIVITY ARTIFACTS")
print("-" * 75)
print(f"Tornado CSV        : {artifacts.tornado_csv.exists()} ({artifacts.tornado_csv})")
print(f"Elasticity CSV     : {artifacts.elasticity_csv.exists()} ({artifacts.elasticity_csv})")
print(f"Sensitivity Excel  : {artifacts.sensitivity_excel.exists()} ({artifacts.sensitivity_excel})")
print(f"Summary JSON       : {artifacts.summary_json.exists()} ({artifacts.summary_json})")

figures = list(artifacts.figures_directory.glob("*.png"))
print(f"\nGenerated Diagnostic Visualizations ({len(figures)} total):")
for fig in sorted(figures):
    print(f"  â€¢ {fig.name}")

assert len(figures) == 5, f"Expected 5 figures, found {len(figures)}."

print(LINE)
print("Sensitivity analysis engine verified successfully âœ“")
print(LINE)