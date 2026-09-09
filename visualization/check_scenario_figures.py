"""
visualization/check_scenario_figures.py
=======================================

Verification script for Scenario & Sensitivity Figures Module (Part 10.7).
"""

from pathlib import Path
import numpy as np
import pandas as pd

from visualization.scenario_figures import ScenarioFigureGenerator

LINE = "=" * 75

print(LINE)
print(" SCENARIO & SENSITIVITY FIGURES VERIFICATION (PART 10.7)")
print(LINE)

scenario_path = Path("backtesting/results/comparison/scenario_ranking.csv")

if scenario_path.exists():
    print("Loading scenario ranking matrix from disk...")
    scenarios_df = pd.read_csv(scenario_path)
else:
    print("Synthesizing 28-scenario experimental catalog dataset for verification...")
    scenarios_data = [
        # Chemistries
        {"scenario_name": "chem_nmc_baseline", "category": "Battery_Chemistry", "net_revenue_usd": 848333.0, "final_soh": 0.9812, "composite_score": 0.72},
        {"scenario_name": "chem_lfp_stationary", "category": "Battery_Chemistry", "net_revenue_usd": 932814.0, "final_soh": 0.9891, "composite_score": 0.81},
        {"scenario_name": "chem_lto_heavy_cycle", "category": "Battery_Chemistry", "net_revenue_usd": 994495.0, "final_soh": 0.9951, "composite_score": 0.88},
        # Horizons
        {"scenario_name": "horizon_12h", "category": "Forecast_Horizon", "net_revenue_usd": 680384.0, "final_soh": 0.9830, "composite_score": 0.61},
        {"scenario_name": "horizon_24h", "category": "Forecast_Horizon", "net_revenue_usd": 795400.0, "final_soh": 0.9820, "composite_score": 0.68},
        {"scenario_name": "horizon_48h_base", "category": "Forecast_Horizon", "net_revenue_usd": 848333.0, "final_soh": 0.9812, "composite_score": 0.72},
        {"scenario_name": "horizon_72h", "category": "Forecast_Horizon", "net_revenue_usd": 901735.0, "final_soh": 0.9805, "composite_score": 0.76},
        # Sizing
        {"scenario_name": "size_25mw_50mwh", "category": "System_Sizing", "net_revenue_usd": 424150.0, "final_soh": 0.9812, "composite_score": 0.45},
        {"scenario_name": "size_100mw_200mwh", "category": "System_Sizing", "net_revenue_usd": 1696600.0, "final_soh": 0.9812, "composite_score": 0.95},
        # Temperatures
        {"scenario_name": "temp_15c_subcooled", "category": "Thermal_Sensitivity", "net_revenue_usd": 886355.0, "final_soh": 0.9845, "composite_score": 0.75},
        {"scenario_name": "temp_25c_reference", "category": "Thermal_Sensitivity", "net_revenue_usd": 848333.0, "final_soh": 0.9812, "composite_score": 0.72},
        {"scenario_name": "temp_45c_stress", "category": "Thermal_Sensitivity", "net_revenue_usd": 683395.0, "final_soh": 0.9690, "composite_score": 0.58},
        # Efficiencies
        {"scenario_name": "eff_85pct_aged", "category": "Efficiency_Sensitivity", "net_revenue_usd": 727080.0, "final_soh": 0.9812, "composite_score": 0.64},
        {"scenario_name": "eff_95pct_nextgen", "category": "Efficiency_Sensitivity", "net_revenue_usd": 936460.0, "final_soh": 0.9812, "composite_score": 0.78},
    ]
    scenarios_df = pd.DataFrame(scenarios_data)

generator = ScenarioFigureGenerator(
    output_directory="visualization/figures/scenarios",
    formats=("png", "pdf", "svg"),
    dpi=300,
)

artifacts = generator.generate_all(scenarios_df=scenarios_df)

print("-" * 75)
print("GENERATED PUBLICATION SCENARIO & SENSITIVITY FIGURES (8 TOTAL)")
print("-" * 75)

figure_groups = [
    ("Figure 10.7.1: Pareto Frontier", artifacts.pareto_frontier),
    ("Figure 10.7.2: Tornado Sensitivity", artifacts.tornado_sensitivity),
    ("Figure 10.7.3: Scenario Ranking", artifacts.scenario_ranking),
    ("Figure 10.7.4: Cross-Category Heatmap", artifacts.scenario_heatmap),
    ("Figure 10.7.5: Horizon & Benchmark Comparison", artifacts.horizon_comparison),
    ("Figure 10.7.6: Battery Chemistry Comparison", artifacts.chemistry_comparison),
    ("Figure 10.7.7: Thermal Sensitivity", artifacts.temperature_sensitivity),
    ("Figure 10.7.8: Efficiency Sensitivity", artifacts.efficiency_sensitivity),
]

total_files = 0
for name, paths in figure_groups:
    print(f"\n{name}:")
    for fmt, p in paths.items():
        assert p.exists() and p.stat().st_size > 0, f"Missing file: {p}"
        total_files += 1
        print(f"  • {fmt.upper():<4}: {p.name} ({p.stat().st_size:>8,} bytes)")

print("\n" + "-" * 75)
print(f"Total Vector & Raster Artifacts Verified: {total_files} files")
assert total_files == 24, f"Expected 24 files (8 figures x 3 formats), found {total_files}."

print(LINE)
print("Scenario & sensitivity visualization package verified successfully ✓")
print(LINE)