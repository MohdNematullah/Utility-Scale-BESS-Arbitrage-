"""
visualization/check_financial_figures.py
========================================

Verification script for Financial Figures Module (Part 10.5).
"""

from pathlib import Path
import numpy as np
import pandas as pd

from visualization.financial_figures import FinancialFigureGenerator

LINE = "=" * 75

print(LINE)
print(" FINANCIAL FIGURES VERIFICATION (PART 10.5)")
print(LINE)

dispatch_path = Path("backtesting/results/dispatch_history.csv")
degradation_path = Path("backtesting/results/degradation_history.csv")
metrics_path = Path("backtesting/results/metrics_summary.csv")

if dispatch_path.exists() and "actual_price" in pd.read_csv(dispatch_path, nrows=2).columns:
    print("Loading empirical simulation dispatch and degradation datasets from disk...")
    dispatch_df = pd.read_csv(dispatch_path)
    degradation_df = pd.read_csv(degradation_path) if degradation_path.exists() else None
else:
    print("Synthesizing 8400-hour realistic market dataset for verification...")
    n_hours = 8400
    base_signal = 35.0 + 15.0 * np.sin(np.linspace(0, 350 * 2 * np.pi, n_hours))
    actual_prices = np.clip(base_signal + np.random.normal(0, 4.0, n_hours), 5.0, 150.0)
    chg = np.array([50.0 if (i % 24) in [2, 3] else 0.0 for i in range(n_hours)])
    dis = np.array([45.0 if (i % 24) in [18, 19] else 0.0 for i in range(n_hours)])

    dispatch_df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n_hours, freq="h", tz="UTC"),
        "actual_price": actual_prices,
        "charge_power_mw": chg,
        "discharge_power_mw": dis,
        "net_revenue_usd": (dis - chg) * actual_prices,
    })

    degradation_df = pd.DataFrame({
        "rolling_window": range(1, 351),
        "degradation_cost_usd": [725.0] * 350,
    })

summary_dict = {
    "gross_revenue_usd": 1102091.72,
    "degradation_cost_usd": 253758.72,
    "fixed_om_cost_usd": 359589.04,
    "variable_om_cost_usd": 19017.97,
    "net_operating_profit_usd": 469725.99,
}

generator = FinancialFigureGenerator(
    output_directory="visualization/figures/finance",
    formats=("png", "pdf", "svg"),
    dpi=300,
)

artifacts = generator.generate_all(
    dispatch_df=dispatch_df,
    degradation_df=degradation_df,
    summary_dict=summary_dict,
)

print("-" * 75)
print("GENERATED PUBLICATION FINANCIAL FIGURES (6 TOTAL)")
print("-" * 75)

figure_groups = [
    ("Figure 10.5.1: Cumulative Revenue Trajectory", artifacts.cumulative_revenue),
    ("Figure 10.5.2: Daily Revenue Dynamics", artifacts.daily_revenue),
    ("Figure 10.5.3: Financial Value Waterfall", artifacts.revenue_waterfall),
    ("Figure 10.5.4: Revenue Distribution", artifacts.revenue_distribution),
    ("Figure 10.5.5: Monthly Revenue & Spread", artifacts.monthly_revenue),
    ("Figure 10.5.6: Lifetime Value Breakdown", artifacts.lifetime_value_breakdown),
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
assert total_files == 18, f"Expected 18 files (6 figures x 3 formats), found {total_files}."

print(LINE)
print("Financial visualization package verified successfully ✓")
print(LINE)