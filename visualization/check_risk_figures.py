"""
visualization/check_risk_figures.py
===================================

Verification script for Risk Figures Module (Part 10.6).
"""

from pathlib import Path
import numpy as np
import pandas as pd

from visualization.risk_figures import RiskFigureGenerator

LINE = "=" * 75

print(LINE)
print(" RISK FIGURES VERIFICATION (PART 10.6)")
print(LINE)

dispatch_path = Path("backtesting/results/dispatch_history.csv")
degradation_path = Path("backtesting/results/degradation_history.csv")

if dispatch_path.exists() and "actual_price" in pd.read_csv(dispatch_path, nrows=2).columns:
    print("Loading empirical simulation dispatch and degradation datasets from disk...")
    dispatch_df = pd.read_csv(dispatch_path)
    degradation_df = pd.read_csv(degradation_path) if degradation_path.exists() else None
else:
    print("Synthesizing 350-day realistic risk series for verification...")
    n_days = 350
    n_hours = n_days * 24
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
        "rolling_window": range(1, n_days + 1),
        "degradation_cost_usd": [725.0] * n_days,
    })

generator = RiskFigureGenerator(
    output_directory="visualization/figures/risk",
    formats=("png", "pdf", "svg"),
    dpi=300,
)

artifacts = generator.generate_all(
    dispatch_df=dispatch_df,
    degradation_df=degradation_df,
)

print("-" * 75)
print("GENERATED PUBLICATION RISK FIGURES (5 TOTAL)")
print("-" * 75)

figure_groups = [
    ("Figure 10.6.1: Daily Profit Distribution", artifacts.daily_profit_dist),
    ("Figure 10.6.2: Drawdown & High-Water Mark", artifacts.drawdown_curve),
    ("Figure 10.6.3: VaR / CVaR Tail Risk", artifacts.var_cvar_tail),
    ("Figure 10.6.4: Rolling Volatility & Sharpe", artifacts.rolling_volatility),
    ("Figure 10.6.5: Return Distribution Moments", artifacts.return_distribution),
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
assert total_files == 15, f"Expected 15 files (5 figures x 3 formats), found {total_files}."

print(LINE)
print("Risk visualization package verified successfully ✓")
print(LINE)