"""
visualization/check_dispatch_figures.py
=======================================

Verification script for Dispatch Figures Module (Part 10.3).
"""

from pathlib import Path
import numpy as np
import pandas as pd

from visualization.dispatch_figures import DispatchFigureGenerator

LINE = "=" * 75

print(LINE)
print(" DISPATCH FIGURES VERIFICATION (PART 10.3)")
print(LINE)

dispatch_path = Path("backtesting/results/dispatch_history.csv")

if dispatch_path.exists() and "charge_power_mw" in pd.read_csv(dispatch_path, nrows=2).columns:
    print("Loading empirical simulation dispatch dataset from disk...")
    dispatch_df = pd.read_csv(dispatch_path)
else:
    print("Synthesizing 8400-hour realistic dispatch profile for verification...")
    n_hours = 8400
    base_signal = 35.0 + 15.0 * np.sin(np.linspace(0, 350 * 2 * np.pi, n_hours))
    actual_prices = np.clip(base_signal + np.random.normal(0, 4.0, n_hours), 5.0, 150.0)

    # 2 hours charging at night, 2 hours discharging at evening peaks
    chg = np.array([50.0 if (i % 24) in [2, 3] else 0.0 for i in range(n_hours)])
    dis = np.array([45.0 if (i % 24) in [18, 19] else 0.0 for i in range(n_hours)])
    soc = np.clip(50.0 + np.cumsum(chg * 0.9 - dis / 0.9) * 0.1, 5.0, 95.0) / 100.0

    dispatch_df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n_hours, freq="h", tz="UTC"),
        "actual_price": actual_prices,
        "charge_power_mw": chg,
        "discharge_power_mw": dis,
        "soc": soc,
    })

generator = DispatchFigureGenerator(
    output_directory="visualization/figures/dispatch",
    formats=("png", "pdf", "svg"),
    dpi=300,
)

artifacts = generator.generate_all(dispatch_df=dispatch_df)

print("-" * 75)
print("GENERATED PUBLICATION DISPATCH FIGURES (7 TOTAL)")
print("-" * 75)

figure_groups = [
    ("Figure 10.3.1: Price & Power Dispatch", artifacts.price_power_dispatch),
    ("Figure 10.3.2: State of Charge (SOC)", artifacts.state_of_charge),
    ("Figure 10.3.3: Diurnal Dispatch Heatmap", artifacts.dispatch_heatmap),
    ("Figure 10.3.4: Daily Energy Throughput", artifacts.daily_throughput),
    ("Figure 10.3.5: Rolling Window Timeline", artifacts.rolling_window_timeline),
    ("Figure 10.3.6: SOC Density Distribution", artifacts.soc_density),
    ("Figure 10.3.7: Arbitrage Spread Capture", artifacts.price_spread_capture),
]

total_files = 0
for name, paths in figure_groups:
    print(f"\n{name}:")
    for fmt, p in paths.items():
        assert p.exists() and p.stat().st_size > 0, f"Missing file: {p}"
        total_files += 1
        print(f"  * {fmt.upper():<4}: {p.name} ({p.stat().st_size:>8,} bytes)")

print("\n" + "-" * 75)
print(f"Total Vector & Raster Artifacts Verified: {total_files} files")
assert total_files == 21, f"Expected 21 files (7 figures x 3 formats), found {total_files}."

print(LINE)
print("Dispatch visualization package verified successfully [OK]")
print(LINE)