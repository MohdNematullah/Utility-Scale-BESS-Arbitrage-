"""
visualization/check_forecast_figures.py
=======================================

Verification script for Forecast Figures Module (Part 10.2).
"""

from pathlib import Path
import numpy as np
import pandas as pd

from visualization.forecast_figures import ForecastFigureGenerator

LINE = "=" * 75

print(LINE)
print(" FORECAST FIGURES VERIFICATION (PART 10.2)")
print(LINE)

dispatch_path = Path("backtesting/results/dispatch_history.csv")

if dispatch_path.exists() and "actual_price" in pd.read_csv(dispatch_path, nrows=2).columns:
    print("Loading empirical simulation dispatch history from disk...")
    dispatch_df = pd.read_csv(dispatch_path)
else:
    print("Synthesizing 8400-hour realistic market dataset for verification...")
    n_hours = 8400
    base_signal = 35.0 + 15.0 * np.sin(np.linspace(0, 350 * 2 * np.pi, n_hours))
    actual_prices = base_signal + np.random.normal(0, 4.0, n_hours)
    forecast_prices = base_signal + np.random.normal(0, 5.5, n_hours)

    dispatch_df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n_hours, freq="h", tz="UTC"),
        "actual_price": actual_prices,
        "forecast_price": forecast_prices,
    })

generator = ForecastFigureGenerator(
    output_directory="visualization/figures/forecast",
    formats=("png", "pdf", "svg"),
    dpi=300,
)

artifacts = generator.generate_all(dispatch_df=dispatch_df)

print("-" * 75)
print("GENERATED PUBLICATION FORECAST FIGURES (6 TOTAL)")
print("-" * 75)

figure_groups = [
    ("Figure 10.2.1: Forecast vs Actual", artifacts.forecast_vs_actual),
    ("Figure 10.2.2: Residual Distribution", artifacts.residual_histogram),
    ("Figure 10.2.3: Residual Timeseries", artifacts.residual_timeseries),
    ("Figure 10.2.4: Horizon Accuracy", artifacts.forecast_horizon_accuracy),
    ("Figure 10.2.5: Parity Scatter", artifacts.forecast_scatter),
    ("Figure 10.2.6: Diurnal Heatmap", artifacts.forecast_heatmap),
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
assert total_files == 18, f"Expected 18 files (6 figures x 3 formats), found {total_files}."

print(LINE)
print("Forecast visualization package verified successfully [OK]")
print(LINE)