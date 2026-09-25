"""
visualization/check_degradation_figures.py
==========================================

Verification script for Battery Degradation Figures Module (Part 10.4).
"""

from pathlib import Path
import numpy as np
import pandas as pd

from visualization.degradation_figures import DegradationFigureGenerator

LINE = "=" * 75

print(LINE)
print(" BATTERY DEGRADATION FIGURES VERIFICATION (PART 10.4)")
print(LINE)

degradation_path = Path("backtesting/results/degradation_history.csv")

if degradation_path.exists() and "soh_end" in pd.read_csv(degradation_path, nrows=2).columns:
    print("Loading empirical simulation degradation dataset from disk...")
    degradation_df = pd.read_csv(degradation_path)
else:
    print("Synthesizing 350-window realistic degradation trajectory for verification...")
    n_windows = 350
    daily_fade = 0.0188 / n_windows
    soh_curve = 1.0 - np.cumsum(np.random.normal(daily_fade, daily_fade * 0.1, n_windows))
    soh_curve = np.clip(soh_curve, 0.90, 1.0)

    degradation_df = pd.DataFrame({
        "rolling_window": range(1, n_windows + 1),
        "soh_start": np.insert(soh_curve[:-1], 0, 1.0),
        "soh_end": soh_curve,
        "calendar_loss": np.full(n_windows, 0.0188 * 0.38 / n_windows),
        "cycle_loss": np.full(n_windows, 0.0188 * 0.62 / n_windows),
        "window_efc": np.random.normal(0.543, 0.05, n_windows),
        "cumulative_efc": np.cumsum(np.random.normal(0.543, 0.05, n_windows)),
        "degradation_cost_usd": np.random.normal(725.0, 30.0, n_windows),
        "cumulative_degradation_cost_usd": np.cumsum(np.random.normal(725.0, 30.0, n_windows)),
    })

generator = DegradationFigureGenerator(
    output_directory="visualization/figures/degradation",
    formats=("png", "pdf", "svg"),
    dpi=300,
)

artifacts = generator.generate_all(degradation_df=degradation_df)

print("-" * 75)
print("GENERATED PUBLICATION DEGRADATION FIGURES (6 TOTAL)")
print("-" * 75)

figure_groups = [
    ("Figure 10.4.1: SOH Degradation Curve", artifacts.soh_curve),
    ("Figure 10.4.2: Usable Capacity Fade", artifacts.capacity_fade),
    ("Figure 10.4.3: Calendar vs Cycle Loss Breakdown", artifacts.calendar_cycle_loss),
    ("Figure 10.4.4: Rainflow Cycle DOD Histogram", artifacts.rainflow_histogram),
    ("Figure 10.4.5: Equivalent Full Cycles (EFC)", artifacts.efc_curve),
    ("Figure 10.4.6: Degradation Cost Accumulation", artifacts.degradation_cost_curve),
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
print("Battery degradation visualization package verified successfully [OK]")
print(LINE)