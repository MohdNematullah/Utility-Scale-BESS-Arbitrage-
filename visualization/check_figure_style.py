"""
visualization/check_figure_style.py
===================================

Verification script for Figure Style & Theme Framework (Part 10.1).
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from visualization.figure_style import (
    COLOR_PALETTE,
    format_axes,
    get_figure_dimensions,
    save_publication_figure,
    set_style,
)

LINE = "=" * 75

print(LINE)
print(" FIGURE STYLE & THEME CHECK (PART 10.1)")
print(LINE)

# 1. Apply Style
set_style()
print("/Nature rcParams successfully applied to Matplotlib engine.")

# 2. Build multi-panel demonstration figure showcasing all semantic roles
dims = get_figure_dimensions("full_tall")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=dims, gridspec_kw={"height_ratios": [1.5, 1.0]})

# Generate smooth synthetic test data
x = np.linspace(0, 48, 200)
actual_price = 35.0 + 15.0 * np.sin(x * np.pi / 12) + np.sin(x * np.pi / 3) * 4.0
forecast_price = actual_price + np.cos(x * np.pi / 6) * 3.0
perfect_price = actual_price + 1.5

ax1.plot(x, actual_price, color=COLOR_PALETTE.actual, label="Actual Price", linewidth=2.0)
ax1.plot(x, forecast_price, color=COLOR_PALETTE.forecast, linestyle="--", label="Forecast (ML)")
ax1.plot(x, perfect_price, color=COLOR_PALETTE.perfect_foresight, linestyle=":", label="Clairvoyant Upper Bound")
ax1.fill_between(x, forecast_price - 5.0, forecast_price + 5.0, color=COLOR_PALETTE.forecast, alpha=0.15, label="Â±1 MAE Band")

format_axes(ax1, title="Dispatch Price Trajectory & Forecast Confidence", ylabel="Price ($/MWh)")
ax1.legend(loc="upper left")

# Dispatch Power & Degradation Signals
charge_pwr = np.clip(-np.sin(x * np.pi / 12) * 50.0, 0, 50.0)
discharge_pwr = np.clip(np.sin(x * np.pi / 12) * 45.0, 0, 45.0)

ax2.fill_between(x, discharge_pwr, 0, color=COLOR_PALETTE.discharge, alpha=0.8, label="Discharging (MW)")
ax2.fill_between(x, -charge_pwr, 0, color=COLOR_PALETTE.charge, alpha=0.8, label="Charging (MW)")
ax2.axhline(0, color=COLOR_PALETTE.actual, linewidth=0.8)

format_axes(ax2, title="BESS Dispatch Response Profile", xlabel="Time (Hours)", ylabel="Net Power (MW)")
ax2.legend(loc="lower left")

# 3. Export across all four production formats
out_dir = Path("backtesting/results/test_figures_theme")
saved = save_publication_figure(fig, out_dir / "theme_verification_demo", formats=("png", "pdf", "svg", "tiff"), dpi=300)
plt.close(fig)

print("-" * 75)
print("EXPORTED MULTI-FORMAT VERIFICATION FIGURES")
print("-" * 75)
for fmt, path in saved.items():
    print(f"Format: {fmt.upper():<5} | Exists: {path.exists()} | Size: {path.stat().st_size:>8,} bytes | Path: {path}")

assert len(saved) == 4, "Expected exactly 4 formats (PNG, PDF, SVG, TIFF)."
assert all(p.exists() and p.stat().st_size > 1000 for p in saved.values()), "All export files must be non-empty."

print(LINE)
print("Figure style & multi-format export verified successfully [OK]")
print(LINE)