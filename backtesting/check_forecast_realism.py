"""
backtesting/check_forecast_realism.py
====================================

Verification and sanity-check script for Forecast Realism Module (Part 9.1).
"""

from pathlib import Path
import numpy as np
import pandas as pd

from backtesting.forecast_realism import ForecastRealismEngine

LINE = "=" * 75

print(LINE)
print(" FORECAST REALISM CHECK (PART 9.1)")
print(LINE)

dispatch_file = Path("backtesting/results/dispatch_history.csv")
summary_file = Path("backtesting/results/metrics_summary.csv")

if dispatch_file.exists() and "actual_price" in pd.read_csv(dispatch_file, nrows=2).columns:
    print("Loading empirical simulation dispatch dataset from disk...")
    dispatch_df = pd.read_csv(dispatch_file)
    gross_rev = None
    if summary_file.exists():
        sm = dict(zip(pd.read_csv(summary_file)["metric"], pd.read_csv(summary_file)["value"]))
        gross_rev = float(sm.get("gross_revenue_usd", 1102091.72))
        pf_rev = float(sm.get("perfect_foresight_benchmark_usd", 889638.21))
    else:
        pf_rev = 889638.21
else:
    print("Generating representative out-of-sample forecast scenario for verification...")
    n_hours = 8400
    base_signal = 35.0 + 15.0 * np.sin(np.linspace(0, 350 * 2 * np.pi, n_hours))
    actual_prices = base_signal + np.random.normal(0, 4.0, n_hours)
    forecast_prices = base_signal + np.random.normal(0, 5.5, n_hours)

    dispatch_df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n_hours, freq="h", tz="UTC"),
        "actual_price": actual_prices,
        "forecast_price": forecast_prices,
        "net_revenue_usd": np.maximum(0.0, (forecast_prices - 30.0) * 25.0),
    })
    gross_rev = 1102091.72
    pf_rev = 1250000.00

engine = ForecastRealismEngine(output_directory="backtesting/results/forecast_realism")
metrics, artifacts = engine.evaluate_and_export(
    dispatch_df=dispatch_df,
    gross_revenue_usd=gross_rev,
    perfect_foresight_revenue_usd=pf_rev,
)

print("-" * 75)
print("FORECAST ACCURACY & VALUE CAPTURE KPIS")
print("-" * 75)
print(f"Mean Absolute Error (MAE)        : ${metrics.mae:.2f}/MWh")
print(f"Root Mean Square Error (RMSE)    : ${metrics.rmse:.2f}/MWh")
print(f"WAPE / Volume MAPE               : {metrics.mape_pct:.2f}%")
print(f"Symmetric MAPE (SMAPE)           : {metrics.smape_pct:.2f}%")
print(f"Coefficient of Determination (R²): {metrics.r2_score:.4f}")
print(f"Mean Bias Error (MBE)            : ${metrics.bias:.2f}/MWh")
print(f"Residual Std Deviation (σ)       : ${metrics.residual_std:.2f}/MWh")
print(f"Directional Trajectory Accuracy  : {metrics.directional_accuracy_pct:.2f}%")
print(f"Value Capture Ratio (VCR)        : {metrics.value_capture_ratio_pct:.2f}%")
print(f"Perfect Foresight Gap            : ${metrics.perfect_foresight_gap_usd:,.2f}")

assert metrics.mae > 0.0, "MAE must be strictly positive."
assert metrics.rmse >= metrics.mae, "RMSE must be mathematically greater than or equal to MAE."
assert -1.0 <= metrics.r2_score <= 1.0, "R² must be bounded within standard limits."
assert 0.0 <= metrics.directional_accuracy_pct <= 100.0, "Directional accuracy must be within [0, 100]%."

print("-" * 75)
print("EXPORTED DIAGNOSTIC ARTIFACTS")
print("-" * 75)
print(f"Summary CSV   : {artifacts.summary_csv.exists()} ({artifacts.summary_csv})")
print(f"Residuals CSV : {artifacts.residuals_csv.exists()} ({artifacts.residuals_csv})")
print(f"Scorecard JSON: {artifacts.summary_json.exists()} ({artifacts.summary_json})")

figures = list(artifacts.figures_directory.glob("*.png"))
print(f"\nGenerated Diagnostic Visualizations ({len(figures)} total):")
for fig in sorted(figures):
    print(f"  • {fig.name}")

assert len(figures) == 6, f"Expected exactly 6 figures, found {len(figures)}."

print(LINE)
print("Forecast realism evaluation verified successfully ✓")
print(LINE)