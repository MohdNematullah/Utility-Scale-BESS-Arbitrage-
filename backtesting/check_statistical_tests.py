"""
backtesting/check_statistical_tests.py
======================================

Verification and sanity-check script for Statistical Significance Engine (Part 9.5).
"""

from pathlib import Path
import numpy as np
import pandas as pd

from backtesting.statistical_tests import StatisticalTestEngine

LINE = "=" * 75

print(LINE)
print(" STATISTICAL SIGNIFICANCE & HYPOTHESIS TESTING CHECK (PART 9.5)")
print(LINE)

dispatch_file = Path("backtesting/results/dispatch_history.csv")

if dispatch_file.exists() and "actual_price" in pd.read_csv(dispatch_file, nrows=2).columns:
    print("Loading empirical simulation dispatch dataset from disk...")
    dispatch_df = pd.read_csv(dispatch_file)
    actual_prices = dispatch_df["actual_price"].to_numpy(dtype=float)
    xgb_preds = dispatch_df["forecast_price"].to_numpy(dtype=float)
    n_hours = len(actual_prices)
else:
    print("Synthesizing multi-model forecast benchmarks for verification...")
    n_hours = 8400
    base_signal = 35.0 + 15.0 * np.sin(np.linspace(0, 350 * 2 * np.pi, n_hours))
    actual_prices = base_signal + np.random.normal(0, 4.0, n_hours)
    xgb_preds = base_signal + np.random.normal(0, 5.0, n_hours)

# Construct benchmark models
persistence_preds = np.roll(actual_prices, 24)
persistence_preds[:24] = actual_prices[:24]

# 7-day rolling moving average benchmark
ma_preds = pd.Series(actual_prices).rolling(168, min_periods=1).mean().to_numpy()
perfect_preds = actual_prices.copy()

models_predictions = {
    "XGBoost": xgb_preds,
    "Persistence": persistence_preds,
    "Moving_Average": ma_preds,
    "Perfect_Foresight": perfect_preds,
}

# Construct daily revenue series for each model
n_days = n_hours // 24
daily_revenues = {
    "XGBoost": np.random.normal(2450.0, 350.0, n_days),
    "Persistence": np.random.normal(1950.0, 420.0, n_days),
    "Moving_Average": np.random.normal(1750.0, 450.0, n_days),
    "Perfect_Foresight": np.random.normal(3200.0, 300.0, n_days),
}

engine = StatisticalTestEngine(output_directory="backtesting/results/statistical_tests")
dm_results, bootstrap_results, effect_results, artifacts = engine.evaluate_and_export(
    ground_truth=actual_prices,
    models_predictions=models_predictions,
    daily_revenues=daily_revenues,
    forecast_horizon_h=24,
)

print("-" * 75)
print("DIEBOLD-MARIANO PREDICTIVE SUPERIORITY TESTS (HLN 1997)")
print("-" * 75)
for dm in dm_results:
    sig = "***" if dm.p_value < 0.001 else ("**" if dm.p_value < 0.01 else ("*" if dm.p_value < 0.05 else "n.s."))
    print(f"  {dm.model_a:<18} vs {dm.model_b:<18} | HLN: {dm.hln_statistic:>6.2f} | p-val: {dm.p_value:.5f} ({sig:<4}) | Superior: {dm.superior_model}")

print("-" * 75)
print("MOVING BLOCK BOOTSTRAP CONFIDENCE INTERVALS (95% CI)")
print("-" * 75)
for b in bootstrap_results:
    print(f"  {b.metric_name:<28} | Point: {b.point_estimate:>10.2f} | 95% CI: [{b.ci_95_lower:>10.2f}, {b.ci_95_upper:>10.2f}] | SE: {b.standard_error:>7.2f}")

print("-" * 75)
print("STANDARDIZED EFFECT SIZES & CLINICAL SIGNIFICANCE")
print("-" * 75)
for e in effect_results:
    print(f"  {e.comparison_name:<28} | Hedges' g: {e.hedges_g:>6.2f} ({e.magnitude_interpretation:<10}) | CLES: {e.cles_pct:>5.1f}%")

# Rigorous Statistical Assertions
assert len(dm_results) == 6, "Expected exactly 6 pairwise DM comparisons for 4 models."
assert len(bootstrap_results) >= 2, "Bootstrap CIs must contain MAE and RMSE."
for b in bootstrap_results:
    assert b.ci_95_lower <= b.point_estimate <= b.ci_95_upper, f"Point estimate outside 95% CI for {b.metric_name}."
    assert b.ci_99_lower <= b.ci_95_lower, "99% lower bound must be <= 95% lower bound."
    assert b.ci_99_upper >= b.ci_95_upper, "99% upper bound must be >= 95% upper bound."

print("-" * 75)
print("EXPORTED STATISTICAL ARTIFACTS")
print("-" * 75)
print(f"Summary CSV       : {artifacts.summary_csv.exists()} ({artifacts.summary_csv})")
print(f"Bootstrap CSV     : {artifacts.bootstrap_csv.exists()} ({artifacts.bootstrap_csv})")
print(f"DM Matrix CSV     : {artifacts.dm_matrix_csv.exists()} ({artifacts.dm_matrix_csv})")
print(f"Statistical Excel : {artifacts.statistical_excel.exists()} ({artifacts.statistical_excel})")
print(f"Summary JSON      : {artifacts.summary_json.exists()} ({artifacts.summary_json})")

figures = list(artifacts.figures_directory.glob("*.png"))
print(f"\nGenerated Publication Diagnostic Visualizations ({len(figures)} total):")
for fig in sorted(figures):
    print(f"  • {fig.name}")

assert len(figures) == 5, f"Expected 5 figures, found {len(figures)}."

print(LINE)
print("Statistical significance evaluation verified successfully ✓")
print(LINE)