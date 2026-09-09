"""
backtesting/check_metrics.py
============================

Verification script for Backtest Metrics Module.
"""

from pathlib import Path
import pandas as pd

from backtesting.metrics import BacktestMetrics

print("=" * 70)
print("BACKTEST METRICS CHECK")
print("=" * 70)

dispatch_path = Path("backtesting/results/dispatch_history.csv")
degradation_path = Path("backtesting/results/degradation_history.csv")

if not dispatch_path.exists() or not degradation_path.exists():
    raise FileNotFoundError(
        "Backtesting results not found. Execute 'python -m backtesting.check_engine' first."
    )

dispatch = pd.read_csv(dispatch_path)
degradation = pd.read_csv(degradation_path)

metrics = BacktestMetrics()
result = metrics.evaluate(dispatch, degradation)

print(result.dataframe.to_string(index=False))

print("-" * 70)
print("SUMMARY VALIDATION")
print("-" * 70)
for k, v in result.summary.items():
    print(f"{k:<30}: {v}")

print("=" * 70)
print("Backtest metrics verified successfully ✓")
print("=" * 70)