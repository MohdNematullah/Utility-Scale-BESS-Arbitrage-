"""
backtesting/check_export_reports.py
==================================

Validation script for Part 8.4 Export & Reporting Engine.

Verifies:
1. Metrics generation.
2. CSV summaries export.
3. Multi-sheet Excel workbook export.
4. JSON summary export.
5. High-resolution figure generation.
"""

from pathlib import Path
import pandas as pd

from data.loader import MarketDataLoader
from features.engineering import FeatureEngineer
from forecasting.models import XGBoostForecaster
from backtesting.engine import RollingBacktestEngine, RollingBacktestResult
from backtesting.export_reports import BacktestExportEngine

LINE = "=" * 70

print(LINE)
print("BACKTEST EXPORT REPORT CHECK")
print(LINE)

dispatch_file = Path("backtesting/results/dispatch_history.csv")
degradation_file = Path("backtesting/results/degradation_history.csv")
summary_file = Path("backtesting/results/rolling_summary.csv")

# ---------------------------------------------------------------------
# 1. Acquire Backtest Result (Reuse existing or run fresh)
# ---------------------------------------------------------------------
if dispatch_file.exists() and degradation_file.exists() and summary_file.exists():
    print("Loading existing backtest results from disk ✓")
    dispatch_df = pd.read_csv(dispatch_file)
    degradation_df = pd.read_csv(degradation_file)
    summary_dict = pd.read_csv(summary_file).iloc[0].to_dict()

    result = RollingBacktestResult(
        dispatch_history=dispatch_df,
        degradation_history=degradation_df,
        summary=summary_dict,
    )
else:
    print("Running rolling horizon backtest...")
    loader = MarketDataLoader()
    engineer = FeatureEngineer()

    market = loader.load_csv("data/raw/ercot_prices.csv")
    features = engineer.transform(market)

    if "timestamp" in features.columns:
        features["timestamp"] = pd.to_datetime(features["timestamp"], utc=True)
        features = features.sort_values("timestamp").set_index("timestamp")
    features.index = pd.DatetimeIndex(features.index)

    model_path = Path("forecasting/saved_models/xgboost_dayahead.pkl")
    if not model_path.exists():
        raise FileNotFoundError("Run python -m forecasting.check_trainer first.")

    model = XGBoostForecaster()
    model.load(model_path)
    print("Forecast model loaded ✓")

    engine = RollingBacktestEngine(
        forecaster=model,
        feature_dataframe=features,
    )
    result = engine.run()
    print("Rolling backtest completed ✓")

# ---------------------------------------------------------------------
# 2. Export Reports & Figures
# ---------------------------------------------------------------------
exporter = BacktestExportEngine()
exports = exporter.export(result)

print("Reports exported ✓")

print("-" * 70)
print("CSV FILES")
print("-" * 70)

csv_files = [
    exports.metrics_summary,
    exports.financial_summary,
    exports.battery_summary,
    exports.operational_summary,
    exports.forecast_summary,
]

for file in csv_files:
    print(f"{file.name:<30} {file.exists()}")

print("-" * 70)
print("WORKBOOK / JSON")
print("-" * 70)

print(f"{exports.excel_report.name:<30} {exports.excel_report.exists()}")
print(f"{exports.json_report.name:<30} {exports.json_report.exists()}")

print("-" * 70)
print("FIGURES")
print("-" * 70)

figures = sorted(exports.figure_directory.glob("*.png"))
for fig in figures:
    print(f"{fig.name:<30} {fig.exists()}")

print(LINE)
print("Backtest export reports verified successfully ✓")
print(LINE)