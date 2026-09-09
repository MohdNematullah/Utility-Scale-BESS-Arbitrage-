"""
tests/backtesting/test_export_reports.py
========================================

Unit tests for Part 8.4/8.5 Export & Reporting Engine.
"""

from pathlib import Path
import json
import pytest
import openpyxl
import pandas as pd

from data.loader import MarketDataLoader
from features.engineering import FeatureEngineer
from forecasting.models import XGBoostForecaster
from backtesting.engine import RollingBacktestEngine, RollingBacktestResult
from backtesting.export_reports import BacktestExportEngine


# ---------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def export_artifacts():
    dispatch_file = Path("backtesting/results/dispatch_history.csv")
    degradation_file = Path("backtesting/results/degradation_history.csv")
    summary_file = Path("backtesting/results/rolling_summary.csv")

    if dispatch_file.exists() and degradation_file.exists() and summary_file.exists():
        dispatch_df = pd.read_csv(dispatch_file)
        degradation_df = pd.read_csv(degradation_file)
        summary_dict = pd.read_csv(summary_file).iloc[0].to_dict()

        result = RollingBacktestResult(
            dispatch_history=dispatch_df,
            degradation_history=degradation_df,
            summary=summary_dict,
        )
    else:
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
            pytest.skip("Trained forecast model not found.")

        model = XGBoostForecaster()
        model.load(model_path)

        engine = RollingBacktestEngine(
            forecaster=model,
            feature_dataframe=features,
        )
        result = engine.run()

    exporter = BacktestExportEngine()
    return exporter.export(result)


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_metrics_summary_exists(export_artifacts):
    assert export_artifacts.metrics_summary.exists()


def test_financial_summary_exists(export_artifacts):
    assert export_artifacts.financial_summary.exists()


def test_battery_summary_exists(export_artifacts):
    assert export_artifacts.battery_summary.exists()


def test_operational_summary_exists(export_artifacts):
    assert export_artifacts.operational_summary.exists()


def test_forecast_summary_exists(export_artifacts):
    assert export_artifacts.forecast_summary.exists()


def test_daily_summary_exists(export_artifacts):
    assert export_artifacts.daily_summary.exists()


def test_monthly_summary_exists(export_artifacts):
    assert export_artifacts.monthly_summary.exists()


def test_excel_created(export_artifacts):
    assert export_artifacts.excel_report.exists()


def test_excel_sheets(export_artifacts):
    wb = openpyxl.load_workbook(export_artifacts.excel_report, read_only=True)
    expected = {
        "Summary",
        "Dispatch",
        "Degradation",
        "Financial",
        "Daily_Summary",
        "Monthly_Summary",
        "Battery",
        "Operations",
        "Forecast",
    }
    actual_sheets = set(wb.sheetnames)
    wb.close()
    assert expected.issubset(actual_sheets)


def test_json_created(export_artifacts):
    assert export_artifacts.json_report.exists()


def test_json_keys(export_artifacts):
    with open(export_artifacts.json_report, encoding="utf-8") as file:
        report = json.load(file)

    assert "simulation" in report
    assert "financial" in report
    assert "battery" in report
    assert "forecast_performance" in report

    assert report["financial"]["net_revenue_usd"] > 0
    assert report["battery"]["final_soh"] > 0.80


def test_all_9_figures_created(export_artifacts):
    figure_dir = export_artifacts.figure_directory
    expected = {
        "cumulative_revenue.png",
        "soh_curve.png",
        "soc_profile.png",
        "price_dispatch.png",
        "degradation_breakdown.png",
        "daily_revenue.png",
        "utilization_heatmap.png",
        "monthly_degradation_chart.png",
        "forecast_error_overlay.png",
    }
    produced = {p.name for p in figure_dir.glob("*.png")}
    assert expected.issubset(produced)