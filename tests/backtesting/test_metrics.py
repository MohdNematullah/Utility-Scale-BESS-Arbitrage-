"""
tests/backtesting/test_metrics.py
=================================

Unit tests for Backtest Metrics Module.
"""

from pathlib import Path
import pytest
import pandas as pd

from backtesting.metrics import BacktestMetrics


@pytest.fixture(scope="module")
def evaluation_result():
    dispatch_path = Path("backtesting/results/dispatch_history.csv")
    degradation_path = Path("backtesting/results/degradation_history.csv")

    if not dispatch_path.exists() or not degradation_path.exists():
        pytest.skip("Backtesting result CSVs not found. Run check_engine first.")

    dispatch = pd.read_csv(dispatch_path)
    degradation = pd.read_csv(degradation_path)
    metrics = BacktestMetrics()
    return metrics.evaluate(dispatch, degradation), len(dispatch)


def test_summary_created(evaluation_result):
    result, _ = evaluation_result
    assert isinstance(result.summary, dict)


def test_dataframe_created(evaluation_result):
    result, _ = evaluation_result
    assert len(result.dataframe) > 0


def test_net_revenue(evaluation_result):
    result, _ = evaluation_result
    assert result.summary["net_revenue_usd"] > 0


def test_capacity_fade(evaluation_result):
    result, _ = evaluation_result
    assert result.summary["capacity_fade"] > 0


def test_final_soh(evaluation_result):
    result, _ = evaluation_result
    assert result.summary["final_soh"] < 1.0


def test_cycles_positive(evaluation_result):
    result, _ = evaluation_result
    assert result.summary["equivalent_full_cycles"] > 0


def test_profit_factor_positive(evaluation_result):
    result, _ = evaluation_result
    assert result.summary["profit_factor"] > 0


def test_operational_hours(evaluation_result):
    result, dispatch_len = evaluation_result
    total_hours = (
        result.summary["charging_hours"]
        + result.summary["discharging_hours"]
        + result.summary["idle_hours"]
    )
    assert total_hours == dispatch_len