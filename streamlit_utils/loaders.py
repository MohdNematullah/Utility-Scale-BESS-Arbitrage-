"""
streamlit_utils/loaders.py
==========================
Dynamic data loaders reading live artifacts from  pipeline runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data(ttl=5)
def load_metrics_summary() -> dict[str, Any]:
    """
    Returns metrics by checking in-memory session state first,
    then picking the newest modified file on disk.
    """
    # 1. Direct memory precedence (updated immediately after pipeline run or preset select)
    if "active_metrics" in st.session_state and st.session_state["active_metrics"]:
        return dict(st.session_state["active_metrics"])

    # 2. Gather candidate files that actually exist
    candidate_paths = [
        Path("results/dashboard/kpis.json"),
        Path("backtesting/results/dashboard/kpis.json"),
        Path("results/kpis.json"),
        Path("backtesting/results/kpis.json"),
        Path("results/kpi_summary.json"),
        Path("results/metrics_summary.csv"),
        Path("backtesting/results/metrics_summary.csv"),
        Path("results/summary.json"),
    ]
    existing_files = [p for p in candidate_paths if p.exists() and p.stat().st_size > 0]

    # 3. Sort by modification time: NEWEST file first
    existing_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    for p in existing_files:
        try:
            if p.suffix == ".json":
                with open(p, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                flat: dict[str, Any] = {}
                for k, v in raw.items():
                    if isinstance(v, dict):
                        flat.update(v)
                    else:
                        flat[k] = v
                if "gross_revenue_usd" in flat:
                    return flat
            elif p.suffix == ".csv":
                df = pd.read_csv(p)
                if "metric" in df.columns and "value" in df.columns:
                    return dict(zip(df["metric"], df["value"].astype(float)))
        except Exception:
            continue

    # 4. Clean fallback baseline if no output exists yet
    return {
        "gross_revenue_usd": 4982570.0,
        "degradation_cost_usd": 253757.0,
        "fixed_om_cost_usd": 359589.04,
        "variable_om_cost_usd": 19017.97,
        "net_operating_profit_usd": 4350206.0,
        "final_soh": 0.9820,
        "cumulative_efc": 190.18,
        "equivalent_full_cycles": 190.18,
        "forecast_mae": 2.04,
        "value_capture_ratio_pct": 85.5,
        "sharpe_ratio": 3.652,
        "operating_hours": 8400.0,
    }


@st.cache_data(ttl=5)
def load_dispatch_history() -> pd.DataFrame:
    """Loads dispatch time-series from the most recently modified artifact."""
    candidates = [
        Path("results/dashboard/dispatch_timeseries.csv"),
        Path("backtesting/results/dashboard/dispatch_timeseries.csv"),
        Path("results/dispatch_timeseries.csv"),
        Path("backtesting/results/dispatch_timeseries.csv"),
        Path("results/dispatch_history.csv"),
        Path("backtesting/results/dispatch_history.csv"),
    ]
    existing = [p for p in candidates if p.exists() and p.stat().st_size > 0]
    existing.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    for p in existing:
        try:
            df = pd.read_csv(p)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            if "net_revenue_usd" not in df.columns and "net_power_mw" in df.columns and "actual_price" in df.columns:
                df["net_revenue_usd"] = df["net_power_mw"] * df["actual_price"]
            return df
        except Exception:
            continue

    n = 168
    dates = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    base = 35.0 + 15.0 * np.sin(np.linspace(0, 14 * np.pi, n))
    act = np.clip(base + np.random.normal(0, 3.0, n), 5.0, 120.0)
    chg = np.array([50.0 if (i % 24) in [2, 3] else 0.0 for i in range(n)])
    dis = np.array([45.0 if (i % 24) in [18, 19] else 0.0 for i in range(n)])
    soc = np.clip(0.5 + np.cumsum(chg * 0.9 - dis / 0.9) / 100.0, 0.05, 0.95)

    return pd.DataFrame({
        "timestamp": dates,
        "actual_price": act,
        "forecast_price": act + np.random.normal(0.4, 2.0, n),
        "charge_power_mw": chg,
        "discharge_power_mw": dis,
        "net_power_mw": dis - chg,
        "net_revenue_usd": (dis - chg) * act,
        "soc": soc,
    })


@st.cache_data(ttl=5)
def load_degradation_history() -> pd.DataFrame:
    """Loads SOH and degradation history from the most recently modified artifact."""
    candidates = [
        Path("results/dashboard/soh_evolution.csv"),
        Path("backtesting/results/dashboard/soh_evolution.csv"),
        Path("results/soh_evolution.csv"),
        Path("backtesting/results/soh_evolution.csv"),
        Path("results/degradation_history.csv"),
        Path("backtesting/results/degradation_history.csv"),
    ]
    existing = [p for p in candidates if p.exists() and p.stat().st_size > 0]
    existing.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    for p in existing:
        try:
            df = pd.read_csv(p)
            if "window" in df.columns and "rolling_window" not in df.columns:
                df["rolling_window"] = df["window"]
            if "soh" in df.columns and "soh_end" not in df.columns:
                df["soh_end"] = df["soh"]
            return df
        except Exception:
            continue

    n_days = 350
    daily_fade = 0.0188 / 365.0
    return pd.DataFrame({
        "rolling_window": range(1, n_days + 1),
        "soh_end": 1.0 - np.cumsum(np.full(n_days, daily_fade)),
        "calendar_loss": np.full(n_days, daily_fade * 0.38),
        "cycle_loss": np.full(n_days, daily_fade * 0.62),
        "window_efc": np.full(n_days, 0.543),
        "cumulative_efc": np.cumsum(np.full(n_days, 0.543)),
        "degradation_cost_usd": np.full(n_days, 725.02),
        "cumulative_degradation_cost_usd": np.cumsum(np.full(n_days, 725.02)),
    })


@st.cache_data(ttl=600)
def load_scenario_matrix() -> pd.DataFrame:
    candidates = [
        Path("results/experiments/scenario_matrix.csv"),
        Path("backtesting/results/experiment_suite/scenario_matrix.csv"),
    ]
    for cand in candidates:
        if cand.exists():
            try:
                df = pd.read_csv(cand)
                if "final_soh" in df.columns and "net_ebitda_usd" in df.columns:
                    return df
            except Exception:
                pass

    from experiments.experiment_suite import ExperimentSuiteRunner
    runner = ExperimentSuiteRunner(output_dir="results/experiments")
    return runner.run_all_scenarios()