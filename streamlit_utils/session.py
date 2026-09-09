"""
streamlit_utils/session.py
==========================
Session State initialization and non-destructive parameter management for .
"""

from __future__ import annotations
from pathlib import Path
import streamlit as st

DEFAULTS = {
    "power_mw": 50.0,
    "capacity_mwh": 100.0,
    "chemistry": "NMC",
    "rte_pct": 90.25,
    "horizon_h": 48,
    "step_h": 24,
    "cell_temp_c": 25.0,
    "wear_hurdle_usd": 10.0,
    "forecast_model": "Recursive XGBoost (Dynamic)",
    "degradation_mode": "Rainflow + Arrhenius (Physical)",
    "selected_scenario_id": "chem_nmc_baseline",
    "results_dir": Path("results"),
    "pipeline_executed": False,
    "experiment_executed": False,
    "last_run_config": None,
}


def init_session_state() -> None:
    """Initializes global simulation defaults in Streamlit session state."""
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_configuration() -> None:
    """
    Surgically resets user controls to baseline defaults without
    destroying pipeline cache, telemetry datasets, or navigation state.
    """
    config_keys = [
        "power_mw",
        "capacity_mwh",
        "chemistry",
        "rte_pct",
        "horizon_h",
        "step_h",
        "cell_temp_c",
        "wear_hurdle_usd",
        "forecast_model",
        "degradation_mode",
        "selected_scenario_id",
    ]
    for k in config_keys:
        st.session_state[k] = DEFAULTS[k]