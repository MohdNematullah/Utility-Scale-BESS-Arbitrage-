"""
pages/12_Experiment_Runner.py
=============================
Interactive Experiment Suite Runner.
"""

from __future__ import annotations

import streamlit as st
from streamlit_utils.theme import apply_theme
from experiments.experiment_suite import ExperimentSuiteRunner

apply_theme()
st.header("🧪 Automated Experiment Suite (28 Scenarios)")

st.markdown(
    "Executes the full 28-scenario sensitivity sweep across forecast horizons, "
    "predictive models, cell chemistries, operating temperatures, asset sizing, "
    "round-trip efficiencies, and degradation wear hurdles."
)

if st.button("Execute All 28 Scenarios Batch Sweep"):
    with st.spinner("Executing simulation sweeps across 28 scenarios..."):
        try:
            runner = ExperimentSuiteRunner(output_dir="results/experiments")
            df_res = runner.run_all_scenarios()
            st.success(f"Successfully completed all 28 experimental scenarios ({len(df_res)} records generated)!")
            st.dataframe(df_res, use_container_width=True)
        except Exception as exc:
            st.error(f"Experiment execution failed: {exc}")