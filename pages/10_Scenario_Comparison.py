"""
pages/10_Scenario_Comparison.py
===============================
Multi-Scenario Sensitivity & Pareto Frontier.
"""

import streamlit as st
from streamlit_utils.charts import plot_pareto_interactive
from streamlit_utils.loaders import load_scenario_matrix
from streamlit_utils.theme import apply_theme

apply_theme()
st.header("⚖️ Multi-Scenario Sensitivity & Pareto Optimization")

df_scen = load_scenario_matrix()
st.plotly_chart(plot_pareto_interactive(df_scen), use_container_width=True)

st.subheader("28-Scenario Experimental Matrix")
st.dataframe(df_scen, use_container_width=True)