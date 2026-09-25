"""
pages/05_Dispatch_Optimization.py
=================================
Rolling Mathematical Dispatch Optimization.
"""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from streamlit_utils.charts import plot_dispatch_trajectory
from streamlit_utils.loaders import load_dispatch_history
from streamlit_utils.theme import apply_theme

apply_theme()
st.header("⚡ Rolling Mathematical Dispatch Optimization (Pyomo)")

df = load_dispatch_history()

if df is not None and not df.empty:
    df_slice = df.iloc[:168].copy()
    st.plotly_chart(plot_dispatch_trajectory(df_slice), use_container_width=True)

    st.subheader("State of Charge (SOC) Storage Dynamics")
    soc_col = "soc" if "soc" in df_slice.columns else df_slice.columns[0]
    fig_soc = px.line(
        df_slice,
        x="timestamp",
        y=soc_col,
        title="Hourly State of Charge Trajectory",
        labels={soc_col: "State of Charge (SOC Fraction)"},
    )
    fig_soc.add_hline(y=0.9, line_dash="dash", line_color="red", annotation_text="Upper Bound (90%)")
    fig_soc.add_hline(y=0.1, line_dash="dash", line_color="red", annotation_text="Lower Bound (10%)")
    st.plotly_chart(fig_soc, use_container_width=True)
else:
    st.warning("Dispatch trajectory data is unavailable.")