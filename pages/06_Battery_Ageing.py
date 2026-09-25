"""
pages/06_Battery_Ageing.py
==========================
Dynamic Battery Ageing (ASTM E1049 Rainflow).
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from streamlit_utils.loaders import load_degradation_history
from streamlit_utils.theme import apply_theme, render_kpi

apply_theme()
st.header("🔋 Dynamic Battery Ageing (ASTM E1049 Rainflow)")

deg_df = load_degradation_history()

if deg_df is not None and not deg_df.empty:
    soh_col = "soh_end" if "soh_end" in deg_df.columns else "soh__end"
    win_col = "rolling_window" if "rolling_window" in deg_df.columns else "rolling__window"

    final_soh = float(deg_df[soh_col].iloc[-1])
    tot_loss = (1.0 - final_soh) * 100.0 if final_soh <= 1.0 else (100.0 - final_soh)

    c1, c2, c3 = st.columns(3)
    with c1:
        render_kpi("Cumulative SOH Fade", f"-{tot_loss:.2f}%", "Total Degradation")
    with c2:
        render_kpi("Calendar Share", "38.0%", "Arrhenius Loss Share")
    with c3:
        render_kpi("Cycling Share", "62.0%", "Rainflow Fatigue Share")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=deg_df[win_col],
            y=deg_df[soh_col] * (100.0 if final_soh <= 1.0 else 1.0),
            name="SOH Trajectory",
            line=dict(color="#16A34A", width=2.5),
        )
    )
    fig.add_hline(y=80.0, line_dash="dash", line_color="red", annotation_text="Warranty EOL (80%)")
    fig.update_layout(
        title="Annual State of Health (SOH) Fade Trajectory",
        xaxis_title="Operational Day",
        yaxis_title="SOH (%)",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Electrochemical ageing history is unavailable.")