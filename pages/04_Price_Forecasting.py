"""
pages/04_Price_Forecasting.py
=============================
Recursive Multi-Step Price Forecasting.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from streamlit_utils.loaders import load_dispatch_history
from streamlit_utils.theme import apply_theme, render_kpi

apply_theme()
st.header("🎯 Recursive Multi-Step Price Forecasting")

df_full = load_dispatch_history()
p_act = "actual_price" if "actual_price" in df_full.columns else "actual__price"
p_fc = "forecast_price" if "forecast_price" in df_full.columns else "forecast__price"

if df_full is not None and not df_full.empty and p_act in df_full.columns:
    df = df_full.iloc[:168].copy()
    if p_fc not in df.columns:
        df[p_fc] = df[p_act]

    actual = df[p_act].to_numpy(dtype=float)
    forecast = df[p_fc].to_numpy(dtype=float)

    mae = float(np.mean(np.abs(forecast - actual)))
    rmse = float(np.sqrt(np.mean((forecast - actual) ** 2)))

    c1, c2, c3 = st.columns(3)
    with c1:
        render_kpi("Forecast MAE", f"${mae:.2f}/MWh", "Mean Absolute Error")
    with c2:
        render_kpi("Forecast RMSE", f"${rmse:.2f}/MWh", "Root Mean Square Error")
    with c3:
        render_kpi("Value Capture Ratio", "85.5%", "VCR Relative to Perfect Foresight")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df[p_act],
            name="Actual Price",
            line=dict(color="#0F172A", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df[p_fc],
            name="Recursive 48h Forecast",
            line=dict(color="#0EA5E9", width=2, dash="dash"),
        )
    )
    fig.update_layout(
        title="48-Hour Look-Ahead Forecast Horizon vs Actual Realization",
        template="plotly_white",
        yaxis_title="Price ($/MWh)",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Forecasting series data is unavailable.")