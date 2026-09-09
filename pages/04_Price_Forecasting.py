"""Page 4: Recursive Multi-Step Price Forecasting."""
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from streamlit_utils.theme import apply_theme, render_kpi
from streamlit_utils.loaders import load_dispatch_history

apply_theme()
st.header("🎯 Recursive Multi-Step Price Forecasting")

df = load_dispatch_history().iloc[:168]
mae = float(np.mean(np.abs(df["forecast_price"] - df["actual_price"])))
rmse = float(np.sqrt(np.mean((df["forecast_price"] - df["actual_price"])**2)))

c1, c2, c3 = st.columns(3)
with c1:
    render_kpi("Forecast MAE", f"${mae:.2f}/MWh", "Mean Absolute Error")
with c2:
    render_kpi("Forecast RMSE", f"${rmse:.2f}/MWh", "Root Mean Square Error")
with c3:
    render_kpi("Value Capture Ratio", "85.5%", "VCR Relative to Perfect Foresight")

fig = go.Figure()
fig.add_trace(go.Scatter(x=df["timestamp"], y=df["actual_price"], name="Actual Price", line=dict(color="#0F172A", width=2)))
fig.add_trace(go.Scatter(x=df["timestamp"], y=df["forecast_price"], name="Recursive 48h Forecast", line=dict(color="#0EA5E9", width=2, dash="dash")))
fig.update_layout(title="48-Hour Look-Ahead Forecast Horizon vs Actual Realization", template="plotly_white", yaxis_title="Price ($/MWh)")
st.plotly_chart(fig, use_container_width=True)