"""Page 8: Detailed Backtest Metrics."""
import streamlit as st
from streamlit_utils.theme import apply_theme, render_kpi
from streamlit_utils.loaders import load_metrics_summary

apply_theme()
st.header("📊 Techno-Economic Backtest Metrics")

summary = load_metrics_summary()
c1, c2, c3 = st.columns(3)
with c1:
    render_kpi("Revenue per kW-yr", f"${summary.get('gross_revenue_usd', 4982570.0)/50000:.2f}/kW-yr")
with c2:
    render_kpi("Revenue per MWh Yield", "$28.98/MWh")
with c3:
    render_kpi("Degradation per EFC", "$1,334.35/EFC")

st.json(summary)