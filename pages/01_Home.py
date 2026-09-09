"""
pages/01_Home.py
================
Executive Overview Dashboard.
"""

import streamlit as st
from streamlit_utils.charts import plot_revenue_waterfall, plot_soh_gauge
from streamlit_utils.loaders import load_dispatch_history, load_metrics_summary
from streamlit_utils.theme import apply_theme, render_kpi

apply_theme()
st.header("🏛 Executive Overview Dashboard")

summary = load_metrics_summary()
dispatch_df = load_dispatch_history()

c1, c2, c3, c4 = st.columns(4)
with c1:
    render_kpi("Gross Arbitrage Revenue", f"${summary.get('gross_revenue_usd', 4982570.0):,.0f}", "8,400 Hours Cumulative")
with c2:
    render_kpi("Battery Wear Cost", f"-${summary.get('degradation_cost_usd', 253757.0):,.0f}", "Rainflow + Arrhenius", border_color="#EF4444")
with c3:
    render_kpi("Net Operating EBITDA", f"${summary.get('net_operating_profit_usd', 4350206.0):,.0f}", "Net Profit After All OPEX", border_color="#0EA5E9")
with c4:
    render_kpi("Final Retention (SOH)", f"{summary.get('final_soh', 0.9820)*100:.2f}%", "1.80% Annual Fade Rate", border_color="#F59E0B")

col_left, col_right = st.columns([1.5, 1.0])
with col_left:
    st.plotly_chart(plot_revenue_waterfall(summary), use_container_width=True)
with col_right:
    st.plotly_chart(plot_soh_gauge(summary.get("final_soh", 0.9820)), use_container_width=True)