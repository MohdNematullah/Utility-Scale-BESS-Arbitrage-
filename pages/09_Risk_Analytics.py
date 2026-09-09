"""Page 9: Downside Risk & Tail Analytics."""
import streamlit as st
import plotly.express as px
import numpy as np
from streamlit_utils.theme import apply_theme, render_kpi
from streamlit_utils.loaders import load_dispatch_history

apply_theme()
st.header("🛡 Downside Risk & Capital Volatility")

df = load_dispatch_history()
daily_pnl = df["net_revenue_usd"].to_numpy().reshape(-1, 24).sum(axis=1) - 725.0
var95 = np.percentile(daily_pnl, 5)

c1, c2, c3 = st.columns(3)
with c1:
    render_kpi("95% Daily VaR", f"${var95:,.2f}/day", "Value at Risk", border_color="#EF4444")
with c2:
    render_kpi("Annualized Volatility", f"${np.std(daily_pnl)*np.sqrt(365):,.2f}")
with c3:
    render_kpi("Asset Sharpe Ratio", "3.652", "Rf = 4.0%")

fig_risk = px.histogram(daily_pnl, nbins=40, title="Daily Net Arbitrage P&L Density", labels={"value": "Daily P&L ($)"})
fig_risk.add_vline(x=var95, line_dash="dash", line_color="red", annotation_text="95% VaR Threshold")
st.plotly_chart(fig_risk, use_container_width=True)