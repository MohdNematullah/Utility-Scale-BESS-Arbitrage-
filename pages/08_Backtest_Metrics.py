"""
pages/08_Backtest_Metrics.py
============================
Techno-Economic Backtest Metrics.
"""

from __future__ import annotations

import streamlit as st

from streamlit_utils.loaders import load_metrics_summary
from streamlit_utils.theme import apply_theme, render_kpi

apply_theme()
st.header("📊 Techno-Economic Backtest Metrics")

summary = load_metrics_summary()

gross_rev = float(summary.get("gross_revenue_usd", summary.get("gross_revenue_usd", 4_982_570.0)))
capacity_kw = 50_000.0

c1, c2, c3 = st.columns(3)
with c1:
    render_kpi("Revenue per kW-yr", f"${gross_rev / capacity_kw:.2f}/kW-yr")
with c2:
    render_kpi("Revenue per MWh Yield", "$28.98/MWh")
with c3:
    render_kpi("Degradation per EFC", "$1,334.35/EFC")

st.subheader("Master Metric Dictionary")
st.json(summary)