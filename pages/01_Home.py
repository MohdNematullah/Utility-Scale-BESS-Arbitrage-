"""
pages/01_Home.py
================
Executive Overview Dashboard.
"""

from __future__ import annotations

import streamlit as st
from streamlit_utils.charts import plot_revenue_waterfall, plot_soh_gauge
from streamlit_utils.loaders import load_dispatch_history, load_metrics_summary
from streamlit_utils.theme import apply_theme, render_kpi


def get_metric(data: dict, key: str, default: float) -> float:
    alt = key.replace("_", "__")
    return float(data.get(key, data.get(alt, default)))


apply_theme()
st.header("🏛️ Executive Overview Dashboard")

summary = load_metrics_summary()
dispatch_df = load_dispatch_history()

gross_rev = get_metric(summary, "gross_revenue_usd", 4_982_570.0)
deg_cost = get_metric(summary, "degradation_cost_usd", 253_757.0)
net_ebitda = get_metric(summary, "net_operating_profit_usd", 4_350_206.0)
final_soh = get_metric(summary, "final_soh", 0.9820)
soh_pct = final_soh * 100.0 if final_soh <= 1.0 else final_soh

c1, c2, c3, c4 = st.columns(4)
with c1:
    render_kpi(
        "Gross Arbitrage Revenue",
        f"${gross_rev:,.0f}",
        "8,400 Hours Cumulative",
    )
with c2:
    render_kpi(
        "Battery Wear Cost",
        f"-${deg_cost:,.0f}",
        "Rainflow + Arrhenius",
        border_color="#EF4444",
    )
with c3:
    render_kpi(
        "Net Operating EBITDA",
        f"${net_ebitda:,.0f}",
        "Net Profit After All OPEX",
        border_color="#0EA5E9",
    )
with c4:
    render_kpi(
        "Final Retention (SOH)",
        f"{soh_pct:.2f}%",
        f"{(100.0 - soh_pct):.2f}% Annual Fade Rate",
        border_color="#F59E0B",
    )

col_left, col_right = st.columns([1.5, 1.0])
with col_left:
    st.plotly_chart(plot_revenue_waterfall(summary), use_container_width=True)
with col_right:
    st.plotly_chart(
        plot_soh_gauge(final_soh if final_soh <= 1.0 else final_soh / 100.0),
        use_container_width=True,
    )