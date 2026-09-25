"""
pages/07_Rolling_Backtesting.py
===============================
Rolling Horizon Backtesting Chronology.
"""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from streamlit_utils.loaders import load_dispatch_history
from streamlit_utils.theme import apply_theme

apply_theme()
st.header("🔄 Rolling-Horizon Backtesting Chronology")

df = load_dispatch_history()
rev_col = "net_revenue_usd" if "net_revenue_usd" in df.columns else "net_revenue_usd"

if df is not None and not df.empty and rev_col in df.columns:
    df = df.copy()
    df["cumulative_profit"] = df[rev_col].cumsum()

    fig_rev = px.line(
        df,
        x="timestamp",
        y="cumulative_profit",
        title="Cumulative Arbitrage Profit Trajectory ($ USD)",
        labels={"cumulative_profit": "Gross Revenue ($)"},
    )
    st.plotly_chart(fig_rev, use_container_width=True)
else:
    st.warning("Backtesting revenue history is unavailable.")