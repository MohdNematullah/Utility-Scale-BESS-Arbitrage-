"""
pages/02_Data_Explorer.py
=========================
Wholesale Market Data Explorer.
"""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from streamlit_utils.loaders import load_dispatch_history
from streamlit_utils.theme import apply_theme

apply_theme()
st.header("📈 Wholesale Market Price Explorer")

df = load_dispatch_history()
price_col = "actual_price" if "actual_price" in df.columns else "actual__price"

if df is not None and not df.empty and price_col in df.columns:
    st.sidebar.subheader("Time Filters")
    max_slider = max(0, len(df) - 168)
    start_idx = (
        st.sidebar.slider("Start Hour", 0, max_slider, 0)
        if max_slider > 0
        else 0
    )
    df_slice = df.iloc[start_idx : start_idx + 168]

    fig_ts = px.line(
        df_slice,
        x="timestamp",
        y=price_col,
        title="Wholesale Spot Prices (168-Hour Window)",
        labels={price_col: "Price ($/MWh)"},
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_hist = px.histogram(
            df,
            x=price_col,
            nbins=60,
            title="Annual Price Distribution",
            labels={price_col: "Price ($/MWh)"},
            color_discrete_sequence=["#0EA5E9"],
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    with c2:
        st.subheader("Statistical Summary")
        st.dataframe(df[price_col].describe().to_frame(), use_container_width=True)
else:
    st.warning("Dispatch time series data is unavailable.")