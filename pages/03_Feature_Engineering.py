"""
pages/03_Feature_Engineering.py
===============================
Temporal Feature Engineering & Signal Decomposition.
"""

from __future__ import annotations

import plotly.express as px
import pandas as pd
import streamlit as st

from streamlit_utils.loaders import load_dispatch_history
from streamlit_utils.theme import apply_theme

apply_theme()
st.header("🧪 Feature Engineering & Signal Decomposition")

df = load_dispatch_history().copy()
price_col = "actual_price" if "actual_price" in df.columns else "actual__price"

if df is not None and not df.empty and price_col in df.columns:
    # Ensure timestamp is parsed as datetime for dt accessors
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["price_lag_24h"] = df[price_col].shift(24).bfill()
    df["price_roll_24h"] = df[price_col].rolling(24, min_periods=1).mean()

    st.subheader("Predictor Correlation Heatmap")
    feature_cols = [price_col, "price_lag_24h", "price_roll_24h", "hour", "day_of_week"]
    corr = df[feature_cols].corr()

    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="Viridis",
        title="Predictive Signal Cross-Correlation Matrix",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Sample Engineered Predictor Features")
    st.dataframe(
        df[["timestamp"] + feature_cols].head(100),
        use_container_width=True,
    )
else:
    st.warning("Feature data is currently unavailable.")