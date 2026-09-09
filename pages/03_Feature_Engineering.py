"""Page 3: Temporal Feature Engineering."""
import streamlit as st
import plotly.express as px
from streamlit_utils.theme import apply_theme
from streamlit_utils.loaders import load_dispatch_history

apply_theme()
st.header("🧪 Feature Engineering & Signal Decomposition")

df = load_dispatch_history().copy()
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["price_lag_24h"] = df["actual_price"].shift(24).bfill()
df["price_roll_24h"] = df["actual_price"].rolling(24, min_periods=1).mean()

st.subheader("Predictor Correlation Heatmap")
corr = df[["actual_price", "price_lag_24h", "price_roll_24h", "hour", "day_of_week"]].corr()
fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="Viridis")
st.plotly_chart(fig_corr, use_container_width=True)

st.subheader("Sample Engineered Predictor Features")
st.dataframe(df[["timestamp", "actual_price", "price_lag_24h", "price_roll_24h", "hour", "day_of_week"]].head(100), use_container_width=True)