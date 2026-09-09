"""Page 7: Rolling Horizon Backtesting."""
import streamlit as st
import plotly.express as px
from streamlit_utils.theme import apply_theme
from streamlit_utils.loaders import load_dispatch_history

apply_theme()
st.header("🔄 Rolling-Horizon Backtesting Chronology")

df = load_dispatch_history()
df["cumulative_profit"] = df["net_revenue_usd"].cumsum()

fig_rev = px.line(df, x="timestamp", y="cumulative_profit", title="Cumulative Arbitrage Profit Trajectory ($ USD)", labels={"cumulative_profit": "Gross Revenue ($)"})
st.plotly_chart(fig_rev, use_container_width=True)