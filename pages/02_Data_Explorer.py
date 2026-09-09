"""Page 2: Wholesale Market Data Explorer."""
import streamlit as st
import plotly.express as px
from streamlit_utils.theme import apply_theme
from streamlit_utils.loaders import load_dispatch_history

apply_theme()
st.header("📈 Wholesale Market Price Explorer")

df = load_dispatch_history()
st.sidebar.subheader("Time Filters")
start_idx = st.sidebar.slider("Start Hour", 0, len(df) - 168, 0)
df_slice = df.iloc[start_idx : start_idx + 168]

fig_ts = px.line(df_slice, x="timestamp", y="actual_price", title="Wholesale Spot Prices (168-Hour Window)", labels={"actual_price": "Price ($/MWh)"})
st.plotly_chart(fig_ts, use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    fig_hist = px.histogram(df, x="actual_price", nbins=60, title="Annual Price Distribution", labels={"actual_price": "Price ($/MWh)"}, color_discrete_sequence=["#0EA5E9"])
    st.plotly_chart(fig_hist, use_container_width=True)
with c2:
    st.subheader("Statistical Summary")
    st.dataframe(df["actual_price"].describe().to_frame(), use_container_width=True)