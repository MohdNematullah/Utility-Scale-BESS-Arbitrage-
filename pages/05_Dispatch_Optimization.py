"""Page 5: Rolling Mathematical Dispatch Optimization."""
import streamlit as st
from streamlit_utils.theme import apply_theme
from streamlit_utils.loaders import load_dispatch_history
from streamlit_utils.charts import plot_dispatch_trajectory

apply_theme()
st.header("⚡ Rolling Mathematical Dispatch Optimization (Pyomo)")

df = load_dispatch_history().iloc[:168]
st.plotly_chart(plot_dispatch_trajectory(df), use_container_width=True)

st.subheader("State of Charge (SOC) Storage Dynamics")
import plotly.express as px
fig_soc = px.line(df, x="timestamp", y="soc", title="Hourly State of Charge Trajectory", labels={"soc": "State of Charge (SOC Fraction)"})
fig_soc.add_hline(y=0.9, line_dash="dash", line_color="red", annotation_text="Upper Bound (90%)")
fig_soc.add_hline(y=0.1, line_dash="dash", line_color="red", annotation_text="Lower Bound (10%)")
st.plotly_chart(fig_soc, use_container_width=True)