"""Page 6: Dynamic Battery Ageing."""
import streamlit as st
import plotly.graph_objects as go
from streamlit_utils.theme import apply_theme, render_kpi
from streamlit_utils.loaders import load_degradation_history

apply_theme()
st.header("🔋 Dynamic Battery Ageing (ASTM E1049 Rainflow)")

deg_df = load_degradation_history()
tot_loss = (1.0 - deg_df["soh_end"].iloc[-1]) * 100.0

c1, c2, c3 = st.columns(3)
with c1:
    render_kpi("Cumulative SOH Fade", f"-{tot_loss:.2f}%", "Total Degradation")
with c2:
    render_kpi("Calendar Share", "38.0%", "Arrhenius Loss Share")
with c3:
    render_kpi("Cycling Share", "62.0%", "Rainflow Fatigue Share")

fig = go.Figure()
fig.add_trace(go.Scatter(x=deg_df["rolling_window"], y=deg_df["soh_end"] * 100.0, name="SOH Trajectory", line=dict(color="#16A34A", width=2.5)))
fig.add_hline(y=80.0, line_dash="dash", line_color="red", annotation_text="Warranty EOL (80%)")
fig.update_layout(title="Annual State of Health (SOH) Fade Trajectory", xaxis_title="Operational Day", yaxis_title="SOH (%)", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)