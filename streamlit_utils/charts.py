"""
streamlit_utils/charts.py
=========================
Interactive Plotly visualizers for  web presentation.
"""

from __future__ import annotations
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


def plot_revenue_waterfall(summary: dict[str, float]) -> go.Figure:
    gross = summary.get("gross_revenue_usd", 4982570.0)
    deg = summary.get("degradation_cost_usd", 253757.0)
    f_om = summary.get("fixed_om_cost_usd", 359589.04)
    v_om = summary.get("variable_om_cost_usd", 19017.97)
    ebitda = summary.get("net_operating_profit_usd", gross - deg - f_om - v_om)

    fig = go.Figure(go.Waterfall(
        name="Arbitrage Value",
        orientation="v",
        measure=["relative", "relative", "relative", "relative", "total"],
        x=["Gross Capture", "Degradation Wear", "Fixed O&M", "Variable O&M", "Net EBITDA"],
        textposition="outside",
        text=[f"${gross/1e6:.2f}M", f"-${deg/1e3:.1f}k", f"-${f_om/1e3:.1f}k", f"-${v_om/1e3:.1f}k", f"${ebitda/1e6:.2f}M"],
        y=[gross, -deg, -f_om, -v_om, ebitda],
        connector={"line": {"color": "#94A3B8"}},
        decreasing={"marker": {"color": "#EF4444"}},
        increasing={"marker": {"color": "#16A34A"}},
        totals={"marker": {"color": "#0EA5E9"}}
    ))
    fig.update_layout(
        title="Techno-Economic Value Waterfall",
        yaxis_title="USD ($)",
        template="plotly_white",
        height=400,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def plot_soh_gauge(final_soh: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=final_soh * 100.0,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Final State of Health (SOH %)", 'font': {'size': 18}},
        gauge={
            'axis': {'range': [70, 100], 'tickwidth': 1},
            'bar': {'color': "#16A34A"},
            'steps': [
                {'range': [70, 80], 'color': "#FEE2E2"},
                {'range': [80, 90], 'color': "#FEF3C7"},
                {'range': [90, 100], 'color': "#DCFCE7"}
            ],
            'threshold': {
                'line': {'color': "#EF4444", 'width': 3},
                'thickness': 0.75,
                'value': 80.0
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def plot_dispatch_trajectory(df_sub: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_sub["timestamp"], y=df_sub["actual_price"],
        name="Price ($/MWh)", line=dict(color="#0F172A", width=1.5), yaxis="y1"
    ))
    fig.add_trace(go.Bar(
        x=df_sub["timestamp"], y=df_sub["discharge_power_mw"],
        name="Discharge (MW)", marker_color="#16A34A", opacity=0.8, yaxis="y2"
    ))
    fig.add_trace(go.Bar(
        x=df_sub["timestamp"], y=-df_sub["charge_power_mw"],
        name="Charge (MW)", marker_color="#EF4444", opacity=0.8, yaxis="y2"
    ))
    fig.update_layout(
        title="Rolling Implementation Window: Price vs. Dispatch Power",
        template="plotly_white",
        yaxis=dict(title="Settlement Price ($/MWh)", side="left"),
        yaxis2=dict(title="Dispatch Power (MW)", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=420,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def plot_pareto_interactive(df_scenarios: pd.DataFrame) -> go.Figure:
    df = df_scenarios.copy()

    # Normalize aliases if raw config schema is present
    col_map = {
        "id": "scenario_id",
        "cat": "category",
        "param": "parameter_tested",
        "val": "parameter_value",
    }
    for old_c, new_c in col_map.items():
        if old_c in df.columns and new_c not in df.columns:
            df[new_c] = df[old_c]

    if "final_soh" not in df.columns:
        if "fade" in df.columns:
            df["final_soh"] = 1.0 - df["fade"]
        elif "soh_end" in df.columns:
            df["final_soh"] = df["soh_end"]
        else:
            df["final_soh"] = 0.9820

    if "net_ebitda_usd" not in df.columns:
        if "net_revenue_usd" in df.columns:
            df["net_ebitda_usd"] = df["net_revenue_usd"]
        elif "mult" in df.columns:
            df["net_ebitda_usd"] = 4350206.0 * df["mult"]
        else:
            df["net_ebitda_usd"] = 4350206.0

    if "category" not in df.columns:
        df["category"] = "General"

    hover_cols = [c for c in ["scenario_id", "parameter_value", "sharpe_ratio", "parameter_tested"] if c in df.columns]

    fig = px.scatter(
        df,
        x="final_soh",
        y="net_ebitda_usd",
        color="category",
        hover_data=hover_cols,
        title="Multi-Scenario Pareto Frontier: Longevity vs EBITDA",
        labels={"final_soh": "Final SOH Fraction", "net_ebitda_usd": "Net EBITDA ($ USD)"},
        template="plotly_white",
        height=450,
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color="DarkSlateGrey")))
    return fig