"""
streamlit_utils/theme.py
========================
Styling and layout utilities for  Streamlit Portal.
"""

from __future__ import annotations
import streamlit as st

PRIMARY_COLOR = "#16A34A"
SECONDARY_COLOR = "#0EA5E9"
ACCENT_COLOR = "#F59E0B"
DANGER_COLOR = "#EF4444"
DARK_TEXT = "#0F172A"
LIGHT_BG = "#F8FAFC"


def apply_theme() -> None:
    """Injects institutional styling across components."""
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: #FFFFFF;
            color: {DARK_TEXT};
        }}
        .kpi-card {{
            background: {LIGHT_BG};
            padding: 1.25rem;
            border-radius: 0.5rem;
            border-left: 4px solid {PRIMARY_COLOR};
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
            margin-bottom: 1rem;
        }}
        .kpi-label {{
            font-size: 0.825rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748B;
        }}
        .kpi-value {{
            font-size: 1.75rem;
            font-weight: 700;
            color: {DARK_TEXT};
            margin-top: 0.25rem;
        }}
        .kpi-subtext {{
            font-size: 0.75rem;
            color: #94A3B8;
            margin-top: 0.25rem;
        }}
        div[data-testid="stSidebarNav"] {{
            padding-top: 1rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_kpi(label: str, value: str, subtext: str = "", border_color: str = PRIMARY_COLOR) -> None:
    """Renders a formatted card component."""
    st.markdown(
        f"""
        <div class="kpi-card" style="border-left-color: {border_color};">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-subtext">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )