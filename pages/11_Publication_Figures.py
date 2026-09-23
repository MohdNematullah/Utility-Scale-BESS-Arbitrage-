"""
pages/11_Publication_Figures.py
===============================
Publication Figures Inspection and Multi-Format Export Suite.
"""

from pathlib import Path
import streamlit as st
from streamlit_utils.theme import apply_theme

apply_theme()
st.header("ðŸ–¼ Publication Figures Suite")

fig_dirs = [
    Path("results/_figures/png"),
    Path("results/figures/png"),
    Path("results/_figures"),
    Path("results/figures"),
]

fig_dir = next((d for d in fig_dirs if d.exists() and any(d.glob("*.png"))), None)

if fig_dir is not None:
    figs = sorted(list(fig_dir.glob("*.png")))
    st.write(f"Found **{len(figs)}** publication-grade figures on disk.")

    col_select, _ = st.columns([2, 1])
    with col_select:
        sel_fig_name = st.selectbox("Select Figure to Inspect", [f.name for f in figs])

    if sel_fig_name:
        sel_path = fig_dir / sel_fig_name
        st.image(str(sel_path), caption=sel_fig_name, use_container_width=True)

        with open(sel_path, "rb") as f:
            st.download_button(
                label=f"â¬‡ Download {sel_fig_name} (PNG)",
                data=f.read(),
                file_name=sel_fig_name,
                mime="image/png",
            )
else:
    st.warning("Figures not found on disk. Click 'â–¶ Run Full' in the sidebar or run `python main.py --figures`.")