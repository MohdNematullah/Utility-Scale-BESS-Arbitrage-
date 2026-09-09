"""
streamlit_utils/downloads.py
============================
Helper utilities for exporting in-memory tables, manifests, and ZIP archives.
"""

from __future__ import annotations
import io
import zipfile
from pathlib import Path
import pandas as pd
import streamlit as st


def to_excel_buffer(dfs: dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for sheet_name, df in dfs.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    return buf.getvalue()


def create_zip_archive(files: list[Path]) -> bytes:
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            if f.exists() and f.is_file():
                zf.write(f, arcname=f.name)
    return zip_buf.getvalue()