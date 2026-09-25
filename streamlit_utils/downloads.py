"""
streamlit_utils/downloads.py
============================
Helper utilities for exporting in-memory tables, manifests, and ZIP archives.
"""

from __future__ import annotations

import io
from pathlib import Path
import zipfile
import pandas as pd


def to_excel_buffer(dfs: dict[str, pd.DataFrame]) -> bytes:
    """Converts a dictionary of dataframes into a multi-tab Excel workbook byte stream."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for sheet_name, df in dfs.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    return buf.getvalue()


def create_zip_archive(files: list[Path]) -> bytes:
    """Bundles a list of disk files into a single ZIP archive byte stream."""
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            path_obj = Path(f)
            if path_obj.exists() and path_obj.is_file():
                zf.write(path_obj, arcname=path_obj.name)
    return zip_buf.getvalue()