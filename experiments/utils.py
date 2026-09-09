"""
experiments/utils.py
====================

Shared utilities for Part 12 Research & Reproducibility Suite.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def compute_file_sha256(filepath: Path | str) -> str:
    """Computes standard SHA256 checksum of a file on disk."""
    p = Path(filepath)
    if not p.is_file():
        raise FileNotFoundError(f"File not found for hashing: {p}")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_dataframe_sha256(df: pd.DataFrame) -> str:
    """Computes a deterministic hash of a pandas DataFrame."""
    buf = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(buf).hexdigest()


def compute_dict_sha256(data: dict[str, Any]) -> str:
    """Computes a deterministic hash of a dictionary with sorted keys."""
    serialized = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def format_currency(val: float) -> str:
    """Formats float to currency string without logging format bugs."""
    return f"${val:,.2f}"


def format_percentage(val: float, decimals: int = 2) -> str:
    """Formats float to clean percentage string."""
    return f"{val:.{decimals}f}%"