"""
visualization/figure_style.py
=============================

Global IEEE/Elsevier/Nature Figure Style & Theming Framework (Part 10.1)



Capabilities:
1. Publication rcParams Engine:
   - Formatted for IEEE Trans. on Smart Grid / Energy Storage journals.
   - Standardized font typography: DejaVu Sans with STIX mathtext rendering.
   - 600 DPI print-ready export resolution and 300 DPI screen previews.
2. Color-Blind Safe Semantic Color Palette (Okabe-Ito & IEEE BESS Standards):
   - Actual Price: Charcoal Black (#111111)
   - Forecast Price: Royal Blue (#0072B2)
   - Perfect Foresight: Steel Cyan (#56B4E9)
   - Battery Charging: Emerald Green (#009E73)
   - Battery Discharging: Vermillion / Orange (#D55E00)
   - State of Health (SOH): Royal Purple (#7B1FA2)
   - Net Arbitrage Profit: Deep Teal (#008080)
   - Degradation Wear & Loss: Crimson Red (#D62728)
   - Auxiliary Grid Lines: Light Slate (#E0E0E0)
3. Journal Layout & Golden Ratio Dimensioning:
   - IEEE single column: 3.5 in (88.9 mm)
   - IEEE double column: 7.0 in (177.8 mm)
   - Thesis full-page: 6.5 in (165.1 mm)
   - Thesis half-page: 4.8 in (121.9 mm)
4. Multi-Format Exporter:
   - Synchronous export to PNG, PDF, SVG, and TIFF formats.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ============================================================================
# Color Palette Definitions (Color-Blind Safe & Semantic Roles)
# ============================================================================

@dataclass(frozen=True, slots=True)
class ColorPalette:
    actual: str = "#111111"             # Primary market ground truth
    forecast: str = "#0072B2"           # Day-ahead recursive prediction
    perfect_foresight: str = "#56B4E9"  # Upper bound clairvoyant dispatch
    charge: str = "#009E73"             # BESS energy injection (absorption)
    discharge: str = "#D55E00"          # BESS energy extraction (generation)
    soh: str = "#7B1FA2"                # Capacity retention & health
    revenue: str = "#008080"            # Gross & net financial cash flow
    loss: str = "#D62728"               # Degradation wear & operating penalty
    amber: str = "#E69F00"              # Mid-level threshold / warning
    grid: str = "#E0E0E0"               # Neutral axis & secondary grid
    background: str = "#FFFFFF"         # Pure white canvas background
    text_dark: str = "#222222"          # High-contrast label typography


COLOR_PALETTE = ColorPalette()


# ============================================================================
# Dimension Standards (Inches)
# ============================================================================

FIGURE_DIMENSIONS = {
    "ieee_single": (3.5, 2.16),         # 3.5 inches width (Golden ratio ~ 0.618)
    "ieee_single_tall": (3.5, 2.80),    # 3.5 inches width (Taller aspect)
    "ieee_double": (7.0, 3.80),         # 7.0 inches width (Full span double-column)
    "ieee_double_tall": (7.0, 5.00),    # 7.0 inches width (Two-row stacked subplots)
    "thesis_full": (6.5, 4.00),         # Standard Thesis page width
    "thesis_full_tall": (6.5, 6.20),    # Multi-panel stacked analysis
    "thesis_half": (4.8, 3.20),         # Compact thesis insert
}


# ============================================================================
# Theme Engine
# ============================================================================

class FigureTheme:
    """Configures global Matplotlib styles according to scientific standards."""

    @staticmethod
    def apply_ieee_theme(font_scale: float = 1.0) -> None:
        """Sets global matplotlib rcParams for IEEE/Nature publication standards."""
        base_size = 9.0 * font_scale

        theme_params: dict[str, Any] = {
            # Render Backend & DPI
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "figure.facecolor": COLOR_PALETTE.background,
            "axes.facecolor": COLOR_PALETTE.background,

            # Typography
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "font.size": base_size,
            "axes.labelsize": base_size + 1.0,
            "axes.titlesize": base_size + 1.5,
            "axes.titleweight": "bold",
            "axes.labelweight": "normal",
            "xtick.labelsize": base_size - 1.0,
            "ytick.labelsize": base_size - 1.0,
            "legend.fontsize": base_size - 1.0,
            "legend.title_fontsize": base_size,

            # Mathematical Notation (Clean STIX engine without requiring system TeX)
            "mathtext.fontset": "stix",
            "text.usetex": False,

            # Lines, Markers, Patches
            "lines.linewidth": 1.75,
            "lines.markersize": 5.0,
            "patch.linewidth": 0.8,
            "patch.edgecolor": COLOR_PALETTE.actual,

            # Axes & Gridlines
            "axes.linewidth": 0.9,
            "axes.edgecolor": "#333333",
            "axes.grid": True,
            "axes.grid.which": "major",
            "axes.axisbelow": True,
            "grid.color": COLOR_PALETTE.grid,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.85,

            # Tick Formatting
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,

            # Legend Placement
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.edgecolor": "#CCCCCC",
            "legend.fancybox": False,
            "legend.borderpad": 0.4,
            "legend.labelspacing": 0.3,
        }

        plt.rcParams.update(theme_params)


def set_ieee_style(font_scale: float = 1.0) -> None:
    """Convenience functional wrapper to configure IEEE typography and colors."""
    FigureTheme.apply_ieee_theme(font_scale=font_scale)


def get_figure_dimensions(layout: str = "thesis_full") -> tuple[float, float]:
    """Retrieves standard golden-ratio figure dimensions in inches."""
    return FIGURE_DIMENSIONS.get(layout, FIGURE_DIMENSIONS["thesis_full"])


def format_axes(
    ax: plt.Axes,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    hide_top_right: bool = True,
) -> None:
    """Applies standardized gridlines, label pads, and border styling."""
    if title:
        ax.set_title(title, pad=7.0)
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=4.0)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=4.0)

    if hide_top_right:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.85, color=COLOR_PALETTE.grid)


def save_publication_figure(
    fig: plt.Figure,
    output_base_path: Path | str,
    formats: Sequence[str] = ("png", "pdf", "svg"),
    dpi: int = 600,
) -> dict[str, Path]:
    """
    Saves a Matplotlib figure in multiple publication formats.
    Supported extensions: 'png', 'pdf', 'svg', 'tiff'.
    """
    base_path = Path(output_base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)

    stem = base_path.parent / base_path.stem
    saved_paths: dict[str, Path] = {}

    for fmt in formats:
        clean_fmt = fmt.lower().lstrip(".")
        target_path = stem.with_suffix(f".{clean_fmt}")

        fig.savefig(
            target_path,
            format=clean_fmt,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.04,
            facecolor=fig.get_facecolor(),
            edgecolor="none",
        )
        saved_paths[clean_fmt] = target_path

    return saved_paths