"""
visualization/figure_style.py
=============================
Figure styling, color palette, dimensions, and multi-format export engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True, slots=True)
class ColorPalette:
    actual: str = "#111111"
    forecast: str = "#0072B2"
    perfect_foresight: str = "#56B4E9"
    charge: str = "#009E73"
    discharge: str = "#D55E00"
    soh: str = "#7B1FA2"
    revenue: str = "#008080"
    loss: str = "#D62728"
    amber: str = "#E69F00"
    grid: str = "#E0E0E0"
    background: str = "#FFFFFF"
    text_dark: str = "#222222"


COLOR_PALETTE = ColorPalette()

FIGURE_DIMENSIONS: dict[str, tuple[float, float]] = {
    "single": (3.5, 2.16),
    "single_tall": (3.5, 2.80),
    "double": (7.0, 3.80),
    "double_tall": (7.0, 5.00),
    "full": (6.5, 4.00),
    "full_tall": (6.5, 6.20),
    "half": (4.8, 3.20),
}


class FigureTheme:
    """Applies standardized publication styles to matplotlib rcParams."""

    @staticmethod
    def apply_theme(font_scale: float = 1.0) -> None:
        base_size = 9.0 * font_scale

        theme_params: dict[str, Any] = {
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "figure.facecolor": COLOR_PALETTE.background,
            "axes.facecolor": COLOR_PALETTE.background,
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
            "mathtext.fontset": "stix",
            "text.usetex": False,
            "lines.linewidth": 1.75,
            "lines.markersize": 5.0,
            "patch.linewidth": 0.8,
            "patch.edgecolor": COLOR_PALETTE.actual,
            "axes.linewidth": 0.9,
            "axes.edgecolor": "#333333",
            "axes.grid": True,
            "axes.grid.which": "major",
            "axes.axisbelow": True,
            "grid.color": COLOR_PALETTE.grid,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.85,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.edgecolor": "#CCCCCC",
            "legend.fancybox": False,
            "legend.borderpad": 0.4,
            "legend.labelspacing": 0.3,
        }
        plt.rcParams.update(theme_params)


def set_style(font_scale: float = 1.0) -> None:
    FigureTheme.apply_theme(font_scale=font_scale)


def get_figure_dimensions(layout: str = "full") -> tuple[float, float]:
    return FIGURE_DIMENSIONS.get(layout, FIGURE_DIMENSIONS["full"])


def format_axes(
    ax: plt.Axes,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    hide_top_right: bool = True,
) -> None:
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
    base_path = Path(output_base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    stem_name = base_path.stem

    saved_paths: dict[str, Path] = {}
    for fmt in formats:
        clean_fmt = fmt.lower().lstrip(".")
        target_path = base_path.parent / f"{stem_name}.{clean_fmt}"

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


__all__ = [
    "COLOR_PALETTE",
    "ColorPalette",
    "FIGURE_DIMENSIONS",
    "FigureTheme",
    "format_axes",
    "get_figure_dimensions",
    "save_publication_figure",
    "set_style",
]