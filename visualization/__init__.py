"""
visualization
=============

Publication Visualization Package

Implements compliant visual rendering, publication exports,
multi-format vector graphics (PNG/PDF/SVG/TIFF), and figure themes.
"""

from visualization.figure_style import (
    COLOR_PALETTE,
    FIGURE_DIMENSIONS,
    FigureTheme,
    format_axes,
    get_figure_dimensions,
    save_publication_figure,
    set_style,
)

__all__ = [
    "COLOR_PALETTE",
    "FIGURE_DIMENSIONS",
    "FigureTheme",
    "format_axes",
    "get_figure_dimensions",
    "save_publication_figure",
    "set_style",
]