"""
tests/visualization/test_figure_style.py
========================================

Unit Test Suite for Figure Style & Visualization Engine (Part 10.1).
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pytest

from visualization.figure_style import (
    COLOR_PALETTE,
    FIGURE_DIMENSIONS,
    format_axes,
    get_figure_dimensions,
    save_publication_figure,
    set_ieee_style,
)


class TestFigureStyleEngine:
    def test_ieee_style_application(self):
        set_ieee_style(font_scale=1.2)
        assert plt.rcParams["mathtext.fontset"] == "stix"
        assert plt.rcParams["text.usetex"] is False
        assert plt.rcParams["figure.dpi"] == 300
        assert plt.rcParams["savefig.dpi"] == 600
        assert "DejaVu Sans" in plt.rcParams["font.sans-serif"]

    def test_palette_definitions(self):
        for field_name in COLOR_PALETTE.__dataclass_fields__:
            color_val = getattr(COLOR_PALETTE, field_name)
            assert isinstance(color_val, str)
            assert color_val.startswith("#")
            assert len(color_val) == 7

    def test_dimensions_lookup(self):
        w, h = get_figure_dimensions("ieee_single")
        assert w == 3.5
        assert h == 2.16

        w_full, h_full = get_figure_dimensions("thesis_full")
        assert w_full == 6.5
        assert h_full == 4.0

        # Fallback handling
        w_def, h_def = get_figure_dimensions("non_existent_layout")
        assert w_def == 6.5
        assert h_def == 4.0

    def test_format_axes_helper(self):
        set_ieee_style()
        fig, ax = plt.subplots()
        format_axes(ax, title="Test Plot", xlabel="Time (h)", ylabel="Value ($)")

        assert ax.get_title() == "Test Plot"
        assert ax.get_xlabel() == "Time (h)"
        assert ax.get_ylabel() == "Value ($)"
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()
        plt.close(fig)

    def test_multi_format_saving(self, tmp_path):
        set_ieee_style()
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot([0, 1, 2], [10, 20, 30], color=COLOR_PALETTE.revenue, label="Trajectory")
        ax.legend()

        export_target = tmp_path / "plot_demo"
        saved = save_publication_figure(fig, export_target, formats=("png", "pdf", "svg", "tiff"), dpi=150)
        plt.close(fig)

        assert set(saved.keys()) == {"png", "pdf", "svg", "tiff"}
        for fmt, file_path in saved.items():
            assert file_path.exists()
            assert file_path.stat().st_size > 0
            assert file_path.suffix.lower() == f".{fmt}"