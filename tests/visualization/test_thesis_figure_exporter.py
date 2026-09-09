"""
tests/visualization/test_thesis_figure_exporter.py
==================================================

Unit Test Suite for Master Thesis Figure Exporter (Part 10.8 / 10.10).
"""

import json
from pathlib import Path
import pytest

from visualization.thesis_figure_exporter import (
    ThesisExportManifest,
    ThesisFigureExporter,
    ThesisFigureGenerator,
)


class TestThesisFigureExporterPipeline:
    def test_alias_equivalence(self):
        assert ThesisFigureGenerator is ThesisFigureExporter

    def test_master_export_pipeline(self, tmp_path):
        exporter = ThesisFigureExporter(
            output_directory=tmp_path / "thesis_out",
            formats=("png", "pdf", "svg", "tiff"),
            dpi=100,  # Fast execution for CI/CD test
        )

        manifest = exporter.export_all_figures()

        assert isinstance(manifest, ThesisExportManifest)
        assert manifest.total_figures_count == 44
        assert set(manifest.formats_exported) == {"png", "pdf", "svg", "tiff"}

        # Verify all subdirectories contain exactly 44 figures
        for fmt in ("png", "pdf", "svg", "tiff"):
            dir_path = Path(manifest.subdirectories[fmt])
            assert dir_path.exists()
            files = list(dir_path.glob(f"*.{fmt}"))
            assert len(files) == 44, f"Expected 44 .{fmt} files, found {len(files)}"

        # Verify Manifest JSON
        manifest_json_path = Path(manifest.output_directory) / "thesis_figures_manifest.json"
        assert manifest_json_path.exists()
        with open(manifest_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["total_figures_count"] == 44
        assert len(data["figures_inventory"]) == 44

        # Verify Figure Catalog Markdown
        catalog_md_path = Path(manifest.output_directory) / "FIGURE_CATALOG.md"
        assert catalog_md_path.exists()
        md_text = catalog_md_path.read_text(encoding="utf-8")
        assert "Figure 10.2.1" in md_text
        assert "Figure 10.8.6" in md_text
        assert "Master Thesis Publication Figure Catalog" in md_text

    def test_isolated_executive_figure(self, tmp_path):
        exporter = ThesisFigureExporter(output_directory=tmp_path, formats=["png"], dpi=100)
        res = exporter.plot_system_architecture_energy_balance()
        assert "png" in res
        assert res["png"].exists()
        assert res["png"].stat().st_size > 1000