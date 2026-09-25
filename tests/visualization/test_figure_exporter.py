"""Unit test suite for the publication figure export pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from visualization.figure_exporter import (
    ExportManifest,
    FigureExporter,
    FigureGenerator,
    ThesisFigureExporter,
)


class TestFigureExporterPipeline:
    def test_alias_equivalence(self):
        assert FigureGenerator is FigureExporter
        assert ThesisFigureExporter is FigureExporter

    def test_master_export_pipeline(self, tmp_path: Path):
        exporter = FigureExporter(
            output_directory=tmp_path / "_out",
            formats=("png", "pdf", "svg", "tiff"),
            dpi=100,
        )

        manifest = exporter.export_all_figures()

        assert isinstance(manifest, ExportManifest)
        assert manifest.total_figures_count == 44
        assert set(manifest.formats_exported) == {"png", "pdf", "svg", "tiff"}

        for fmt in ("png", "pdf", "svg", "tiff"):
            dir_path = Path(manifest.subdirectories[fmt])
            assert dir_path.exists()
            files = list(dir_path.glob(f"*.{fmt}"))
            assert len(files) == 44, f"Expected 44 .{fmt} files, found {len(files)}"

        manifest_json_path = Path(manifest.output_directory) / "_figures_manifest.json"
        assert manifest_json_path.exists()
        with open(manifest_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["total_figures_count"] == 44
        assert len(data["figures_inventory"]) == 44

        catalog_md_path = Path(manifest.output_directory) / "FIGURE_CATALOG.md"
        assert catalog_md_path.exists()
        md_text = catalog_md_path.read_text(encoding="utf-8")
        assert "Figure 10.2.1" in md_text
        assert "Figure 10.8.6" in md_text
        assert "Master Publication Figure Catalog" in md_text

    def test_isolated_executive_figure(self, tmp_path: Path):
        exporter = FigureExporter(output_directory=tmp_path, formats=["png"], dpi=100)
        res = exporter.plot_system_architecture_energy_balance()
        assert "png" in res
        assert res["png"].exists()
        assert res["png"].stat().st_size > 1000