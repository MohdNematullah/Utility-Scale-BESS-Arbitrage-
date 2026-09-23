"""
visualization/check_visualization.py
====================================

Master Production Verification & QA Script for Part 10 Visualization Package.
"""

from pathlib import Path
from visualization._figure_exporter import ThesisFigureExporter

LINE = "=" * 60

print(LINE)
print("PUBLICATION VISUALIZATION CHECK")
print(LINE + "\n")

exporter = ThesisFigureExporter(
    output_directory="results/_figures",
    formats=("png", "pdf", "svg", "tiff"),
    dpi=300,
)

manifest = exporter.export_all_figures()

# 1. Section Verifications
forecast_files = list(Path("results/_figures/png").glob("Figure_10_2_*.*"))
dispatch_files = list(Path("results/_figures/png").glob("Figure_10_3_*.*"))
battery_files = list(Path("results/_figures/png").glob("Figure_10_4_*.*"))
finance_files = list(Path("results/_figures/png").glob("Figure_10_5_*.*"))
risk_files = list(Path("results/_figures/png").glob("Figure_10_6_*.*"))
scenario_files = list(Path("results/_figures/png").glob("Figure_10_7_*.*"))

print(f"Forecast Figures ............ {'PASS' if len(forecast_files) == 6 else 'FAIL'}")
print(f"Dispatch Figures ............ {'PASS' if len(dispatch_files) == 7 else 'FAIL'}")
print(f"Battery Figures ............. {'PASS' if len(battery_files) == 6 else 'FAIL'}")
print(f"Financial Figures ........... {'PASS' if len(finance_files) == 6 else 'FAIL'}")
print(f"Risk Figures ................ {'PASS' if len(risk_files) == 5 else 'FAIL'}")
print(f"Scenario Figures ............ {'PASS' if len(scenario_files) == 8 else 'FAIL'}\n")

# 2. Format Verifications
png_count = len(list(Path("results/_figures/png").glob("*.png")))
pdf_count = len(list(Path("results/_figures/pdf").glob("*.pdf")))
svg_count = len(list(Path("results/_figures/svg").glob("*.svg")))
tiff_count = len(list(Path("results/_figures/tiff").glob("*.tiff")))

print(f"PNG Export ................. {'PASS' if png_count == 44 else 'FAIL'}")
print(f"PDF Export ................. {'PASS' if pdf_count == 44 else 'FAIL'}")
print(f"SVG Export ................. {'PASS' if svg_count == 44 else 'FAIL'}")
print(f"TIFF Export ................ {'PASS' if tiff_count == 44 else 'FAIL'}\n")

print(f"Total Figures Exported: {manifest.total_figures_count}\n")

catalog_file = Path("results/_figures/FIGURE_CATALOG.md")
manifest_file = Path("results/_figures/_figures_manifest.json")

assert manifest.total_figures_count == 44, f"Expected 44 figures, got {manifest.total_figures_count}"
assert png_count == 44 and pdf_count == 44 and svg_count == 44 and tiff_count == 44
assert catalog_file.exists() and catalog_file.stat().st_size > 1000
assert manifest_file.exists() and manifest_file.stat().st_size > 1000

print(LINE)
print("Visualization package verified successfully.")
print(LINE)