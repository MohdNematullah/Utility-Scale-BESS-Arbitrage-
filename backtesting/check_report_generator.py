"""
backtesting/check_report_generator.py
=====================================

Verification and sanity-check script for Report Generator Module (Part 9.6).
"""

from pathlib import Path
from backtesting.report_generator import ThesisReportGenerator

LINE = "=" * 75

print(LINE)
print(" REPORT & PUBLICATION SYNTHESIZER CHECK (PART 9.6)")
print(LINE)

generator = ThesisReportGenerator(output_directory="backtesting/results/_report")
artifacts = generator.generate_all_reports()

print("-" * 75)
print("GENERATED PUBLICATION ARTIFACTS")
print("-" * 75)
print(f"Markdown Report : {artifacts.report_markdown.exists()} ({artifacts.report_markdown})")
print(f"Master Excel Workbook  : {artifacts.master_excel.exists()} ({artifacts.master_excel})")
print(f"Manifest JSON Index    : {artifacts.manifest_json.exists()} ({artifacts.manifest_json})")
print(f"LaTeX Tables Directory : {artifacts.latex_tables_dir.exists()} ({artifacts.latex_tables_dir})")

tex_files = list(artifacts.latex_tables_dir.glob("*.tex"))
print(f"\nGenerated Booktabs LaTeX Tables ({len(tex_files)} total):")
for f in sorted(tex_files):
    print(f"  â€¢ {f.name}")

assert len(tex_files) == 5, f"Expected 5 LaTeX tables, found {len(tex_files)}."
assert artifacts.report_markdown.stat().st_size > 1000, "Markdown chapter too small or unpopulated."
assert artifacts.master_excel.stat().st_size > 2000, "Master Excel workbook too small or unpopulated."

print("\n" + "-" * 75)
print("PREVIEW: CHAPTER 5 STRUCTURE (FIRST 25 LINES)")
print("-" * 75)
lines = artifacts.report_markdown.read_text(encoding="utf-8").splitlines()[:25]
for line in lines:
    print(line)

print("\n" + LINE)
print("report generator verified successfully âœ“")
print(LINE)