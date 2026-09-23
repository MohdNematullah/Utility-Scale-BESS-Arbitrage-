"""
tests/backtesting/test_report_generator.py
=========================================

Unit Test Suite for Report Generator Module (Part 9.6).
"""

import json
from pathlib import Path
import openpyxl
import pytest

from backtesting.report_generator import (
    ThesisReportArtifacts,
    ThesisReportGenerator,
)


@pytest.fixture
def mock_evaluation_payloads(tmp_path):
    f_json = tmp_path / "forecast.json"
    a_json = tmp_path / "arbitrage.json"
    r_json = tmp_path / "risk.json"
    s_json = tmp_path / "sensitivity.json"
    t_json = tmp_path / "statistical.json"

    f_json.write_text(json.dumps({
        "mae": 7.93, "rmse": 14.32, "mape_pct": 22.19, "r2_score": 0.2241,
        "directional_accuracy_pct": 67.83, "value_capture_ratio_pct": 55.30,
        "perfect_foresight_gap_usd": 891010.04
    }))

    a_json.write_text(json.dumps({
        "gross_revenue_usd": 1102091.72, "degradation_cost_usd": 253758.72,
        "net_revenue_usd": 848333.00, "fixed_om_cost_usd": 359589.04,
        "variable_om_cost_usd": 19017.97, "net_operating_profit_usd": 469725.99,
        "net_arbitrage_margin_pct": 76.97, "revenue_per_kw_year": 22.99
    }))

    r_json.write_text(json.dumps({
        "sharpe_ratio": 77.326, "sortino_ratio": 99.990, "max_drawdown_usd": 0.0,
        "historical_var_95_usd": -1654.10, "cvar_95_usd": -1469.08
    }))

    s_json.write_text(json.dumps({
        "tornado_spectrum": [
            {"sensitivity_rank": 1, "parameter": "System Sizing", "low_revenue_usd": 424150.0, "high_revenue_usd": 1696600.0, "swing_usd": 1272450.0}
        ]
    }))

    t_json.write_text(json.dumps({
        "effect_sizes": [
            {"comparison_name": "XGBoost vs Persistence", "hln_statistic": -12.17, "p_value": 0.00000, "superior_model": "XGBoost"}
        ]
    }))

    return f_json, a_json, r_json, s_json, t_json


class TestReportGeneratorPipeline:
    def test_complete_report_generation(self, mock_evaluation_payloads, tmp_path):
        f_json, a_json, r_json, s_json, t_json = mock_evaluation_payloads
        output_dir = tmp_path / "_out"

        generator = ThesisReportGenerator(output_directory=output_dir)
        artifacts = generator.generate_all_reports(
            forecast_json_path=f_json,
            arbitrage_json_path=a_json,
            risk_json_path=r_json,
            sensitivity_json_path=s_json,
            statistical_json_path=t_json,
        )

        assert isinstance(artifacts, ThesisReportArtifacts)
        assert artifacts.report_markdown.exists()
        assert artifacts.master_excel.exists()
        assert artifacts.manifest_json.exists()

        # Check all 5 LaTeX tables exist
        tex_files = list(artifacts.latex_tables_dir.glob("*.tex"))
        assert len(tex_files) == 5
        names = {f.name for f in tex_files}
        assert names == {
            "table_forecast_accuracy.tex",
            "table_arbitrage_kpis.tex",
            "table_risk_tail_metrics.tex",
            "table_statistical_tests.tex",
            "table_sensitivity_tornado.tex",
        }

        # Check Markdown content integrity
        md_text = artifacts.report_markdown.read_text(encoding="utf-8")
        assert "Chapter 5: Empirical Results" in md_text
        assert "1,102,091.72" in md_text
        assert "table_forecast_accuracy.tex" in md_text

        # Check Excel workbook tabs
        wb = openpyxl.load_workbook(artifacts.master_excel, read_only=True)
        assert "Forecast_Realism" in wb.sheetnames
        assert "Arbitrage_Economics" in wb.sheetnames
        assert "Risk_Analytics" in wb.sheetnames
        assert "Tornado_Sensitivity" in wb.sheetnames
        assert "Statistical_Significance" in wb.sheetnames
        wb.close()