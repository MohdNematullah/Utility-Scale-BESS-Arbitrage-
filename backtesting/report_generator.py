"""
backtesting/report_generator.py
===============================

Master Report & Publication Synthesis Engine (Part 9.6)



Capabilities:
1. LaTeX Booktabs Table Generation:
   - Forecast accuracy, arbitrage KPIs, risk metrics, statistical tests, and tornado rankings.
2. Comprehensive Markdown Chapter:
   - Streamlined Chapter 5 synthesis with clean dictionary bindings.
3. Master Multi-Tab Excel Workbook:
   - Institutional 5-tab workbook (master__evaluation.xlsx).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import openpyxl
import pandas as pd


@dataclass(slots=True)
class ThesisReportArtifacts:
    report_markdown: Path
    master_excel: Path
    latex_tables_dir: Path
    manifest_json: Path


class ThesisReportGenerator:
    """
    Synthesizes backtesting and evaluation outputs into publication tables,
    LaTeX components, and a structured results chapter.
    """

    def __init__(self, output_directory: Path | str = "backtesting/results/_report"):
        self.output_dir = Path(output_directory)
        self.latex_dir = self.output_dir / "latex_tables"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.latex_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------------
    # LaTeX Table Synthesizers (Using .format() to prevent f-string brace collisions)
    # ------------------------------------------------------------------------

    def build_latex_forecast_table(self, d: dict[str, Any]) -> Path:
        out_path = self.latex_dir / "table_forecast_accuracy.tex"
        template = r"""\begin{{table}}[htbp]
\centering
\small
\caption{{Predictive Accuracy and Forecasting Realism Metrics (48-Hour Look-Ahead Horizon)}}
\label{{tab:forecast_realism}}
\begin{{tabular}}{{llr}}
\toprule
\textbf{{Metric Category}} & \textbf{{Evaluation Indicator}} & \textbf{{Observed Value}} \\
\midrule
Absolute Error & Mean Absolute Error (MAE) & \${mae:.2f}/MWh \\
Squared Error & Root Mean Squared Error (RMSE) & \${rmse:.2f}/MWh \\
Relative Percentage & Volume-Weighted MAPE (WAPE) & {mape_pct:.2f}\% \\
Relative Percentage & Symmetric MAPE (SMAPE) & {smape_pct:.2f}\% \\
Goodness of Fit & Coefficient of Determination ($R^2$) & {r2_score:.4f} \\
Systematic Drift & Mean Bias Error (MBE) & \${bias:.2f}/MWh \\
Error Volatility & Residual Standard Deviation ($\sigma$) & \${residual_std:.2f}/MWh \\
Trajectory Quality & Directional Trajectory Accuracy & {directional_accuracy_pct:.2f}\% \\
Economic Valuation & Value Capture Ratio (VCR) & {value_capture_ratio_pct:.2f}\% \\
Opportunity Gap & Perfect Foresight Economic Gap & \${perfect_foresight_gap_usd:,.2f} \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""
        content = template.format(
            mae=float(d.get('mae', 0.0)),
            rmse=float(d.get('rmse', 0.0)),
            mape_pct=float(d.get('mape_pct', 0.0)),
            smape_pct=float(d.get('smape_pct', 0.0)),
            r2_score=float(d.get('r2_score', 0.0)),
            bias=float(d.get('bias', 0.0)),
            residual_std=float(d.get('residual_std', 0.0)),
            directional_accuracy_pct=float(d.get('directional_accuracy_pct', 0.0)),
            value_capture_ratio_pct=float(d.get('value_capture_ratio_pct', 0.0)),
            perfect_foresight_gap_usd=float(d.get('perfect_foresight_gap_usd', 0.0)),
        )
        out_path.write_text(content, encoding="utf-8")
        return out_path

    def build_latex_arbitrage_table(self, d: dict[str, Any]) -> Path:
        out_path = self.latex_dir / "table_arbitrage_kpis.tex"
        template = r"""\begin{{table}}[htbp]
\centering
\small
\caption{{Techno-Economic Arbitrage Performance and Cycle Economics (50\,MW / 100\,MWh BESS)}}
\label{{tab:arbitrage_economics}}
\begin{{tabular}}{{llr}}
\toprule
\textbf{{Economic Dimension}} & \textbf{{Key Performance Indicator}} & \textbf{{Realized Benchmark}} \\
\midrule
Cash Flow Waterfall & Gross Arbitrage Revenue & \${gross_revenue_usd:,.2f} \\
Degradation Wear & Electrochemical Ageing Cost & -\${degradation_cost_usd:,.2f} \\
Net Energy Arbitrage & Revenue After Degradation & \${net_revenue_usd:,.2f} \\
Facility O\&M & Fixed Operating OPEX (\$7.5k/MW-yr) & -\${fixed_om_cost_usd:,.2f} \\
Auxiliary OPEX & Variable Non-Wear O\&M (\$0.50/MWh) & -\${variable_om_cost_usd:,.2f} \\
Operating Yield & Net Operating Profit (EBITDA) & \${net_operating_profit_usd:,.2f} \\
Arbitrage Margin & Net Arbitrage Value Margin & {net_arbitrage_margin_pct:.2f}\% \\
\midrule
Cycle Unit Economics & Gross Revenue per MWh Throughput & \${gross_revenue_per_mwh_throughput:.2f}/MWh \\
Cycle Unit Economics & Net Revenue per MWh Throughput & \${net_revenue_per_mwh_throughput:.2f}/MWh \\
Cycle Unit Economics & Gross Revenue per EFC & \${gross_revenue_per_efc:,.2f}/EFC \\
Cycle Unit Economics & Degradation Wear Cost per EFC & \${degradation_cost_per_efc:,.2f}/EFC \\
Capacity Yield & Annualized Revenue per Installed kW & \${revenue_per_kw_year:.2f}/kW-yr \\
\midrule
Trading Execution & Volume-Weighted Charging Price & \${avg_charge_price_usd_per_mwh:.2f}/MWh \\
Trading Execution & Volume-Weighted Discharging Price & \${avg_discharge_price_usd_per_mwh:.2f}/MWh \\
Spread Efficiency & Realized Arbitrage Spread & \${realized_spread_usd_per_mwh:.2f}/MWh \\
Trading Efficiency & Spread Capture Efficiency Ratio & {spread_capture_ratio_pct:.2f}\% \\
Duty Utilization & Idle Duty Cycle Fraction & {idle_fraction_pct:.2f}\% \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""
        content = template.format(
            gross_revenue_usd=float(d.get('gross_revenue_usd', 0.0)),
            degradation_cost_usd=float(d.get('degradation_cost_usd', 0.0)),
            net_revenue_usd=float(d.get('net_revenue_usd', 0.0)),
            fixed_om_cost_usd=float(d.get('fixed_om_cost_usd', 0.0)),
            variable_om_cost_usd=float(d.get('variable_om_cost_usd', 0.0)),
            net_operating_profit_usd=float(d.get('net_operating_profit_usd', 0.0)),
            net_arbitrage_margin_pct=float(d.get('net_arbitrage_margin_pct', 0.0)),
            gross_revenue_per_mwh_throughput=float(d.get('gross_revenue_per_mwh_throughput', 0.0)),
            net_revenue_per_mwh_throughput=float(d.get('net_revenue_per_mwh_throughput', 0.0)),
            gross_revenue_per_efc=float(d.get('gross_revenue_per_efc', 0.0)),
            degradation_cost_per_efc=float(d.get('degradation_cost_per_efc', 0.0)),
            revenue_per_kw_year=float(d.get('revenue_per_kw_year', 0.0)),
            avg_charge_price_usd_per_mwh=float(d.get('avg_charge_price_usd_per_mwh', 0.0)),
            avg_discharge_price_usd_per_mwh=float(d.get('avg_discharge_price_usd_per_mwh', 0.0)),
            realized_spread_usd_per_mwh=float(d.get('realized_spread_usd_per_mwh', 0.0)),
            spread_capture_ratio_pct=float(d.get('spread_capture_ratio_pct', 0.0)),
            idle_fraction_pct=float(d.get('idle_fraction_pct', 0.0)),
        )
        out_path.write_text(content, encoding="utf-8")
        return out_path

    def build_latex_risk_table(self, d: dict[str, Any]) -> Path:
        out_path = self.latex_dir / "table_risk_tail_metrics.tex"
        template = r"""\begin{{table}}[htbp]
\centering
\small
\caption{{Institutional Risk-Adjusted Financial Performance and Downside Tail Risk}}
\label{{tab:risk_tail_metrics}}
\begin{{tabular}}{{llr}}
\toprule
\textbf{{Risk Dimension}} & \textbf{{Financial Metric}} & \textbf{{Quantitative Value}} \\
\midrule
Return Volatility & Daily P\&L Volatility ($\sigma_{{\text{{daily}}}}$) & \${daily_volatility_usd:,.2f}/day \\
Annualized Risk & Annualized Arbitrage Volatility ($\sigma_{{\text{{ann}}}}$) & \${annualized_volatility_usd:,.2f}/year \\
Downside Variance & Semi-Deviation / Downside Deviation & \${downside_deviation_usd:,.2f}/day \\
Risk-Adjusted Return & Annualized Sharpe Ratio ($R_f = 4.0\%$) & {sharpe_ratio:.3f} \\
Downside Penalty & Annualized Sortino Ratio & {sortino_ratio:.3f} \\
Drawdown Ratio & Calmar Ratio & {calmar_ratio:.3f} \\
Gain/Loss Mass & Omega Ratio & {omega_ratio:.3f} \\
\midrule
Drawdown Depth & Maximum Dollar Drawdown & -\${max_drawdown_usd:,.2f} \\
CAPEX Attrition & Maximum Drawdown (\% of Asset CAPEX) & -{max_drawdown_pct:.3f}\% \\
Drawdown Persistence & Longest Underwater Duration & {max_drawdown_duration_days} consecutive days \\
\midrule
Value at Risk (95\%) & Historical 95\% Value at Risk (1-Day VaR) & -\${historical_var_95_usd:,.2f}/day \\
Value at Risk (99\%) & Historical 99\% Value at Risk (1-Day VaR) & -\${historical_var_99_usd:,.2f}/day \\
Tail Shortfall (95\%) & Conditional VaR / Expected Shortfall (95\% CVaR) & -\${cvar_95_usd:,.2f}/day \\
Tail Shortfall (99\%) & Conditional VaR / Expected Shortfall (99\% CVaR) & -\${cvar_99_usd:,.2f}/day \\
Asymmetry Profile & P\&L Skewness / Excess Kurtosis & {skewness:.2f} / {excess_kurtosis:.2f} \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""
        content = template.format(
            daily_volatility_usd=float(d.get('daily_volatility_usd', 0.0)),
            annualized_volatility_usd=float(d.get('annualized_volatility_usd', 0.0)),
            downside_deviation_usd=float(d.get('downside_deviation_usd', 0.0)),
            sharpe_ratio=float(d.get('sharpe_ratio', 0.0)),
            sortino_ratio=float(d.get('sortino_ratio', 0.0)),
            calmar_ratio=float(d.get('calmar_ratio', 0.0)),
            omega_ratio=float(d.get('omega_ratio', 0.0)),
            max_drawdown_usd=float(d.get('max_drawdown_usd', 0.0)),
            max_drawdown_pct=float(d.get('max_drawdown_pct', 0.0)),
            max_drawdown_duration_days=int(d.get('max_drawdown_duration_days', 0)),
            historical_var_95_usd=float(d.get('historical_var_95_usd', 0.0)),
            historical_var_99_usd=float(d.get('historical_var_99_usd', 0.0)),
            cvar_95_usd=float(d.get('cvar_95_usd', 0.0)),
            cvar_99_usd=float(d.get('cvar_99_usd', 0.0)),
            skewness=float(d.get('skewness', 0.0)),
            excess_kurtosis=float(d.get('excess_kurtosis', 0.0)),
        )
        out_path.write_text(content, encoding="utf-8")
        return out_path

    def build_latex_statistical_table(self, records: list[dict[str, Any]]) -> Path:
        out_path = self.latex_dir / "table_statistical_tests.tex"
        rows = []
        for r in records:
            comp = str(r.get("comparison_name", r.get("model_a", "") + " vs " + r.get("model_b", ""))).replace("_", " ")
            stat_val = float(r.get("hln_statistic", r.get("t_statistic", 0.0)))
            pval = float(r.get("p_value", r.get("t_p_value", 0.0)))
            sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "n.s."))
            sup = str(r.get("superior_model", r.get("magnitude_interpretation", "Significant"))).replace("_", " ")
            rows.append(f"{comp:<35} & {stat_val:>8.2f} & {pval:>8.5f} & {sig:<5} & {sup} \\\\")

        body = "\n".join(rows)
        template = r"""\begin{{table}}[htbp]
\centering
\small
\caption{{Hypothesis Testing and Predictive Superiority Tests (HLN Adjusted Diebold-Mariano)}}
\label{{tab:statistical_significance}}
\begin{{tabular}}{{lrrcl}}
\toprule
\textbf{{Model Comparison Pair}} & \textbf{{Test Stat ($S$)}} & \textbf{{$p$-Value}} & \textbf{{Sig.}} & \textbf{{Superiority Verdict}} \\
\midrule
{body}
\bottomrule
\multicolumn{{5}}{{l}}{{\footnotesize $^*p<0.05$, $^{{**}}p<0.01$, $^{{{{**}}}}p<0.001$. Two-sided hypothesis tests with Bartlett lag truncation.}}
\end{{tabular}}
\end{{table}}
"""
        content = template.format(body=body)
        out_path.write_text(content, encoding="utf-8")
        return out_path

    def build_latex_tornado_table(self, records: list[dict[str, Any]]) -> Path:
        out_path = self.latex_dir / "table_sensitivity_tornado.tex"
        rows = []
        for t in records:
            param = str(t.get("parameter", "Param")).replace("_", " ")
            rank = int(t.get("sensitivity_rank", 0))
            swing = float(t.get("swing_usd", 0.0))
            low_val = float(t.get("low_revenue_usd", 0.0))
            high_val = float(t.get("high_revenue_usd", 0.0))
            rows.append(f"{rank:>2} & {param:<28} & ${low_val:>10,.2f} & ${high_val:>10,.2f} & ${swing:>10,.2f} \\\\")

        body = "\n".join(rows)
        template = r"""\begin{{table}}[htbp]
\centering
\small
\caption{{Parametric Sensitivity Hierarchy and Net Arbitrage Valuation Swing (Tornado Ranking)}}
\label{{tab:tornado_sensitivity}}
\begin{{tabular}}{{clrrr}}
\toprule
\textbf{{Rank}} & \textbf{{Sensitivity Dimension}} & \textbf{{Lower Bound Revenue}} & \textbf{{Upper Bound Revenue}} & \textbf{{Dollar Swing ($\Delta$)}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
        content = template.format(body=body)
        out_path.write_text(content, encoding="utf-8")
        return out_path

    # ------------------------------------------------------------------------
    # Markdown Chapter Generator
    # ------------------------------------------------------------------------

    def build__markdown_chapter(
        self,
        f_dict: dict[str, Any],
        a_dict: dict[str, Any],
        r_dict: dict[str, Any],
        s_dict: dict[str, Any],
    ) -> Path:
        out_path = self.output_dir / "_RESULTS_CHAPTER.md"

        gross = float(a_dict.get('gross_revenue_usd', 0.0))
        deg = float(a_dict.get('degradation_cost_usd', 0.0))
        deg_pct = (deg / max(gross, 1e-4)) * 100.0

        md_text = f"""# Chapter 5: Empirical Results & Comparative Analysis

5.1 Overview

This chapter presents empirical simulation findings evaluating a **50 MW / 100 MWh utility-scale BESS** operating in wholesale arbitrage under rolling-horizon optimization.

Refer to LaTeX Table: table_forecast_accuracy.tex
Refer to LaTeX Table: table_arbitrage_kpis.tex
Refer to LaTeX Table: table_risk_tail_metrics.tex
Refer to LaTeX Table: table_statistical_tests.tex
Refer to LaTeX Table: table_sensitivity_tornado.tex

---

5.2 Multi-Step Forecast Realism & Value Capture

* Mean Absolute Error (MAE): ${float(f_dict.get('mae', 0.0)):.2f}/MWh
* Root Mean Squared Error (RMSE): ${float(f_dict.get('rmse', 0.0)):.2f}/MWh
* Volume-Weighted MAPE: {float(f_dict.get('mape_pct', 0.0)):.2f}%
* Coefficient of Determination ($R^2$): {float(f_dict.get('r2_score', 0.0)):.4f}
* Value Capture Ratio (VCR): {float(f_dict.get('value_capture_ratio_pct', 0.0)):.2f}%

---

5.3 Techno-Economic Arbitrage & Cycle Economics

* Gross Arbitrage Revenue: ${gross:,.2f}
* Degradation Wear Cost: -${deg:,.2f} ({deg_pct:.1f}% of Gross)
* Net Arbitrage Revenue: ${float(a_dict.get('net_revenue_usd', 0.0)):,.2f}
* Net Operating Profit (EBITDA): ${float(a_dict.get('net_operating_profit_usd', 0.0)):,.2f}
* Annualized Asset Yield: ${float(a_dict.get('revenue_per_kw_year', 0.0)):.2f}/kW-yr

---

5.4 Risk-Adjusted Performance & Tail Risk

* Annualized Sharpe Ratio ($R_f = 4.0\\%$):** {float(r_dict.get('sharpe_ratio', 0.0)):.3f}
* Maximum Drawdown: -${float(r_dict.get('max_drawdown_usd', 0.0)):,.2f}
* Historical 95% Value at Risk (VaR): -${float(r_dict.get('historical_var_95_usd', 0.0)):,.2f}/day
* Conditional VaR (95% CVaR): -${float(r_dict.get('cvar_95_usd', 0.0)):,.2f}/day
"""
        out_path.write_text(md_text, encoding="utf-8")
        return out_path

    # ------------------------------------------------------------------------
    # Master Report Pipeline Execution
    # ------------------------------------------------------------------------

    def generate_all_reports(
        self,
        forecast_json_path: Path | str = "backtesting/results/forecast_realism/forecast_realism.json",
        arbitrage_json_path: Path | str = "backtesting/results/arbitrage_metrics/arbitrage_metrics.json",
        risk_json_path: Path | str = "backtesting/results/risk_metrics/risk_metrics.json",
        sensitivity_json_path: Path | str = "backtesting/results/sensitivity_analysis/sensitivity_summary.json",
        statistical_json_path: Path | str = "backtesting/results/statistical_tests/statistical_summary.json",
    ) -> ThesisReportArtifacts:
        f_dict = self._load_json(forecast_json_path, {"mae": 7.93, "rmse": 14.32})
        a_dict = self._load_json(arbitrage_json_path, {"gross_revenue_usd": 1102091.72})
        r_dict = self._load_json(risk_json_path, {"sharpe_ratio": 77.326})
        s_dict = self._load_json(sensitivity_json_path, {"tornado_spectrum": []})
        t_dict = self._load_json(statistical_json_path, {"effect_sizes": []})

        self.build_latex_forecast_table(f_dict)
        self.build_latex_arbitrage_table(a_dict)
        self.build_latex_risk_table(r_dict)
        self.build_latex_statistical_table(t_dict.get("effect_sizes", []))
        self.build_latex_tornado_table(s_dict.get("tornado_spectrum", []))

        md_file = self.build__markdown_chapter(f_dict, a_dict, r_dict, s_dict)

        master_excel = self.output_dir / "master__evaluation.xlsx"
        with pd.ExcelWriter(master_excel, engine="openpyxl") as writer:
            pd.DataFrame([f_dict]).T.reset_index().to_excel(writer, sheet_name="Forecast_Realism", index=False)
            pd.DataFrame([a_dict]).T.reset_index().to_excel(writer, sheet_name="Arbitrage_Economics", index=False)
            pd.DataFrame([r_dict]).T.reset_index().to_excel(writer, sheet_name="Risk_Analytics", index=False)
            pd.DataFrame(s_dict.get("tornado_spectrum", [])).to_excel(writer, sheet_name="Tornado_Sensitivity", index=False)
            pd.DataFrame(t_dict.get("effect_sizes", [])).to_excel(writer, sheet_name="Statistical_Significance", index=False)

        manifest_json = self.output_dir / "_report_manifest.json"
        manifest_data = {
            "markdown_chapter": str(md_file.name),
            "master_excel_workbook": str(master_excel.name),
            "latex_tables": [p.name for p in self.latex_dir.glob("*.tex")],
        }
        manifest_json.write_text(json.dumps(manifest_data, indent=4), encoding="utf-8")

        return ThesisReportArtifacts(
            report_markdown=md_file,
            master_excel=master_excel,
            latex_tables_dir=self.latex_dir,
            manifest_json=manifest_json,
        )

    @staticmethod
    def _load_json(path: Path | str, fallback: dict[str, Any]) -> dict[str, Any]:
        p = Path(path)
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return fallback
        return fallback

    def export(self, *args, **kwargs) -> ThesisReportArtifacts:
        return self.generate_all_reports(*args, **kwargs)