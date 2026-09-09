"""
backtesting/sensitivity_analysis.py
==================================

Research-Grade Sensitivity Analysis & Parametric Elasticity Engine (Part 9.4)



Capabilities:
1. Multi-Dimensional Sensitivity Evaluation:
   - Forecast Horizon Look-Ahead Elasticity (12h -> 72h)
   - Thermal Degradation Arrhenius Sensitivity (15°C -> 45°C)
   - Round-Trip Efficiency (RTE) Elasticity (85% -> 95%)
   - Battery Chemistry Architecture Trade-Offs (NMC vs. LFP vs. LTO)
   - Power & Energy Duration Sizing (25MW/50MWh -> 100MW/200MWh)
   - Wear Hurdle Cost Disincentive Sensitivity ($0 -> $25/MWh)
2. Quantitative Elasticity Analytics:
   - Point & Arc Elasticity (% Delta Net Revenue / % Delta Parameter)
   - Marginal Value per Unit Variation ($/Hour look-ahead, $/°C, $/1% RTE)
3. Tornado Impact Spectrum:
   - High-to-low parametric rank swing relative to baseline net revenue
4. Multi-Criteria Radar / Spider Formulation:
   - 6-axis normalized radar benchmarking across chemistries and sizing options
5. Publication Diagnostic Visualizations:
   - Tornado Sensitivity Chart
   - Multi-Criteria Radar / Spider Chart
   - Horizon Look-Ahead Elasticity & Marginal Return Curve
   - Thermal Degradation & Capacity Fade Surface
   - Round-Trip Efficiency vs. Net Revenue Elasticity
6. Multi-Format Tabular Reporting:
   - Comprehensive multi-tab Excel workbook (sensitivity_report.xlsx)
   - Summary CSVs, Elasticity Matrix CSV, and structured JSON scorecard
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd

plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"


# ============================================================================
# Dataclasses
# ============================================================================

@dataclass(slots=True)
class TornadoParameterRecord:
    parameter: str
    baseline_value: float
    low_value: float
    high_value: float
    low_revenue_usd: float
    high_revenue_usd: float
    swing_usd: float
    sensitivity_rank: int


@dataclass(slots=True)
class ElasticityRecord:
    parameter_dimension: str
    parameter_variation: str
    base_param: float
    varied_param: float
    pct_delta_param: float
    base_net_revenue_usd: float
    varied_net_revenue_usd: float
    pct_delta_revenue: float
    arc_elasticity: float
    marginal_rate_usd: float


@dataclass(slots=True)
class SensitivityArtifacts:
    summary_csv: Path
    elasticity_csv: Path
    tornado_csv: Path
    sensitivity_excel: Path
    summary_json: Path
    figure_tornado: Path
    figure_radar: Path
    figure_horizon: Path
    figure_thermal: Path
    figure_efficiency: Path
    figures_directory: Path


# ============================================================================
# Sensitivity Analysis Engine
# ============================================================================

class SensitivityAnalysisEngine:
    """
    Computes cross-parameter sensitivities, elasticities, tornado rankings,
    and multi-objective spider visualizations.
    """

    def __init__(self, output_directory: Path | str = "backtesting/results/sensitivity_analysis"):
        self.output_dir = Path(output_directory)
        self.figure_dir = self.output_dir / "figures"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figure_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------------
    # Quantitative Elasticity Formulations
    # ------------------------------------------------------------------------

    @staticmethod
    def calculate_arc_elasticity(
        x_base: float,
        x_var: float,
        y_base: float,
        y_var: float,
        epsilon: float = 1e-6,
    ) -> float:
        """
        Computes midpoint arc elasticity:
        E = ((Y1 - Y0) / ((Y1 + Y0) / 2)) / ((X1 - X0) / ((X1 + X0) / 2))
        """
        delta_x = x_var - x_base
        delta_y = y_var - y_base

        if abs(delta_x) < epsilon:
            return 0.0

        avg_x = (x_base + x_var) / 2.0
        avg_y = (y_base + y_var) / 2.0

        if abs(avg_y) < epsilon or abs(avg_x) < epsilon:
            return 0.0

        pct_x = delta_x / avg_x
        pct_y = delta_y / avg_y

        return float(pct_y / pct_x)

    def evaluate_elasticities(
        self,
        scenarios_df: pd.DataFrame,
        baseline_scenario_name: str = "chem_nmc_baseline",
    ) -> list[ElasticityRecord]:
        """
        Extracts point and arc elasticities across horizon, temperature, and efficiency.
        """
        df = scenarios_df.copy()
        clean_name = df["scenario_name"].astype(str)

        base_match = df[clean_name.str.contains(baseline_scenario_name, case=False)]
        if base_match.empty:
            base_match = df[clean_name.str.contains("baseline", case=False)]

        base_row = base_match.iloc[0] if not base_match.empty else df.iloc[0]
        base_rev = float(base_row["net_revenue_usd"])

        records: list[ElasticityRecord] = []

        # 1. Horizon Elasticity
        hor_df = df[df["category"].astype(str).str.contains("Horizon", case=False)].sort_values("forecast_horizon")
        if len(hor_df) > 1:
            base_h_row = hor_df[hor_df["forecast_horizon"] == 48]
            base_h_row = base_h_row.iloc[0] if not base_h_row.empty else hor_df.iloc[len(hor_df) // 2]
            h0 = float(base_h_row["forecast_horizon"])
            y0 = float(base_h_row["net_revenue_usd"])

            for _, row in hor_df.iterrows():
                h1 = float(row["forecast_horizon"])
                y1 = float(row["net_revenue_usd"])
                if h1 == h0:
                    continue
                pct_h = ((h1 - h0) / h0) * 100.0
                pct_y = ((y1 - y0) / y0) * 100.0
                e = self.calculate_arc_elasticity(h0, h1, y0, y1)
                marginal = (y1 - y0) / (h1 - h0)
                records.append(ElasticityRecord(
                    parameter_dimension="Forecast_Horizon",
                    parameter_variation=f"{h1:.0f}h vs {h0:.0f}h",
                    base_param=h0,
                    varied_param=h1,
                    pct_delta_param=round(pct_h, 2),
                    base_net_revenue_usd=round(y0, 2),
                    varied_net_revenue_usd=round(y1, 2),
                    pct_delta_revenue=round(pct_y, 2),
                    arc_elasticity=round(e, 4),
                    marginal_rate_usd=round(marginal, 2),
                ))

        # 2. Thermal Elasticity
        thm_df = df[df["category"].astype(str).str.contains("Thermal", case=False)].copy()
        if len(thm_df) > 1:
            thm_df["temp_c"] = thm_df["scenario_name"].str.extract(r"(\d+)c")[0].astype(float)
            thm_df = thm_df.dropna(subset=["temp_c"]).sort_values("temp_c")

            if len(thm_df) > 1:
                base_t_row = thm_df[thm_df["temp_c"] == 25.0]
                base_t_row = base_t_row.iloc[0] if not base_t_row.empty else thm_df.iloc[0]
                t0 = float(base_t_row["temp_c"])
                y0 = float(base_t_row["net_revenue_usd"])

                for _, row in thm_df.iterrows():
                    t1 = float(row["temp_c"])
                    y1 = float(row["net_revenue_usd"])
                    if t1 == t0:
                        continue
                    pct_t = ((t1 - t0) / t0) * 100.0
                    pct_y = ((y1 - y0) / y0) * 100.0
                    e = self.calculate_arc_elasticity(t0, t1, y0, y1)
                    marginal = (y1 - y0) / (t1 - t0)
                    records.append(ElasticityRecord(
                        parameter_dimension="Thermal_Sensitivity",
                        parameter_variation=f"{t1:.0f}°C vs {t0:.0f}°C",
                        base_param=t0,
                        varied_param=t1,
                        pct_delta_param=round(pct_t, 2),
                        base_net_revenue_usd=round(y0, 2),
                        varied_net_revenue_usd=round(y1, 2),
                        pct_delta_revenue=round(pct_y, 2),
                        arc_elasticity=round(e, 4),
                        marginal_rate_usd=round(marginal, 2),
                    ))

        # 3. Efficiency Elasticity
        eff_df = df[df["category"].astype(str).str.contains("Efficiency", case=False)].copy()
        if len(eff_df) > 1:
            eff_df["rte_pct"] = eff_df["scenario_name"].str.extract(r"(\d+)pct")[0].astype(float)
            eff_df = eff_df.dropna(subset=["rte_pct"]).sort_values("rte_pct")

            if len(eff_df) > 1:
                base_e_row = eff_df[eff_df["rte_pct"] == 90.0]
                base_e_row = base_e_row.iloc[0] if not base_e_row.empty else eff_df.iloc[0]
                e0 = float(base_e_row["rte_pct"])
                y0 = float(base_e_row["net_revenue_usd"])

                for _, row in eff_df.iterrows():
                    e1 = float(row["rte_pct"])
                    y1 = float(row["net_revenue_usd"])
                    if e1 == e0:
                        continue
                    pct_e = ((e1 - e0) / e0) * 100.0
                    pct_y = ((y1 - y0) / y0) * 100.0
                    arc_e = self.calculate_arc_elasticity(e0, e1, y0, y1)
                    marginal = (y1 - y0) / (e1 - e0)
                    records.append(ElasticityRecord(
                        parameter_dimension="Efficiency_Sensitivity",
                        parameter_variation=f"{e1:.1f}% vs {e0:.1f}%",
                        base_param=e0,
                        varied_param=e1,
                        pct_delta_param=round(pct_e, 2),
                        base_net_revenue_usd=round(y0, 2),
                        varied_net_revenue_usd=round(y1, 2),
                        pct_delta_revenue=round(pct_y, 2),
                        arc_elasticity=round(arc_e, 4),
                        marginal_rate_usd=round(marginal, 2),
                    ))

        return records

    # ------------------------------------------------------------------------
    # Tornado Sensitivity Formulation
    # ------------------------------------------------------------------------

    def evaluate_tornado(
        self,
        scenarios_df: pd.DataFrame,
        baseline_revenue_usd: float = 848333.00,
    ) -> list[TornadoParameterRecord]:
        df = scenarios_df.copy()
        tornado_list: list[TornadoParameterRecord] = []

        categories = {
            "Forecast Look-Ahead": ("Forecast_Horizon", 48.0, 12.0, 72.0),
            "System Duration / Sizing": ("System_Sizing", 100.0, 50.0, 200.0),
            "Operating Temperature": ("Thermal_Sensitivity", 25.0, 15.0, 45.0),
            "Round-Trip Efficiency": ("Efficiency_Sensitivity", 90.25, 85.0, 95.0),
            "Battery Chemistry": ("Battery_Chemistry", 1.0, 0.5, 2.0),
            "Degradation Hurdle Penalty": ("Degradation_Modeling", 10.0, 0.0, 25.0),
        }

        for param_label, (cat_name, b_val, l_val, h_val) in categories.items():
            sub = df[df["category"].astype(str).str.contains(cat_name, case=False)]
            if sub.empty:
                continue

            revs = sub["net_revenue_usd"].to_numpy(dtype=float)
            min_rev = float(np.min(revs))
            max_rev = float(np.max(revs))
            swing = max_rev - min_rev

            tornado_list.append(TornadoParameterRecord(
                parameter=param_label,
                baseline_value=b_val,
                low_value=l_val,
                high_value=h_val,
                low_revenue_usd=round(min_rev, 2),
                high_revenue_usd=round(max_rev, 2),
                swing_usd=round(swing, 2),
                sensitivity_rank=0,
            ))

        tornado_list.sort(key=lambda x: x.swing_usd, reverse=True)
        for rank_idx, item in enumerate(tornado_list, 1):
            item.sensitivity_rank = rank_idx

        return tornado_list

    # ------------------------------------------------------------------------
    # Publication Visualizations
    # ------------------------------------------------------------------------

    def plot_tornado_chart(
        self,
        tornado_records: list[TornadoParameterRecord],
        baseline_revenue_usd: float = 848333.00,
        filename: str = "tornado_sensitivity_chart.png",
    ) -> Path:
        fig, ax = plt.subplots(figsize=(10, 6))

        params = [r.parameter for r in reversed(tornado_records)]
        y_pos = np.arange(len(params))

        low_deltas = [(r.low_revenue_usd - baseline_revenue_usd) / 1000.0 for r in reversed(tornado_records)]
        high_deltas = [(r.high_revenue_usd - baseline_revenue_usd) / 1000.0 for r in reversed(tornado_records)]

        ax.barh(y_pos, low_deltas, align="center", color="#d62728", alpha=0.85, label="Adverse / Low Bound Delta ($k)")
        ax.barh(y_pos, high_deltas, align="center", color="#2ca02c", alpha=0.85, label="Favorable / High Bound Delta ($k)")

        ax.axvline(0.0, color="black", linewidth=1.2, linestyle="-")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(params, fontsize=9.5)
        ax.set_xlabel("Net Revenue Deviation from Baseline ($k USD)", fontsize=10)
        ax.set_title(
            f"Parametric Tornado Diagram: Net Arbitrage Sensitivity (Baseline: ${baseline_revenue_usd:,.0f})",
            fontsize=11,
            fontweight="bold",
        )
        ax.grid(True, linestyle="--", alpha=0.5, axis="x")

        for i, (l_val, h_val) in enumerate(zip(low_deltas, high_deltas)):
            ax.text(l_val - 10.0, i, f"${l_val:+,.0f}k", va="center", ha="right", fontsize=8)
            ax.text(h_val + 10.0, i, f"${h_val:+,.0f}k", va="center", ha="left", fontsize=8)

        ax.legend(loc="lower right", framealpha=0.95)

        out_path = self.figure_dir / filename
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_spider_radar(
        self,
        scenarios_df: pd.DataFrame,
        filename: str = "spider_radar_chart.png",
    ) -> Path:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, polar=True)

        categories = [
            "Net Profit ($)",
            "Asset Longevity (SOH)",
            "Risk Stability (Sharpe)",
            "Cycling Activity (EFC)",
            "Gross Margin (%)",
        ]
        num_vars = len(categories)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        chem_specs = [
            ("chem_nmc_baseline", "NMC Baseline", "#1f77b4"),
            ("chem_lfp_stationary", "LFP Stationary", "#2ca02c"),
            ("chem_lto_heavy_cycle", "LTO Heavy Duty", "#ff7f0e"),
        ]

        df = scenarios_df.copy()
        clean_name = df["scenario_name"].astype(str)

        p_min, p_max = df["net_revenue_usd"].min(), df["net_revenue_usd"].max()
        s_min = df["final_soh"].min() if "final_soh" in df.columns else 0.90
        s_max = df["final_soh"].max() if "final_soh" in df.columns else 1.00
        sh_min = df["sharpe_ratio"].min() if "sharpe_ratio" in df.columns else 1.0
        sh_max = df["sharpe_ratio"].max() if "sharpe_ratio" in df.columns else 3.0
        e_min = df["cumulative_efc"].min() if "cumulative_efc" in df.columns else 100.0
        e_max = df["cumulative_efc"].max() if "cumulative_efc" in df.columns else 250.0

        for pattern, label, color in chem_specs:
            match = df[clean_name.str.contains(pattern, case=False)]
            if match.empty:
                continue
            row = match.iloc[0]

            v_prof = ((row["net_revenue_usd"] - p_min) / max(p_max - p_min, 1e-4)) * 100.0
            v_soh = ((row.get("final_soh", 0.98) - s_min) / max(s_max - s_min, 1e-4)) * 100.0
            v_sh = ((row.get("sharpe_ratio", 2.0) - sh_min) / max(sh_max - sh_min, 1e-4)) * 100.0
            v_efc = ((row.get("cumulative_efc", 180.0) - e_min) / max(e_max - e_min, 1e-4)) * 100.0
            v_grm = (row["net_revenue_usd"] / max(row.get("gross_revenue_usd", row["net_revenue_usd"]), 1e-4)) * 100.0

            values = [v_prof, v_soh, v_sh, v_efc, v_grm]
            values += values[:1]

            ax.plot(angles, values, color=color, linewidth=2.0, label=label)
            ax.fill(angles, values, color=color, alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9.5, fontweight="bold")
        ax.set_ylim(0, 105)
        ax.set_title("Multi-Criteria Techno-Economic Chemistry Radar", fontsize=12, fontweight="bold", y=1.08)
        ax.grid(True, linestyle="--", alpha=0.5)

        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.25, 1.1), framealpha=0.95)

        out_path = self.figure_dir / filename
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_horizon_elasticity_curve(
        self,
        scenarios_df: pd.DataFrame,
        filename: str = "horizon_elasticity_curve.png",
    ) -> Path:
        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax2 = ax1.twinx()

        hor_df = scenarios_df[scenarios_df["category"].astype(str).str.contains("Horizon", case=False)].copy()
        if hor_df.empty:
            ax1.text(0.5, 0.5, "No Horizon Data", ha="center", va="center")
            out_path = self.figure_dir / filename
            plt.savefig(out_path)
            plt.close(fig)
            return out_path

        hor_df = hor_df.sort_values("forecast_horizon")
        x = hor_df["forecast_horizon"].to_numpy(dtype=float)
        y_rev = hor_df["net_revenue_usd"].to_numpy(dtype=float) / 1000.0

        marginal_gain = np.gradient(y_rev, x) * 1000.0

        line1 = ax1.plot(x, y_rev, color="#1f77b4", marker="o", linewidth=2.2, label="Net Revenue ($k)")
        bar2 = ax2.bar(x, marginal_gain, width=3.5, color="#ff7f0e", alpha=0.35, label="Marginal Gain ($/Hour Look-Ahead)")

        ax1.set_xlabel("Forecast Look-Ahead Horizon (Hours)", fontsize=10)
        ax1.set_ylabel("Net Arbitrage Revenue ($k USD)", color="#1f77b4", fontsize=10)
        ax2.set_ylabel("Marginal Economic Gain ($/Hour)", color="#ff7f0e", fontsize=10)
        ax1.set_xticks(x)
        ax1.grid(True, linestyle="--", alpha=0.5)

        lines = line1 + [bar2]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper left", framealpha=0.95)

        ax1.set_title("Forecast Horizon Look-Ahead Valuation & Marginal Yield Curve", fontsize=11, fontweight="bold")

        out_path = self.figure_dir / filename
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_thermal_surface(
        self,
        scenarios_df: pd.DataFrame,
        filename: str = "thermal_degradation_surface.png",
    ) -> Path:
        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax2 = ax1.twinx()

        thm_df = scenarios_df[scenarios_df["category"].astype(str).str.contains("Thermal", case=False)].copy()
        if thm_df.empty:
            ax1.text(0.5, 0.5, "No Thermal Data", ha="center", va="center")
            out_path = self.figure_dir / filename
            plt.savefig(out_path)
            plt.close(fig)
            return out_path

        thm_df["temp_c"] = thm_df["scenario_name"].str.extract(r"(\d+)c")[0].astype(float)
        thm_df = thm_df.dropna(subset=["temp_c"]).sort_values("temp_c")

        if thm_df.empty:
            ax1.text(0.5, 0.5, "No Temperature Data Found in Scenario Names", ha="center", va="center")
            out_path = self.figure_dir / filename
            plt.savefig(out_path)
            plt.close(fig)
            return out_path

        t = thm_df["temp_c"].to_numpy(dtype=float)

        # Defensive extraction of degradation cost
        if "degradation_cost_usd" in thm_df.columns:
            deg_cost = thm_df["degradation_cost_usd"].to_numpy(dtype=float) / 1000.0
        elif "gross_revenue_usd" in thm_df.columns and "net_revenue_usd" in thm_df.columns:
            deg_cost = (thm_df["gross_revenue_usd"].to_numpy(dtype=float) - thm_df["net_revenue_usd"].to_numpy(dtype=float)) / 1000.0
        else:
            deg_cost = np.zeros(len(thm_df))

        fade_pct = ((1.0 - thm_df["final_soh"].to_numpy(dtype=float)) * 100.0) if "final_soh" in thm_df.columns else np.zeros(len(thm_df))

        line1 = ax1.plot(t, deg_cost, color="#d62728", marker="s", linewidth=2.0, label="Degradation Cost ($k)")
        line2 = ax2.plot(t, fade_pct, color="#e1974c", marker="^", linestyle="--", linewidth=2.0, label="Capacity Fade (% Loss)")

        ax1.set_xlabel("Ambient Operating Cell Temperature (°C)", fontsize=10)
        ax1.set_ylabel("Degradation Wear Cost ($k USD)", color="#d62728", fontsize=10)
        ax2.set_ylabel("Annual Capacity Fade (%)", color="#e1974c", fontsize=10)
        ax1.set_xticks(t)
        ax1.grid(True, linestyle="--", alpha=0.5)

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper left", framealpha=0.95)

        ax1.set_title("Arrhenius Thermal Acceleration: Temperature vs Degradation Cost & SOH Loss", fontsize=11, fontweight="bold")

        out_path = self.figure_dir / filename
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_efficiency_elasticity(
        self,
        scenarios_df: pd.DataFrame,
        filename: str = "efficiency_revenue_elasticity.png",
    ) -> Path:
        fig, ax = plt.subplots(figsize=(9, 5))

        eff_df = scenarios_df[scenarios_df["category"].astype(str).str.contains("Efficiency", case=False)].copy()
        if eff_df.empty:
            ax.text(0.5, 0.5, "No Efficiency Data", ha="center", va="center")
            out_path = self.figure_dir / filename
            plt.savefig(out_path)
            plt.close(fig)
            return out_path

        eff_df["rte_pct"] = eff_df["scenario_name"].str.extract(r"(\d+)pct")[0].astype(float)
        eff_df = eff_df.dropna(subset=["rte_pct"]).sort_values("rte_pct")

        if eff_df.empty:
            ax.text(0.5, 0.5, "No RTE Data Found in Scenario Names", ha="center", va="center")
            out_path = self.figure_dir / filename
            plt.savefig(out_path)
            plt.close(fig)
            return out_path

        x = eff_df["rte_pct"].to_numpy(dtype=float)
        y = eff_df["net_revenue_usd"].to_numpy(dtype=float) / 1000.0

        ax.plot(x, y, color="#2ca02c", marker="o", linewidth=2.2, label="Net Arbitrage Revenue ($k)")

        if len(x) > 1 and float(np.ptp(x)) > 1e-4:
            poly = np.polyfit(x, y, 1)
            x_seq = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_seq, np.polyval(poly, x_seq), "k--", linewidth=1.2, label=f"Elasticity Slope: +${poly[0]:.2f}k / 1% RTE")

        ax.set_title("AC-AC Round-Trip Efficiency (RTE) Revenue Elasticity Profile", fontsize=11, fontweight="bold")
        ax.set_xlabel("Round-Trip Efficiency (RTE %)", fontsize=10)
        ax.set_ylabel("Net Revenue ($k USD)", fontsize=10)
        ax.set_xticks(x)
        ax.grid(True, linestyle="--", alpha=0.5)

        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(handles, labels, loc="upper left", framealpha=0.95)

        out_path = self.figure_dir / filename
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
        return out_path

    # ------------------------------------------------------------------------
    # Master Pipeline Execution
    # ------------------------------------------------------------------------

    def evaluate_and_export(
        self,
        scenarios_df: pd.DataFrame,
        baseline_revenue_usd: float = 848333.00,
    ) -> tuple[list[ElasticityRecord], list[TornadoParameterRecord], SensitivityArtifacts]:
        elasticities = self.evaluate_elasticities(scenarios_df)
        tornado_records = self.evaluate_tornado(scenarios_df, baseline_revenue_usd=baseline_revenue_usd)

        # 1. Export CSV Artifacts
        tornado_csv = self.output_dir / "tornado_parameters.csv"
        df_tor = pd.DataFrame([asdict(t) for t in tornado_records])
        df_tor.to_csv(tornado_csv, index=False)

        elasticity_csv = self.output_dir / "elasticity_matrix.csv"
        df_ela = pd.DataFrame([asdict(e) for e in elasticities])
        df_ela.to_csv(elasticity_csv, index=False)

        summary_csv = self.output_dir / "sensitivity_summary.csv"
        df_tor[["parameter", "swing_usd", "sensitivity_rank"]].to_csv(summary_csv, index=False)

        # 2. Export JSON Scorecard
        summary_json = self.output_dir / "sensitivity_summary.json"
        summary_dict = {
            "most_sensitive_parameter": tornado_records[0].parameter if tornado_records else "None",
            "max_parametric_swing_usd": tornado_records[0].swing_usd if tornado_records else 0.0,
            "total_elasticities_evaluated": len(elasticities),
            "tornado_spectrum": [asdict(t) for t in tornado_records],
        }
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, indent=4)

        # 3. Export Multi-Tab Excel Workbook
        excel_path = self.output_dir / "sensitivity_report.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df_tor.to_excel(writer, sheet_name="Tornado_Spectrum", index=False)
            df_ela.to_excel(writer, sheet_name="Elasticities", index=False)
            scenarios_df.to_excel(writer, sheet_name="All_Scenarios", index=False)

        # 4. Export Diagnostic Figures
        fig_tor = self.plot_tornado_chart(tornado_records, baseline_revenue_usd=baseline_revenue_usd)
        fig_rad = self.plot_spider_radar(scenarios_df)
        fig_hor = self.plot_horizon_elasticity_curve(scenarios_df)
        fig_thm = self.plot_thermal_surface(scenarios_df)
        fig_eff = self.plot_efficiency_elasticity(scenarios_df)

        artifacts = SensitivityArtifacts(
            summary_csv=summary_csv,
            elasticity_csv=elasticity_csv,
            tornado_csv=tornado_csv,
            sensitivity_excel=excel_path,
            summary_json=summary_json,
            figure_tornado=fig_tor,
            figure_radar=fig_rad,
            figure_horizon=fig_hor,
            figure_thermal=fig_thm,
            figure_efficiency=fig_eff,
            figures_directory=self.figure_dir,
        )

        return elasticities, tornado_records, artifacts