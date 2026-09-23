"""
backtesting/arbitrage_metrics.py
================================

Battery Arbitrage Metrics & Cycle Economics Module (Part 9.2)



Capabilities:
1. Revenue and Profit Decomposition:
   - Gross Arbitrage Revenue ($)
   - Degradation Cost ($)
   - Net Arbitrage Revenue ($)
   - Fixed and Variable O&M ($)
   - Net Operating Profit (EBITDA $)
   - Net Arbitrage Value Margin (%)
2. Cycle Economics and Unit-Normalized KPIs:
   - Gross and Net Revenue per MWh Throughput ($/MWh)
   - Gross and Net Revenue per Equivalent Full Cycle ($/EFC)
   - Degradation Wear Cost per EFC ($/EFC)
   - Annualized Capacity Yield ($/kW-year and $/kWh-year)
3. Trading Spread Capture and Execution Quality:
   - Volume-Weighted Charging Price ($/MWh)
   - Volume-Weighted Discharging Price ($/MWh)
   - Realized Price Spread ($/MWh)
   - Theoretical Maximum Daily Spread ($/MWh)
   - Spread Capture Ratio (%)
4. Duty Cycle and Utilization Telemetry:
   - Equivalent Full Cycles (EFC) and Average Daily Cycling Rate
   - Capacity Factor (%) and Utilization Factor (%)
   - Charging, Discharging, and Idle Duty Fractions (%)
   - Realized AC-AC Round-Trip Efficiency (RTE %)
5. Granular Temporal Breakdowns and Diagnostic Visualizations:
   - Daily and Monthly Performance DataFrames
   - Publication-ready Matplotlib figures
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
import pandas as pd

plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"


# ============================================================================
# Dataclasses
# ============================================================================

@dataclass(slots=True)
class ArbitrageKPIs:
    gross_revenue_usd: float
    degradation_cost_usd: float
    net_revenue_usd: float
    fixed_om_cost_usd: float
    variable_om_cost_usd: float
    net_operating_profit_usd: float
    net_arbitrage_margin_pct: float
    total_energy_charged_mwh: float
    total_energy_discharged_mwh: float
    energy_throughput_mwh: float
    equivalent_full_cycles: float
    avg_daily_cycles: float
    capacity_factor_pct: float
    utilization_factor_pct: float
    round_trip_efficiency_pct: float
    avg_charge_price_usd_per_mwh: float
    avg_discharge_price_usd_per_mwh: float
    realized_spread_usd_per_mwh: float
    theoretical_max_spread_usd_per_mwh: float
    spread_capture_ratio_pct: float
    gross_revenue_per_mwh_throughput: float
    net_revenue_per_mwh_throughput: float
    gross_revenue_per_efc: float
    net_revenue_per_efc: float
    degradation_cost_per_efc: float
    revenue_per_kw_year: float
    revenue_per_kwh_year: float
    charging_hours: int
    discharging_hours: int
    idle_hours: int
    idle_fraction_pct: float
    simulated_days: float


@dataclass(slots=True)
class ArbitrageArtifacts:
    kpis_csv: Path
    monthly_csv: Path
    daily_csv: Path
    kpis_json: Path
    figure_waterfall: Path
    figure_monthly: Path
    figure_cycle_economics: Path
    figure_spread_dist: Path
    figure_duration_curve: Path
    figures_directory: Path


# ============================================================================
# Core Arbitrage Metrics Engine
# ============================================================================

class ArbitrageMetricsEngine:
    """
    Evaluates detailed BESS arbitrage performance, cycle economics, and spread capture.
    """

    def __init__(self, output_directory: Path | str = "backtesting/results/arbitrage_metrics"):
        self.output_dir = Path(output_directory)
        self.figure_dir = self.output_dir / "figures"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figure_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------------
    # Core Mathematical & Economic Evaluation
    # ------------------------------------------------------------------------

    def evaluate(
        self,
        dispatch_df: pd.DataFrame,
        degradation_df: pd.DataFrame | None = None,
        nominal_capacity_mwh: float = 100.0,
        rated_power_mw: float = 50.0,
        fixed_om_per_mw_year: float = 7500.0,
        variable_om_per_mwh: float = 0.50,
    ) -> tuple[ArbitrageKPIs, pd.DataFrame, pd.DataFrame]:
        df = dispatch_df.copy()

        # 1. Price and Power Signal Resolution
        price_col = next((c for c in ["actual_price", "price", "settlement_price", "forecast_price"] if c in df.columns), None)
        if price_col is None:
            raise KeyError("Dispatch DataFrame must contain a valid price column.")

        prices = df[price_col].to_numpy(dtype=float)

        if "charge_power_mw" in df.columns:
            chg_pwr = df["charge_power_mw"].to_numpy(dtype=float)
            dis_pwr = df["discharge_power_mw"].to_numpy(dtype=float)
        elif "energy_charged_mwh" in df.columns:
            chg_pwr = df["energy_charged_mwh"].to_numpy(dtype=float)
            dis_pwr = df["energy_discharged_mwh"].to_numpy(dtype=float)
        else:
            raise KeyError("Dispatch DataFrame must contain charge and discharge power columns.")

        # 2. Energy Throughput and Efficiency
        total_charged = float(np.sum(chg_pwr))
        total_discharged = float(np.sum(dis_pwr))
        throughput = total_charged + total_discharged

        rte_pct = (total_discharged / total_charged * 100.0) if total_charged > 0 else 0.0

        # 3. Revenue Accounting
        rev_col = next((c for c in ["net_revenue_usd", "net_revenue_$"] if c in df.columns), None)
        if rev_col is not None:
            gross_revenue = float(df[rev_col].sum())
        else:
            gross_revenue = float(np.sum((dis_pwr - chg_pwr) * prices))

        if degradation_df is not None and "degradation_cost_usd" in degradation_df.columns:
            degradation_cost = float(degradation_df["degradation_cost_usd"].sum())
        elif "degradation_cost_usd" in df.columns:
            degradation_cost = float(df["degradation_cost_usd"].sum())
        else:
            degradation_cost = 0.0

        net_revenue = gross_revenue - degradation_cost

        # 4. Duty Cycles & Operational Hours
        EPS = 1e-4
        charging_mask = chg_pwr > EPS
        discharging_mask = dis_pwr > EPS
        idle_mask = (~charging_mask) & (~discharging_mask)

        n_hours = len(df)
        sim_days = max(n_hours / 24.0, 1.0)
        n_charging = int(np.sum(charging_mask))
        n_discharging = int(np.sum(discharging_mask))
        n_idle = int(np.sum(idle_mask))
        idle_pct = (n_idle / max(n_hours, 1)) * 100.0

        # 5. Fixed & Variable O&M
        fixed_om = rated_power_mw * fixed_om_per_mw_year * (sim_days / 365.0)
        variable_om = throughput * variable_om_per_mwh
        net_profit = net_revenue - (fixed_om + variable_om)
        net_margin = (net_revenue / gross_revenue * 100.0) if gross_revenue > 0 else 0.0

        # 6. EFC and Utilization
        if degradation_df is not None and "window_efc" in degradation_df.columns:
            efc = float(degradation_df["window_efc"].sum())
        elif degradation_df is not None and "cumulative_efc" in degradation_df.columns:
            efc = float(degradation_df["cumulative_efc"].iloc[-1])
        else:
            efc = throughput / (2.0 * nominal_capacity_mwh)

        daily_cycles = efc / sim_days
        capacity_factor = (total_discharged / (rated_power_mw * n_hours)) * 100.0 if (rated_power_mw * n_hours) > 0 else 0.0
        utilization_factor = throughput / (2.0 * nominal_capacity_mwh * sim_days) * 100.0

        # 7. Volume-Weighted Prices and Realized Spread
        avg_chg_price = float(np.sum(prices * chg_pwr) / total_charged) if total_charged > 0 else 0.0
        avg_dis_price = float(np.sum(prices * dis_pwr) / total_discharged) if total_discharged > 0 else 0.0
        realized_spread = avg_dis_price - avg_chg_price

        # 8. Daily Arbitrage Aggregation & Theoretical Max Spread
        df["_hour_idx"] = np.arange(len(df))
        if "timestamp" in df.columns:
            ts_series = pd.to_datetime(df["timestamp"])
            df["_date"] = ts_series.dt.date
            df["_year_month"] = ts_series.dt.strftime("%Y-%m")
        else:
            df["_date"] = df["_hour_idx"] // 24
            df["_year_month"] = (df["_hour_idx"] // (24 * 30)).astype(str)

        daily_groups = []
        for date_key, group in df.groupby("_date"):
            p_grp = group[price_col].values
            c_grp = group["charge_power_mw"].values if "charge_power_mw" in group.columns else (
                group["energy_charged_mwh"].values if "energy_charged_mwh" in group.columns else chg_pwr[group.index]
            )
            d_grp = group["discharge_power_mw"].values if "discharge_power_mw" in group.columns else (
                group["energy_discharged_mwh"].values if "energy_discharged_mwh" in group.columns else dis_pwr[group.index]
            )

            day_chg = float(np.sum(c_grp))
            day_dis = float(np.sum(d_grp))
            day_gross = float(np.sum((d_grp - c_grp) * p_grp)) if rev_col is None else float(group[rev_col].sum())
            day_thp = day_chg + day_dis
            day_efc = day_thp / (2.0 * nominal_capacity_mwh)

            p_max = float(np.max(p_grp))
            p_min = float(np.min(p_grp))
            theo_spread = max(p_max - p_min, 0.0)

            chg_w_p = float(np.sum(p_grp * c_grp) / day_chg) if day_chg > 0 else 0.0
            dis_w_p = float(np.sum(p_grp * d_grp) / day_dis) if day_dis > 0 else 0.0
            day_real_spread = dis_w_p - chg_w_p

            spread_cap_pct = (day_real_spread / theo_spread * 100.0) if theo_spread > 0 and day_chg > 0 and day_dis > 0 else 0.0

            daily_groups.append({
                "date": str(date_key),
                "gross_revenue_usd": round(day_gross, 2),
                "energy_charged_mwh": round(day_chg, 2),
                "energy_discharged_mwh": round(day_dis, 2),
                "throughput_mwh": round(day_thp, 2),
                "efc": round(day_efc, 3),
                "avg_charge_price": round(chg_w_p, 2),
                "avg_discharge_price": round(dis_w_p, 2),
                "realized_spread": round(day_real_spread, 2),
                "theoretical_max_spread": round(theo_spread, 2),
                "spread_capture_pct": round(spread_cap_pct, 2),
            })

        daily_df = pd.DataFrame(daily_groups)
        theo_max_spread_overall = float(daily_df["theoretical_max_spread"].mean()) if not daily_df.empty else 0.0
        overall_spread_cap_pct = (realized_spread / theo_max_spread_overall * 100.0) if theo_max_spread_overall > 0 else 0.0

        # 9. Monthly Aggregation
        monthly_groups = []
        for ym_key, grp in df.groupby("_year_month"):
            m_chg = float(grp["charge_power_mw"].sum()) if "charge_power_mw" in grp.columns else float(grp["energy_charged_mwh"].sum())
            m_dis = float(grp["discharge_power_mw"].sum()) if "discharge_power_mw" in grp.columns else float(grp["energy_discharged_mwh"].sum())
            m_rev = float(grp[rev_col].sum()) if rev_col else float(np.sum((m_dis - m_chg) * grp[price_col]))
            m_thp = m_chg + m_dis
            m_efc = m_thp / (2.0 * nominal_capacity_mwh)
            m_chg_p = float(np.sum(grp[price_col] * grp["charge_power_mw"]) / m_chg) if m_chg > 0 else 0.0
            m_dis_p = float(np.sum(grp[price_col] * grp["discharge_power_mw"]) / m_dis) if m_dis > 0 else 0.0
            m_spread = m_dis_p - m_chg_p

            monthly_groups.append({
                "year_month": str(ym_key),
                "gross_revenue_usd": round(m_rev, 2),
                "energy_discharged_mwh": round(m_dis, 2),
                "throughput_mwh": round(m_thp, 2),
                "efc": round(m_efc, 2),
                "realized_spread_usd": round(m_spread, 2),
                "revenue_per_efc": round(m_rev / m_efc, 2) if m_efc > 0 else 0.0,
            })
        monthly_df = pd.DataFrame(monthly_groups)

        df.drop(columns=["_hour_idx", "_date", "_year_month"], errors="ignore", inplace=True)

        # 10. Unit Normalized Indicators
        gross_per_mwh = gross_revenue / throughput if throughput > 0 else 0.0
        net_per_mwh = net_revenue / throughput if throughput > 0 else 0.0
        gross_per_efc = gross_revenue / efc if efc > 0 else 0.0
        net_per_efc = net_revenue / efc if efc > 0 else 0.0
        deg_per_efc = degradation_cost / efc if efc > 0 else 0.0

        annual_scale = 365.0 / sim_days
        rev_per_kw_yr = (gross_revenue * annual_scale) / (rated_power_mw * 1000.0)
        rev_per_kwh_yr = (gross_revenue * annual_scale) / (nominal_capacity_mwh * 1000.0)

        kpis = ArbitrageKPIs(
            gross_revenue_usd=round(gross_revenue, 2),
            degradation_cost_usd=round(degradation_cost, 2),
            net_revenue_usd=round(net_revenue, 2),
            fixed_om_cost_usd=round(fixed_om, 2),
            variable_om_cost_usd=round(variable_om, 2),
            net_operating_profit_usd=round(net_profit, 2),
            net_arbitrage_margin_pct=round(net_margin, 2),
            total_energy_charged_mwh=round(total_charged, 2),
            total_energy_discharged_mwh=round(total_discharged, 2),
            energy_throughput_mwh=round(throughput, 2),
            equivalent_full_cycles=round(efc, 2),
            avg_daily_cycles=round(daily_cycles, 3),
            capacity_factor_pct=round(capacity_factor, 2),
            utilization_factor_pct=round(utilization_factor, 2),
            round_trip_efficiency_pct=round(rte_pct, 2),
            avg_charge_price_usd_per_mwh=round(avg_chg_price, 2),
            avg_discharge_price_usd_per_mwh=round(avg_dis_price, 2),
            realized_spread_usd_per_mwh=round(realized_spread, 2),
            theoretical_max_spread_usd_per_mwh=round(theo_max_spread_overall, 2),
            spread_capture_ratio_pct=round(min(overall_spread_cap_pct, 100.0), 2),
            gross_revenue_per_mwh_throughput=round(gross_per_mwh, 2),
            net_revenue_per_mwh_throughput=round(net_per_mwh, 2),
            gross_revenue_per_efc=round(gross_per_efc, 2),
            net_revenue_per_efc=round(net_per_efc, 2),
            degradation_cost_per_efc=round(deg_per_efc, 2),
            revenue_per_kw_year=round(rev_per_kw_yr, 2),
            revenue_per_kwh_year=round(rev_per_kwh_yr, 2),
            charging_hours=n_charging,
            discharging_hours=n_discharging,
            idle_hours=n_idle,
            idle_fraction_pct=round(idle_pct, 2),
            simulated_days=round(sim_days, 1),
        )

        return kpis, daily_df, monthly_df

    # ------------------------------------------------------------------------
    # Publication Visualizations
    # ------------------------------------------------------------------------

    def plot_waterfall(self, kpis: ArbitrageKPIs, filename: str = "arbitrage_waterfall_breakdown.png") -> Path:
        fig, ax = plt.subplots(figsize=(10, 5))

        labels = [
            "Gross Arbitrage\nRevenue",
            "Battery Cell\nDegradation",
            "Fixed Facility\nO&M",
            "Variable\nO&M",
            "Net Operating\nProfit (EBITDA)",
        ]
        values = [
            kpis.gross_revenue_usd,
            -kpis.degradation_cost_usd,
            -kpis.fixed_om_cost_usd,
            -kpis.variable_om_cost_usd,
            kpis.net_operating_profit_usd,
        ]
        colors = ["#2ca02c", "#d62728", "#ff7f0e", "#7293cb", "#1f77b4"]

        bars = ax.bar(labels, [abs(v) / 1000.0 for v in values], color=colors, width=0.55, alpha=0.9, edgecolor="black")

        for bar, val in zip(bars, values):
            y_val = bar.get_height()
            prefix = "+" if val > 0 else "-"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                y_val + 15.0,
                f"{prefix}${abs(val):,.0f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )

        ax.set_ylabel("Capital Flow ($k USD)", fontsize=10)
        ax.set_title("Techno-Economic Value Waterfall: Gross Capture to Net EBITDA", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5, axis="y")
        ax.set_ylim(0, max(kpis.gross_revenue_usd / 1000.0 * 1.25, 50.0))

        out_path = self.figure_dir / filename
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_monthly_spread_vs_revenue(self, monthly_df: pd.DataFrame, filename: str = "monthly_spread_vs_revenue.png") -> Path:
        fig, ax1 = plt.subplots(figsize=(11, 5))
        ax2 = ax1.twinx()

        x = np.arange(len(monthly_df))
        width = 0.4

        ax1.bar(x, monthly_df["gross_revenue_usd"] / 1000.0, width=width, color="#2ca02c", alpha=0.85, label="Gross Revenue ($k)")
        ax2.plot(x, monthly_df["realized_spread_usd"], color="#d62728", marker="o", linewidth=2.0, label="Realized Spread ($/MWh)")

        ax1.set_xlabel("Month of Simulation", fontsize=10)
        ax1.set_ylabel("Gross Arbitrage Revenue ($k USD)", color="#2ca02c", fontsize=10)
        ax2.set_ylabel("Realized Price Spread ($/MWh)", color="#d62728", fontsize=10)
        ax1.set_xticks(x)
        ax1.set_xticklabels(monthly_df["year_month"], rotation=30, ha="right", fontsize=9)
        ax1.grid(True, linestyle="--", alpha=0.4)

        ax1.set_title("Monthly Energy Arbitrage Dynamics: Realized Spread vs Revenue Yield", fontsize=11, fontweight="bold")

        out_path = self.figure_dir / filename
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_cycle_economics(self, daily_df: pd.DataFrame, filename: str = "cycle_economics_scatter.png") -> Path:
        fig, ax = plt.subplots(figsize=(9, 5))

        scatter = ax.scatter(
            daily_df["efc"],
            daily_df["gross_revenue_usd"],
            c=daily_df["realized_spread"],
            cmap="viridis",
            s=45,
            alpha=0.85,
            edgecolors="none",
            label="Daily Cycles",
        )
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Realized Price Spread ($/MWh)", fontsize=9)

        x_efc = daily_df["efc"].to_numpy(dtype=float)
        y_rev = daily_df["gross_revenue_usd"].to_numpy(dtype=float)

        # Guard against zero-variance conditioning rank warnings
        if len(x_efc) > 1 and float(np.ptp(x_efc)) > 1e-4:
            poly = np.polyfit(x_efc, y_rev, 1)
            x_seq = np.linspace(x_efc.min(), x_efc.max(), 100)
            ax.plot(x_seq, np.polyval(poly, x_seq), "r--", linewidth=1.5, label=f"Marginal Cycle Slope (${poly[0]:,.0f}/EFC)")

        ax.set_title("Daily Cycle Economics: Cycling Intensity (EFC) vs Daily Revenue", fontsize=11, fontweight="bold")
        ax.set_xlabel("Daily Cycling Intensity (Equivalent Full Cycles)", fontsize=10)
        ax.set_ylabel("Daily Gross Revenue ($ USD)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)

        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(handles, labels, loc="upper left", framealpha=0.95)

        out_path = self.figure_dir / filename
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_spread_capture_distribution(self, daily_df: pd.DataFrame, filename: str = "daily_spread_capture_distribution.png") -> Path:
        fig, ax = plt.subplots(figsize=(9, 5))

        ratios = daily_df["spread_capture_pct"].values
        mean_r = float(np.mean(ratios))
        median_r = float(np.median(ratios))

        ax.hist(ratios, bins=35, color="#7293cb", edgecolor="white", alpha=0.85, density=True)
        ax.axvline(mean_r, color="red", linestyle="--", linewidth=1.8, label=f"Mean Capture: {mean_r:.1f}%")
        ax.axvline(median_r, color="black", linestyle=":", linewidth=1.8, label=f"Median Capture: {median_r:.1f}%")

        ax.set_title("Daily Arbitrage Execution Efficiency: Spread Capture Ratio Distribution", fontsize=11, fontweight="bold")
        ax.set_xlabel("Spread Capture Ratio [% = Realized Spread / Theoretical Max Spread]", fontsize=10)
        ax.set_ylabel("Probability Density", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)

        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(handles, labels, loc="upper right", framealpha=0.95)

        out_path = self.figure_dir / filename
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_power_duration_curve(self, dispatch_df: pd.DataFrame, filename: str = "dispatch_duration_curve.png") -> Path:
        fig, ax = plt.subplots(figsize=(9, 5))

        if "charge_power_mw" in dispatch_df.columns:
            net_power = (dispatch_df["discharge_power_mw"] - dispatch_df["charge_power_mw"]).to_numpy(dtype=float)
        else:
            net_power = np.zeros(len(dispatch_df))

        sorted_net = np.sort(net_power)[::-1]
        x_pct = (np.arange(len(sorted_net)) / max(len(sorted_net), 1)) * 100.0

        ax.plot(x_pct, sorted_net, color="#1f77b4", linewidth=2.0, label="Net Power Duration Profile")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)

        ax.fill_between(x_pct, sorted_net, 0, where=(sorted_net > 0), color="#2ca02c", alpha=0.25, label="Discharge Regime")
        ax.fill_between(x_pct, sorted_net, 0, where=(sorted_net < 0), color="#ff7f0e", alpha=0.25, label="Charge Regime")

        ax.set_title("BESS Dispatch Power Duration Curve", fontsize=11, fontweight="bold")
        ax.set_xlabel("Percentage of Simulation Hours (%)", fontsize=10)
        ax.set_ylabel("Net Power [Discharge (+) / Charge (âˆ’)] (MW)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)

        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(handles, labels, loc="upper right", framealpha=0.95)

        out_path = self.figure_dir / filename
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
        return out_path

    # ------------------------------------------------------------------------
    # Master Execution & Export
    # ------------------------------------------------------------------------

    def evaluate_and_export(
        self,
        dispatch_df: pd.DataFrame,
        degradation_df: pd.DataFrame | None = None,
        nominal_capacity_mwh: float = 100.0,
        rated_power_mw: float = 50.0,
    ) -> tuple[ArbitrageKPIs, ArbitrageArtifacts]:
        kpis, daily_df, monthly_df = self.evaluate(
            dispatch_df=dispatch_df,
            degradation_df=degradation_df,
            nominal_capacity_mwh=nominal_capacity_mwh,
            rated_power_mw=rated_power_mw,
        )

        # 1. Export CSV Summaries
        kpis_csv = self.output_dir / "arbitrage_kpis_summary.csv"
        df_kpi = pd.DataFrame([asdict(kpis)]).T.reset_index()
        df_kpi.columns = ["metric", "value"]
        df_kpi.to_csv(kpis_csv, index=False)

        daily_csv = self.output_dir / "daily_arbitrage_summary.csv"
        daily_df.to_csv(daily_csv, index=False)

        monthly_csv = self.output_dir / "monthly_arbitrage_summary.csv"
        monthly_df.to_csv(monthly_csv, index=False)

        # 2. Export JSON Summary
        kpis_json = self.output_dir / "arbitrage_metrics.json"
        with open(kpis_json, "w", encoding="utf-8") as f:
            json.dump(asdict(kpis), f, indent=4)

        # 3. Export Visualizations
        fig_wf = self.plot_waterfall(kpis)
        fig_mon = self.plot_monthly_spread_vs_revenue(monthly_df)
        fig_cyc = self.plot_cycle_economics(daily_df)
        fig_dist = self.plot_spread_capture_distribution(daily_df)
        fig_dur = self.plot_power_duration_curve(dispatch_df)

        artifacts = ArbitrageArtifacts(
            kpis_csv=kpis_csv,
            monthly_csv=monthly_csv,
            daily_csv=daily_csv,
            kpis_json=kpis_json,
            figure_waterfall=fig_wf,
            figure_monthly=fig_mon,
            figure_cycle_economics=fig_cyc,
            figure_spread_dist=fig_dist,
            figure_duration_curve=fig_dur,
            figures_directory=self.figure_dir,
        )

        return kpis, artifacts