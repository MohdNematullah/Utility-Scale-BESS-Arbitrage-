"""
backtesting/export_reports.py
=============================

Research-Grade Backtest Export & Reporting Engine



Exports:
1. CSV summaries (Metrics, Financial, Battery, Operations, Forecast, Daily, Monthly)
2. Multi-tab Excel workbook with daily and monthly aggregations
3. Structured JSON summary report
4. Publication-quality figures:
   - 3-Curve Financial Trajectory (Gross, Degradation, Net Revenue)
   - Forecast Error Overlay with Residual Distribution
   - 2D Dispatch Intensity Heatmap
   - Monthly Degradation & SOH Dynamics
   - Price vs Dispatch Power (Dual Axis)
   - SOH Trajectory & SOC Profile
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtesting.engine import RollingBacktestResult
from backtesting.metrics import BacktestMetrics, BacktestMetricsResult

plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"


# ---------------------------------------------------------------------
# Result Dataclass
# ---------------------------------------------------------------------

@dataclass(slots=True)
class ExportReportResult:
    metrics_summary: Path
    financial_summary: Path
    battery_summary: Path
    operational_summary: Path
    forecast_summary: Path
    daily_summary: Path
    monthly_summary: Path
    excel_report: Path
    json_report: Path
    figure_directory: Path


# ---------------------------------------------------------------------
# Export Engine
# ---------------------------------------------------------------------

class BacktestExportEngine:
    """
    Research-grade exporter for rolling-horizon backtesting results.
    """

    def __init__(self, output_directory: str = "backtesting/results"):
        self.output_dir = Path(output_directory)
        self.figure_dir = self.output_dir / "figures"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figure_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_engine = BacktestMetrics()

    # ===============================================================
    # Public API
    # ===============================================================

    def export(self, result: RollingBacktestResult) -> ExportReportResult:
        metrics: BacktestMetricsResult = self.metrics_engine.evaluate(
            result.dispatch_history,
            result.degradation_history,
        )

        daily_df, monthly_df = self._build_aggregations(result)
        csv_tables = self._export_csv(result, metrics, daily_df, monthly_df)
        self._export_excel(result, metrics, csv_tables, daily_df, monthly_df)
        self._export_json(result, metrics)
        self._generate_figures(result, monthly_df)

        return ExportReportResult(
            metrics_summary=self.output_dir / "metrics_summary.csv",
            financial_summary=self.output_dir / "financial_summary.csv",
            battery_summary=self.output_dir / "battery_summary.csv",
            operational_summary=self.output_dir / "operational_summary.csv",
            forecast_summary=self.output_dir / "forecast_summary.csv",
            daily_summary=self.output_dir / "daily_financial_summary.csv",
            monthly_summary=self.output_dir / "monthly_financial_summary.csv",
            excel_report=self.output_dir / "backtest_report.xlsx",
            json_report=self.output_dir / "backtest_summary.json",
            figure_directory=self.figure_dir,
        )

    # ===============================================================
    # Daily & Monthly Aggregations
    # ===============================================================

    def _build_aggregations(self, result: RollingBacktestResult) -> tuple[pd.DataFrame, pd.DataFrame]:
        dispatch = result.dispatch_history.copy()
        degradation = result.degradation_history.copy()

        rev_col = "net_revenue_usd" if "net_revenue_usd" in dispatch.columns else "net_revenue_$"

        if "timestamp" in dispatch.columns:
            ts = pd.to_datetime(dispatch["timestamp"])
            dispatch["date"] = ts.dt.date
            # Avoid PeriodArray warning by dropping tz localization first
            dispatch["year_month"] = ts.dt.tz_localize(None).dt.to_period("M").astype(str)
        else:
            dispatch["date"] = dispatch.index // 24
            dispatch["year_month"] = dispatch.index // (24 * 30)

        # Aggregate daily dispatch
        daily_disp = dispatch.groupby("date").agg(
            gross_revenue_usd=(rev_col, "sum"),
            charge_energy_mwh=("charge_power_mw", "sum") if "charge_power_mw" in dispatch.columns else (rev_col, lambda x: 0.0),
            discharge_energy_mwh=("discharge_power_mw", "sum") if "discharge_power_mw" in dispatch.columns else (rev_col, lambda x: 0.0),
        ).reset_index()

        # Merge with daily degradation metrics
        if "rolling_window" in degradation.columns and len(degradation) == len(daily_disp):
            daily_df = daily_disp.copy()
            daily_df["degradation_cost_usd"] = degradation["degradation_cost_usd"].values
            daily_df["net_revenue_usd"] = daily_df["gross_revenue_usd"] - daily_df["degradation_cost_usd"]
            daily_df["efc"] = degradation["window_efc"].values if "window_efc" in degradation.columns else 0.0
            daily_df["remaining_soh"] = degradation["remaining_soh"].values if "remaining_soh" in degradation.columns else degradation["soh_end"].values
        else:
            daily_df = daily_disp.copy()
            daily_df["degradation_cost_usd"] = 0.0
            daily_df["net_revenue_usd"] = daily_df["gross_revenue_usd"]
            daily_df["efc"] = 0.0
            daily_df["remaining_soh"] = 1.0

        daily_df = daily_df.round(4)

        # Aggregate monthly table
        dispatch["date_dt"] = pd.to_datetime(dispatch["date"].astype(str))
        monthly_groups = []
        for period, group in dispatch.groupby(dispatch["date_dt"].dt.to_period("M")):
            idx_range = group.index
            start_win = idx_range[0] // 24
            end_win = (idx_range[-1] // 24) + 1
            m_deg = degradation.iloc[start_win:end_win] if len(degradation) >= len(dispatch) // 24 else degradation

            m_gross = group[rev_col].sum()
            m_deg_cost = m_deg["degradation_cost_usd"].sum() if "degradation_cost_usd" in m_deg.columns else 0.0
            m_cal_loss = m_deg["calendar_loss"].sum() * 100.0 if "calendar_loss" in m_deg.columns else 0.0
            m_cyc_loss = m_deg["cycle_loss"].sum() * 100.0 if "cycle_loss" in m_deg.columns else 0.0
            end_soh = m_deg["remaining_soh"].iloc[-1] if "remaining_soh" in m_deg.columns else m_deg["soh_end"].iloc[-1]

            monthly_groups.append({
                "year_month": str(period),
                "gross_revenue_usd": round(m_gross, 2),
                "degradation_cost_usd": round(m_deg_cost, 2),
                "net_revenue_usd": round(m_gross - m_deg_cost, 2),
                "calendar_fade_pct": round(m_cal_loss, 4),
                "cycle_fade_pct": round(m_cyc_loss, 4),
                "total_fade_pct": round(m_cal_loss + m_cyc_loss, 4),
                "end_soh": round(end_soh, 4),
            })

        monthly_df = pd.DataFrame(monthly_groups)
        return daily_df, monthly_df

    # ===============================================================
    # CSV EXPORTS
    # ===============================================================

    def _export_csv(
        self,
        result: RollingBacktestResult,
        metrics: BacktestMetricsResult,
        daily_df: pd.DataFrame,
        monthly_df: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        summary_df = metrics.dataframe.copy()
        summary_df.to_csv(self.output_dir / "metrics_summary.csv", index=False)

        financial = summary_df[
            summary_df["metric"].str.contains(
                "revenue|profit|cost|om|asset_value|sharpe|sortino|drawdown|benchmark|capture",
                case=False,
            )
        ].copy()

        battery = summary_df[
            summary_df["metric"].str.contains(
                "soh|fade|cycle|throughput|utilization|efficiency",
                case=False,
            )
        ].copy()

        operations = summary_df[
            summary_df["metric"].str.contains(
                "charging|discharging|idle|soc",
                case=False,
            )
        ].copy()

        forecast_rows = [
            {"metric": "forecast_horizon_hours", "value": result.summary.get("forecast_horizon_hours", 48)},
            {"metric": "implementation_horizon_hours", "value": result.summary.get("implementation_horizon_hours", 24)},
            {"metric": "rolling_step_hours", "value": result.summary.get("rolling_step_hours", 24)},
        ]
        forecast_metrics_df = summary_df[
            summary_df["metric"].str.startswith("forecast_")
        ]
        forecast = pd.concat([pd.DataFrame(forecast_rows), forecast_metrics_df], ignore_index=True)

        financial.to_csv(self.output_dir / "financial_summary.csv", index=False)
        battery.to_csv(self.output_dir / "battery_summary.csv", index=False)
        operations.to_csv(self.output_dir / "operational_summary.csv", index=False)
        forecast.to_csv(self.output_dir / "forecast_summary.csv", index=False)

        daily_df.to_csv(self.output_dir / "daily_financial_summary.csv", index=False)
        monthly_df.to_csv(self.output_dir / "monthly_financial_summary.csv", index=False)

        return {
            "financial": financial,
            "battery": battery,
            "operations": operations,
            "forecast": forecast,
        }

    # ===============================================================
    # EXCEL REPORT
    # ===============================================================

    def _export_excel(
        self,
        result: RollingBacktestResult,
        metrics: BacktestMetricsResult,
        csv_tables: dict[str, pd.DataFrame],
        daily_df: pd.DataFrame,
        monthly_df: pd.DataFrame,
    ):
        workbook = self.output_dir / "backtest_report.xlsx"

        with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
            metrics.dataframe.to_excel(writer, sheet_name="Summary", index=False)
            csv_tables["financial"].to_excel(writer, sheet_name="Financial", index=False)
            daily_df.to_excel(writer, sheet_name="Daily_Summary", index=False)
            monthly_df.to_excel(writer, sheet_name="Monthly_Summary", index=False)
            csv_tables["battery"].to_excel(writer, sheet_name="Battery", index=False)
            csv_tables["operations"].to_excel(writer, sheet_name="Operations", index=False)
            csv_tables["forecast"].to_excel(writer, sheet_name="Forecast", index=False)
            result.dispatch_history.to_excel(writer, sheet_name="Dispatch", index=False)
            result.degradation_history.to_excel(writer, sheet_name="Degradation", index=False)

    # ===============================================================
    # JSON REPORT
    # ===============================================================

    def _export_json(
        self,
        result: RollingBacktestResult,
        metrics: BacktestMetricsResult,
    ):
        engine_summary = result.summary
        metrics_summary = metrics.summary

        report = {
            "simulation": {
                "rolling_windows": engine_summary.get("rolling_windows", len(result.degradation_history)),
                "hours_executed": engine_summary.get("hours_executed", len(result.dispatch_history)),
                "forecast_horizon_hours": engine_summary.get("forecast_horizon_hours", 48),
                "implementation_horizon_hours": engine_summary.get("implementation_horizon_hours", 24),
                "rolling_step_hours": engine_summary.get("rolling_step_hours", 24),
            },
            "financial": {
                "gross_revenue_usd": metrics_summary.get("gross_revenue_usd"),
                "degradation_cost_usd": metrics_summary.get("degradation_cost_usd"),
                "net_revenue_usd": metrics_summary.get("net_revenue_usd"),
                "fixed_om_cost_usd": metrics_summary.get("fixed_om_cost_usd"),
                "variable_om_cost_usd": metrics_summary.get("variable_om_cost_usd"),
                "total_operating_cost_usd": metrics_summary.get("total_operating_cost_usd"),
                "net_operating_profit_usd": metrics_summary.get("net_operating_profit_usd"),
                "profit_factor": metrics_summary.get("profit_factor"),
                "sharpe_ratio": metrics_summary.get("sharpe_ratio"),
                "sortino_ratio": metrics_summary.get("sortino_ratio"),
                "max_drawdown_usd": metrics_summary.get("max_drawdown_usd"),
                "max_drawdown_pct": metrics_summary.get("max_drawdown_pct"),
                "perfect_foresight_benchmark_usd": metrics_summary.get("perfect_foresight_benchmark_usd"),
                "revenue_capture_ratio_pct": metrics_summary.get("revenue_capture_ratio_pct"),
            },
            "battery": {
                "initial_soh": metrics_summary.get("initial_soh"),
                "final_soh": metrics_summary.get("final_soh"),
                "capacity_fade": metrics_summary.get("capacity_fade"),
                "equivalent_full_cycles": metrics_summary.get("equivalent_full_cycles"),
                "round_trip_efficiency_pct": metrics_summary.get("round_trip_efficiency_pct"),
                "terminal_asset_value_usd": metrics_summary.get("terminal_asset_value_usd"),
            },
            "forecast_performance": {
                "mae": metrics_summary.get("forecast_mae"),
                "rmse": metrics_summary.get("forecast_rmse"),
                "mape_pct": metrics_summary.get("forecast_mape_pct"),
                "bias": metrics_summary.get("forecast_bias"),
                "r2_score": metrics_summary.get("forecast_r2"),
            },
        }

        with open(self.output_dir / "backtest_summary.json", "w", encoding="utf-8") as file:
            json.dump(report, file, indent=4)

    # ===============================================================
    # FIGURES (9 Publication-Grade Visualizations)
    # ===============================================================

    def _generate_figures(self, result: RollingBacktestResult, monthly_df: pd.DataFrame):
        dispatch = result.dispatch_history.copy()
        degradation = result.degradation_history.copy()

        if "timestamp" in dispatch.columns:
            dispatch["timestamp"] = pd.to_datetime(dispatch["timestamp"])

        self._plot_cumulative_revenue(dispatch, degradation)
        self._plot_soh(degradation)
        self._plot_soc(dispatch)
        self._plot_price_dispatch(dispatch)
        self._plot_degradation(degradation)
        self._plot_daily_revenue(dispatch)
        self._plot_utilization_heatmap(dispatch)
        self._plot_monthly_degradation(monthly_df)
        self._plot_forecast_error_overlay(dispatch)

    def _save_plot(self, filename: str):
        plt.tight_layout()
        plt.savefig(self.figure_dir / filename)
        plt.close()

    def _plot_cumulative_revenue(self, dispatch: pd.DataFrame, degradation: pd.DataFrame):
        """
        Plots the 3 economic curves: Gross Revenue, Degradation Cost, and Net Revenue.
        """
        plt.figure(figsize=(11, 5))
        rev_col = "net_revenue_usd" if "net_revenue_usd" in dispatch.columns else "net_revenue_$"

        if "timestamp" in dispatch.columns:
            daily_gross = dispatch.groupby(dispatch["timestamp"].dt.date)[rev_col].sum().values
        else:
            daily_gross = dispatch[rev_col].groupby(dispatch.index // 24).sum().values

        n_days = min(len(daily_gross), len(degradation))
        days = np.arange(1, n_days + 1)

        cum_gross = np.cumsum(daily_gross[:n_days])
        cum_deg = np.cumsum(degradation["degradation_cost_usd"].iloc[:n_days].values)
        cum_net = cum_gross - cum_deg

        plt.plot(days, cum_gross, label="Cumulative Gross Arbitrage Revenue", color="#1f77b4", linewidth=2.0)
        plt.plot(days, cum_deg, label="Cumulative Degradation Cost", color="#d62728", linewidth=1.8, linestyle="--")
        plt.plot(days, cum_net, label="Cumulative Net Revenue (After Wear)", color="#2ca02c", linewidth=2.2)

        plt.fill_between(days, cum_net, cum_gross, color="#d62728", alpha=0.12, label="Degradation Impact Gap")

        plt.title("Cumulative Financial Performance & Degradation Impact", fontsize=12, fontweight="bold")
        plt.xlabel("Simulation Day", fontsize=10)
        plt.ylabel("USD ($)", fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="upper left", framealpha=0.95)

        stats_text = (
            f"Gross: ${cum_gross[-1]:,.2f}\n"
            f"Degradation: -${cum_deg[-1]:,.2f}\n"
            f"Net P&L: ${cum_net[-1]:,.2f}"
        )
        plt.gca().text(
            0.97, 0.05, stats_text,
            transform=plt.gca().transAxes,
            fontsize=9,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cccccc", alpha=0.9),
        )

        self._save_plot("cumulative_revenue.png")

    def _plot_forecast_error_overlay(self, dispatch: pd.DataFrame):
        """
        Plots Forecast vs Actual settlement price with an error band and residual distribution.
        """
        sample_len = min(len(dispatch), 168)
        sample = dispatch.iloc[:sample_len].copy()

        actual_col = next((c for c in ["actual_price", "price", "settlement_price"] if c in sample.columns), None)

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]}
        )

        x = np.arange(sample_len)
        forecast = sample["forecast_price"].values

        if actual_col:
            actual = sample[actual_col].values
            residuals = forecast - actual
            mae = float(np.mean(np.abs(residuals)))
            rmse = float(np.sqrt(np.mean(residuals**2)))

            ax1.plot(x, actual, label="Actual Settlement Price", color="#1f77b4", linewidth=1.5)
            ax1.plot(x, forecast, label="Day-Ahead Forecast", color="#ff7f0e", linestyle="--", linewidth=1.5)
            ax1.fill_between(
                x, forecast - mae, forecast + mae, color="#ff7f0e", alpha=0.2, label=f"±1 MAE Confidence Band (${mae:.2f})"
            )

            ax2.bar(x, residuals, color=np.where(residuals >= 0, "#e1974c", "#7293cb"), width=0.8, alpha=0.8)
            ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)
            ax2.set_ylabel("Error ($/MWh)", fontsize=9)
            ax2.set_xlabel("Hour of Week", fontsize=10)
            ax2.grid(True, linestyle="--", alpha=0.5)

            title_text = f"Price Forecast Accuracy & Error Residuals (MAE: ${mae:.2f}, RMSE: ${rmse:.2f})"
        else:
            rolling_dispersion = pd.Series(forecast).rolling(24, min_periods=1).std().fillna(5.0).values
            ax1.plot(x, forecast, label="Day-Ahead Forecast", color="#ff7f0e", linewidth=1.5)
            ax1.fill_between(
                x,
                forecast - 1.96 * rolling_dispersion,
                forecast + 1.96 * rolling_dispersion,
                color="#ff7f0e",
                alpha=0.2,
                label="95% Look-Ahead Dispersion Band",
            )

            ax2.plot(x, rolling_dispersion, color="#7293cb", linewidth=1.2)
            ax2.set_ylabel("Std Dev ($)", fontsize=9)
            ax2.set_xlabel("Hour of Week", fontsize=10)
            ax2.grid(True, linestyle="--", alpha=0.5)
            title_text = "Price Forecast Look-Ahead Profile with 95% Dispersion Band"

        ax1.set_title(title_text, fontsize=11, fontweight="bold")
        ax1.set_ylabel("Price ($/MWh)", fontsize=10)
        ax1.legend(loc="upper right", framealpha=0.95)
        ax1.grid(True, linestyle="--", alpha=0.5)

        self._save_plot("forecast_error_overlay.png")

    def _plot_soh(self, degradation: pd.DataFrame):
        plt.figure(figsize=(11, 4))
        soh_col = "remaining_soh" if "remaining_soh" in degradation.columns else "soh_end"

        plt.plot(degradation[soh_col].values, color="#d62728", linewidth=1.5)
        plt.title("Battery State of Health (SOH) Trajectory")
        plt.xlabel("Rolling Window (Day)")
        plt.ylabel("SOH Fraction")
        plt.ylim(0.95, 1.005)
        plt.grid(True, linestyle="--", alpha=0.6)
        self._save_plot("soh_curve.png")

    def _plot_soc(self, dispatch: pd.DataFrame):
        plt.figure(figsize=(11, 4))
        sample_window = min(len(dispatch), 168)
        plt.plot(dispatch["soc_fraction"].iloc[:sample_window].values, color="#2ca02c", linewidth=1.2)
        plt.title(f"State of Charge (SOC) Profile (First {sample_window} Hours)")
        plt.xlabel("Hour of Week")
        plt.ylabel("SOC Fraction")
        plt.ylim(-0.05, 1.05)
        plt.grid(True, linestyle="--", alpha=0.6)
        self._save_plot("soc_profile.png")

    def _plot_price_dispatch(self, dispatch: pd.DataFrame):
        sample_len = min(len(dispatch), 168)
        sample = dispatch.iloc[:sample_len]

        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax2 = ax1.twinx()

        line1 = ax1.plot(sample["forecast_price"].values, color="#1f77b4", label="Forecast Price ($/MWh)", linewidth=1.2)
        ax1.set_xlabel("Hour")
        ax1.set_ylabel("Price ($/MWh)", color="#1f77b4")
        ax1.tick_params(axis="y", labelcolor="#1f77b4")

        net_power = sample["discharge_power_mw"].values - sample["charge_power_mw"].values
        line2 = ax2.plot(net_power, color="#ff7f0e", label="Net Dispatch Power (MW)", linewidth=1.2, linestyle="--")
        ax2.set_ylabel("Dispatch Power (MW)", color="#ff7f0e")
        ax2.tick_params(axis="y", labelcolor="#ff7f0e")
        ax2.axhline(0, color="gray", linestyle=":", linewidth=0.8)

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper right")
        plt.title("Sample Co-Optimized Dispatch vs Market Price (168-Hour Window)")
        ax1.grid(True, linestyle="--", alpha=0.5)
        self._save_plot("price_dispatch.png")

    def _plot_degradation(self, degradation: pd.DataFrame):
        plt.figure(figsize=(10, 4))
        cum_cal = degradation["calendar_loss"].cumsum() * 100.0
        cum_cyc = degradation["cycle_loss"].cumsum() * 100.0

        plt.stackplot(
            range(len(degradation)),
            cum_cal,
            cum_cyc,
            labels=["Calendar Fade (%)", "Cycle Fade (%)"],
            colors=["#aec7e8", "#ffbb78"],
            alpha=0.85,
        )
        plt.legend(loc="upper left")
        plt.title("Cumulative Capacity Degradation Breakdown (% Fade)")
        plt.xlabel("Rolling Window (Day)")
        plt.ylabel("Total Capacity Loss (%)")
        plt.grid(True, linestyle="--", alpha=0.5)
        self._save_plot("degradation_breakdown.png")

    def _plot_daily_revenue(self, dispatch: pd.DataFrame):
        if "timestamp" not in dispatch.columns:
            return

        rev_col = "net_revenue_usd" if "net_revenue_usd" in dispatch.columns else "net_revenue_$"
        daily = dispatch.groupby(dispatch["timestamp"].dt.date)[rev_col].sum()

        plt.figure(figsize=(12, 4))
        plt.bar(range(len(daily)), daily.values, color="#2ca02c", width=0.8, alpha=0.8)
        plt.axhline(0, color="black", linewidth=0.8)
        plt.title("Daily Arbitrage Gross Revenue")
        plt.xlabel("Simulation Day")
        plt.ylabel("Daily Gross Revenue ($)")
        plt.grid(True, linestyle="--", alpha=0.5)
        self._save_plot("daily_revenue.png")

    def _plot_utilization_heatmap(self, dispatch: pd.DataFrame):
        df = dispatch.copy()
        if "timestamp" not in df.columns:
            df["hour"] = df.index % 24
            df["month"] = (df.index // (24 * 30)) + 1
        else:
            df["hour"] = df["timestamp"].dt.hour
            df["month"] = df["timestamp"].dt.month

        net_power = df["discharge_power_mw"] - df["charge_power_mw"]
        df["net_power"] = net_power

        pivot = df.pivot_table(index="hour", columns="month", values="net_power", aggfunc="mean").fillna(0.0)

        plt.figure(figsize=(10, 5))
        c = plt.imshow(pivot.values, cmap="coolwarm", aspect="auto", origin="lower")
        plt.colorbar(c, label="Mean Net Power (MW) [Discharge > 0, Charge < 0]")
        plt.title("BESS Dispatch Intensity Heatmap (Hour of Day vs Month)")
        plt.xlabel("Month of Year")
        plt.ylabel("Hour of Day (0-23)")
        plt.xticks(range(len(pivot.columns)), pivot.columns)
        plt.yticks(range(0, 24, 2))
        self._save_plot("utilization_heatmap.png")

    def _plot_monthly_degradation(self, monthly_df: pd.DataFrame):
        if monthly_df.empty:
            return

        fig, ax1 = plt.subplots(figsize=(11, 4.5))
        ax2 = ax1.twinx()

        x = np.arange(len(monthly_df))
        width = 0.55

        ax1.bar(x, monthly_df["calendar_fade_pct"], width, label="Calendar Fade (%)", color="#7293cb", alpha=0.9)
        ax1.bar(x, monthly_df["cycle_fade_pct"], width, bottom=monthly_df["calendar_fade_pct"], label="Cycle Fade (%)", color="#e1974c", alpha=0.9)

        ax1.set_xlabel("Simulation Month")
        ax1.set_ylabel("Monthly Capacity Fade (%)", color="#333333")
        ax1.set_xticks(x)
        ax1.set_xticklabels(monthly_df["year_month"], rotation=45)
        ax1.legend(loc="upper left")

        ax2.plot(x, monthly_df["end_soh"], color="#d62728", marker="o", linewidth=1.5, label="End SOH")
        ax2.set_ylabel("State of Health (SOH)", color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728")
        ax2.set_ylim(0.97, 1.002)

        plt.title("Monthly Degradation Dynamics & SOH Trajectory")
        ax1.grid(True, linestyle="--", alpha=0.5)
        self._save_plot("monthly_degradation_chart.png")