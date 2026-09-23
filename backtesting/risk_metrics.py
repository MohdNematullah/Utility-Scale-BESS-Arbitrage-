"""
backtesting/risk_metrics.py
===========================

Financial Risk & Downside Volatility Engine (Part 9.3)



Capabilities:
1. Daily Return and P&L Time-Series Formulation
2. Volatility Analytics:
   - Daily and Annualized Return Volatility
   - Return Skewness and Excess Kurtosis
   - Semi-Deviation and Downside Volatility (below target hurdle)
3. Institutional Risk-Adjusted Return Metrics:
   - Annualized Sharpe Ratio
   - Sortino Ratio (Downside Penalized)
   - Calmar Ratio (Return to Max Drawdown)
   - Omega Ratio (Gains to Losses Probability Mass)
4. Peak-to-Trough Drawdown Dynamics:
   - Cumulative Equity and High-Water Mark Trajectory
   - Maximum Drawdown ($ and % of Peak Capital)
   - Longest Drawdown Duration (Days Underwater)
   - Average Drawdown Depth and Recovery Velocity
5. Tail Risk and Capital Adequacy (Basel / Extreme Value):
   - Historical Value at Risk (VaR 95%, VaR 99%)
   - Parametric Gaussian Value at Risk (VaR 95%, VaR 99%)
   - Conditional Value at Risk / Expected Shortfall (CVaR 95%, CVaR 99%)
6. Publication Diagnostic Visualizations:
   - Underwater Drawdown Equity Curve
   - P&L Tail Risk Distribution with VaR / CVaR Thresholds
   - Rolling 30-Day Sharpe Ratio and Annualized Volatility Profile
   - Monthly Risk-Return Trade-Off Scatter
   - Drawdown Episodes and Recovery Duration Profile
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
class RiskMetricsSummary:
    annualized_return_usd: float
    daily_volatility_usd: float
    annualized_volatility_usd: float
    downside_deviation_usd: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    max_drawdown_usd: float
    max_drawdown_pct: float
    max_drawdown_duration_days: int
    historical_var_95_usd: float
    historical_var_99_usd: float
    parametric_var_95_usd: float
    parametric_var_99_usd: float
    cvar_95_usd: float
    cvar_99_usd: float
    skewness: float
    excess_kurtosis: float
    profitable_days_pct: float
    total_evaluated_days: int


@dataclass(slots=True)
class RiskArtifacts:
    summary_csv: Path
    daily_risk_csv: Path
    summary_json: Path
    figure_underwater: Path
    figure_tail_risk: Path
    figure_rolling_risk: Path
    figure_monthly_scatter: Path
    figure_drawdown_profile: Path
    figures_directory: Path


# ============================================================================
# Core Risk Metrics Engine
# ============================================================================

class RiskMetricsEngine:
    """
    Evaluates financial return distributions, drawdown mechanics, and tail risk.
    """

    def __init__(self, output_directory: Path | str = "backtesting/results/risk_metrics"):
        self.output_dir = Path(output_directory)
        self.figure_dir = self.output_dir / "figures"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figure_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------------
    # Core Mathematical & Financial Risk Formulation
    # ------------------------------------------------------------------------

    def evaluate(
        self,
        dispatch_df: pd.DataFrame,
        degradation_df: pd.DataFrame | None = None,
        risk_free_rate_pct: float = 4.0,
        fixed_facility_capex_usd: float = 35000000.0,
    ) -> tuple[RiskMetricsSummary, pd.DataFrame]:
        df = dispatch_df.copy()

        # 1. Resolve Daily Net Revenue / Cash Flow
        rev_col = next((c for c in ["net_revenue_usd", "net_revenue_$", "gross_revenue_usd"] if c in df.columns), None)
        if rev_col is None:
            raise KeyError("Dispatch DataFrame must contain a valid revenue column.")

        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"])
            df["_date"] = ts.dt.date
            df["_month"] = ts.dt.strftime("%Y-%m")
        else:
            df["_date"] = df.index // 24
            df["_month"] = (df.index // (24 * 30)).astype(str)

        daily_gross = df.groupby("_date")[rev_col].sum()

        if degradation_df is not None and "degradation_cost_usd" in degradation_df.columns:
            n_days = min(len(daily_gross), len(degradation_df))
            daily_deg = degradation_df["degradation_cost_usd"].iloc[:n_days].values
            daily_net = daily_gross.iloc[:n_days].values - daily_deg
            date_index = daily_gross.index[:n_days]
        else:
            daily_net = daily_gross.values
            date_index = daily_gross.index

        daily_pnl = np.asarray(daily_net, dtype=float)
        n_days = len(daily_pnl)
        if n_days < 2:
            raise ValueError("Risk evaluation requires at least 2 simulation days.")

        # 2. Return Analytics & Volatility
        mean_daily_pnl = float(np.mean(daily_pnl))
        ann_return = mean_daily_pnl * 365.0
        daily_vol = float(np.std(daily_pnl, ddof=1))
        ann_vol = daily_vol * np.sqrt(365.0)

        # 3. Downside Deviation & Sortino Ratio
        target_hurdle = 0.0
        negative_deviations = np.minimum(0.0, daily_pnl - target_hurdle)
        downside_dev = float(np.sqrt(np.mean(negative_deviations ** 2)))

        rf_daily = (risk_free_rate_pct / 100.0) / 365.0 * (fixed_facility_capex_usd / 365.0)
        excess_daily_return = mean_daily_pnl - rf_daily

        sharpe = float((excess_daily_return / daily_vol) * np.sqrt(365.0)) if daily_vol > 1e-6 else 99.99
        sortino = float((excess_daily_return / downside_dev) * np.sqrt(365.0)) if downside_dev > 1e-6 else 99.99

        # 4. Omega Ratio
        positive_pnl_mass = float(np.sum(np.maximum(0.0, daily_pnl)))
        negative_pnl_mass = float(np.sum(np.maximum(0.0, -daily_pnl)))
        omega = (positive_pnl_mass / negative_pnl_mass) if negative_pnl_mass > 1e-6 else 99.99

        # 5. Drawdown Mechanics (Cumulative Equity & Water Mark)
        cum_equity = np.cumsum(daily_pnl)
        high_water_mark = np.maximum.accumulate(cum_equity)
        drawdown_usd = high_water_mark - cum_equity

        max_dd_usd = float(np.max(drawdown_usd))
        max_dd_pct = float((max_dd_usd / fixed_facility_capex_usd) * 100.0)
        calmar = (ann_return / max_dd_usd) if max_dd_usd > 1e-6 else 99.99

        # Compute Duration of Drawdowns (Underwater periods)
        is_underwater = drawdown_usd > 1e-4
        max_dd_duration = 0
        cur_duration = 0
        for uw in is_underwater:
            if uw:
                cur_duration += 1
                max_dd_duration = max(max_dd_duration, cur_duration)
            else:
                cur_duration = 0

        # 6. Tail Risk: Value at Risk (VaR) & Expected Shortfall (CVaR)
        losses = -daily_pnl

        hist_var_95 = float(np.percentile(losses, 95.0))
        hist_var_99 = float(np.percentile(losses, 99.0))

        param_var_95 = float(-mean_daily_pnl + 1.64485 * daily_vol)
        param_var_99 = float(-mean_daily_pnl + 2.32635 * daily_vol)

        tail_losses_95 = losses[losses >= hist_var_95]
        cvar_95 = float(np.mean(tail_losses_95)) if len(tail_losses_95) > 0 else hist_var_95

        tail_losses_99 = losses[losses >= hist_var_99]
        cvar_99 = float(np.mean(tail_losses_99)) if len(tail_losses_99) > 0 else hist_var_99

        # 7. Distribution Moments
        centered = daily_pnl - mean_daily_pnl
        skewness = float((np.sum(centered ** 3) / n_days) / (daily_vol ** 3)) if daily_vol > 1e-6 else 0.0
        excess_kurt = float((np.sum(centered ** 4) / n_days) / (daily_vol ** 4) - 3.0) if daily_vol > 1e-6 else 0.0
        pct_profitable = float(np.mean(daily_pnl > 0.0) * 100.0)

        summary = RiskMetricsSummary(
            annualized_return_usd=round(ann_return, 2),
            daily_volatility_usd=round(daily_vol, 2),
            annualized_volatility_usd=round(ann_vol, 2),
            downside_deviation_usd=round(downside_dev, 2),
            sharpe_ratio=round(min(sharpe, 99.99), 3),
            sortino_ratio=round(min(sortino, 99.99), 3),
            calmar_ratio=round(min(calmar, 99.99), 3),
            omega_ratio=round(min(omega, 99.99), 3),
            max_drawdown_usd=round(max_dd_usd, 2),
            max_drawdown_pct=round(max_dd_pct, 3),
            max_drawdown_duration_days=int(max_dd_duration),
            historical_var_95_usd=round(hist_var_95, 2),
            historical_var_99_usd=round(hist_var_99, 2),
            parametric_var_95_usd=round(param_var_95, 2),
            parametric_var_99_usd=round(param_var_99, 2),
            cvar_95_usd=round(cvar_95, 2),
            cvar_99_usd=round(cvar_99, 2),
            skewness=round(skewness, 3),
            excess_kurtosis=round(excess_kurt, 3),
            profitable_days_pct=round(pct_profitable, 2),
            total_evaluated_days=n_days,
        )

        daily_df = pd.DataFrame({
            "date": [str(d) for d in date_index],
            "daily_net_pnl_usd": np.round(daily_pnl, 2),
            "cumulative_pnl_usd": np.round(cum_equity, 2),
            "high_water_mark_usd": np.round(high_water_mark, 2),
            "drawdown_usd": np.round(drawdown_usd, 2),
            "is_drawdown": is_underwater,
        })

        return summary, daily_df

    # ------------------------------------------------------------------------
    # Publication Visualizations
    # ------------------------------------------------------------------------

    def plot_underwater_curve(
        self,
        daily_df: pd.DataFrame,
        summary: RiskMetricsSummary,
        filename: str = "cumulative_drawdown_underwater.png",
    ) -> Path:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
        )

        x = np.arange(len(daily_df))

        # Upper panel: Equity vs High Water Mark
        ax1.plot(x, daily_df["cumulative_pnl_usd"] / 1000.0, color="#1f77b4", linewidth=2.0, label="Cumulative Net Profit")
        ax1.plot(x, daily_df["high_water_mark_usd"] / 1000.0, color="#2ca02c", linestyle="--", linewidth=1.5, label="High-Water Mark")
        ax1.set_ylabel("Cumulative P&L ($k USD)", fontsize=10)
        ax1.set_title("BESS Equity Growth & High-Water Mark Trajectory", fontsize=11, fontweight="bold")
        ax1.grid(True, linestyle="--", alpha=0.5)

        handles1, labels1 = ax1.get_legend_handles_labels()
        if labels1:
            ax1.legend(handles1, labels1, loc="upper left", framealpha=0.95)

        # Lower panel: Underwater Drawdown
        ax2.fill_between(x, -daily_df["drawdown_usd"] / 1000.0, 0, color="#d62728", alpha=0.4, label="Drawdown Depth")
        ax2.plot(x, -daily_df["drawdown_usd"] / 1000.0, color="#d62728", linewidth=1.2)
        ax2.set_ylabel("Drawdown ($k)", fontsize=9)
        ax2.set_xlabel("Simulation Day", fontsize=10)
        ax2.set_title(f"Underwater Drawdown Profile (Max Drawdown: -${summary.max_drawdown_usd:,.2f})", fontsize=10, fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.5)

        handles2, labels2 = ax2.get_legend_handles_labels()
        if labels2:
            ax2.legend(handles2, labels2, loc="lower left", framealpha=0.95)

        out_path = self.figure_dir / filename
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_tail_risk_distribution(
        self,
        daily_df: pd.DataFrame,
        summary: RiskMetricsSummary,
        filename: str = "tail_risk_var_cvar_dist.png",
    ) -> Path:
        fig, ax = plt.subplots(figsize=(9, 5))

        pnl = daily_df["daily_net_pnl_usd"].values
        ax.hist(pnl, bins=40, density=True, alpha=0.6, color="#7293cb", edgecolor="white", label="Daily P&L Distribution")

        ax.axvline(-summary.historical_var_95_usd, color="#ff7f0e", linestyle="--", linewidth=1.8, label=f"95% VaR (-${summary.historical_var_95_usd:,.0f})")
        ax.axvline(-summary.cvar_95_usd, color="#d62728", linestyle="-", linewidth=2.0, label=f"95% CVaR (-${summary.cvar_95_usd:,.0f})")
        ax.axvline(0.0, color="black", linestyle=":", linewidth=1.0)

        ax.set_title("Daily Net Profit Distribution & Extreme Tail Risk Thresholds", fontsize=11, fontweight="bold")
        ax.set_xlabel("Daily Net Arbitrage P&L ($ USD)", fontsize=10)
        ax.set_ylabel("Probability Density", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)

        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(handles, labels, loc="upper left", framealpha=0.95)

        out_path = self.figure_dir / filename
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_rolling_risk_ratios(
        self,
        daily_df: pd.DataFrame,
        window: int = 30,
        filename: str = "rolling_risk_ratios.png",
    ) -> Path:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)

        pnl = daily_df["daily_net_pnl_usd"]
        rolling_mean = pnl.rolling(window, min_periods=5).mean()
        rolling_std = pnl.rolling(window, min_periods=5).std().replace(0, np.nan)
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(365.0)
        rolling_ann_vol = rolling_std * np.sqrt(365.0)

        x = np.arange(len(daily_df))

        ax1.plot(x, rolling_sharpe, color="#2ca02c", linewidth=1.8, label=f"Rolling {window}-Day Sharpe Ratio")
        ax1.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax1.set_ylabel("Sharpe Ratio", fontsize=9)
        ax1.set_title(f"Rolling {window}-Day Risk-Adjusted Stability & Volatility", fontsize=11, fontweight="bold")
        ax1.grid(True, linestyle="--", alpha=0.5)

        handles1, labels1 = ax1.get_legend_handles_labels()
        if labels1:
            ax1.legend(handles1, labels1, loc="upper left", framealpha=0.95)

        ax2.plot(x, rolling_ann_vol / 1000.0, color="#d62728", linewidth=1.8, label="Rolling Annualized Volatility ($k)")
        ax2.set_ylabel("Volatility ($k)", fontsize=9)
        ax2.set_xlabel("Simulation Day", fontsize=10)
        ax2.grid(True, linestyle="--", alpha=0.5)

        handles2, labels2 = ax2.get_legend_handles_labels()
        if labels2:
            ax2.legend(handles2, labels2, loc="upper left", framealpha=0.95)

        out_path = self.figure_dir / filename
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_monthly_risk_scatter(
        self,
        dispatch_df: pd.DataFrame,
        filename: str = "monthly_risk_return_scatter.png",
    ) -> Path:
        fig, ax = plt.subplots(figsize=(8, 5))

        df = dispatch_df.copy()
        if "timestamp" in df.columns:
            df["_month"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m")
        else:
            df["_month"] = (df.index // (24 * 30)).astype(str)

        rev_col = next((c for c in ["net_revenue_usd", "net_revenue_$"] if c in df.columns), df.columns[0])

        monthly_stats = df.groupby("_month")[rev_col].agg(["sum", "std"]).fillna(0.0)
        monthly_stats["annualized_vol"] = monthly_stats["std"] * np.sqrt(24 * 30)

        scatter = ax.scatter(
            monthly_stats["annualized_vol"] / 1000.0,
            monthly_stats["sum"] / 1000.0,
            color="#1f77b4",
            s=80,
            edgecolors="black",
            alpha=0.85,
            label="Calendar Months",
        )

        for month_name, row in monthly_stats.iterrows():
            ax.annotate(
                str(month_name),
                (row["annualized_vol"] / 1000.0, row["sum"] / 1000.0),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

        ax.set_title("Monthly Return vs Volatility Efficiency Map", fontsize=11, fontweight="bold")
        ax.set_xlabel("Monthly Volatility ($k USD)", fontsize=10)
        ax.set_ylabel("Monthly Net Revenue ($k USD)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)

        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(handles, labels, loc="upper left", framealpha=0.95)

        out_path = self.figure_dir / filename
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_drawdown_duration_profile(
        self,
        daily_df: pd.DataFrame,
        filename: str = "drawdown_duration_profile.png",
    ) -> Path:
        fig, ax = plt.subplots(figsize=(9, 4.5))

        dd = daily_df["drawdown_usd"].values
        sorted_dd = np.sort(dd)[::-1] / 1000.0
        x_pct = (np.arange(len(sorted_dd)) / max(len(sorted_dd), 1)) * 100.0

        ax.plot(x_pct, sorted_dd, color="#d62728", linewidth=2.0, label="Capital Drawdown Exceedance")
        ax.fill_between(x_pct, sorted_dd, 0, color="#d62728", alpha=0.2)

        ax.set_title("Capital Drawdown Duration & Depth Exceedance Curve", fontsize=11, fontweight="bold")
        ax.set_xlabel("Percentage of Operational Days (%)", fontsize=10)
        ax.set_ylabel("Capital Below Peak ($k USD)", fontsize=10)
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
        risk_free_rate_pct: float = 4.0,
        fixed_facility_capex_usd: float = 35000000.0,
    ) -> tuple[RiskMetricsSummary, RiskArtifacts]:
        summary, daily_df = self.evaluate(
            dispatch_df=dispatch_df,
            degradation_df=degradation_df,
            risk_free_rate_pct=risk_free_rate_pct,
            fixed_facility_capex_usd=fixed_facility_capex_usd,
        )

        # 1. Export CSVs
        summary_csv = self.output_dir / "risk_metrics_summary.csv"
        df_sum = pd.DataFrame([asdict(summary)]).T.reset_index()
        df_sum.columns = ["metric", "value"]
        df_sum.to_csv(summary_csv, index=False)

        daily_risk_csv = self.output_dir / "daily_returns_and_drawdowns.csv"
        daily_df.to_csv(daily_risk_csv, index=False)

        # 2. Export JSON
        summary_json = self.output_dir / "risk_metrics.json"
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(asdict(summary), f, indent=4)

        # 3. Export Visualizations
        fig_uw = self.plot_underwater_curve(daily_df, summary)
        fig_tr = self.plot_tail_risk_distribution(daily_df, summary)
        fig_rr = self.plot_rolling_risk_ratios(daily_df)
        fig_ms = self.plot_monthly_risk_scatter(dispatch_df)
        fig_dp = self.plot_drawdown_duration_profile(daily_df)

        artifacts = RiskArtifacts(
            summary_csv=summary_csv,
            daily_risk_csv=daily_risk_csv,
            summary_json=summary_json,
            figure_underwater=fig_uw,
            figure_tail_risk=fig_tr,
            figure_rolling_risk=fig_rr,
            figure_monthly_scatter=fig_ms,
            figure_drawdown_profile=fig_dp,
            figures_directory=self.figure_dir,
        )

        return summary, artifacts