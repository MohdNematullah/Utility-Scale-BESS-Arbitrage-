"""
visualization/risk_figures.py
=============================

Publication-Grade Risk & Downside Volatility Visualizations (Part 10.6)



Generates the 5 core financial risk figures for Chapter 8:
1. Figure 10.6.1 — Daily Net Profit Distribution (Empirical histogram, normal density fit, win rate %)
2. Figure 10.6.2 — Drawdown & High-Water Mark Trajectory (Cumulative equity curve vs. underwater depth)
3. Figure 10.6.3 — Extreme Tail Risk & Value at Risk (Loss distribution with 95%/99% VaR and CVaR cutoffs)
4. Figure 10.6.4 — Rolling Risk Dynamics (30-day rolling annualized volatility vs. rolling Sharpe ratio)
5. Figure 10.6.5 — Return Distribution Moments (Kurtosis, skewness, Gaussian reference parity)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization.figure_style import (
    COLOR_PALETTE,
    format_axes,
    get_figure_dimensions,
    save_publication_figure,
    set_ieee_style,
)


@dataclass(slots=True)
class RiskFigureArtifacts:
    daily_profit_dist: dict[str, Path]
    drawdown_curve: dict[str, Path]
    var_cvar_tail: dict[str, Path]
    rolling_volatility: dict[str, Path]
    return_distribution: dict[str, Path]
    output_directory: Path


class RiskFigureGenerator:
    """
    Produces publication-grade multi-format vector graphics
    for financial risk assessment, drawdown profiles, and extreme tail risk metrics.
    """

    def __init__(
        self,
        output_directory: Path | str = "visualization/figures/risk",
        formats: Sequence[str] = ("png", "pdf", "svg"),
        dpi: int = 600,
    ):
        self.output_dir = Path(output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.formats = tuple(formats)
        self.dpi = dpi
        set_ieee_style()

    # ------------------------------------------------------------------------
    # Helper: Resolve Daily P&L Array
    # ------------------------------------------------------------------------
    @staticmethod
    def _extract_daily_pnl(
        dispatch_df: pd.DataFrame,
        degradation_df: pd.DataFrame | None = None,
    ) -> np.ndarray:
        rev_col = next(
            (c for c in ["net_revenue_usd", "net_revenue_$", "gross_revenue_usd"] if c in dispatch_df.columns),
            dispatch_df.columns[0],
        )
        revs = np.asarray(dispatch_df[rev_col], dtype=float)
        n_days = len(revs) // 24
        if n_days == 0:
            return np.array([float(np.sum(revs))])

        daily_gross = revs[: n_days * 24].reshape(n_days, 24).sum(axis=1)

        if degradation_df is not None and "degradation_cost_usd" in degradation_df.columns:
            deg = np.asarray(degradation_df["degradation_cost_usd"], dtype=float)
            m = min(n_days, len(deg))
            daily_net = daily_gross[:m] - deg[:m]
        else:
            daily_net = daily_gross * 0.7697

        return np.asarray(daily_net, dtype=float)

    # ------------------------------------------------------------------------
    # Figure 10.6.1 — Daily Profit Distribution & Win Rate
    # ------------------------------------------------------------------------
    def plot_daily_profit_distribution(
        self,
        daily_pnl: np.ndarray,
        filename_stem: str = "Figure_10_6_1_Daily_Profit_Distribution",
    ) -> dict[str, Path]:
        dims = get_figure_dimensions("thesis_full")
        fig, ax = plt.subplots(figsize=dims)

        pnl = np.asarray(daily_pnl, dtype=float)
        mu = float(np.mean(pnl))
        med = float(np.median(pnl))
        sigma = float(np.std(pnl, ddof=1))
        win_rate = float(np.mean(pnl > 0.0)) * 100.0

        # Empirical Distribution
        ax.hist(pnl, bins=40, density=True, color="#7293cb", alpha=0.65, edgecolor="white", label="Empirical Daily P&L")

        # Gaussian Reference Fit
        if sigma > 0:
            x_seq = np.linspace(np.min(pnl), np.max(pnl), 250)
            pdf = (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(-0.5 * ((x_seq - mu) / sigma) ** 2)
            ax.plot(x_seq, pdf, color=COLOR_PALETTE.revenue, linestyle="--", linewidth=1.8, label=rf"Normal Density ($\mu=\${mu:,.0f}, \sigma=\${sigma:,.0f}$)")

        ax.axvline(0.0, color=COLOR_PALETTE.actual, linestyle="-", linewidth=1.0, label="Breakeven ($0/day)")
        ax.axvline(mu, color=COLOR_PALETTE.charge, linestyle=":", linewidth=1.8, label=rf"Mean Yield (\${mu:,.0f}/day)")
        ax.axvline(med, color=COLOR_PALETTE.amber, linestyle="--", linewidth=1.6, label=rf"Median Yield (\${med:,.0f}/day)")

        format_axes(
            ax,
            title=rf"Daily Arbitrage Net Cash Flow Distribution (Win Rate: {win_rate:.1f}\%)",
            xlabel="Daily Net Profit ($ USD)",
            ylabel="Probability Density",
        )
        ax.legend(loc="upper left", framealpha=0.95)

        callout = (
            rf"$\mathrm{{Profitable\ Days}}: {win_rate:.1f}\%$" "\n"
            rf"$\mu\ \mathrm{{(Mean)}}: \${mu:,.2f}$" "\n"
            rf"$\sigma\ \mathrm{{(Daily\ Vol)}}: \${sigma:,.2f}$"
        )
        ax.text(
            0.97, 0.94, callout,
            transform=ax.transAxes,
            fontsize=8.5,
            va="top",
            ha="right",
            bbox=dict(boxstyle="square,pad=0.4", facecolor="#FAFAFA", alpha=0.9, edgecolor="#CCCCCC"),
        )

        saved = save_publication_figure(fig, self.output_dir / filename_stem, formats=self.formats, dpi=self.dpi)
        plt.close(fig)
        return saved

    # ------------------------------------------------------------------------
    # Figure 10.6.2 — Drawdown & High-Water Mark Curve
    # ------------------------------------------------------------------------
    def plot_drawdown_curve(
        self,
        daily_pnl: np.ndarray,
        filename_stem: str = "Figure_10_6_2_Drawdown_Curve",
    ) -> dict[str, Path]:
        dims = get_figure_dimensions("thesis_full_tall")
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=dims, sharex=True, gridspec_kw={"height_ratios": [1.8, 1.0]}
        )

        pnl = np.asarray(daily_pnl, dtype=float)
        x_days = np.arange(1, len(pnl) + 1)
        cum_equity = np.cumsum(pnl) / 1000.0
        hwm = np.maximum.accumulate(cum_equity)
        drawdown = hwm - cum_equity

        max_dd_k = float(np.max(drawdown))
        max_idx = int(np.argmax(drawdown))

        # Upper: Equity Growth vs High-Water Mark
        ax1.plot(x_days, cum_equity, color=COLOR_PALETTE.forecast, linewidth=2.0, label="Cumulative Net Profit ($k)")
        ax1.plot(x_days, hwm, color=COLOR_PALETTE.charge, linestyle="--", linewidth=1.5, label="High-Water Mark ($k)")
        format_axes(ax1, title="Cumulative Capital Growth & Peak High-Water Mark", ylabel="Cumulative P&L ($k USD)")
        ax1.legend(loc="upper left", framealpha=0.95)

        # Lower: Underwater Drawdown Profile
        ax2.fill_between(x_days, -drawdown, 0, color=COLOR_PALETTE.loss, alpha=0.45, label="Underwater Drawdown ($k)")
        ax2.plot(x_days, -drawdown, color=COLOR_PALETTE.loss, linewidth=1.2)
        ax2.axhline(0, color=COLOR_PALETTE.actual, linewidth=0.8)

        # Mark Max Drawdown Point
        if max_dd_k > 0:
            ax2.scatter([x_days[max_idx]], [-max_dd_k], color=COLOR_PALETTE.loss, s=45, zorder=5)
            ax2.annotate(
                rf"Max DD: -\${max_dd_k*1000:,.0f}",
                xy=(x_days[max_idx], -max_dd_k),
                xytext=(x_days[max_idx] + 8, -max_dd_k * 0.75),
                fontsize=8.5,
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=COLOR_PALETTE.loss, lw=1.2),
            )

        format_axes(
            ax2,
            title=rf"Capital Drawdown Depth (Maximum Loss Below Peak: \${max_dd_k*1000:,.2f})",
            xlabel="Operational Horizon (Days)",
            ylabel="Drawdown ($k USD)",
        )
        ax2.legend(loc="lower left", framealpha=0.95)

        saved = save_publication_figure(fig, self.output_dir / filename_stem, formats=self.formats, dpi=self.dpi)
        plt.close(fig)
        return saved

    # ------------------------------------------------------------------------
    # Figure 10.6.3 — VaR / CVaR Extreme Tail Risk
    # ------------------------------------------------------------------------
    def plot_var_cvar_tail(
        self,
        daily_pnl: np.ndarray,
        filename_stem: str = "Figure_10_6_3_VaR_CVaR_Tail",
    ) -> dict[str, Path]:
        dims = get_figure_dimensions("thesis_full")
        fig, ax = plt.subplots(figsize=dims)

        pnl = np.asarray(daily_pnl, dtype=float)
        losses = -pnl  # Loss convention: positive means losing money

        var_95 = float(np.percentile(losses, 95.0))
        var_99 = float(np.percentile(losses, 99.0))
        tail_95 = losses[losses >= var_95]
        tail_99 = losses[losses >= var_99]
        cvar_95 = float(np.mean(tail_95)) if len(tail_95) > 0 else var_95
        cvar_99 = float(np.mean(tail_99)) if len(tail_99) > 0 else var_99

        # Plot on P&L scale (Losses converted back to P&L for intuitive presentation)
        counts, bins, patches = ax.hist(
            pnl, bins=45, density=True, color="#7293cb", alpha=0.55, edgecolor="white", label="Daily Net Return Distribution"
        )

        # Highlight Tail Region (< -VaR_95)
        cutoff_val = -var_95
        for patch, left_edge in zip(patches, bins[:-1]):
            if left_edge <= cutoff_val:
                patch.set_facecolor(COLOR_PALETTE.loss)
                patch.set_alpha(0.7)

        # Threshold Markers
        ax.axvline(0.0, color=COLOR_PALETTE.actual, linestyle="-", linewidth=0.9)
        ax.axvline(-var_95, color=COLOR_PALETTE.discharge, linestyle="--", linewidth=1.8, label=rf"95\% VaR: -\${var_95:,.0f}/day")
        ax.axvline(-cvar_95, color=COLOR_PALETTE.loss, linestyle="-", linewidth=2.0, label=rf"95\% CVaR (Expected Shortfall): -\${cvar_95:,.0f}/day")
        ax.axvline(-var_99, color=COLOR_PALETTE.soh, linestyle=":", linewidth=1.8, label=rf"99\% VaR: -\${var_99:,.0f}/day")
        ax.axvline(-cvar_99, color="#8B0000", linestyle="-.", linewidth=1.8, label=rf"99\% CVaR: -\${cvar_99:,.0f}/day")

        format_axes(
            ax,
            title="Extreme Downside Risk: Value at Risk (VaR) & Expected Shortfall (CVaR)",
            xlabel="Daily Net Profit ($ USD)",
            ylabel="Probability Density",
        )
        ax.legend(loc="upper left", framealpha=0.95)

        saved = save_publication_figure(fig, self.output_dir / filename_stem, formats=self.formats, dpi=self.dpi)
        plt.close(fig)
        return saved

    # ------------------------------------------------------------------------
    # Figure 10.6.4 — Rolling Volatility & Sharpe Profile
    # ------------------------------------------------------------------------
    def plot_rolling_volatility(
        self,
        daily_pnl: np.ndarray,
        window: int = 30,
        risk_free_rate_pct: float = 4.0,
        capex_usd: float = 35000000.0,
        filename_stem: str = "Figure_10_6_4_Rolling_Volatility",
    ) -> dict[str, Path]:
        dims = get_figure_dimensions("thesis_full")
        fig, ax1 = plt.subplots(figsize=dims)
        ax2 = ax1.twinx()

        pnl_s = pd.Series(np.asarray(daily_pnl, dtype=float))
        n = len(pnl_s)
        x_days = np.arange(1, n + 1)

        roll_mean = pnl_s.rolling(window, min_periods=7).mean().to_numpy()
        roll_std = pnl_s.rolling(window, min_periods=7).std().replace(0, np.nan).to_numpy()

        ann_vol_k = (roll_std * np.sqrt(365.0)) / 1000.0

        # Excess return over risk-free
        daily_rf = (risk_free_rate_pct / 100.0) / 365.0 * (capex_usd / 365.0)
        roll_sharpe = np.where(roll_std > 1e-4, ((roll_mean - daily_rf) / roll_std) * np.sqrt(365.0), 0.0)
        roll_sharpe = np.clip(roll_sharpe, -10.0, 100.0)

        # Primary Axis: Rolling Volatility ($k)
        l1 = ax1.plot(x_days, ann_vol_k, color=COLOR_PALETTE.loss, linewidth=1.9, label=rf"{window}-Day Annualized Volatility ($\sigma_{{\mathrm{{ann}}}}$)")
        format_axes(
            ax1,
            title=rf"Rolling Risk-Adjusted Stability ({window}-Day Annualized Metrics)",
            xlabel="Operational Horizon (Days)",
            ylabel="Annualized Volatility ($k USD/year)",
            hide_top_right=False,
        )

        # Secondary Axis: Rolling Sharpe Ratio
        l2 = ax2.plot(x_days, roll_sharpe, color=COLOR_PALETTE.revenue, linestyle="--", linewidth=1.9, label=rf"{window}-Day Sharpe Ratio ($R_f={risk_free_rate_pct}\%$)")
        ax2.axhline(0, color="gray", linestyle=":", linewidth=0.8)
        ax2.set_ylabel("Annualized Sharpe Ratio", color=COLOR_PALETTE.revenue)
        ax2.spines["top"].set_visible(False)

        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper right", framealpha=0.95)

        saved = save_publication_figure(fig, self.output_dir / filename_stem, formats=self.formats, dpi=self.dpi)
        plt.close(fig)
        return saved

    # ------------------------------------------------------------------------
    # Figure 10.6.5 — Return Distribution Moments (Skewness & Kurtosis)
    # ------------------------------------------------------------------------
    def plot_return_distribution(
        self,
        daily_pnl: np.ndarray,
        filename_stem: str = "Figure_10_6_5_Return_Distribution",
    ) -> dict[str, Path]:
        dims = get_figure_dimensions("thesis_full")
        fig, ax = plt.subplots(figsize=dims)

        pnl = np.asarray(daily_pnl, dtype=float)
        n = len(pnl)
        mu = float(np.mean(pnl))
        sigma = float(np.std(pnl, ddof=1))
        skew = float((np.sum((pnl - mu) ** 3) / n) / (sigma ** 3)) if sigma > 0 else 0.0
        kurt = float((np.sum((pnl - mu) ** 4) / n) / (sigma ** 4)) if sigma > 0 else 3.0
        excess_kurt = kurt - 3.0

        # Normalized standardization
        z_scores = (pnl - mu) / sigma if sigma > 0 else np.zeros_like(pnl)

        ax.hist(z_scores, bins=45, density=True, color="#A6C8E0", alpha=0.6, edgecolor="white", label="Empirical Standardized Returns")

        # Gaussian parity line
        x_seq = np.linspace(-4.0, 4.0, 200)
        norm_pdf = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * (x_seq ** 2))
        ax.plot(x_seq, norm_pdf, color=COLOR_PALETTE.actual, linestyle="--", linewidth=1.8, label="Standard Normal Density N(0, 1)")

        # Zero mean line
        ax.axvline(0.0, color="gray", linestyle=":", linewidth=1.0)

        format_axes(
            ax,
            title="Standardized Arbitrage Return Distribution & Higher Moments",
            xlabel=r"Standardized Return Deviation ($\frac{R - \mu}{\sigma}$)",
            ylabel="Probability Density",
        )
        ax.set_xlim(-4.2, 4.2)
        ax.legend(loc="upper right", framealpha=0.95)

        # Statistical Moments Callout
        callout = (
            rf"$\mathrm{{Skewness}}\ (S): {skew:+.3f}$" "\n"
            rf"$\mathrm{{Kurtosis}}\ (K): {kurt:.3f}$" "\n"
            rf"$\mathrm{{Excess\ Kurtosis}}: {excess_kurt:+.3f}$"
        )
        ax.text(
            0.03, 0.94, callout,
            transform=ax.transAxes,
            fontsize=8.5,
            va="top",
            bbox=dict(boxstyle="square,pad=0.4", facecolor="#FAFAFA", alpha=0.9, edgecolor="#CCCCCC"),
        )

        saved = save_publication_figure(fig, self.output_dir / filename_stem, formats=self.formats, dpi=self.dpi)
        plt.close(fig)
        return saved

    # ------------------------------------------------------------------------
    # Master Pipeline Execution
    # ------------------------------------------------------------------------
    def generate_all(
        self,
        dispatch_df: pd.DataFrame,
        degradation_df: pd.DataFrame | None = None,
        daily_pnl: np.ndarray | None = None,
    ) -> RiskFigureArtifacts:
        pnl = daily_pnl if daily_pnl is not None else self._extract_daily_pnl(dispatch_df, degradation_df)

        f_1 = self.plot_daily_profit_distribution(pnl)
        f_2 = self.plot_drawdown_curve(pnl)
        f_3 = self.plot_var_cvar_tail(pnl)
        f_4 = self.plot_rolling_volatility(pnl)
        f_5 = self.plot_return_distribution(pnl)

        return RiskFigureArtifacts(
            daily_profit_dist=f_1,
            drawdown_curve=f_2,
            var_cvar_tail=f_3,
            rolling_volatility=f_4,
            return_distribution=f_5,
            output_directory=self.output_dir,
        )