"""
backtesting/forecast_realism.py
===============================

Forecast Realism & Predictive Quality Evaluation Module (Part 9.1 Enhanced)



Enhancements:
- Added 30-day moving window rolling RMSE trajectory (plot_rolling_rmse)
- Added Quantile-Quantile normality diagnostic plot (plot_residual_qqplot)
- Clean raw-string LaTeX labels preventing syntax warnings in modern Python
- Robust date/hour resolution supporting both Series and DatetimeIndex structures
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"


@dataclass(slots=True)
class ForecastMetricsSummary:
    mae: float
    rmse: float
    mape_pct: float
    smape_pct: float
    r2_score: float
    bias: float
    residual_std: float
    residual_skewness: float
    residual_kurtosis: float
    directional_accuracy_pct: float
    value_capture_ratio_pct: float
    perfect_foresight_gap_usd: float
    sample_count: int


@dataclass(slots=True)
class ForecastRealismArtifacts:
    summary_csv: Path
    residuals_csv: Path
    summary_json: Path
    figure_forecast_vs_actual: Path
    figure_residual_dist: Path
    figure_residual_qq: Path
    figure_horizon_heatmap: Path
    figure_scatter: Path
    figure_rolling_rmse: Path
    figures_directory: Path


class ForecastRealismEngine:
    """
    Evaluates recursive price forecasting realism, residual errors,
    and market trajectory predictability.
    """

    def __init__(self, output_directory: Path | str = "backtesting/results/forecast_realism"):
        self.output_dir = Path(output_directory)
        self.figure_dir = self.output_dir / "figures"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figure_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-6) -> float:
        sum_actual = np.sum(np.abs(y_true))
        if sum_actual < epsilon:
            return 0.0
        return float((np.sum(np.abs(y_true - y_pred)) / sum_actual) * 100.0)

    @staticmethod
    def symmetric_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-6) -> float:
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + epsilon
        return float(np.mean(np.abs(y_pred - y_true) / denominator) * 100.0)

    @staticmethod
    def coefficient_of_determination(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0.0:
            return 1.0 if ss_res == 0.0 else 0.0
        return float(1.0 - (ss_res / ss_tot))

    @staticmethod
    def mean_bias_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(y_pred - y_true))

    @staticmethod
    def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if len(y_true) < 2:
            return 100.0
        actual_delta = np.diff(y_true)
        forecast_delta = y_pred[1:] - y_true[:-1]
        correct_directions = np.equal(np.sign(actual_delta), np.sign(forecast_delta))
        return float(np.mean(correct_directions) * 100.0)

    @staticmethod
    def residual_higher_moments(residuals: np.ndarray) -> tuple[float, float, float]:
        std_dev = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
        centered = residuals - np.mean(residuals)
        n = len(residuals)
        if n < 3 or std_dev == 0.0:
            return std_dev, 0.0, 3.0

        skewness = float((np.sum(centered ** 3) / n) / (std_dev ** 3))
        kurtosis = float((np.sum(centered ** 4) / n) / (std_dev ** 4))
        return std_dev, skewness, kurtosis

    def evaluate_metrics(
        self,
        y_true: Sequence[float] | np.ndarray,
        y_pred: Sequence[float] | np.ndarray,
        gross_revenue_usd: float | None = None,
        perfect_foresight_revenue_usd: float | None = None,
    ) -> ForecastMetricsSummary:
        actual = np.asarray(y_true, dtype=float)
        forecast = np.asarray(y_pred, dtype=float)

        if len(actual) != len(forecast):
            raise ValueError(f"Length mismatch: y_true ({len(actual)}) vs y_pred ({len(forecast)}).")
        if len(actual) == 0:
            raise ValueError("Input price series cannot be empty.")

        residuals = forecast - actual
        std_err, skew, kurt = self.residual_higher_moments(residuals)

        if gross_revenue_usd is not None and perfect_foresight_revenue_usd is not None and perfect_foresight_revenue_usd > 0.0:
            vcr = min((gross_revenue_usd / perfect_foresight_revenue_usd) * 100.0, 100.0)
            pf_gap = max(perfect_foresight_revenue_usd - gross_revenue_usd, 0.0)
        else:
            vcr = 100.0
            pf_gap = 0.0

        return ForecastMetricsSummary(
            mae=round(self.mean_absolute_error(actual, forecast), 4),
            rmse=round(self.root_mean_squared_error(actual, forecast), 4),
            mape_pct=round(self.mean_absolute_percentage_error(actual, forecast), 2),
            smape_pct=round(self.symmetric_mape(actual, forecast), 2),
            r2_score=round(self.coefficient_of_determination(actual, forecast), 4),
            bias=round(self.mean_bias_error(actual, forecast), 4),
            residual_std=round(std_err, 4),
            residual_skewness=round(skew, 3),
            residual_kurtosis=round(kurt, 3),
            directional_accuracy_pct=round(self.directional_accuracy(actual, forecast), 2),
            value_capture_ratio_pct=round(vcr, 2),
            perfect_foresight_gap_usd=round(pf_gap, 2),
            sample_count=len(actual),
        )

    def plot_forecast_vs_actual(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        timestamps: Any | None = None,
        sample_hours: int = 168,
        filename: str = "forecast_vs_actual.png",
    ) -> Path:
        n = min(len(y_true), sample_hours)
        actual_sub = y_true[:n]
        forecast_sub = y_pred[:n]
        x_axis = np.arange(n)

        mae_local = float(np.mean(np.abs(actual_sub - forecast_sub)))

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]}
        )

        ax1.plot(x_axis, actual_sub, label="Actual Settlement Price", color="#1f77b4", linewidth=1.5)
        ax1.plot(x_axis, forecast_sub, label="Day-Ahead Forecast", color="#ff7f0e", linestyle="--", linewidth=1.5)
        ax1.fill_between(
            x_axis,
            forecast_sub - mae_local,
            forecast_sub + mae_local,
            color="#ff7f0e",
            alpha=0.2,
            label=f"Â±1 MAE Confidence Band (${mae_local:.2f})",
        )
        ax1.set_ylabel("Price ($/MWh)", fontsize=10)
        ax1.set_title(f"Day-Ahead Forecast vs Realized Settlement Price (First {n} Hours)", fontsize=11, fontweight="bold")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="upper right", framealpha=0.95)

        residuals_sub = forecast_sub - actual_sub
        colors = np.where(residuals_sub >= 0, "#e1974c", "#7293cb")
        ax2.bar(x_axis, residuals_sub, color=colors, width=0.8, alpha=0.85)
        ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax2.set_ylabel("Error ($/MWh)", fontsize=9)
        ax2.set_xlabel("Time Horizon (Hours)", fontsize=10)
        ax2.grid(True, linestyle="--", alpha=0.5)

        if timestamps is not None and len(timestamps) >= n:
            step = max(1, n // 7)
            tick_locs = np.arange(0, n, step)
            if hasattr(timestamps, "iloc"):
                tick_labels = [str(timestamps.iloc[i])[:16] for i in tick_locs]
            elif hasattr(timestamps, "__getitem__"):
                tick_labels = [str(timestamps[i])[:16] for i in tick_locs]
            else:
                tick_labels = [str(i) for i in tick_locs]
            ax2.set_xticks(tick_locs)
            ax2.set_xticklabels(tick_labels, rotation=20, ha="right", fontsize=8)
            ax2.set_xlabel("Timestamp", fontsize=10)

        out_path = self.figure_dir / filename
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_residual_distribution(
        self,
        residuals: np.ndarray,
        metrics: ForecastMetricsSummary,
        filename: str = "residual_distribution.png",
    ) -> Path:
        fig, ax = plt.subplots(figsize=(9, 5))

        ax.hist(
            residuals,
            bins=50,
            density=True,
            alpha=0.65,
            color="#7293cb",
            edgecolor="white",
            label="Empirical Error Distribution",
        )

        mu, sigma = float(np.mean(residuals)), float(np.std(residuals))
        if sigma > 0:
            x_norm = np.linspace(min(residuals), max(residuals), 200)
            p_norm = (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x_norm - mu) / sigma) ** 2)
            ax.plot(x_norm, p_norm, "r--", linewidth=1.8, label=rf"Gaussian Fit ($\mu={mu:.2f}, \sigma={sigma:.2f}$)")

        ax.axvline(0.0, color="black", linestyle=":", linewidth=1.0)
        ax.set_title("Forecast Error Residual Distribution & Kurtosis Profile", fontsize=11, fontweight="bold")
        ax.set_xlabel("Residual Error [Forecast âˆ’ Actual] ($/MWh)", fontsize=10)
        ax.set_ylabel("Probability Density", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", framealpha=0.95)

        stat_box = (
            f"MAE:  ${metrics.mae:.2f}\n"
            f"RMSE: ${metrics.rmse:.2f}\n"
            f"Bias: ${metrics.bias:.2f}\n"
            f"Skew: {metrics.residual_skewness:.2f}\n"
            f"Kurt: {metrics.residual_kurtosis:.2f}"
        )
        ax.text(
            0.03, 0.95, stat_box,
            transform=ax.transAxes,
            fontsize=9,
            fontfamily="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="#cccccc"),
        )

        out_path = self.figure_dir / filename
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_residual_qqplot(self, residuals: np.ndarray, filename: str = "residual_qqplot.png") -> Path:
        fig, ax = plt.subplots(figsize=(8, 5))

        clean_res = np.sort(residuals)
        n = len(clean_res)
        probabilities = (np.arange(1, n + 1) - 0.5) / n
        q_norm = np.sqrt(2.0) * np.vectorize(self._erfinv)(2.0 * probabilities - 1.0)

        std_dev = np.std(clean_res)
        std_res = (clean_res - np.mean(clean_res)) / std_dev if std_dev > 0 else clean_res

        ax.scatter(q_norm, std_res, alpha=0.5, color="#1f77b4", s=15, label="Standardized Residuals")
        line_lims = [min(float(np.min(q_norm)), float(np.min(std_res))), max(float(np.max(q_norm)), float(np.max(std_res)))]
        ax.plot(line_lims, line_lims, "r--", linewidth=1.5, label="Normal Parity Reference (y = x)")

        ax.set_title("Quantile-Quantile (Q-Q) Normality Diagnostic Plot", fontsize=11, fontweight="bold")
        ax.set_xlabel("Theoretical Normal Quantiles", fontsize=10)
        ax.set_ylabel("Sample Standardized Quantiles", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", framealpha=0.95)

        out_path = self.figure_dir / filename
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
        return out_path

    @staticmethod
    def _erfinv(x: float) -> float:
        a = 0.147
        if x == 0.0:
            return 0.0
        log_term = np.log(1.0 - x * x)
        term1 = 2.0 / (np.pi * a) + log_term / 2.0
        inner = term1 * term1 - log_term / a
        sign = 1.0 if x > 0 else -1.0
        return float(sign * np.sqrt(np.sqrt(inner) - term1))

    def plot_forecast_horizon_heatmap(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        timestamps: Any | None = None,
        filename: str = "forecast_horizon_heatmap.png",
    ) -> Path:
        fig, ax = plt.subplots(figsize=(10, 5))
        n = len(y_true)

        if timestamps is not None and len(timestamps) == n:
            if hasattr(timestamps, "dt"):
                hours = timestamps.dt.hour.to_numpy()
                months = timestamps.dt.month.to_numpy()
            elif isinstance(timestamps, pd.DatetimeIndex):
                hours = timestamps.hour.to_numpy()
                months = timestamps.month.to_numpy()
            else:
                ts_dt = pd.DatetimeIndex(pd.to_datetime(timestamps))
                hours = ts_dt.hour.to_numpy()
                months = ts_dt.month.to_numpy()
        else:
            hours = np.arange(n) % 24
            months = (np.arange(n) // (24 * 30)) + 1

        abs_error = np.abs(y_pred - y_true)
        df_error = pd.DataFrame({"hour": hours, "month": months, "abs_error": abs_error})
        pivot = df_error.pivot_table(index="hour", columns="month", values="abs_error", aggfunc="mean")
        pivot = pivot.reindex(index=range(24)).fillna(0.0)

        im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto", origin="lower")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Mean Absolute Error ($/MWh)", fontsize=9)

        ax.set_title("Forecast MAE Across Hour of Day & Month", fontsize=11, fontweight="bold")
        ax.set_xlabel("Month of Year", fontsize=10)
        ax.set_ylabel("Hour of Day (0â€“23)", fontsize=10)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(0, 24, 2))

        out_path = self.figure_dir / filename
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_forecast_scatter(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        r2_score: float,
        filename: str = "forecast_scatter.png",
    ) -> Path:
        fig, ax = plt.subplots(figsize=(7, 7))

        ax.scatter(y_true, y_pred, alpha=0.4, color="#2ca02c", s=18, edgecolors="none", label="Hourly Observations")

        mn = min(float(np.min(y_true)), float(np.min(y_pred)))
        mx = max(float(np.max(y_true)), float(np.max(y_pred)))
        ax.plot([mn, mx], [mn, mx], "k--", linewidth=1.5, label="Ideal Parity (y = x)")

        if len(y_true) > 1:
            poly = np.polyfit(y_true, y_pred, 1)
            x_seq = np.linspace(mn, mx, 100)
            ax.plot(x_seq, np.polyval(poly, x_seq), "r-", linewidth=1.5, label=f"Fit (Slope: {poly[0]:.2f})")

        ax.set_title(rf"Actual vs Predicted Price Parity ($R^2 = {r2_score:.4f}$)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Actual Settlement Price ($/MWh)", fontsize=10)
        ax.set_ylabel("Forecast Look-Ahead Price ($/MWh)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", framealpha=0.95)
        ax.set_xlim(mn - 2, mx + 2)
        ax.set_ylim(mn - 2, mx + 2)

        out_path = self.figure_dir / filename
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_rolling_rmse(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        window: int = 720,
        filename: str = "rolling_rmse.png",
    ) -> Path:
        """Evaluates longitudinal forecast error stability across a 30-day window."""
        fig, ax = plt.subplots(figsize=(10, 4.5))
        residuals_sq = (y_pred - y_true) ** 2
        rolling_rmse = np.sqrt(
            pd.Series(residuals_sq).rolling(window=window, min_periods=24).mean().to_numpy()
        )
        x = np.arange(len(y_true))

        ax.plot(x, rolling_rmse, color="#1f77b4", linewidth=1.8, label=f"Rolling {window}h Moving RMSE")
        ax.axhline(float(np.sqrt(np.mean(residuals_sq))), color="red", linestyle="--", linewidth=1.2, label="Full Horizon RMSE")
        ax.set_title(f"Longitudinal Predictive Error Stability ({window}-Hour Moving Window)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Simulation Hour", fontsize=10)
        ax.set_ylabel("RMSE ($/MWh)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", framealpha=0.95)

        out_path = self.figure_dir / filename
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
        return out_path

    def evaluate_and_export(
        self,
        dispatch_df: pd.DataFrame,
        gross_revenue_usd: float | None = None,
        perfect_foresight_revenue_usd: float | None = None,
    ) -> tuple[ForecastMetricsSummary, ForecastRealismArtifacts]:
        actual_col = next((c for c in ["actual_price", "price", "settlement_price"] if c in dispatch_df.columns), None)
        if not actual_col or "forecast_price" not in dispatch_df.columns:
            raise KeyError(
                "Dispatch DataFrame must contain 'forecast_price' and a valid ground truth actual price column."
            )

        y_true = dispatch_df[actual_col].to_numpy(dtype=float)
        y_pred = dispatch_df["forecast_price"].to_numpy(dtype=float)

        if gross_revenue_usd is None:
            rev_col = next((c for c in ["net_revenue_usd", "net_revenue_$"] if c in dispatch_df.columns), None)
            if rev_col:
                gross_revenue_usd = float(dispatch_df[rev_col].sum())

        metrics = self.evaluate_metrics(
            y_true=y_true,
            y_pred=y_pred,
            gross_revenue_usd=gross_revenue_usd,
            perfect_foresight_revenue_usd=perfect_foresight_revenue_usd,
        )

        residuals = y_pred - y_true
        ts_series = dispatch_df["timestamp"] if "timestamp" in dispatch_df.columns else None

        # 1. Summary CSV
        summary_csv = self.output_dir / "forecast_accuracy_summary.csv"
        metrics_df = pd.DataFrame([asdict(metrics)]).T.reset_index()
        metrics_df.columns = ["metric", "value"]
        metrics_df.to_csv(summary_csv, index=False)

        # 2. Residuals Series CSV
        residuals_csv = self.output_dir / "forecast_residuals.csv"
        res_df = pd.DataFrame({
            "step": range(len(y_true)),
            "actual_price": np.round(y_true, 4),
            "forecast_price": np.round(y_pred, 4),
            "residual_error": np.round(residuals, 4),
            "abs_error": np.round(np.abs(residuals), 4),
        })
        if ts_series is not None:
            res_df["timestamp"] = ts_series.values
        res_df.to_csv(residuals_csv, index=False)

        # 3. JSON Scorecard
        summary_json = self.output_dir / "forecast_realism.json"
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(asdict(metrics), f, indent=4)

        # 4. Generate 6 Diagnostic Figures
        fig_vs = self.plot_forecast_vs_actual(y_true, y_pred, ts_series)
        fig_dist = self.plot_residual_distribution(residuals, metrics)
        fig_qq = self.plot_residual_qqplot(residuals)
        fig_hm = self.plot_forecast_horizon_heatmap(y_true, y_pred, ts_series)
        fig_scat = self.plot_forecast_scatter(y_true, y_pred, metrics.r2_score)
        fig_roll = self.plot_rolling_rmse(y_true, y_pred, window=720)

        artifacts = ForecastRealismArtifacts(
            summary_csv=summary_csv,
            residuals_csv=residuals_csv,
            summary_json=summary_json,
            figure_forecast_vs_actual=fig_vs,
            figure_residual_dist=fig_dist,
            figure_residual_qq=fig_qq,
            figure_horizon_heatmap=fig_hm,
            figure_scatter=fig_scat,
            figure_rolling_rmse=fig_roll,
            figures_directory=self.figure_dir,
        )

        return metrics, artifacts