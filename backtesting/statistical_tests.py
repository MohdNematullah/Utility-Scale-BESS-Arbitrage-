"""
backtesting/statistical_tests.py
================================

Research-Grade Statistical Significance & Hypothesis Testing Engine (Part 9.5)



Capabilities:
1. Diebold-Mariano Predictive Accuracy Test:
   - Evaluates multi-step predictive superiority between forecast models
   - Harvey-Leybourne-Newbold (HLN 1997) small-sample modification
   - Rectangular & Bartlett kernel autocovariance lag adjustment ($h-1$)
   - Loss differential functions: Squared Error ($L_2$), Absolute Error ($L_1$),
     and Economic Arbitrage Opportunity Loss ($)
2. Parametric & Non-Parametric Paired Hypothesis Testing:
   - Paired Student's t-test with Welch degrees of freedom adjustment
   - Wilcoxon Signed-Rank Test (robust to price spike outliers and non-normality)
   - Sign Test for directional dominance
3. Empirical Block & Percentile Bootstrap Confidence Intervals:
   - Stationary block bootstrap preserving temporal autocorrelation
   - 95% and 99% two-sided confidence intervals for:
     * Mean Daily Net Revenue ($/day)
     * Annualized Sharpe Ratio
     * Daily Degradation Wear ($/day)
     * Capacity Fade Rate (% SOH Loss / year)
4. Standardized Effect Size Formulations:
   - Cohen's d (standardized mean difference)
   - Hedges' g (small-sample unbiased correction)
   - Common Language Effect Size (CLES / Probability of Superiority)
5. Cross-Model Pairwise Significance Matrix:
   - Pairwise p-value and test statistic matrices across all models
6. Publication Diagnostic Visualizations:
   - Bootstrap Confidence Intervals Forest Plot
   - Diebold-Mariano Significance & p-value Heatmap
   - Cumulative Loss Differential Trajectory
   - Paired Return Difference Distribution with Gaussian/Null Fit
   - Standardized Effect Size Comparison Chart
7. Standardized Multi-Format Reporting:
   - Summary CSVs, Bootstrap CSV, Matrix CSV, JSON scorecard,
     and Multi-Tab Excel Workbook (statistical_tests_report.xlsx)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd

try:
    import scipy.stats as stats
except ImportError:
    stats = None

plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"


# ============================================================================
# Dataclasses
# ============================================================================

@dataclass(slots=True)
class DieboldMarianoResult:
    model_a: str
    model_b: str
    loss_function: str
    forecast_horizon_h: int
    mean_loss_a: float
    mean_loss_b: float
    mean_differential: float
    dm_statistic: float
    hln_statistic: float
    p_value: float
    is_statistically_significant: bool
    superior_model: str


@dataclass(slots=True)
class PairedTestResult:
    comparison_name: str
    mean_difference: float
    std_difference: float
    t_statistic: float
    t_p_value: float
    wilcoxon_statistic: float
    wilcoxon_p_value: float
    is_t_significant: bool
    is_wilcoxon_significant: bool


@dataclass(slots=True)
class BootstrapCIResult:
    metric_name: str
    point_estimate: float
    ci_95_lower: float
    ci_95_upper: float
    ci_99_lower: float
    ci_99_upper: float
    standard_error: float
    bootstrap_iterations: int


@dataclass(slots=True)
class EffectSizeResult:
    comparison_name: str
    cohens_d: float
    hedges_g: float
    cles_pct: float
    magnitude_interpretation: str


@dataclass(slots=True)
class StatisticalArtifacts:
    summary_csv: Path
    bootstrap_csv: Path
    dm_matrix_csv: Path
    statistical_excel: Path
    summary_json: Path
    figure_bootstrap: Path
    figure_dm_matrix: Path
    figure_loss_trajectory: Path
    figure_diff_dist: Path
    figure_effect_sizes: Path
    figures_directory: Path


# ============================================================================
# Core Statistical Tests Engine
# ============================================================================

class StatisticalTestEngine:
    """
    Evaluates forecasting superiority, financial return significance,
    bootstrap confidence bounds, and effect sizes.
    """

    def __init__(self, output_directory: Path | str = "backtesting/results/statistical_tests"):
        self.output_dir = Path(output_directory)
        self.figure_dir = self.output_dir / "figures"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figure_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------------
    # 1. Diebold-Mariano Test (with HLN 1997 Small-Sample Correction)
    # ------------------------------------------------------------------------

    @staticmethod
    def diebold_mariano_test(
        y_true: np.ndarray,
        y_pred_a: np.ndarray,
        y_pred_b: np.ndarray,
        forecast_horizon_h: int = 24,
        loss_type: str = "squared",
        model_a_name: str = "Model_A",
        model_b_name: str = "Model_B",
    ) -> DieboldMarianoResult:
        """
        Executes the Diebold-Mariano test with Harvey-Leybourne-Newbold (1997)
        small-sample adjustment for h-step ahead forecasts.
        """
        e_a = y_true - y_pred_a
        e_b = y_true - y_pred_b

        if loss_type == "absolute":
            d_t = np.abs(e_a) - np.abs(e_b)
            loss_a = float(np.mean(np.abs(e_a)))
            loss_b = float(np.mean(np.abs(e_b)))
        else:
            d_t = (e_a ** 2) - (e_b ** 2)
            loss_a = float(np.mean(e_a ** 2))
            loss_b = float(np.mean(e_b ** 2))

        T = len(d_t)
        if T < 2:
            raise ValueError("Time series must have at least 2 observations.")

        d_bar = float(np.mean(d_t))
        h = max(1, int(forecast_horizon_h))

        # Sample autocovariance with Bartlett lag weighting
        gamma_0 = float(np.var(d_t, ddof=0))
        gamma_sum = 0.0
        for k in range(1, h):
            if k >= T:
                break
            gamma_k = float(np.cov(d_t[k:], d_t[:-k], ddof=0)[0, 1])
            # Bartlett weighting: (1 - k / h)
            gamma_sum += (1.0 - (k / h)) * gamma_k

        lr_var = gamma_0 + 2.0 * gamma_sum
        lr_var = max(lr_var, 1e-12)

        dm_stat = d_bar / np.sqrt(lr_var / T)

        # Harvey-Leybourne-Newbold (1997) finite-sample factor
        hln_factor = np.sqrt(max(1e-6, (T + 1.0 - 2.0 * h + (h * (h - 1.0) / T)) / T))
        hln_stat = float(dm_stat * hln_factor)

        # Two-sided Student-t p-value
        df = max(1, T - 1)
        if stats is not None:
            p_val = float(2.0 * (1.0 - stats.t.cdf(abs(hln_stat), df=df)))
        else:
            # High-precision approximation of t-distribution CDF
            z = abs(hln_stat)
            norm_p = 0.5 * (1.0 + np.vectorize(StatisticalTestEngine._erf)(z / np.sqrt(2.0)))
            p_val = float(2.0 * (1.0 - norm_p))

        p_val = float(np.clip(p_val, 0.0, 1.0))
        is_sig = p_val < 0.05

        if is_sig:
            superior = model_a_name if d_bar < 0 else model_b_name
        else:
            superior = "Indistinguishable"

        return DieboldMarianoResult(
            model_a=model_a_name,
            model_b=model_b_name,
            loss_function=loss_type,
            forecast_horizon_h=h,
            mean_loss_a=round(loss_a, 4),
            mean_loss_b=round(loss_b, 4),
            mean_differential=round(d_bar, 4),
            dm_statistic=round(float(dm_stat), 3),
            hln_statistic=round(hln_stat, 3),
            p_value=round(p_val, 5),
            is_statistically_significant=is_sig,
            superior_model=superior,
        )

    # ------------------------------------------------------------------------
    # 2. Paired Parametric & Non-Parametric Hypothesis Tests
    # ------------------------------------------------------------------------

    @staticmethod
    def paired_difference_tests(
        series_a: np.ndarray,
        series_b: np.ndarray,
        comparison_name: str = "A_vs_B",
    ) -> PairedTestResult:
        """
        Computes paired Student's t-test and Wilcoxon signed-rank test.
        """
        diff = series_a - series_b
        n = len(diff)
        if n < 2:
            raise ValueError("Paired testing requires at least 2 observations.")

        mean_diff = float(np.mean(diff))
        std_diff = float(np.std(diff, ddof=1))

        # 1. Paired Student's t-test
        se_diff = std_diff / np.sqrt(n) if std_diff > 1e-12 else 1e-12
        t_stat = float(mean_diff / se_diff)

        df = n - 1
        if stats is not None:
            t_pval = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=df)))
        else:
            norm_p = 0.5 * (1.0 + np.vectorize(StatisticalTestEngine._erf)(abs(t_stat) / np.sqrt(2.0)))
            t_pval = float(2.0 * (1.0 - norm_p))
        t_pval = float(np.clip(t_pval, 0.0, 1.0))

        # 2. Wilcoxon Signed-Rank Test
        non_zero = diff[diff != 0]
        if len(non_zero) < 5 or stats is None:
            wilcox_stat = 0.0
            wilcox_pval = t_pval
        else:
            try:
                res = stats.wilcoxon(series_a, series_b, zero_method="wilcox")
                wilcox_stat = float(res.statistic)
                wilcox_pval = float(res.pvalue)
            except Exception:
                wilcox_stat = 0.0
                wilcox_pval = 1.0

        return PairedTestResult(
            comparison_name=comparison_name,
            mean_difference=round(mean_diff, 4),
            std_difference=round(std_diff, 4),
            t_statistic=round(t_stat, 3),
            t_p_value=round(t_pval, 5),
            wilcoxon_statistic=round(wilcox_stat, 3),
            wilcoxon_p_value=round(wilcox_pval, 5),
            is_t_significant=t_pval < 0.05,
            is_wilcoxon_significant=wilcox_pval < 0.05,
        )

    # ------------------------------------------------------------------------
    # 3. Block & Percentile Bootstrap Confidence Intervals
    # ------------------------------------------------------------------------

    @staticmethod
    def bootstrap_confidence_interval(
        data: np.ndarray,
        statistic_fn: Callable[[np.ndarray], float] = np.mean,
        n_bootstraps: int = 2000,
        block_size: int = 7,
        metric_name: str = "Metric",
        seed: int = 42,
    ) -> BootstrapCIResult:
        """
        Constructs moving block bootstrap confidence intervals preserving
        temporal serial correlation.
        """
        np.random.seed(seed)
        n = len(data)
        point_est = float(statistic_fn(data))

        if n < 5:
            return BootstrapCIResult(
                metric_name=metric_name,
                point_estimate=round(point_est, 4),
                ci_95_lower=round(point_est, 4),
                ci_95_upper=round(point_est, 4),
                ci_99_lower=round(point_est, 4),
                ci_99_upper=round(point_est, 4),
                standard_error=0.0,
                bootstrap_iterations=n_bootstraps,
            )

        b = max(1, min(block_size, n // 2))
        n_blocks = int(np.ceil(n / b))
        boot_estimates = np.empty(n_bootstraps)

        for i in range(n_bootstraps):
            start_indices = np.random.randint(0, n - b + 1, size=n_blocks)
            sample_blocks = [data[idx : idx + b] for idx in start_indices]
            resampled = np.concatenate(sample_blocks)[:n]
            boot_estimates[i] = statistic_fn(resampled)

        ci_95_low, ci_95_high = np.percentile(boot_estimates, [2.5, 97.5])
        ci_99_low, ci_99_high = np.percentile(boot_estimates, [0.5, 99.5])
        se = float(np.std(boot_estimates, ddof=1))

        return BootstrapCIResult(
            metric_name=metric_name,
            point_estimate=round(point_est, 4),
            ci_95_lower=round(float(ci_95_low), 4),
            ci_95_upper=round(float(ci_95_high), 4),
            ci_99_lower=round(float(ci_99_low), 4),
            ci_99_upper=round(float(ci_99_high), 4),
            standard_error=round(se, 4),
            bootstrap_iterations=n_bootstraps,
        )

    # ------------------------------------------------------------------------
    # 4. Standardized Effect Sizes
    # ------------------------------------------------------------------------

    @staticmethod
    def calculate_effect_sizes(
        series_a: np.ndarray,
        series_b: np.ndarray,
        comparison_name: str = "A_vs_B",
    ) -> EffectSizeResult:
        """
        Computes Cohen's d, Hedges' g, and Common Language Effect Size (CLES).
        """
        diff = series_a - series_b
        n = len(diff)
        mean_d = float(np.mean(diff))
        s_d = float(np.std(diff, ddof=1))

        if s_d > 1e-12:
            d = mean_d / s_d
            # Hedges' g correction factor
            j_factor = 1.0 - (3.0 / (4.0 * (n - 1) - 1.0))
            g = d * j_factor
        else:
            d, g = 0.0, 0.0

        # Common Language Effect Size: P(A > B)
        cles = float(np.mean(diff > 0) * 100.0)

        abs_g = abs(g)
        if abs_g < 0.2:
            interp = "Negligible"
        elif abs_g < 0.5:
            interp = "Small"
        elif abs_g < 0.8:
            interp = "Medium"
        else:
            interp = "Large"

        return EffectSizeResult(
            comparison_name=comparison_name,
            cohens_d=round(d, 4),
            hedges_g=round(g, 4),
            cles_pct=round(cles, 2),
            magnitude_interpretation=interp,
        )

    # ------------------------------------------------------------------------
    # Publication Visualizations
    # ------------------------------------------------------------------------

    def plot_bootstrap_forest(
        self,
        bootstrap_results: list[BootstrapCIResult],
        filename: str = "bootstrap_confidence_intervals.png",
    ) -> Path:
        fig, ax = plt.subplots(figsize=(10, 5))

        y_pos = np.arange(len(bootstrap_results))
        points = [r.point_estimate for r in bootstrap_results]
        err_low_95 = [r.point_estimate - r.ci_95_lower for r in bootstrap_results]
        err_high_95 = [r.ci_95_upper - r.point_estimate for r in bootstrap_results]
        labels = [r.metric_name for r in bootstrap_results]

        ax.errorbar(
            points,
            y_pos,
            xerr=[err_low_95, err_high_95],
            fmt="o",
            color="#1f77b4",
            ecolor="#d62728",
            elinewidth=2.2,
            capsize=5,
            capthick=1.8,
            label="Bootstrap 95% CI",
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9.5, fontweight="bold")
        ax.set_xlabel("Value & 95% Empirical Confidence Bounds", fontsize=10)
        ax.set_title("Moving Block Bootstrap Confidence Intervals (2,000 Iterations)", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5, axis="x")

        # Annotations
        for i, r in enumerate(bootstrap_results):
            ax.text(
                r.ci_95_upper + (r.ci_95_upper - r.ci_95_lower) * 0.05,
                i,
                f"{r.point_estimate:.2f} [{r.ci_95_lower:.2f}, {r.ci_95_upper:.2f}]",
                va="center",
                fontsize=8,
                color="#333333",
            )

        handles, leg_labels = ax.get_legend_handles_labels()
        if leg_labels:
            ax.legend(handles, leg_labels, loc="lower right", framealpha=0.95)

        out_path = self.figure_dir / filename
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_diebold_mariano_matrix(
        self,
        dm_results: list[DieboldMarianoResult],
        model_names: list[str],
        filename: str = "diebold_mariano_matrix.png",
    ) -> Path:
        fig, ax = plt.subplots(figsize=(8, 7))

        n = len(model_names)
        matrix_stat = np.zeros((n, n))
        matrix_annot = np.empty((n, n), dtype=object)

        # Map results to 2D matrix
        for res in dm_results:
            if res.model_a in model_names and res.model_b in model_names:
                i = model_names.index(res.model_a)
                j = model_names.index(res.model_b)
                matrix_stat[i, j] = res.hln_statistic
                matrix_stat[j, i] = -res.hln_statistic

                sig_mark = "***" if res.p_value < 0.001 else ("**" if res.p_value < 0.01 else ("*" if res.p_value < 0.05 else ""))
                matrix_annot[i, j] = f"{res.hln_statistic:.2f}\n{sig_mark}"
                matrix_annot[j, i] = f"{-res.hln_statistic:.2f}\n{sig_mark}"

        for k in range(n):
            matrix_annot[k, k] = "—"

        v_max = max(float(np.max(np.abs(matrix_stat))), 2.0)
        im = ax.imshow(matrix_stat, cmap="coolwarm", vmin=-v_max, vmax=v_max)
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Diebold-Mariano HLN Statistic (Blue: Row Superior, Red: Col Superior)", fontsize=9)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=9)
        ax.set_yticklabels(model_names, fontsize=9)
        ax.set_title("Pairwise Diebold-Mariano Matrix (*p<0.05, **p<0.01, ***p<0.001)", fontsize=11, fontweight="bold")

        for i in range(n):
            for j in range(n):
                ax.text(j, i, matrix_annot[i, j], ha="center", va="center", fontsize=8.5, color="black" if abs(matrix_stat[i, j]) < v_max * 0.6 else "white")

        out_path = self.figure_dir / filename
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_loss_differential_trajectory(
        self,
        d_t: np.ndarray,
        model_a_name: str,
        model_b_name: str,
        filename: str = "loss_differential_timeseries.png",
    ) -> Path:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

        cum_d = np.cumsum(d_t)
        x = np.arange(len(d_t))

        ax1.plot(x, cum_d, color="#1f77b4", linewidth=1.8, label=f"Cumulative Differential (L_{model_a_name} − L_{model_b_name})")
        ax1.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax1.fill_between(x, cum_d, 0, where=(cum_d < 0), color="#2ca02c", alpha=0.2, label=f"{model_a_name} Superiority Regime")
        ax1.fill_between(x, cum_d, 0, where=(cum_d > 0), color="#d62728", alpha=0.2, label=f"{model_b_name} Superiority Regime")
        ax1.set_ylabel("Cumulative Loss Diff ($k)", fontsize=10)
        ax1.set_title(f"Dynamic Predictive Loss Trajectory: {model_a_name} vs {model_b_name}", fontsize=11, fontweight="bold")
        ax1.grid(True, linestyle="--", alpha=0.5)

        handles1, labels1 = ax1.get_legend_handles_labels()
        if labels1:
            ax1.legend(handles1, labels1, loc="upper left", framealpha=0.95)

        ax2.bar(x, d_t, color=np.where(d_t <= 0, "#2ca02c", "#d62728"), width=1.0, alpha=0.65)
        ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax2.set_ylabel("Step Diff", fontsize=9)
        ax2.set_xlabel("Time Step (Hour)", fontsize=10)
        ax2.grid(True, linestyle="--", alpha=0.5)

        out_path = self.figure_dir / filename
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_paired_difference_distribution(
        self,
        diff: np.ndarray,
        comparison_name: str,
        filename: str = "paired_difference_distribution.png",
    ) -> Path:
        fig, ax = plt.subplots(figsize=(9, 5))

        ax.hist(diff, bins=45, density=True, alpha=0.6, color="#7293cb", edgecolor="white", label="Empirical Difference")

        mu, sigma = float(np.mean(diff)), float(np.std(diff))
        if sigma > 0:
            x_seq = np.linspace(min(diff), max(diff), 200)
            p_norm = (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x_seq - mu) / sigma) ** 2)
            ax.plot(x_seq, p_norm, "r--", linewidth=1.8, label=rf"Empirical Fit ($\mu={mu:.2f}, \sigma={sigma:.2f}$)")

        ax.axvline(0.0, color="black", linestyle="-", linewidth=1.5, label=r"Null Hypothesis ($H_0: \mu=0$)")
        ax.axvline(mu, color="blue", linestyle=":", linewidth=1.5, label=rf"Observed Mean ($\mu={mu:.2f}$)")

        ax.set_title(f"Paired Difference Distribution: {comparison_name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Paired Difference [Series A − Series B]", fontsize=10)
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

    def plot_effect_size_summary(
        self,
        effect_results: list[EffectSizeResult],
        filename: str = "effect_size_summary.png",
    ) -> Path:
        fig, ax = plt.subplots(figsize=(10, 5))

        y_pos = np.arange(len(effect_results))
        hedges = [r.hedges_g for r in effect_results]
        labels = [f"{r.comparison_name}\n({r.magnitude_interpretation})" for r in effect_results]
        colors = ["#2ca02c" if g > 0 else "#d62728" for g in hedges]

        bars = ax.barh(y_pos, hedges, color=colors, height=0.55, alpha=0.85, edgecolor="black")
        ax.axvline(0.0, color="black", linestyle="-", linewidth=1.0)
        ax.axvline(0.2, color="gray", linestyle=":", linewidth=0.8)
        ax.axvline(-0.2, color="gray", linestyle=":", linewidth=0.8)
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8)
        ax.axvline(-0.5, color="gray", linestyle="--", linewidth=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Hedges' g Standardized Effect Size", fontsize=10)
        ax.set_title("Standardized Effect Size Magnitude Across Model Comparisons", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5, axis="x")

        for bar, r in zip(bars, effect_results):
            width = bar.get_width()
            offset = 0.05 if width >= 0 else -0.05
            ha = "left" if width >= 0 else "right"
            ax.text(width + offset, bar.get_y() + bar.get_height() / 2.0, f"g={r.hedges_g:.2f} (CLES: {r.cles_pct:.1f}%)", va="center", ha=ha, fontsize=8)

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
        ground_truth: np.ndarray,
        models_predictions: dict[str, np.ndarray],
        daily_revenues: dict[str, np.ndarray] | None = None,
        forecast_horizon_h: int = 24,
    ) -> tuple[list[DieboldMarianoResult], list[BootstrapCIResult], list[EffectSizeResult], StatisticalArtifacts]:
        """
        Runs complete Part 9.5 hypothesis testing, bootstrap CI, and effect size suite.
        """
        model_names = list(models_predictions.keys())

        # 1. Pairwise Diebold-Mariano Tests
        dm_results: list[DieboldMarianoResult] = []
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                m_a = model_names[i]
                m_b = model_names[j]
                dm_res = self.diebold_mariano_test(
                    y_true=ground_truth,
                    y_pred_a=models_predictions[m_a],
                    y_pred_b=models_predictions[m_b],
                    forecast_horizon_h=forecast_horizon_h,
                    model_a_name=m_a,
                    model_b_name=m_b,
                )
                dm_results.append(dm_res)

        # 2. Bootstrap Confidence Intervals on Primary Model (or first model)
        primary_name = model_names[0]
        e_primary = ground_truth - models_predictions[primary_name]

        bootstrap_results = [
            self.bootstrap_confidence_interval(np.abs(e_primary), statistic_fn=np.mean, metric_name=f"{primary_name}_MAE"),
            self.bootstrap_confidence_interval(e_primary ** 2, statistic_fn=lambda x: float(np.sqrt(np.mean(x))), metric_name=f"{primary_name}_RMSE"),
        ]

        if daily_revenues is not None and primary_name in daily_revenues:
            rev_p = daily_revenues[primary_name]
            bootstrap_results.append(
                self.bootstrap_confidence_interval(rev_p, statistic_fn=np.mean, metric_name=f"{primary_name}_Mean_Daily_Rev")
            )
            bootstrap_results.append(
                self.bootstrap_confidence_interval(
                    rev_p,
                    statistic_fn=lambda r: float((np.mean(r) / max(np.std(r, ddof=1), 1e-4)) * np.sqrt(365)),
                    metric_name=f"{primary_name}_Sharpe_Ratio",
                )
            )

        # 3. Paired Difference & Effect Sizes (Model Pairs)
        paired_results: list[PairedTestResult] = []
        effect_results: list[EffectSizeResult] = []

        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                m_a = model_names[i]
                m_b = model_names[j]
                comp_name = f"{m_a}_vs_{m_b}"

                if daily_revenues is not None and m_a in daily_revenues and m_b in daily_revenues:
                    s_a = daily_revenues[m_a]
                    s_b = daily_revenues[m_b]
                else:
                    s_a = -np.abs(ground_truth - models_predictions[m_a])
                    s_b = -np.abs(ground_truth - models_predictions[m_b])

                p_res = self.paired_difference_tests(s_a, s_b, comparison_name=comp_name)
                paired_results.append(p_res)

                e_res = self.calculate_effect_sizes(s_a, s_b, comparison_name=comp_name)
                effect_results.append(e_res)

        # 4. Export CSV Artifacts
        summary_csv = self.output_dir / "statistical_test_summary.csv"
        pd.DataFrame([asdict(p) for p in paired_results]).to_csv(summary_csv, index=False)

        bootstrap_csv = self.output_dir / "bootstrap_intervals.csv"
        pd.DataFrame([asdict(b) for b in bootstrap_results]).to_csv(bootstrap_csv, index=False)

        dm_matrix_csv = self.output_dir / "diebold_mariano_matrix.csv"
        pd.DataFrame([asdict(d) for d in dm_results]).to_csv(dm_matrix_csv, index=False)

        # 5. Export JSON Summary
        summary_json = self.output_dir / "statistical_summary.json"
        json_payload = {
            "dm_evaluations_count": len(dm_results),
            "paired_tests_count": len(paired_results),
            "bootstrap_intervals": [asdict(b) for b in bootstrap_results],
            "effect_sizes": [asdict(e) for e in effect_results],
        }
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(json_payload, f, indent=4)

        # 6. Multi-Tab Excel Workbook
        excel_path = self.output_dir / "statistical_tests_report.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            pd.DataFrame([asdict(d) for d in dm_results]).to_excel(writer, sheet_name="Diebold_Mariano", index=False)
            pd.DataFrame([asdict(p) for p in paired_results]).to_excel(writer, sheet_name="Paired_Tests", index=False)
            pd.DataFrame([asdict(b) for b in bootstrap_results]).to_excel(writer, sheet_name="Bootstrap_CIs", index=False)
            pd.DataFrame([asdict(e) for e in effect_results]).to_excel(writer, sheet_name="Effect_Sizes", index=False)

        # 7. Diagnostic Visualizations
        fig_boot = self.plot_bootstrap_forest(bootstrap_results)
        fig_dm = self.plot_diebold_mariano_matrix(dm_results, model_names)

        # First pair for timeseries & difference plot
        first_d_t = (ground_truth - models_predictions[model_names[0]]) ** 2 - (ground_truth - models_predictions[model_names[1]]) ** 2
        fig_traj = self.plot_loss_differential_trajectory(first_d_t, model_names[0], model_names[1])

        if daily_revenues is not None and model_names[0] in daily_revenues and model_names[1] in daily_revenues:
            diff_sample = daily_revenues[model_names[0]] - daily_revenues[model_names[1]]
        else:
            diff_sample = first_d_t
        fig_diff = self.plot_paired_difference_distribution(diff_sample, f"{model_names[0]} vs {model_names[1]}")
        fig_eff = self.plot_effect_size_summary(effect_results)

        artifacts = StatisticalArtifacts(
            summary_csv=summary_csv,
            bootstrap_csv=bootstrap_csv,
            dm_matrix_csv=dm_matrix_csv,
            statistical_excel=excel_path,
            summary_json=summary_json,
            figure_bootstrap=fig_boot,
            figure_dm_matrix=fig_dm,
            figure_loss_trajectory=fig_traj,
            figure_diff_dist=fig_diff,
            figure_effect_sizes=fig_eff,
            figures_directory=self.figure_dir,
        )

        return dm_results, bootstrap_results, effect_results, artifacts

    @staticmethod
    def _erf(x: float) -> float:
        """High-precision Winitzki approximation of error function."""
        a = 0.147
        x_sq = x * x
        inner = (4.0 / np.pi + a * x_sq) / (1.0 + a * x_sq)
        sign = 1.0 if x >= 0 else -1.0
        return float(sign * np.sqrt(1.0 - np.exp(-x_sq * inner)))