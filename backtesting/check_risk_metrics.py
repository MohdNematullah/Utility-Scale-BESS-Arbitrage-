"""
backtesting/check_risk_metrics.py
=================================

Verification and sanity-check script for Risk Metrics Module (Part 9.3).
"""

from pathlib import Path
import numpy as np
import pandas as pd

from backtesting.risk_metrics import RiskMetricsEngine

LINE = "=" * 75

print(LINE)
print(" RISK & VOLATILITY METRICS CHECK (PART 9.3)")
print(LINE)

dispatch_file = Path("backtesting/results/dispatch_history.csv")
degradation_file = Path("backtesting/results/degradation_history.csv")

if dispatch_file.exists() and degradation_file.exists():
    print("Loading empirical simulation dispatch and degradation datasets from disk...")
    dispatch_df = pd.read_csv(dispatch_file)
    degradation_df = pd.read_csv(degradation_file)
else:
    print("Generating calibrated 350-day synthetic backtest dataset for verification...")
    n_hours = 8400
    base_signal = 30.0 + 20.0 * np.sin(np.linspace(0, 350 * 2 * np.pi, n_hours))
    actual_prices = np.clip(base_signal + np.random.normal(0, 5.0, n_hours), 5.0, 150.0)

    chg_mw = np.array([50.0 if (i % 24) in [2, 3] else 0.0 for i in range(n_hours)])
    dis_mw = np.array([45.0 if (i % 24) in [18, 19] else 0.0 for i in range(n_hours)])

    dispatch_df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n_hours, freq="h", tz="UTC"),
        "actual_price": actual_prices,
        "charge_power_mw": chg_mw,
        "discharge_power_mw": dis_mw,
        "net_revenue_usd": (dis_mw - chg_mw) * actual_prices,
    })

    degradation_df = pd.DataFrame({
        "rolling_window": range(1, 351),
        "window_efc": [0.543] * 350,
        "degradation_cost_usd": [725.0] * 350,
    })

engine = RiskMetricsEngine(output_directory="backtesting/results/risk_metrics")
summary, artifacts = engine.evaluate_and_export(
    dispatch_df=dispatch_df,
    degradation_df=degradation_df,
    risk_free_rate_pct=4.0,
    fixed_facility_capex_usd=35000000.0,
)

print("-" * 75)
print("INSTITUTIONAL RISK-ADJUSTED PERFORMANCE METRICS")
print("-" * 75)
print(f"Annualized Return          : ${summary.annualized_return_usd:,.2f}")
print(f"Daily P&L Volatility (Ïƒ)   : ${summary.daily_volatility_usd:,.2f}/day")
print(f"Annualized Volatility (Ïƒ)  : ${summary.annualized_volatility_usd:,.2f}/year")
print(f"Downside Deviation         : ${summary.downside_deviation_usd:,.2f}/day")
print(f"Annualized Sharpe Ratio    : {summary.sharpe_ratio:.3f}")
print(f"Annualized Sortino Ratio   : {summary.sortino_ratio:.3f}")
print(f"Calmar Ratio               : {summary.calmar_ratio:.3f}")
print(f"Omega Ratio (Gain/Loss)    : {summary.omega_ratio:.3f}")
print(f"Profitable Operating Days  : {summary.profitable_days_pct:.2f}% ({summary.total_evaluated_days} days)")

print("-" * 75)
print("DRAWDOWN & CAPITAL PRESERVATION DYNAMICS")
print("-" * 75)
print(f"Maximum Drawdown ($)       : -${summary.max_drawdown_usd:,.2f}")
print(f"Maximum Drawdown (% CAPEX) : -{summary.max_drawdown_pct:.3f}%")
print(f"Max Underwater Duration    : {summary.max_drawdown_duration_days} consecutive days")

print("-" * 75)
print("EXTREME TAIL RISK (VALUE AT RISK & EXPECTED SHORTFALL)")
print("-" * 75)
print(f"Historical 95% VaR         : -${summary.historical_var_95_usd:,.2f}/day")
print(f"Historical 99% VaR         : -${summary.historical_var_99_usd:,.2f}/day")
print(f"Conditional VaR (95% CVaR) : -${summary.cvar_95_usd:,.2f}/day")
print(f"Conditional VaR (99% CVaR) : -${summary.cvar_99_usd:,.2f}/day")
print(f"Return Distribution Skew   : {summary.skewness:.3f}")
print(f"Return Excess Kurtosis     : {summary.excess_kurtosis:.3f}")

# Rigorous Assertions
assert summary.annualized_volatility_usd >= summary.daily_volatility_usd
assert summary.cvar_95_usd >= summary.historical_var_95_usd, "CVaR must mathematically exceed VaR."
assert summary.cvar_99_usd >= summary.historical_var_99_usd, "99% CVaR must mathematically exceed 99% VaR."
assert summary.max_drawdown_usd >= 0.0

print("-" * 75)
print("EXPORTED RISK ARTIFACTS")
print("-" * 75)
print(f"Summary CSV   : {artifacts.summary_csv.exists()} ({artifacts.summary_csv})")
print(f"Daily Risk CSV: {artifacts.daily_risk_csv.exists()} ({artifacts.daily_risk_csv})")
print(f"Summary JSON  : {artifacts.summary_json.exists()} ({artifacts.summary_json})")

figures = list(artifacts.figures_directory.glob("*.png"))
print(f"\nGenerated Diagnostic Visualizations ({len(figures)} total):")
for fig in sorted(figures):
    print(f"  â€¢ {fig.name}")

assert len(figures) == 5, f"Expected 5 figures, found {len(figures)}."

print(LINE)
print("Risk metrics evaluation verified successfully âœ“")
print(LINE)