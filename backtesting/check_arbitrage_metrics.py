"""
backtesting/check_arbitrage_metrics.py
=====================================

Verification and sanity-check script for Arbitrage Metrics Module (Part 9.2).
"""

from pathlib import Path
import numpy as np
import pandas as pd

from backtesting.arbitrage_metrics import ArbitrageMetricsEngine

LINE = "=" * 75

print(LINE)
print(" ARBITRAGE METRICS CHECK (PART 9.2)")
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

engine = ArbitrageMetricsEngine(output_directory="backtesting/results/arbitrage_metrics")
kpis, artifacts = engine.evaluate_and_export(
    dispatch_df=dispatch_df,
    degradation_df=degradation_df,
    nominal_capacity_mwh=100.0,
    rated_power_mw=50.0,
)

print("-" * 75)
print("FINANCIAL & TECHNO-ECONOMIC SUMMARY")
print("-" * 75)
print(f"Gross Arbitrage Revenue      : ${kpis.gross_revenue_usd:,.2f}")
print(f"Cell Degradation Cost        : -${kpis.degradation_cost_usd:,.2f}")
print(f"Net Arbitrage Revenue        : ${kpis.net_revenue_usd:,.2f}")
print(f"Fixed Facility O&M           : -${kpis.fixed_om_cost_usd:,.2f}")
print(f"Variable Non-Wear O&M        : -${kpis.variable_om_cost_usd:,.2f}")
print(f"Net Operating Profit (EBITDA): ${kpis.net_operating_profit_usd:,.2f}")
print(f"Net Arbitrage Margin         : {kpis.net_arbitrage_margin_pct:.2f}%")

print("-" * 75)
print("CYCLE ECONOMICS & UNIT MARGINS")
print("-" * 75)
print(f"Gross Revenue / MWh Throughput: ${kpis.gross_revenue_per_mwh_throughput:.2f}/MWh")
print(f"Net Revenue / MWh Throughput  : ${kpis.net_revenue_per_mwh_throughput:.2f}/MWh")
print(f"Gross Revenue / EFC           : ${kpis.gross_revenue_per_efc:,.2f}/EFC")
print(f"Net Revenue / EFC             : ${kpis.net_revenue_per_efc:,.2f}/EFC")
print(f"Degradation Cost / EFC        : ${kpis.degradation_cost_per_efc:,.2f}/EFC")
print(f"Annualized Yield / kW-year    : ${kpis.revenue_per_kw_year:.2f}/kW-yr")
print(f"Annualized Yield / kWh-year   : ${kpis.revenue_per_kwh_year:.2f}/kWh-yr")

print("-" * 75)
print("SPREAD CAPTURE & TRADING EXECUTION")
print("-" * 75)
print(f"Volume-Weighted Charge Price  : ${kpis.avg_charge_price_usd_per_mwh:.2f}/MWh")
print(f"Volume-Weighted Discharge Price: ${kpis.avg_discharge_price_usd_per_mwh:.2f}/MWh")
print(f"Realized Price Spread         : ${kpis.realized_spread_usd_per_mwh:.2f}/MWh")
print(f"Theoretical Mean Max Spread   : ${kpis.theoretical_max_spread_usd_per_mwh:.2f}/MWh")
print(f"Spread Capture Ratio          : {kpis.spread_capture_ratio_pct:.2f}%")
print(f"Realized AC Round-Trip Eff.   : {kpis.round_trip_efficiency_pct:.2f}%")
print(f"Idle Hours Fraction           : {kpis.idle_fraction_pct:.2f}% ({kpis.idle_hours} hrs)")

# Mathematical Assertions
assert np.isclose(kpis.gross_revenue_usd - kpis.degradation_cost_usd, kpis.net_revenue_usd, atol=0.05)
assert kpis.round_trip_efficiency_pct > 0.0, "Round-trip efficiency must be positive."
assert kpis.equivalent_full_cycles > 0.0, "EFC must be strictly positive."
assert (kpis.charging_hours + kpis.discharging_hours + kpis.idle_hours) == len(dispatch_df)

print("-" * 75)
print("EXPORTED DIAGNOSTIC ARTIFACTS")
print("-" * 75)
print(f"KPIs CSV     : {artifacts.kpis_csv.exists()} ({artifacts.kpis_csv})")
print(f"Monthly CSV  : {artifacts.monthly_csv.exists()} ({artifacts.monthly_csv})")
print(f"Daily CSV    : {artifacts.daily_csv.exists()} ({artifacts.daily_csv})")
print(f"KPIs JSON    : {artifacts.kpis_json.exists()} ({artifacts.kpis_json})")

figures = list(artifacts.figures_directory.glob("*.png"))
print(f"\nGenerated Publication Visualizations ({len(figures)} total):")
for fig in sorted(figures):
    print(f"  • {fig.name}")

assert len(figures) == 5, f"Expected 5 figures, found {len(figures)}."

print(LINE)
print("Arbitrage metrics evaluation verified successfully ✓")
print(LINE)