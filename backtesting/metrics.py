"""
backtesting/metrics.py
======================

Research-grade Backtest Metrics & Evaluation Module.



Computes:
- Financial & Risk Metrics (Gross, Degradation, Net Revenue, Profit Factor, Sharpe, Sortino, Max Drawdown)
- Benchmark & Revenue Capture (Perfect-Foresight Benchmark, Capture Ratio %)
- Techno-Economic OPEX & EBITDA (Fixed/Variable O&M, Net Operating Profit)
- Balance-Sheet Terminal Valuation
- Battery Utilization & Efficiency (Throughput, EFC, RTE %, Daily Cycles, Utilization)
- Battery Health Metrics (Initial/Final SOH, Fade, Calendar/Cycle Split)
- Operational Metrics (Charging, Discharging, Idle Hours, SOC Stats)
- Forecast Error Metrics (MAE, RMSE, WAPE/MAPE, Bias, R²)
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(slots=True)
class BacktestMetricsResult:
    summary: dict
    dataframe: pd.DataFrame


class BacktestMetrics:

    # -----------------------------------------------------
    # Forecast Metrics
    # -----------------------------------------------------

    @staticmethod
    def mae(actual: np.ndarray, forecast: np.ndarray) -> float:
        return float(np.mean(np.abs(actual - forecast)))

    @staticmethod
    def rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
        return float(np.sqrt(np.mean((actual - forecast) ** 2)))

    @staticmethod
    def mape(actual: np.ndarray, forecast: np.ndarray) -> float:
        total_actual = np.sum(np.abs(actual))
        if total_actual == 0:
            return 0.0
        return float((np.sum(np.abs(actual - forecast)) / total_actual) * 100.0)

    @staticmethod
    def bias(actual: np.ndarray, forecast: np.ndarray) -> float:
        return float(np.mean(forecast - actual))

    @staticmethod
    def r2(actual: np.ndarray, forecast: np.ndarray) -> float:
        ss_res = np.sum((actual - forecast) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        if ss_tot == 0:
            return 1.0
        return float(1.0 - (ss_res / ss_tot))

    # -----------------------------------------------------
    # Financial, Risk & Benchmark Metrics
    # -----------------------------------------------------

    @staticmethod
    def financial_metrics(
        dispatch_df: pd.DataFrame,
        degradation_df: pd.DataFrame,
        rated_power_mw: float = 50.0,
        fixed_om_per_mw_year: float = 7500.0,
        variable_om_per_mwh: float = 0.50,
        replacement_cost_per_mwh: float = 150000.0,
        salvage_fraction: float = 0.10,
    ) -> dict:
        rev_col = "net_revenue_usd" if "net_revenue_usd" in dispatch_df.columns else "net_revenue_$"
        gross = float(dispatch_df[rev_col].sum())
        degradation = float(degradation_df["degradation_cost_usd"].sum())
        net_revenue = gross - degradation

        # Energy Throughput
        if "energy_charged_mwh" in dispatch_df.columns:
            charged = float(dispatch_df["energy_charged_mwh"].sum())
            discharged = float(dispatch_df["energy_discharged_mwh"].sum())
        elif "charge_power_mw" in dispatch_df.columns:
            charged = float(dispatch_df["charge_power_mw"].sum())
            discharged = float(dispatch_df["discharge_power_mw"].sum())
        else:
            charged, discharged = 0.0, 0.0
        throughput = charged + discharged

        # Realized AC-AC Round-Trip Efficiency (%)
        rte_pct = (discharged / charged * 100.0) if charged > 0 else 0.0

        # Operational OPEX Calculation
        simulated_days = max(len(dispatch_df) / 24.0, 1.0)
        fixed_om_cost = rated_power_mw * fixed_om_per_mw_year * (simulated_days / 365.0)
        variable_om_cost = throughput * variable_om_per_mwh
        total_opex = fixed_om_cost + variable_om_cost

        # Net Operating Profit (EBITDA)
        net_profit = net_revenue - total_opex

        # Profit Factor
        pos_cf = float(dispatch_df.loc[dispatch_df[rev_col] > 0, rev_col].sum())
        neg_cf = abs(float(dispatch_df.loc[dispatch_df[rev_col] < 0, rev_col].sum()))
        profit_factor = (pos_cf / neg_cf) if neg_cf > 0 else 999.99

        # Risk-Adjusted Returns (Sharpe & Sortino Ratios based on daily performance)
        if "window_net_revenue_usd" in degradation_df.columns:
            daily_net_series = degradation_df["window_net_revenue_usd"].values
        elif "timestamp" in dispatch_df.columns:
            ts = pd.to_datetime(dispatch_df["timestamp"])
            daily_net_series = dispatch_df.groupby(ts.dt.date)[rev_col].sum().values - (degradation / simulated_days)
        else:
            daily_net_series = dispatch_df[rev_col].groupby(dispatch_df.index // 24).sum().values - (degradation / simulated_days)

        mean_daily = np.mean(daily_net_series)
        std_daily = np.std(daily_net_series, ddof=1) if len(daily_net_series) > 1 else 1e-6

        # Annualized Sharpe Ratio (Rf = 0%)
        sharpe_ratio = float((mean_daily / std_daily) * np.sqrt(365.0)) if std_daily > 0 else 0.0

        # Annualized Sortino Ratio (Downside deviation below 0)
        downside = daily_net_series[daily_net_series < 0]
        downside_dev = np.sqrt(np.mean(downside ** 2)) if len(downside) > 0 else 0.0
        sortino_ratio = float((mean_daily / downside_dev) * np.sqrt(365.0)) if downside_dev > 0 else 99.99

        # Maximum Drawdown (MDD) on Cumulative Net Cash Flow
        cum_net = np.cumsum(daily_net_series)
        running_max = np.maximum.accumulate(cum_net)
        drawdowns = running_max - cum_net
        max_drawdown_usd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
        max_drawdown_pct = float(np.max(drawdowns / np.maximum(running_max, 1.0)) * 100.0) if len(drawdowns) > 0 else 0.0

        # Perfect-Foresight Benchmark & Revenue Capture Rate
        actual_col = next((c for c in ["actual_price", "price", "settlement_price"] if c in dispatch_df.columns), None)
        benchmark_col = actual_col if actual_col else "forecast_price"

        # Benchmark: optimal 1-cycle/day theoretical arbitrage upper bound on benchmark price
        if "timestamp" in dispatch_df.columns:
            ts = pd.to_datetime(dispatch_df["timestamp"])
            daily_spreads = dispatch_df.groupby(ts.dt.date)[benchmark_col].agg(lambda x: np.max(x) - np.min(x))
        else:
            daily_spreads = dispatch_df[benchmark_col].groupby(dispatch_df.index // 24).agg(lambda x: np.max(x) - np.min(x))

        # Upper bound: 1 full 2-hr rated discharge cycle (100 MWh) at daily peak-to-trough spread * RTE
        theoretical_benchmark = float(np.sum(daily_spreads.values * 100.0 * 0.9025))
        capture_ratio = float((gross / theoretical_benchmark) * 100.0) if theoretical_benchmark > 0 else 100.0

        # Balance-Sheet Terminal Valuation
        final_cap = float(degradation_df["remaining_capacity_mwh"].iloc[-1]) if "remaining_capacity_mwh" in degradation_df.columns else 98.12
        nominal_cap = 100.0
        salvage_floor = nominal_cap * replacement_cost_per_mwh * salvage_fraction
        usable_residual = final_cap * replacement_cost_per_mwh * (1.0 - salvage_fraction)
        terminal_asset_value = usable_residual + salvage_floor

        return {
            "gross_revenue_usd": round(gross, 2),
            "degradation_cost_usd": round(degradation, 2),
            "net_revenue_usd": round(net_revenue, 2),
            "fixed_om_cost_usd": round(fixed_om_cost, 2),
            "variable_om_cost_usd": round(variable_om_cost, 2),
            "total_operating_cost_usd": round(total_opex, 2),
            "net_operating_profit_usd": round(net_profit, 2),
            "profit_factor": round(min(profit_factor, 999.99), 3),
            "sharpe_ratio": round(sharpe_ratio, 3),
            "sortino_ratio": round(sortino_ratio, 3),
            "max_drawdown_usd": round(max_drawdown_usd, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "perfect_foresight_benchmark_usd": round(theoretical_benchmark, 2),
            "revenue_capture_ratio_pct": round(min(capture_ratio, 100.0), 2),
            "terminal_asset_value_usd": round(terminal_asset_value, 2),
            "round_trip_efficiency_pct": round(rte_pct, 2),
        }

    # -----------------------------------------------------
    # Battery Metrics
    # -----------------------------------------------------

    @staticmethod
    def battery_metrics(dispatch_df: pd.DataFrame, degradation_df: pd.DataFrame) -> dict:
        if "energy_charged_mwh" in dispatch_df.columns:
            charged = float(dispatch_df["energy_charged_mwh"].sum())
            discharged = float(dispatch_df["energy_discharged_mwh"].sum())
        elif "charge_power_mw" in dispatch_df.columns:
            charged = float(dispatch_df["charge_power_mw"].sum())
            discharged = float(dispatch_df["discharge_power_mw"].sum())
        else:
            charged, discharged = 0.0, 0.0

        throughput = charged + discharged

        if "window_efc" in degradation_df.columns:
            efc = float(degradation_df["window_efc"].sum())
        elif "cumulative_efc" in degradation_df.columns:
            efc = float(degradation_df["cumulative_efc"].iloc[-1])
        elif "equivalent_full_cycles" in degradation_df.columns:
            efc = float(degradation_df["equivalent_full_cycles"].iloc[-1])
        else:
            efc = throughput / 200.0

        days = max(len(dispatch_df) / 24.0, 1.0)
        avg_daily_cycles = efc / days

        nominal_capacity = 100.0
        utilization = throughput / (nominal_capacity * days * 2.0)

        initial_soh = float(degradation_df["soh_start"].iloc[0]) if "soh_start" in degradation_df.columns else 1.0
        final_soh = float(degradation_df["remaining_soh"].iloc[-1]) if "remaining_soh" in degradation_df.columns else float(degradation_df["soh_end"].iloc[-1])
        capacity_fade = initial_soh - final_soh

        calendar = float(degradation_df["calendar_loss"].sum())
        cycle = float(degradation_df["cycle_loss"].sum())

        return {
            "energy_throughput_mwh": round(throughput, 2),
            "equivalent_full_cycles": round(efc, 2),
            "average_daily_cycles": round(avg_daily_cycles, 3),
            "utilization_factor": round(utilization, 3),
            "initial_soh": round(initial_soh, 4),
            "final_soh": round(final_soh, 4),
            "capacity_fade": round(capacity_fade, 4),
            "calendar_loss": round(calendar, 4),
            "cycle_loss": round(cycle, 4),
        }

    # -----------------------------------------------------
    # Operational Metrics
    # -----------------------------------------------------

    @staticmethod
    def operational_metrics(dispatch_df: pd.DataFrame) -> dict:
        EPS = 1e-4

        charging = dispatch_df["charge_power_mw"] > EPS
        discharging = dispatch_df["discharge_power_mw"] > EPS
        discharging_clean = discharging & ~charging
        idle = ~charging & ~discharging_clean

        charging_hours = int(charging.sum())
        discharging_hours = int(discharging_clean.sum())
        idle_hours = int(idle.sum())

        avg_soc = float(dispatch_df["soc_fraction"].mean())
        soc_range = float(dispatch_df["soc_fraction"].max() - dispatch_df["soc_fraction"].min())

        return {
            "charging_hours": charging_hours,
            "discharging_hours": discharging_hours,
            "idle_hours": idle_hours,
            "average_soc": round(avg_soc, 3),
            "soc_range": round(soc_range, 3),
        }

    # -----------------------------------------------------
    # Forecast Evaluation
    # -----------------------------------------------------

    @staticmethod
    def forecast_metrics(dispatch_df: pd.DataFrame) -> dict:
        actual_col = next((c for c in ["actual_price", "price", "settlement_price"] if c in dispatch_df.columns), None)
        if not actual_col or "forecast_price" not in dispatch_df.columns:
            return {
                "forecast_mae": 0.0,
                "forecast_rmse": 0.0,
                "forecast_mape_pct": 0.0,
                "forecast_bias": 0.0,
                "forecast_r2": 0.0,
            }

        actual = np.asarray(dispatch_df[actual_col].values, dtype=float)
        forecast = np.asarray(dispatch_df["forecast_price"].values, dtype=float)

        return {
            "forecast_mae": round(BacktestMetrics.mae(actual, forecast), 3),
            "forecast_rmse": round(BacktestMetrics.rmse(actual, forecast), 3),
            "forecast_mape_pct": round(BacktestMetrics.mape(actual, forecast), 2),
            "forecast_bias": round(BacktestMetrics.bias(actual, forecast), 3),
            "forecast_r2": round(BacktestMetrics.r2(actual, forecast), 4),
        }

    # -----------------------------------------------------
    # Master Evaluation API
    # -----------------------------------------------------

    def evaluate(self, dispatch_df: pd.DataFrame, degradation_df: pd.DataFrame) -> BacktestMetricsResult:
        summary = {}
        summary.update(self.financial_metrics(dispatch_df, degradation_df))
        summary.update(self.battery_metrics(dispatch_df, degradation_df))
        summary.update(self.operational_metrics(dispatch_df))
        summary.update(self.forecast_metrics(dispatch_df))

        dataframe = pd.DataFrame(
            {
                "metric": list(summary.keys()),
                "value": list(summary.values()),
            }
        )

        return BacktestMetricsResult(summary, dataframe)