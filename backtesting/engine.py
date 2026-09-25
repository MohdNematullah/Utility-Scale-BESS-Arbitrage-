"""
backtesting/engine.py
=====================

Rolling Horizon Backtesting Engine
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyomo.environ as pyo

from backtesting.config import DEFAULT_BACKTEST_CONFIG
from forecasting.models import XGBoostForecaster
from forecasting.recursive_engine import RecursiveForecastEngine
from optimization.model_builder import BatteryOptimizationModelBuilder
from optimization.solver import BatteryOptimizationSolver
from optimization.results import BatteryResultsProcessor

from battery.ageing_engine import BatteryAgeingEngine
from battery.rainflow import RainflowCycleCounter


# ---------------------------------------------------------------------
# Result Dataclass
# ---------------------------------------------------------------------

@dataclass(slots=True)
class RollingBacktestResult:
    dispatch_history: pd.DataFrame
    degradation_history: pd.DataFrame
    summary: dict


# ---------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------

class RollingBacktestEngine:
    """
    rolling horizon battery arbitrage engine with strict
    energy continuity and wear co-optimization.
    """

    def __init__(
        self,
        forecaster: XGBoostForecaster,
        feature_dataframe: pd.DataFrame,
        config=DEFAULT_BACKTEST_CONFIG,
    ):
        self.config = config
        self.features = feature_dataframe.copy()

        if "timestamp" in self.features.columns:
            self.features["timestamp"] = pd.to_datetime(
                self.features["timestamp"],
                utc=True,
            )
            self.features = self.features.set_index("timestamp")

        if not isinstance(self.features.index, pd.DatetimeIndex):
            raise TypeError(
                "RollingBacktestEngine requires a pandas DatetimeIndex. "
                "Run FeatureEngineer.transform() before backtesting."
            )

        self.features = self.features.sort_index()

        self.forecast_engine = RecursiveForecastEngine(forecaster)
        self.builder = BatteryOptimizationModelBuilder()
        self.solver = BatteryOptimizationSolver()
        self.processor = BatteryResultsProcessor()

        self.ageing = BatteryAgeingEngine()
        self.rainflow = RainflowCycleCounter()

    # -----------------------------------------------------------------
    # Generate Forecast Window
    # -----------------------------------------------------------------

    def forecast_window(self, feature_window: pd.DataFrame) -> pd.DataFrame:
        feature_window = feature_window.copy()

        if "timestamp" in feature_window.columns:
            feature_window["timestamp"] = pd.to_datetime(
                feature_window["timestamp"],
                utc=True,
            )
            feature_window = feature_window.set_index("timestamp")

        feature_window = feature_window.sort_index()

        minimum_history = 168
        if len(feature_window) < minimum_history:
            raise ValueError(
                f"Rolling window has only {len(feature_window)} rows. "
                f"Recursive forecasting requires at least {minimum_history} rows."
            )

        return self.forecast_engine.forecast(
            feature_dataframe=feature_window,
            horizon=self.config.forecast_horizon_hours,
        )

    # -----------------------------------------------------------------
    # Solve Optimization Window
    # -----------------------------------------------------------------

    def optimize_window(self, forecast_df: pd.DataFrame, current_soc_mwh: float):
        model = self.builder.build(forecast_df, initial_soc=current_soc_mwh)

        # Calibrated marginal wear hurdle ($10/MWh power = $20/MWh roundtrip hurdle)
        marginal_wear_cost = 10.0

        for obj in model.component_objects(pyo.Objective, active=True):
            obj.deactivate()

        def profit_objective(m):
            return sum(
                (m.discharge_power[t] - m.charge_power[t]) * m.price[t]
                - (m.discharge_power[t] + m.charge_power[t]) * marginal_wear_cost
                for t in m.T
            )

        model.degradation_aware_obj = pyo.Objective(rule=profit_objective, sense=pyo.maximize)

        solver_summary = self.solver.solve(model)
        dispatch = self.processor.dispatch_schedule(model)
        battery_summary = self.processor.battery_summary(model, dispatch)
        return dispatch, battery_summary, solver_summary

    # -----------------------------------------------------------------
    # Execute First Implementation Horizon
    # -----------------------------------------------------------------

    def execute_window(self, dispatch: pd.DataFrame) -> pd.DataFrame:
        return dispatch.iloc[: self.config.implementation_horizon_hours].copy()

    # -----------------------------------------------------------------
    # Main Backtest Loop
    # -----------------------------------------------------------------

    def run(self) -> RollingBacktestResult:
        dispatch_frames = []
        degradation_frames = []

        soh = self.config.initial_soh
        total_revenue = 0.0
        total_deg_cost = 0.0

        cumulative_charged = 0.0
        cumulative_discharged = 0.0

        forecast_horizon = self.config.forecast_horizon_hours
        implementation_horizon = self.config.implementation_horizon_hours
        rolling_step = self.config.rolling_step_hours

        history_hours = 168
        last_start = len(self.features) - forecast_horizon

        nominal_capacity = self.ageing.config.chemistry.nominal_capacity_mwh
        min_soc_mwh = float(self.ageing.config.chemistry.minimum_soc_fraction * nominal_capacity)
        max_soc_mwh = float(self.ageing.config.chemistry.maximum_soc_fraction * nominal_capacity)

        # Initialize battery state of charge from configuration
        current_soc_mwh = nominal_capacity * self.ageing.config.chemistry.initial_soc_fraction
        current_soc_mwh = max(min_soc_mwh, min(max_soc_mwh, current_soc_mwh))

        for start in range(history_hours, last_start + 1, rolling_step):
            history_start = max(0, start - history_hours)
            window = self.features.iloc[history_start:start].copy()

            if len(window) < history_hours:
                continue

            # 1. Forecast & Optimize with strict inter-window SOC continuity
            forecast = self.forecast_window(window)
            dispatch, _, _ = self.optimize_window(forecast, current_soc_mwh=current_soc_mwh)
            executed = self.execute_window(dispatch).copy()

            # Carry terminal SOC forward to become the initial SOC of the next window
            if "soc_mwh" in executed.columns:
                current_soc_mwh = float(executed["soc_mwh"].iloc[-1])
            elif "soc" in executed.columns:
                current_soc_mwh = float(executed["soc"].iloc[-1])

            current_soc_mwh = max(min_soc_mwh, min(max_soc_mwh, current_soc_mwh))

            if "soc_mwh" in executed.columns:
                executed["soc_fraction"] = (
                    executed["soc_mwh"] / nominal_capacity
                ).clip(0.0, 1.0)
            elif "soc" in executed.columns:
                executed["soc_mwh"] = executed["soc"]
                executed["soc_fraction"] = (
                    executed["soc_mwh"] / nominal_capacity
                ).clip(0.0, 1.0)

            # 2. Extract Ground-Truth Actual Settlement Price
            actual_window = self.features.iloc[start : start + implementation_horizon]
            for p_col in ["actual_price", "price", "settlement_price", "system_lambda"]:
                if p_col in actual_window.columns:
                    executed["actual_price"] = actual_window[p_col].iloc[: len(executed)].values
                    break

            # 3. Calculate realised revenue from actual prices.
            required = [
                "actual_price",
                "charge_power_mw",
                "discharge_power_mw",
            ]
            missing = [
                column for column in required
                if column not in executed.columns
            ]
            if missing:
                raise ValueError(f"Missing settlement columns: {missing}")
            if executed[required].isna().any().any():
                raise ValueError("Settlement data contains missing values.")

            # The current optimiser uses one-hour intervals.
            dt_hours = 1.0
            net_energy_mwh = (
                executed["discharge_power_mw"]
                - executed["charge_power_mw"]
            ) * dt_hours

            # Cash flow before battery ageing costs.
            executed["net_revenue_usd"] = (
                executed["actual_price"] * net_energy_mwh
            )

            # Keep the existing report columns consistent.
            executed["net_revenue_$"] = executed["net_revenue_usd"]
            executed["hourly_revenue_$"] = executed["net_revenue_usd"]
            executed["cumulative_revenue_$"] = (
                total_revenue + executed["net_revenue_usd"].cumsum()
            )
            window_revenue = float(executed["net_revenue_usd"].sum())
            total_revenue += window_revenue

            # 4. Incremental Window Accounting
            window_days = implementation_horizon / 24.0

            if "energy_charged_mwh" in executed.columns:
                window_charge = float(executed["energy_charged_mwh"].sum())
                window_discharge = float(executed["energy_discharged_mwh"].sum())
            else:
                window_charge = float(executed["charge_power_mw"].sum())
                window_discharge = float(executed["discharge_power_mw"].sum())

            cumulative_charged += window_charge
            cumulative_discharged += window_discharge

            window_throughput_mwh = window_charge + window_discharge
            window_efc = window_throughput_mwh / (2.0 * nominal_capacity)
            cumulative_efc = (cumulative_charged + cumulative_discharged) / (2.0 * nominal_capacity)

            soc_series = executed["soc_fraction"].tolist()
            window_avg_soc = float(executed["soc_fraction"].mean())
            window_avg_dod = (
                float(max(soc_series) - min(soc_series))
                if len(soc_series) > 1
                else 0.0
            )

            # Pre-degradation state
            soh_start = soh
            capacity_start_mwh = soh_start * nominal_capacity

            # 5. Incremental Ageing Evaluation (Window values only)
            ageing_result = self.ageing.evaluate(
                initial_soh=soh_start,
                calendar_days=window_days,
                average_soc=window_avg_soc,
                temperature_c=self.config.reference_temperature_c,
                charged_energy_mwh=window_charge,
                discharged_energy_mwh=window_discharge,
                average_dod=window_avg_dod,
                average_c_rate=self.config.default_c_rate,
            )

            soh_end = ageing_result.remaining_soh
            capacity_end_mwh = ageing_result.remaining_capacity_mwh
            window_loss = max(soh_start - soh_end, 0.0)

            window_deg_cost = ageing_result.degradation_cost_usd
            total_deg_cost += window_deg_cost

            window_gross_revenue_usd = window_revenue
            window_net_revenue_usd = window_revenue - window_deg_cost

            # Advance SOH monotonically
            soh = soh_end

            window_number = start // rolling_step + 1
            window_origin = window.index[-1]

            executed["rolling_window"] = window_number
            executed["forecast_origin"] = window_origin
            executed["remaining_soh"] = soh
            dispatch_frames.append(executed)

            degradation_frames.append(
                pd.DataFrame(
                    {
                        "rolling_window": [window_number],
                        "forecast_origin": [window_origin],
                        "soh_start": [round(soh_start, 6)],
                        "soh_end": [round(soh_end, 6)],
                        "capacity_start_mwh": [round(capacity_start_mwh, 4)],
                        "capacity_end_mwh": [round(capacity_end_mwh, 4)],
                        "calendar_loss": [ageing_result.calendar_loss_fraction],
                        "cycle_loss": [ageing_result.cycle_loss_fraction],
                        "window_loss": [window_loss],
                        "window_throughput_mwh": [round(window_throughput_mwh, 4)],
                        "window_efc": [round(window_efc, 4)],
                        "cumulative_efc": [round(cumulative_efc, 4)],
                        "window_gross_revenue_usd": [round(window_gross_revenue_usd, 2)],
                        "degradation_cost_usd": [round(window_deg_cost, 2)],
                        "window_net_revenue_usd": [round(window_net_revenue_usd, 2)],
                        "remaining_soh": [soh_end],
                        "remaining_capacity_mwh": [capacity_end_mwh],
                    }
                )
            )

        if not dispatch_frames:
            raise RuntimeError("No rolling windows were executed.")

        dispatch_history = pd.concat(dispatch_frames, ignore_index=True)
        degradation_history = pd.concat(degradation_frames, ignore_index=True)

        summary = {
            "rolling_windows": len(degradation_history),
            "hours_executed": len(dispatch_history),
            "forecast_horizon_hours": forecast_horizon,
            "implementation_horizon_hours": implementation_horizon,
            "rolling_step_hours": rolling_step,
            "gross_revenue_usd": round(total_revenue, 2),
            "degradation_cost_usd": round(total_deg_cost, 2),
            "net_revenue_usd": round(total_revenue - total_deg_cost, 2),
            "initial_soh": self.config.initial_soh,
            "final_soh": round(soh, 4),
            "capacity_remaining_mwh": round(
                degradation_history.iloc[-1]["capacity_end_mwh"], 3
            ),
            "equivalent_full_cycles": round(cumulative_efc, 3),
        }

        return RollingBacktestResult(
            dispatch_history=dispatch_history,
            degradation_history=degradation_history,
            summary=summary,
        )

    # -----------------------------------------------------------------
    # Export Results
    # -----------------------------------------------------------------

    def export(self, result: RollingBacktestResult):
        out_dir = self.config.export_directory
        out_dir.mkdir(parents=True, exist_ok=True)

        dispatch_file = out_dir / "dispatch_history.csv"
        degradation_file = out_dir / "degradation_history.csv"
        summary_file = out_dir / "rolling_summary.csv"

        result.dispatch_history.reset_index().to_csv(dispatch_file, index=False)
        result.degradation_history.reset_index(drop=True).to_csv(degradation_file, index=False)
        pd.DataFrame([result.summary]).to_csv(summary_file, index=False)

        return {
            "dispatch": dispatch_file,
            "degradation": degradation_file,
            "summary": summary_file,
        }