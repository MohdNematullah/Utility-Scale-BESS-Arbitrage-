"""
results.py
==========

Research-grade optimization results processor for .

Extracts solved battery dispatch results and generates
publication-quality experiment outputs.

Author : Mohd Nematullah
Project:  Thesis
"""

from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
import pyomo.environ as pyo


# ============================================================
# Experiment Summary Dataclass
# ============================================================

@dataclass(slots=True)
class BatteryExperimentSummary:

    experiment_name: str
    market: str
    horizon_hours: int

    total_revenue: float
    total_charge_energy: float
    total_discharge_energy: float
    round_trip_efficiency: float

    soc_min: float
    soc_max: float
    soc_initial: float
    soc_terminal: float

    average_price: float
    maximum_price: float
    minimum_price: float


# ============================================================
# Results Processor
# ============================================================

class BatteryResultsProcessor:

    def __init__(
        self,
        output_directory="optimization/results",
    ):

        self.output_directory = Path(output_directory)

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # --------------------------------------------------------
    # Dispatch Schedule
    # --------------------------------------------------------

    def dispatch_schedule(self, model):

        forecast = model.forecast_dataframe.copy()

        dataframe = pd.DataFrame({

            "timestamp": forecast["timestamp"],

            "forecast_price": forecast["prediction"],

            "charge_power_mw": [
                pyo.value(model.charge_power[t])
                for t in model.T
            ],

            "discharge_power_mw": [
                pyo.value(model.discharge_power[t])
                for t in model.T
            ],

            "soc_mwh": [
                pyo.value(model.soc[t])
                for t in model.T
            ],
        })

        dataframe["net_power_mw"] = (
            dataframe["discharge_power_mw"]
            - dataframe["charge_power_mw"]
        )

        dataframe["hourly_revenue_$"] = (
            dataframe["forecast_price"]
            * dataframe["net_power_mw"]
        )

        dataframe["cumulative_revenue_$"] = (
            dataframe["hourly_revenue_$"].cumsum()
        )

        dataframe["energy_charged_mwh"] = dataframe["charge_power_mw"]
        dataframe["energy_discharged_mwh"] = dataframe["discharge_power_mw"]

        numeric_cols = dataframe.select_dtypes(include="number").columns
        dataframe[numeric_cols] = dataframe[numeric_cols].round(4)
        return dataframe

    # --------------------------------------------------------
    # Revenue Breakdown
    # --------------------------------------------------------

    def revenue_breakdown(self, dispatch):

        revenue = dispatch.copy()

        revenue["charging_cost_$"] = (
            revenue["forecast_price"]
            * revenue["charge_power_mw"]
        )

        revenue["discharging_income_$"] = (
            revenue["forecast_price"]
            * revenue["discharge_power_mw"]
        )

        revenue["net_profit_$"] = (
            revenue["discharging_income_$"]
            - revenue["charging_cost_$"]
        )

        numeric_cols = revenue.select_dtypes(include="number").columns
        revenue[numeric_cols] = revenue[numeric_cols].round(4)
        return revenue

    # --------------------------------------------------------
    # Battery Summary
    # --------------------------------------------------------

    def battery_summary(self, model, dispatch):

        battery = model

        summary = BatteryExperimentSummary(

            experiment_name="Rolling_Horizon_Battery_Arbitrage",

            market="ERCOT_DAM",

            horizon_hours=len(model.T),

            total_revenue=float(
                dispatch["hourly_revenue_$"].sum()
            ),

            total_charge_energy=float(
                dispatch["energy_charged_mwh"].sum()
            ),

            total_discharge_energy=float(
                dispatch["energy_discharged_mwh"].sum()
            ),

            round_trip_efficiency=float(
                pyo.value(battery.charge_efficiency)
                * pyo.value(battery.discharge_efficiency)
            ),

            soc_min=float(dispatch["soc_mwh"].min()),

            soc_max=float(dispatch["soc_mwh"].max()),

            soc_initial=float(dispatch["soc_mwh"].iloc[0]),

            soc_terminal=float(dispatch["soc_mwh"].iloc[-1]),

            average_price=float(dispatch["forecast_price"].mean()),

            maximum_price=float(dispatch["forecast_price"].max()),

            minimum_price=float(dispatch["forecast_price"].min()),
        )

        return summary

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    def experiment_metadata(self, model):

        dispatch = self.dispatch_schedule(model)

        metadata = {

            "forecast_start":
                str(dispatch.timestamp.min()),

            "forecast_end":
                str(dispatch.timestamp.max()),

            "rows":
                len(dispatch),

            "objective":
                "Energy Arbitrage",

            "battery_capacity_mwh":
                pyo.value(model.energy_capacity),

            "charge_efficiency":
                pyo.value(model.charge_efficiency),

            "discharge_efficiency":
                pyo.value(model.discharge_efficiency),
        }

        return pd.DataFrame([metadata])

    # --------------------------------------------------------
    # Export All Results
    # --------------------------------------------------------

    def export(self, model):

        dispatch = self.dispatch_schedule(model)

        revenue = self.revenue_breakdown(dispatch)

        summary = self.battery_summary(model, dispatch)

        metadata = self.experiment_metadata(model)

        dispatch.to_csv(
            self.output_directory / "dispatch_schedule.csv",
            index=False,
        )

        revenue.to_csv(
            self.output_directory / "revenue_breakdown.csv",
            index=False,
        )

        pd.DataFrame([asdict(summary)]).to_csv(
            self.output_directory / "battery_summary.csv",
            index=False,
        )

        metadata.to_csv(
            self.output_directory / "experiment_metadata.csv",
            index=False,
        )

        return summary