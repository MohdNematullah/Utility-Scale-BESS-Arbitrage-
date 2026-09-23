"""
optimization/model_builder.py
=============================

optimization model builder for .

Creates a Pyomo ConcreteModel for battery energy arbitrage using
recursive multi-step electricity price forecasts.

Compatible with:
    â€¢ Python 3.14
    â€¢ Pyomo 6.10+
    â€¢  Config (energy_capacity_mwh, charge_power_mw, etc.)
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyomo.environ as pyo

from optimization.config import DEFAULT_CONFIG, OptimizationConfig
from optimization.constraints import attach_constraints
from optimization.objective import attach_objective


# =====================================================================
# MODEL SUMMARY
# =====================================================================

@dataclass(slots=True)
class ModelSummary:
    horizon: int
    variables: int
    parameters: int
    constraint_blocks: int
    objective: str
    forecast_start: str
    forecast_end: str


# =====================================================================
# BATTERY OPTIMIZATION MODEL BUILDER
# =====================================================================

class BatteryOptimizationModelBuilder:
    """
    Creates a complete Pyomo ConcreteModel for rolling-horizon
    battery arbitrage optimization with dynamic inter-window SOC continuity.
    """

    def __init__(
        self,
        config: OptimizationConfig = DEFAULT_CONFIG,
    ):
        self.config = config

    # -----------------------------------------------------------------
    # Optimization Horizon
    # -----------------------------------------------------------------

    @property
    def horizon(self) -> int:
        """
        Returns optimization horizon from ExperimentConfig.
        """

        experiment = self.config.experiment

        if hasattr(experiment, "rolling_window_hours"):
            return int(experiment.rolling_window_hours)

        if hasattr(experiment, "rolling_window"):
            return int(experiment.rolling_window)

        return 24

    # -----------------------------------------------------------------
    # Forecast Validation
    # -----------------------------------------------------------------

    def validate_forecast(self, forecast) -> pd.DataFrame:
        """
        Accept either:
        - pandas DataFrame
        - ForecastHorizon returned by RecursiveForecastEngine
        """

        # --------------------------------------------------
        # ForecastHorizon â†’ DataFrame
        # --------------------------------------------------
        if hasattr(forecast, "dataframe"):
            forecast = forecast.dataframe.copy()

        if not isinstance(forecast, pd.DataFrame):
            raise TypeError(
                "forecast must be a pandas DataFrame or ForecastHorizon."
            )

        forecast = forecast.copy()

        # Handle column naming flexibility
        if "prediction" not in forecast.columns:
            if "forecast_price" in forecast.columns:
                forecast["prediction"] = forecast["forecast_price"]
            elif "price" in forecast.columns:
                forecast["prediction"] = forecast["price"]

        required_columns = {"timestamp", "prediction"}

        missing = required_columns.difference(forecast.columns)

        if missing:
            raise ValueError(
                f"Forecast missing required columns: {sorted(missing)}"
            )

        forecast["timestamp"] = pd.to_datetime(
            forecast["timestamp"],
            utc=True,
        )

        forecast = (
            forecast.sort_values("timestamp")
            .reset_index(drop=True)
        )

        horizon = self.horizon

        if len(forecast) < horizon:
            raise ValueError(
                f"Forecast contains {len(forecast)} rows; "
                f"required {horizon}."
            )

        return forecast.iloc[:horizon].copy()

    # -----------------------------------------------------------------
    # Build ConcreteModel
    # -----------------------------------------------------------------

    def build(
        self,
        forecast: pd.DataFrame,
        initial_soc: float | None = None,
    ) -> pyo.ConcreteModel:
        """
        Constructs the Pyomo model for battery arbitrage dispatch.

        Parameters
        ----------
        forecast : pd.DataFrame
            Forecasted market prices.
        initial_soc : float | None, optional
            Initial battery state of charge in MWh. If None, falls back to
            the default initial SOC fraction defined in config.
        """

        forecast = self.validate_forecast(forecast)

        battery = self.config.battery

        model = pyo.ConcreteModel(
            name="BTA_V5_Battery_Arbitrage_Model"
        )

        # =============================================================
        # TIME SET
        # =============================================================

        model.T = pyo.RangeSet(
            0,
            self.horizon - 1,
        )

        model.delta_t = pyo.Param(
            initialize=1,
        )

        # =============================================================
        # FORECAST PRICE PARAMETERS
        # =============================================================

        price_dict = {
            hour: float(price)
            for hour, price in enumerate(
                forecast["prediction"]
            )
        }

        model.price = pyo.Param(
            model.T,
            initialize=price_dict,
            mutable=True,
        )

        model.forecast_price = pyo.Param(
            model.T,
            initialize=price_dict,
            mutable=True,
        )

        # =============================================================
        # BATTERY PARAMETERS
        # =============================================================

        model.energy_capacity = pyo.Param(
            initialize=float(
                battery.energy_capacity_mwh
            ),
            mutable=True,
        )

        model.max_charge_power = pyo.Param(
            initialize=float(
                battery.max_charge_power_mw
            ),
            mutable=True,
        )

        model.max_discharge_power = pyo.Param(
            initialize=float(
                battery.max_discharge_power_mw
            ),
            mutable=True,
        )

        model.charge_efficiency = pyo.Param(
            initialize=float(
                battery.charge_efficiency
            ),
            mutable=True,
        )

        model.discharge_efficiency = pyo.Param(
            initialize=float(
                battery.discharge_efficiency
            ),
            mutable=True,
        )

        # =============================================================
        # SOC PARAMETERS (Convert fractions â†’ MWh)
        # =============================================================

        capacity = float(battery.energy_capacity_mwh)

        model.soc_min = pyo.Param(
            initialize=capacity * float(battery.soc_min_fraction),
            mutable=True,
        )

        model.soc_max = pyo.Param(
            initialize=capacity * float(battery.soc_max_fraction),
            mutable=True,
        )

        # Resolve initial SOC dynamically for inter-window continuity
        if initial_soc is not None:
            init_soc_val = float(initial_soc)
        else:
            init_soc_val = capacity * float(battery.initial_soc_fraction)

        model.initial_soc = pyo.Param(
            initialize=init_soc_val,
            mutable=True,
        )

        model.terminal_soc = pyo.Param(
            initialize=capacity * float(battery.terminal_soc_fraction),
            mutable=True,
        )

        # =============================================================
        # PART 7 PLACEHOLDERS (BATTERY AGEING)
        # =============================================================

        model.remaining_capacity_fraction = pyo.Param(
            initialize=1.0,
            mutable=True,
        )

        model.degradation_cost = pyo.Param(
            model.T,
            initialize={
                t: 0.0
                for t in range(self.horizon)
            },
            mutable=True,
        )

        # =============================================================
        # DECISION VARIABLES
        # =============================================================

        model.soc = pyo.Var(
            model.T,
            domain=pyo.NonNegativeReals,
        )

        model.charge_power = pyo.Var(
            model.T,
            domain=pyo.NonNegativeReals,
        )

        model.discharge_power = pyo.Var(
            model.T,
            domain=pyo.NonNegativeReals,
        )

        # =============================================================
        # CONSTRAINTS
        # =============================================================

        attach_constraints(model)

        # =============================================================
        # OBJECTIVE FUNCTION
        # =============================================================

        attach_objective(
            model,
            objective_name="energy_arbitrage",
        )

        # =============================================================
        # METADATA
        # =============================================================

        model.forecast_dataframe = forecast

        model.summary = self.summary(model)

        return model

    # -----------------------------------------------------------------
    # MODEL SUMMARY
    # -----------------------------------------------------------------

    def summary(
        self,
        model: pyo.ConcreteModel,
    ) -> ModelSummary:
        """
        Collect optimization model statistics.
        """

        variable_count = sum(
            len(component)
            for component in model.component_map(
                pyo.Var,
                active=True,
            ).values()
        )

        parameter_count = sum(
            len(component)
            if component.is_indexed()
            else 1
            for component in model.component_map(
                pyo.Param,
                active=True,
            ).values()
        )

        constraint_count = len(
            model.component_map(
                pyo.Constraint,
                active=True,
            )
        )

        objective_name = next(
            iter(
                model.component_map(
                    pyo.Objective,
                    active=True,
                )
            )
        )

        return ModelSummary(
            horizon=self.horizon,
            variables=variable_count,
            parameters=parameter_count,
            constraint_blocks=constraint_count,
            objective=str(objective_name),
            forecast_start=str(
                model.forecast_dataframe["timestamp"].min()
            ),
            forecast_end=str(
                model.forecast_dataframe["timestamp"].max()
            ),
        )

    # -----------------------------------------------------------------
    # EXPORT FORECAST PRICES
    # -----------------------------------------------------------------

    def export_prices(
        self,
        model: pyo.ConcreteModel,
        path: str = "optimization/model_prices.csv",
    ) -> str:
        """
        Export optimization forecast prices.
        """

        output_path = Path(path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        model.forecast_dataframe.to_csv(
            output_path,
            index=False,
        )

        return str(output_path)