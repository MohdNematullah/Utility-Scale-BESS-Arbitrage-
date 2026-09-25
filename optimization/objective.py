"""
optimization/objective.py
=========================

Objective functions for BESS arbitrage optimization.

This module defines optimization objectives separately from the model builder:
1. Forecast-based arbitrage (energy_arbitrage / forecast_value)
2. Perfect foresight arbitrage
3. Forecast + electrochemical degradation penalty (degradation_penalty)
4. Multi-market arbitrage
"""

from __future__ import annotations

import pyomo.environ as pyo
from pyomo.environ import Objective, maximize


# ============================================================
# Energy Arbitrage Objective
# ============================================================

def energy_arbitrage_objective(model: pyo.ConcreteModel) -> pyo.Objective:
    """
    Maximize arbitrage profit over the optimization horizon.

    Profit = sum(price[t] * (discharge_power[t] - charge_power[t]) * delta_t)
    """
    dt = model.delta_t

    return Objective(
        expr=sum(
            model.price[t]
            * (model.discharge_power[t] - model.charge_power[t])
            * dt
            for t in model.T
        ),
        sense=maximize,
    )


# ============================================================
# Forecast Value Objective
# ============================================================

def forecast_value_objective(model: pyo.ConcreteModel) -> pyo.Objective:
    """
    Arbitrage objective formulated over look-ahead forecast prices.

    Profit = sum(forecast_price[t] * (discharge_power[t] - charge_power[t]) * delta_t)
    """
    dt = model.delta_t

    return Objective(
        expr=sum(
            model.forecast_price[t]
            * (model.discharge_power[t] - model.charge_power[t])
            * dt
            for t in model.T
        ),
        sense=maximize,
    )


# ============================================================
# Arbitrage + Degradation Cost Penalty
# ============================================================

def degradation_penalty_objective(model: pyo.ConcreteModel) -> pyo.Objective:
    """
    Net arbitrage profit penalizing dynamic battery cell wear.

    Objective = Revenue - Cell Degradation Cost
    """
    dt = model.delta_t

    revenue = sum(
        model.price[t]
        * (model.discharge_power[t] - model.charge_power[t])
        * dt
        for t in model.T
    )

    degradation_cost = sum(
        model.degradation_cost[t]
        for t in model.T
    )

    return Objective(
        expr=revenue - degradation_cost,
        sense=maximize,
    )


# ============================================================
# Objective Factory
# ============================================================

OBJECTIVES = {
    # Standard snake_case identifiers
    "energy_arbitrage": energy_arbitrage_objective,
    "forecast_value": forecast_value_objective,
    "degradation_penalty": degradation_penalty_objective,
    # Backward compatibility mappings
    "energy_arbitrage": energy_arbitrage_objective,
    "forecast_value": forecast_value_objective,
    "degradation_penalty": degradation_penalty_objective,
}


def attach_objective(
    model: pyo.ConcreteModel,
    objective_name: str = "energy_arbitrage",
) -> pyo.ConcreteModel:
    """
    Attaches selected objective expression to the Pyomo concrete model.
    """
    if objective_name not in OBJECTIVES:
        raise ValueError(
            f"Unknown objective '{objective_name}'. Supported: {list(OBJECTIVES.keys())}"
        )

    model.objective = OBJECTIVES[objective_name](model)
    return model