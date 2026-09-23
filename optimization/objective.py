
"""
objective.py
============

objective functions for  battery arbitrage optimization.

This module defines the optimization objective separately from the model
builder so future experiments can compare:

1. Forecast-based arbitrage.
2. Perfect foresight arbitrage.
3. Forecast + degradation cost.
4. Multi-market arbitrage.

References:
- Rolling-horizon BESS arbitrage LP formulation.
- Forecast-aware battery scheduling literature.
"""

from pyomo.environ import Objective, maximize


# ============================================================
# Energy Arbitrage Objective
# ============================================================

def energy_arbitrage_objective(model):
    """
    Maximize arbitrage profit over the optimization horizon.

    Profit = Î£ price Ã— (discharge - charge) Ã— Î”t

    Forecast prices are supplied from the recursive forecasting module.
    """

    dt = model.delta_t

    return Objective(
        expr=sum(
            model.price[t]
            * (
                model.discharge_power[t]
                - model.charge_power[t]
            )
            * dt
            for t in model.T
        ),
        sense=maximize,
    )


# ============================================================
# Forecast Value Objective
# ============================================================

def forecast_value_objective(model):
    """
    Same arbitrage objective but separated for future experiments.

    Used when comparing:
    - Perfect foresight.
    - Forecast-based optimization.
    """

    dt = model.delta_t

    return Objective(
        expr=sum(
            model.forecast_price[t]
            * (
                model.discharge_power[t]
                - model.charge_power[t]
            )
            * dt
            for t in model.T
        ),
        sense=maximize,
    )


# ============================================================
# Arbitrage + Degradation Placeholder
# ============================================================

def degradation_penalty_objective(model):
    """
    Placeholder objective for Part 7.

    Revenue
      - degradation cost.
    """

    dt = model.delta_t

    revenue = sum(
        model.price[t]
        * (
            model.discharge_power[t]
            - model.charge_power[t]
        )
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
    "energy_arbitrage": energy_arbitrage_objective,
    "forecast_value": forecast_value_objective,
    "degradation_penalty": degradation_penalty_objective,
}


def attach_objective(
    model,
    objective_name="energy_arbitrage",
):
    """
    Attach one objective to the model.
    """

    if objective_name not in OBJECTIVES:
        raise ValueError(
            f"Unknown objective: {objective_name}"
        )

    model.objective = OBJECTIVES[
        objective_name
    ](model)

    return model