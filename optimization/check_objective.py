
"""
check_objective.py

Verifies objective attachment.
"""

from pyomo.environ import (
    ConcreteModel,
    RangeSet,
    Param,
    Var,
    NonNegativeReals,
)

from optimization.objective import attach_objective


model = ConcreteModel()

model.T = RangeSet(0, 23)

model.delta_t = Param(initialize=1)

prices = {
    t: 30 + t
    for t in range(24)
}

model.price = Param(
    model.T,
    initialize=prices,
)

model.forecast_price = Param(
    model.T,
    initialize=prices,
)

model.degradation_cost = Param(
    model.T,
    initialize={t: 0 for t in range(24)},
)

model.charge_power = Var(
    model.T,
    domain=NonNegativeReals,
)

model.discharge_power = Var(
    model.T,
    domain=NonNegativeReals,
)

attach_objective(model)

print("=" * 60)
print("OBJECTIVE CREATED SUCCESSFULLY")
print("=" * 60)
print(model.objective.expr)