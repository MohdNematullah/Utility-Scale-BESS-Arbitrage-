"""
optimization/check_objective.py
===============================

Verification script for mathematical objective function attachment.
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pyomo.environ as pyo
from pyomo.core.base.set_types import NonNegativeReals

from optimization.objective import attach_objective


model = pyo.ConcreteModel(name="BESS_Arbitrage_Verification")

# 24-Hour Optimization Horizon
model.T = pyo.RangeSet(0, 23)
model.delta_t = pyo.Param(initialize=1.0)

prices = {t: 30.0 + float(t) for t in range(24)}

model.price = pyo.Param(
    model.T,
    initialize=prices,
)

model.forecast_price = pyo.Param(
    model.T,
    initialize=prices,
)

model.degradation_cost = pyo.Param(
    model.T,
    initialize={t: 0.0 for t in range(24)},
)

model.charge_power = pyo.Var(
    model.T,
    domain=NonNegativeReals,
)

model.discharge_power = pyo.Var(
    model.T,
    domain=NonNegativeReals,
)

# Attach arbitrage objective expression
attach_objective(model, objective_name="energy_arbitrage")

print("=" * 60)
print("OBJECTIVE CREATED SUCCESSFULLY")
print("=" * 60)
print(model.objective.expr)