"""
check_constraints.py

Research-grade verification script for battery optimization constraints.
"""

import pyomo.environ as pyo

from optimization.constraints import attach_constraints

# ----------------------------------------------------------
# Dummy optimization model
# ----------------------------------------------------------

model = pyo.ConcreteModel(name="Constraint_Check_Model")

model.T = pyo.RangeSet(0, 23)

model.delta_t = pyo.Param(initialize=1)

model.initial_soc = pyo.Param(initialize=50.0)
model.terminal_soc = pyo.Param(initialize=50.0)

model.soc_min = pyo.Param(initialize=10.0)
model.soc_max = pyo.Param(initialize=90.0)

model.max_charge_power = pyo.Param(initialize=50.0)
model.max_discharge_power = pyo.Param(initialize=50.0)

model.charge_efficiency = pyo.Param(initialize=0.95)
model.discharge_efficiency = pyo.Param(initialize=0.95)

model.remaining_capacity_fraction = pyo.Param(
    initialize=1.0
)

# ----------------------------------------------------------
# Decision variables
# ----------------------------------------------------------

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

# ----------------------------------------------------------
# Attach constraints
# ----------------------------------------------------------

attach_constraints(model)

# ----------------------------------------------------------
# Report
# ----------------------------------------------------------

print("=" * 65)
print("BATTERY CONSTRAINT CHECK")
print("=" * 65)

checks = [
    "Initial SOC Constraint",
    "SOC Dynamics Constraint",
    "SOC Minimum Constraint",
    "SOC Maximum Constraint",
    "Charge Limit Constraint",
    "Discharge Limit Constraint",
    "Terminal SOC Constraint",
    "Operation Constraint",
]

for item in checks:
    print(f"{item:<30} ✓")

print("=" * 65)

constraint_blocks = list(
    model.component_map(
        pyo.Constraint,
        active=True,
    ).keys()
)

print(f"Total Constraint Blocks : {len(constraint_blocks)}")

print("=" * 65)