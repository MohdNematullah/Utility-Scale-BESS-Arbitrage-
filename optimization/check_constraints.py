"""
check_constraints.py
====================
verification script for battery optimization constraints.
Actively validates constraint attachment and end-of-interval energy balance dynamics.
"""

import pyomo.environ as pyo
from optimization.constraints import attach_constraints

# ----------------------------------------------------------
# Dummy optimization model setup
# ----------------------------------------------------------
model = pyo.ConcreteModel(name="Constraint_Check_Model")

model.T = pyo.RangeSet(0, 23)
model.delta_t = pyo.Param(initialize=1.0)

model.initial_soc = pyo.Param(initialize=50.0)
model.terminal_soc = pyo.Param(initialize=50.0)

model.soc_min = pyo.Param(initialize=10.0)
model.soc_max = pyo.Param(initialize=90.0)

model.max_charge_power = pyo.Param(initialize=50.0)
model.max_discharge_power = pyo.Param(initialize=50.0)

model.charge_efficiency = pyo.Param(initialize=0.95)
model.discharge_efficiency = pyo.Param(initialize=0.95)
model.remaining_capacity_fraction = pyo.Param(initialize=1.0)

# ----------------------------------------------------------
# Decision variables
# ----------------------------------------------------------
model.soc = pyo.Var(model.T, domain=pyo.NonNegativeReals)
model.charge_power = pyo.Var(model.T, domain=pyo.NonNegativeReals)
model.discharge_power = pyo.Var(model.T, domain=pyo.NonNegativeReals)

# ----------------------------------------------------------
# Attach constraints
# ----------------------------------------------------------
attach_constraints(model)

# ----------------------------------------------------------
# Active Component Verification
# ----------------------------------------------------------
print("=" * 65)
print("BATTERY CONSTRAINT VERIFICATION")
print("=" * 65)

expected_constraints = {
    "Initial SOC Constraint": "initial_soc_constraint",
    "SOC Dynamics Constraint": "soc_dynamics_constraint",
    "SOC Minimum Constraint": "soc_min_constraint",
    "SOC Maximum Constraint": "soc_max_constraint",
    "Charge Limit Constraint": "charge_limit_constraint",
    "Discharge Limit Constraint": "discharge_limit_constraint",
    "Terminal SOC Constraint": "terminal_soc_constraint",
    "Operation Constraint": "operation_constraint",
}

all_passed = True
for label, attr_name in expected_constraints.items():
    component = getattr(model, attr_name, None)
    if component is not None and isinstance(component, pyo.Constraint) and component.active:
        print(f"{label:<32} âœ“ (active)")
    else:
        print(f"{label:<32} âœ— (MISSING OR INACTIVE)")
        all_passed = False

# ----------------------------------------------------------
# Verify Mentor's End-of-Hour Dynamics Formulation
# ----------------------------------------------------------
print("-" * 65)
print("MATHEMATICAL FORMULATION AUDIT")
print("-" * 65)

first_t = model.T.first()
second_t = first_t + 1

# 1. Verify initial_soc_constraint includes hour 0 dispatch
init_expr_str = str(model.initial_soc_constraint.expr)
hour_0_included = "charge_power[0]" in init_expr_str and "initial_soc" in init_expr_str
print(f"End-of-Hour-0 Energy Accounting    : {'âœ“ PASS' if hour_0_included else 'âœ— FAIL'}")

# 2. Verify soc_dynamics connects hour 1 to hour 0
dyn_expr_str = str(model.soc_dynamics_constraint[second_t].expr)
inter_hour_linked = "soc[0]" in dyn_expr_str and "charge_power[1]" in dyn_expr_str
print(f"Inter-Hour Continuity (t-1 -> t)  : {'âœ“ PASS' if inter_hour_linked else 'âœ— FAIL'}")

print("=" * 65)
active_blocks = len(model.component_map(pyo.Constraint, active=True))
print(f"Total Active Constraint Blocks     : {active_blocks} / {len(expected_constraints)}")

if all_passed and hour_0_included and inter_hour_linked:
    print("STATUS: ALL CONSTRAINTS RIGOROUSLY VERIFIED âœ“")
else:
    raise AssertionError("Constraint verification failed.")
print("=" * 65)