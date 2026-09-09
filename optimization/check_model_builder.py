"""
check_model_builder.py

Research-grade verification of model_builder.py
"""

import pyomo.environ as pyo
import pandas as pd

from optimization.model_builder import (
    BatteryOptimizationModelBuilder,
)

# -------------------------------------------------------
# Load recursive forecast
# -------------------------------------------------------

forecast = pd.read_csv(
    "forecasting/forecast_24h.csv"
)

builder = BatteryOptimizationModelBuilder()

model = builder.build(forecast)

summary = model.summary

# -------------------------------------------------------
# Report
# -------------------------------------------------------

print("=" * 70)
print("MODEL BUILDER CHECK")
print("=" * 70)

print(f"Optimization Horizon : {summary.horizon}")
print(f"Forecast Start       : {summary.forecast_start}")
print(f"Forecast End         : {summary.forecast_end}")

print("-" * 70)

print(f"Variables            : {summary.variables}")
print(f"Parameters           : {summary.parameters}")
print(f"Constraint Blocks    : {summary.constraint_blocks}")
print(f"Objective            : {summary.objective}")

print("-" * 70)

print("Variable Blocks")

for name, variable in model.component_map(
    pyo.Var,
    active=True,
).items():
    print(f"{name:<20} {len(variable)}")

print("-" * 70)

print("Constraint Blocks")

for name, constraint in model.component_map(
    pyo.Constraint,
    active=True,
).items():
    print(f"{name:<30} {len(constraint)}")

print("=" * 70)

builder.export_prices(model)

print("ConcreteModel Ready ✓")
print("Forecast parameters exported ✓")