
"""
check_config.py

Checks optimization configuration.
"""

from optimization.config import DEFAULT_CONFIG

config = DEFAULT_CONFIG

config.validate()

print("=" * 70)
print(" OPTIMIZATION CONFIGURATION")
print("=" * 70)

print("\nBattery")

battery = config.battery

print(f"Battery Name          : {battery.battery_name}")
print(f"Energy Capacity       : {battery.energy_capacity_mwh} MWh")
print(f"Charge Power          : {battery.max_charge_power_mw} MW")
print(f"Discharge Power       : {battery.max_discharge_power_mw} MW")
print(f"Charge Efficiency     : {battery.charge_efficiency}")
print(f"Discharge Efficiency  : {battery.discharge_efficiency}")
print(f"Round-trip Efficiency : {battery.round_trip_efficiency:.4f}")

print(f"SOC Minimum           : {battery.soc_min_mwh:.2f} MWh")
print(f"SOC Maximum           : {battery.soc_max_mwh:.2f} MWh")
print(f"Initial SOC           : {battery.initial_soc_mwh:.2f} MWh")
print(f"Terminal SOC          : {battery.terminal_soc_mwh:.2f} MWh")

print("\nForecast")

forecast = config.forecast

print(f"Forecast Horizon      : {forecast.forecast_horizon_hours} hours")
print(f"Recursive Forecast    : {forecast.recursive}")
print(f"Forecast Model        : {forecast.forecast_model}")

print("\nSolver")

solver = config.solver

print(f"Solver                : {solver.solver_name}")
print(f"MIP Gap               : {solver.mip_gap}")
print(f"Time Limit            : {solver.time_limit_seconds} seconds")

print("\nExperiment")

experiment = config.experiment

print(f"Experiment            : {experiment.experiment_name}")
print(f"Market                : {experiment.market_name}")
print(f"Rolling Window        : {experiment.rolling_window_hours} hours")

output = config.export()

print("\nConfiguration exported:")
print(output)

print("=" * 70)