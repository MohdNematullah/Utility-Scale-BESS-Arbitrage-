"""
battery/check_config.py

Verification script for Part 7.1 battery configuration.
"""

from battery.config import DEFAULT_BATTERY_CONFIG

cfg = DEFAULT_BATTERY_CONFIG

print("=" * 70)
print("BATTERY AGEING CONFIGURATION")
print("=" * 70)

print("Chemistry")
print("-" * 70)
print("Type                 :", cfg.chemistry.chemistry)
print("Capacity             :", cfg.chemistry.nominal_capacity_mwh, "MWh")
print("Voltage              :", cfg.chemistry.nominal_voltage_v, "V")
print("Nominal Cycles       :", cfg.chemistry.nominal_cycles)
print("Round-trip Efficiency:", cfg.chemistry.nominal_round_trip_efficiency)

print("\nAgeing Parameters")
print("-" * 70)
print("Calendar Loss / Year :", cfg.ageing.calendar_loss_per_year)
print("Cycle Loss / EFC     :", cfg.ageing.cycle_loss_per_equivalent_cycle)
print("Reference DoD        :", cfg.ageing.reference_depth_of_discharge)
print("Minimum SOH          :", cfg.ageing.minimum_soh)
print("End-of-Life SOH      :", cfg.ageing.end_of_life_soh)

print("\nReplacement Parameters")
print("-" * 70)
print("Replacement Cost/MWh :", cfg.replacement.replacement_cost_per_mwh)
print("Replacement Trigger  :", cfg.replacement.replacement_trigger_soh)
print("Salvage Fraction     :", cfg.replacement.salvage_fraction)

cfg.results_directory.mkdir(parents=True, exist_ok=True)

print("\nResults Directory")
print("-" * 70)
print(cfg.results_directory)

print("=" * 70)
print("Battery configuration verified successfully âœ“")
print("=" * 70)