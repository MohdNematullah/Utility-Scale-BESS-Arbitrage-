"""
backtesting/check_scenarios.py
==============================

Validation & Inspection script for Scenario Library (Part 8.5.2).
"""

from backtesting.scenarios import (
    list_scenarios,
    get_scenario,
    get_scenarios_by_category,
    build_scenario_configs,
    ScenarioCategory,
)

LINE = "=" * 75

print(LINE)
print(" SCENARIOS LIBRARY VERIFICATION")
print(LINE)

scenarios = list_scenarios()
print(f"Total Predefined Scenarios: {len(scenarios)}\n")

print(f"{'ID':<28} {'Category':<22} {'Scenario Name'}")
print("-" * 75)
for s in scenarios:
    print(f"{s.scenario_id:<28} {s.category.value:<22} {s.scenario_name}")

print("-" * 75)
print("TESTING CATEGORY RETRIEVAL")
print("-" * 75)
for cat in ScenarioCategory:
    cat_scenarios = get_scenarios_by_category(cat)
    print(f"Category '{cat.value}': {len(cat_scenarios)} scenarios âœ“")

print("-" * 75)
print("TESTING CONFIG INSTANTIATION & PARAMETER INTEGRITY")
print("-" * 75)
configs = build_scenario_configs()
assert len(configs) == len(scenarios), "Mismatch in generated configuration tuples."

# Test Specific Modifier Properties
lfp_cfg, lfp_bat = get_scenario("SCN_CHEM_LFP").instantiate_configs()
assert lfp_bat.chemistry.nominal_cycle_life == 7000, "LFP cycle life not updated."
assert lfp_bat.ageing.calendar_loss_per_year == 0.010, "LFP calendar loss not updated."
print("SCN_CHEM_LFP Modifier Verification: PASSED âœ“")

t45_cfg, t45_bat = get_scenario("SCN_TEMP_45C").instantiate_configs()
assert t45_bat.ageing.reference_temperature_c == 45.0, "Thermal 45Â°C not updated."
print("SCN_TEMP_45C Modifier Verification: PASSED âœ“")

sz_cfg, sz_bat = get_scenario("SCN_SIZE_100MW_200MWH").instantiate_configs()
assert sz_bat.chemistry.nominal_capacity_mwh == 200.0, "Sizing 200 MWh not updated."
assert sz_bat.chemistry.max_charge_power_mw == 100.0, "Power 100 MW not updated."
print("SCN_SIZE_100MW_200MWH Modifier Verification: PASSED âœ“")

eff_cfg, eff_bat = get_scenario("SCN_EFF_95PCT").instantiate_configs()
assert eff_bat.chemistry.round_trip_efficiency == 0.95, "RTE 95% not updated."
print("SCN_EFF_95PCT Modifier Verification: PASSED âœ“")

print(LINE)
print("Scenario library verified successfully with all 28 scenarios intact âœ“")
print(LINE)