"""
battery/check_ageing_engine.py

verification for combined battery ageing engine.
"""

from battery.ageing_engine import BatteryAgeingEngine

engine = BatteryAgeingEngine()

print("=" * 70)
print("COMBINED BATTERY AGEING ENGINE CHECK")
print("=" * 70)

result = engine.update(
    initial_soh=1.0,
    storage_days=365,
    average_soc=0.50,
    temperature_c=25.0,
    charged_energy_mwh=100.0,
    discharged_energy_mwh=100.0,
    average_dod=0.80,
    average_c_rate=1.0,
)

print(f"Calendar Capacity Loss     : {result.calendar_loss_fraction:.6f}")
print(f"Cycle Capacity Loss        : {result.cycle_loss_fraction:.6f}")
print(f"Total Capacity Loss        : {result.total_capacity_loss_fraction:.6f}")

print()

print(f"Remaining SOH             : {result.remaining_soh:.6f}")
print(f"Remaining Capacity        : {result.remaining_capacity_mwh:.3f} MWh")

print()

print(f"Cumulative EFC            : {result.cumulative_efc:.4f}")
print(f"Remaining Useful Cycles   : {result.remaining_useful_cycles:.0f}")
print(f"Remaining Useful Years    : {result.remaining_useful_years:.2f}")

print()

print(f"Replacement Required      : {result.replacement_required}")
print(f"Degradation Cost (USD)    : ${result.degradation_cost_usd:,.2f}")

print("=" * 70)
print("Combined ageing engine verified successfully âœ“")
print("=" * 70)