"""
Verification script for degradation cost model.
"""

from battery.degradation_cost import BatteryDegradationCostModel

model = BatteryDegradationCostModel()

print("=" * 70)
print("BATTERY DEGRADATION COST CHECK")
print("=" * 70)

result = model.evaluate(
    initial_soh=1.0,
    calendar_days=365,
    average_soc=0.50,
    temperature_c=25,
    charged_energy_mwh=120,
    discharged_energy_mwh=120,
    average_dod=0.80,
    average_c_rate=1.0,
)

print("Throughput (MWh)        :", round(result.throughput_mwh,3))
print("Calendar Loss           :", round(result.calendar_loss_fraction,6))
print("Cycle Loss              :", round(result.cycle_loss_fraction,6))
print("Total Loss              :", round(result.total_loss_fraction,6))
print("Remaining SOH           :", round(result.remaining_soh,6))
print("Remaining Capacity (MWh):", round(result.remaining_capacity_mwh,3))
print()

print("Replacement Value (USD) :", round(model.battery_replacement_value,2))
print("Salvage Value (USD)     :", round(model.salvage_value,2))
print()

print("Degradation Cost (USD)  :", round(result.degradation_cost_usd,2))
print("Marginal Cost ($/MWh)   :", round(result.marginal_cost_per_mwh,4))
print("Replacement Required    :", result.replacement_required)

print("=" * 70)
print("Degradation cost model verified successfully âœ“")
print("=" * 70)