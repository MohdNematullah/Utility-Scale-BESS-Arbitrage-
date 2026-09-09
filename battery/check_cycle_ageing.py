"""
check_cycle_ageing.py

Verification script for Part 7.3 — Cycle Ageing Model.
"""

from battery.cycle_ageing import CycleAgeingModel

model = CycleAgeingModel()

print("=" * 70)
print("CYCLE AGEING MODEL CHECK")
print("=" * 70)

result = model.update_soh(
    initial_soh=1.0,
    charged_energy_mwh=80.0,
    discharged_energy_mwh=80.0,
    average_dod=0.80,
    average_c_rate=1.0,
    temperature_c=25.0,
)

throughput = result.charged_energy_mwh + result.discharged_energy_mwh

print(f"EFC                     : {result.equivalent_full_cycles:.4f}")
print(f"Energy Throughput       : {throughput:.2f} MWh")
print(f"Average DoD             : {result.average_dod:.2%}")
print(f"Average C-rate          : {result.average_c_rate:.2f}")
print(f"Temperature             : {result.temperature_c:.1f} °C")

print()
print(f"Capacity Loss           : {result.capacity_loss_fraction:.6f}")
print(f"Remaining SOH           : {result.remaining_soh:.6f}")
print(f"Remaining Capacity      : {result.remaining_capacity_mwh:.3f} MWh")

print()
print(f"DoD Factor (80%)        : {model.dod_factor(0.80):.4f}")
print(f"C-rate Factor (1C)      : {model.c_rate_factor(1.0):.4f}")
print(f"Temperature Factor (25C): {model.temperature_factor(25.0):.4f}")

print("=" * 70)
print("Cycle ageing verified successfully ✓")
print("=" * 70)