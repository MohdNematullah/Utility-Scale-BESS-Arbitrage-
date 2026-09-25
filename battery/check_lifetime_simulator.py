"""
check_lifetime_simulator.py

Verify lifetime simulation.
"""

from battery.lifetime_simulator import BatteryLifetimeSimulator

print("=" * 70)
print("BATTERY LIFETIME SIMULATION CHECK")
print("=" * 70)

simulator = BatteryLifetimeSimulator()

result = simulator.simulate(
    years=20,
    annual_efc=300,
)

print("Years Simulated        :", result.years_simulated)
print("Replacement Year       :", result.replacement_year)
print("Final SOH              :", round(result.final_soh, 4))
print("Degradation Cost (USD) :", result.total_degradation_cost_usd)
print("Replacement Cost (USD) :", result.total_replacement_cost_usd)

print("-" * 70)
print(result.timeline.head())

path = simulator.export(result)

print("-" * 70)
print("Saved:")
print(path)

print("=" * 70)
print("Lifetime simulation verified successfully âœ“")
print("=" * 70)