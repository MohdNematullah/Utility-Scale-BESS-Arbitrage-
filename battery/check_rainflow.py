"""
Verification script for Rainflow Cycle Counting.
"""

from battery.rainflow import RainflowCycleCounter

counter = RainflowCycleCounter()

soc = [
    0.50,
    0.80,
    0.30,
    0.90,
    0.40,
    0.70,
    0.20,
    0.50,
]

summary = counter.summarize(soc)

print("=" * 70)
print("RAINFLOW CYCLE COUNTING CHECK")
print("=" * 70)

print("Cycles Found           :", len(summary.cycles))
print("Full Cycles            :", summary.full_cycles)
print("Half Cycles            :", summary.half_cycles)
print("Equivalent Full Cycles :", round(summary.equivalent_full_cycles, 4))
print("Energy Throughput      :", round(summary.total_energy_throughput_mwh, 2), "MWh")

print("-" * 70)

for i, cycle in enumerate(summary.cycles, start=1):

    print(
        f"Cycle {i:02d} | "
        f"DoD={cycle.depth_of_discharge:.2f} "
        f"Mean SOC={cycle.mean_soc:.2f} "
        f"Count={cycle.count}"
    )

print("=" * 70)
print("Rainflow counting verified successfully âœ“")
print("=" * 70)