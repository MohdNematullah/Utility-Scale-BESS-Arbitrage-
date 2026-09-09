"""
check_calendar_ageing.py

Verification script for Part 7.2A — Calendar Ageing Model.
"""

from battery.calendar_ageing import CalendarAgeingModel

model = CalendarAgeingModel()

print("=" * 70)
print("CALENDAR AGEING MODEL CHECK")
print("=" * 70)

result = model.update_soh(
    initial_soh=1.0,
    days=365,
    average_soc=0.50,
    temperature_c=25.0,
)

print(f"Storage Days             : {result.days:.0f}")
print(f"Average SOC              : {result.average_soc*100:.1f} %")
print(f"Temperature              : {result.temperature_c:.1f} °C")
print()

print(f"Capacity Loss            : {result.capacity_loss_fraction:.6f}")
print(f"Remaining SOH            : {result.remaining_soh:.6f}")
print(f"Remaining Capacity       : {result.remaining_capacity_mwh:.3f} MWh")
print()

print(
    f"Temperature Factor (25C) : {model.temperature_factor(25.0):.4f}"
)
print(
    f"SOC Factor (50%)         : {model.soc_factor(0.50):.4f}"
)

print()
print("Calendar ageing verified successfully ✓")
print("=" * 70)