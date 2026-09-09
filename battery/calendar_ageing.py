"""
battery/calendar_ageing.py
==========================

Calendar ageing model for .

Implements:
1. Arrhenius temperature ageing factor.
2. SOC ageing multiplier.
3. Incremental daily capacity fade for rolling-horizon windows.
4. Monotonic State-of-Health propagation.
"""

from dataclasses import dataclass
from math import exp

from battery.config import DEFAULT_BATTERY_CONFIG

R_GAS = 8.314462618  # J/mol/K


@dataclass(slots=True)
class CalendarAgeingResult:
    days: float
    average_soc: float
    temperature_c: float
    capacity_loss_fraction: float
    remaining_soh: float
    remaining_capacity_mwh: float


class CalendarAgeingModel:
    """
    Incremental calendar ageing model for NMC/LFP Li-ion batteries.
    """

    def __init__(self, config=DEFAULT_BATTERY_CONFIG):
        self.config = config

    def temperature_factor(self, temperature_c: float) -> float:
        kelvin = temperature_c + 273.15
        ea = self.config.ageing.activation_energy_j_per_mol
        reference = self.config.ageing.reference_temperature_c + 273.15
        return exp((-ea / R_GAS) * (1.0 / kelvin - 1.0 / reference))

    def soc_factor(self, average_soc: float) -> float:
        average_soc = max(0.0, min(1.0, average_soc))
        alpha = self.config.ageing.soc_ageing_factor
        deviation = average_soc - 0.50
        return 1.0 + alpha * (deviation ** 2) / 0.25

    def capacity_loss(
        self,
        days: float,
        average_soc: float,
        temperature_c: float,
    ) -> float:
        """
        Calculates incremental capacity fade over an implementation window (days).
        """
        base_annual_rate = self.config.ageing.calendar_loss_per_year
        temp_mult = self.temperature_factor(temperature_c)
        soc_mult = self.soc_factor(average_soc)

        # Incremental calendar loss per window duration
        incremental_loss = base_annual_rate * temp_mult * soc_mult * (days / 365.0)
        return max(0.0, min(incremental_loss, 1.0))

    def update_soh(
        self,
        initial_soh: float,
        days: float,
        average_soc: float,
        temperature_c: float,
    ) -> CalendarAgeingResult:
        loss = self.capacity_loss(
            days=days,
            average_soc=average_soc,
            temperature_c=temperature_c,
        )

        remaining_soh = max(
            initial_soh - loss,
            self.config.ageing.end_of_life_soh,
        )

        remaining_capacity = (
            remaining_soh * self.config.chemistry.nominal_capacity_mwh
        )

        return CalendarAgeingResult(
            days=days,
            average_soc=average_soc,
            temperature_c=temperature_c,
            capacity_loss_fraction=loss,
            remaining_soh=remaining_soh,
            remaining_capacity_mwh=remaining_capacity,
        )