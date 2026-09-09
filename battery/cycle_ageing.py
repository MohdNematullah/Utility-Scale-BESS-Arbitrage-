"""
cycle_ageing.py
===============

Research-grade cycle ageing model for .

Implements:
1. Equivalent Full Cycles (EFC).
2. Energy throughput calculation.
3. Depth-of-Discharge ageing multiplier.
4. Temperature ageing multiplier.
5. C-rate ageing multiplier.
6. SOH update from cycling degradation.

Reference:
Journal of Energy Storage (2023)
Semi-empirical ageing model for LFP and NMC batteries.
"""

from dataclasses import dataclass
from math import exp

from battery.config import DEFAULT_BATTERY_CONFIG


R_GAS = 8.314462618


from dataclasses import dataclass

@dataclass(slots=True)
class CycleAgeingResult:
    """
    Results for one cycle-ageing update.
    """

    charged_energy_mwh: float
    discharged_energy_mwh: float

    equivalent_full_cycles: float

    average_dod: float
    average_c_rate: float
    temperature_c: float

    capacity_loss_fraction: float

    remaining_soh: float
    remaining_capacity_mwh: float

class CycleAgeingModel:
    """
    Equivalent Full Cycle degradation model.
    """

    def __init__(self, config=DEFAULT_BATTERY_CONFIG):
        self.config = config

    # -----------------------------------------------------
    # Equivalent Full Cycles
    # -----------------------------------------------------

    def equivalent_full_cycles(
        self,
        charged_energy_mwh: float,
        discharged_energy_mwh: float,
    ) -> float:
        """
        EFC = throughput / (2 × nominal capacity)
        """

        throughput = charged_energy_mwh + discharged_energy_mwh

        capacity = self.config.chemistry.nominal_capacity_mwh

        return throughput / (2 * capacity)

    # -----------------------------------------------------
    # Depth of Discharge multiplier
    # -----------------------------------------------------

    def dod_factor(self, average_dod: float) -> float:
        """
        Depth-of-discharge ageing multiplier.

        Returns 1.0 at the reference DoD.
        """

        reference = self.config.ageing.reference_depth_of_discharge
        exponent = self.config.ageing.dod_ageing_exponent

        average_dod = max(0.05, min(1.0, average_dod))

        return (average_dod / reference) ** exponent

    # -----------------------------------------------------
    # Temperature multiplier
    # -----------------------------------------------------

    def temperature_factor(self, temperature_c: float) -> float:
        """
        Cycling temperature degradation multiplier.
        """

        beta = self.config.ageing.cycle_temperature_factor
        reference = self.config.ageing.reference_temperature_c

        delta = max(0.0, temperature_c - reference)

        return 1.0 + beta * delta

    # -----------------------------------------------------
    # C-rate multiplier
    # -----------------------------------------------------

    def c_rate_factor(self, average_c_rate: float) -> float:
        """
        C-rate degradation multiplier.
        """

        factor = self.config.ageing.c_rate_ageing_factor

        average_c_rate = max(0.1, average_c_rate)

        return 1.0 + factor * (average_c_rate - 1.0)

    # -----------------------------------------------------
    # Capacity loss
    # -----------------------------------------------------

    def capacity_loss(
            self,
            charged_energy_mwh: float,
            discharged_energy_mwh: float,
            average_dod: float,
            average_c_rate: float,
            temperature_c: float,
    ) -> float:
        """
        Capacity loss from cycling.
        """

        efc = self.equivalent_full_cycles(
            charged_energy_mwh,
            discharged_energy_mwh,
        )

        base = self.config.ageing.cycle_loss_per_efc

        loss = (
                efc
                * base
                * self.dod_factor(average_dod)
                * self.c_rate_factor(average_c_rate)
                * self.temperature_factor(temperature_c)
        )

        return min(loss, 1.0)

    # -----------------------------------------------------
    # Update SOH
    # -----------------------------------------------------

    def update_soh(
        self,
        initial_soh: float,
        charged_energy_mwh: float,
        discharged_energy_mwh: float,
        average_dod: float,
        average_c_rate: float,
        temperature_c: float,
    ) -> CycleAgeingResult:

        loss = self.capacity_loss(
            charged_energy_mwh,
            discharged_energy_mwh,
            average_dod,
            average_c_rate,
            temperature_c,
        )

        remaining_soh = max(
            initial_soh - loss,
            self.config.ageing.end_of_life_soh,
        )

        remaining_capacity = (
                remaining_soh
                * self.config.chemistry.nominal_capacity_mwh
        )

        return CycleAgeingResult(
            charged_energy_mwh=charged_energy_mwh,
            discharged_energy_mwh=discharged_energy_mwh,
            equivalent_full_cycles=self.equivalent_full_cycles(
                charged_energy_mwh,
                discharged_energy_mwh,
            ),
            average_dod=average_dod,
            average_c_rate=average_c_rate,
            temperature_c=temperature_c,
            capacity_loss_fraction=loss,
            remaining_soh=remaining_soh,
            remaining_capacity_mwh=remaining_capacity,
        )