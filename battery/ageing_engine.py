"""
battery/ageing_engine.py
========================

Combined Battery Ageing Engine for .
"""

from dataclasses import dataclass

from battery.calendar_ageing import CalendarAgeingModel
from battery.cycle_ageing import CycleAgeingModel
from battery.config import DEFAULT_BATTERY_CONFIG


@dataclass(slots=True)
class BatteryAgeingResult:
    calendar_loss_fraction: float
    cycle_loss_fraction: float
    total_capacity_loss_fraction: float
    remaining_soh: float
    remaining_capacity_mwh: float
    cumulative_efc: float
    degradation_cost_usd: float
    replacement_required: bool
    remaining_useful_cycles: float
    remaining_useful_years: float


class BatteryAgeingEngine:

    def __init__(self, config=DEFAULT_BATTERY_CONFIG):
        self.config = config
        self.calendar_model = CalendarAgeingModel(config)
        self.cycle_model = CycleAgeingModel(config)

    def evaluate(
        self,
        initial_soh: float,
        calendar_days: float,
        average_soc: float,
        temperature_c: float,
        charged_energy_mwh: float,
        discharged_energy_mwh: float,
        average_dod: float,
        average_c_rate: float,
    ) -> BatteryAgeingResult:
        return self.update(
            initial_soh=initial_soh,
            storage_days=calendar_days,
            average_soc=average_soc,
            temperature_c=temperature_c,
            charged_energy_mwh=charged_energy_mwh,
            discharged_energy_mwh=discharged_energy_mwh,
            average_dod=average_dod,
            average_c_rate=average_c_rate,
        )

    def update(
        self,
        initial_soh: float,
        storage_days: float,
        average_soc: float,
        temperature_c: float,
        charged_energy_mwh: float,
        discharged_energy_mwh: float,
        average_dod: float,
        average_c_rate: float,
    ) -> BatteryAgeingResult:
        calendar = self.calendar_model.update_soh(
            initial_soh=initial_soh,
            days=storage_days,
            average_soc=average_soc,
            temperature_c=temperature_c,
        )
        calendar_loss = calendar.capacity_loss_fraction

        cycle = self.cycle_model.update_soh(
            initial_soh=initial_soh,
            charged_energy_mwh=charged_energy_mwh,
            discharged_energy_mwh=discharged_energy_mwh,
            average_dod=average_dod,
            average_c_rate=average_c_rate,
            temperature_c=temperature_c,
        )
        cycle_loss = cycle.capacity_loss_fraction

        total_loss = calendar_loss + cycle_loss
        remaining_soh = max(
            initial_soh - total_loss,
            self.config.ageing.end_of_life_soh,
        )
        remaining_capacity = (
            remaining_soh * self.config.chemistry.nominal_capacity_mwh
        )

        efc = self.cycle_model.equivalent_full_cycles(
            charged_energy_mwh,
            discharged_energy_mwh,
        )

        replacement_value = (
            self.config.chemistry.nominal_capacity_mwh
            * self.config.replacement.replacement_cost_per_mwh
        )
        usable_replacement_value = replacement_value * (
            1.0 - self.config.replacement.salvage_fraction
        )

        # Standard physical replacement valuation without the 5x divisor
        degradation_cost = total_loss * usable_replacement_value

        replacement_required = (
            remaining_soh <= self.config.replacement.replacement_trigger_soh
        )

        remaining_cycles = max(
            (remaining_soh - self.config.ageing.end_of_life_soh)
            * self.config.chemistry.nominal_cycle_life,
            0.0,
        )

        remaining_years = max(
            (remaining_soh - self.config.ageing.end_of_life_soh)
            / self.config.ageing.calendar_loss_per_year,
            0.0,
        )

        return BatteryAgeingResult(
            calendar_loss_fraction=calendar_loss,
            cycle_loss_fraction=cycle_loss,
            total_capacity_loss_fraction=total_loss,
            remaining_soh=remaining_soh,
            remaining_capacity_mwh=remaining_capacity,
            cumulative_efc=efc,
            degradation_cost_usd=degradation_cost,
            replacement_required=replacement_required,
            remaining_useful_cycles=remaining_cycles,
            remaining_useful_years=remaining_years,
        )