"""
battery/degradation_cost.py
===========================

Battery degradation cost model for .

Converts capacity fade into economic valuation scaled by usable battery lifespan.
"""

from dataclasses import dataclass

from battery.ageing_engine import BatteryAgeingEngine
from battery.config import DEFAULT_BATTERY_CONFIG


@dataclass(slots=True)
class DegradationCostResult:
    throughput_mwh: float
    calendar_loss_fraction: float
    cycle_loss_fraction: float
    total_loss_fraction: float
    degradation_cost_usd: float
    marginal_cost_per_mwh: float
    remaining_soh: float
    remaining_capacity_mwh: float
    replacement_required: bool


class BatteryDegradationCostModel:
    """
    Standardized economic accounting module for BESS degradation.
    """

    def __init__(self, config=DEFAULT_BATTERY_CONFIG):
        self.config = config
        self.engine = BatteryAgeingEngine(config)

    @property
    def battery_replacement_value(self) -> float:
        return (
            self.config.chemistry.nominal_capacity_mwh
            * self.config.replacement.replacement_cost_per_mwh
        )

    @property
    def salvage_value(self) -> float:
        return (
            self.battery_replacement_value
            * self.config.replacement.salvage_fraction
        )

    @property
    def usable_replacement_value(self) -> float:
        return self.battery_replacement_value - self.salvage_value

    @property
    def usable_capacity_fraction(self) -> float:
        return 1.0 - self.config.ageing.end_of_life_soh

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
    ) -> DegradationCostResult:
        ageing = self.engine.evaluate(
            initial_soh=initial_soh,
            calendar_days=calendar_days,
            average_soc=average_soc,
            temperature_c=temperature_c,
            charged_energy_mwh=charged_energy_mwh,
            discharged_energy_mwh=discharged_energy_mwh,
            average_dod=average_dod,
            average_c_rate=average_c_rate,
        )

        throughput = charged_energy_mwh + discharged_energy_mwh

        # Economic allocation across usable capacity window
        degradation_cost = (
                ageing.total_capacity_loss_fraction * self.usable_replacement_value
        )

        marginal_cost = (
            degradation_cost / throughput
            if throughput > 0
            else 0.0
        )

        return DegradationCostResult(
            throughput_mwh=throughput,
            calendar_loss_fraction=ageing.calendar_loss_fraction,
            cycle_loss_fraction=ageing.cycle_loss_fraction,
            total_loss_fraction=ageing.total_capacity_loss_fraction,
            degradation_cost_usd=degradation_cost,
            marginal_cost_per_mwh=marginal_cost,
            remaining_soh=ageing.remaining_soh,
            remaining_capacity_mwh=ageing.remaining_capacity_mwh,
            replacement_required=ageing.replacement_required,
        )

    def penalty_per_mwh(
        self,
        initial_soh: float,
        calendar_days: float,
        average_soc: float,
        temperature_c: float,
        charged_energy_mwh: float,
        discharged_energy_mwh: float,
        average_dod: float,
        average_c_rate: float,
    ) -> float:
        return self.evaluate(
            initial_soh,
            calendar_days,
            average_soc,
            temperature_c,
            charged_energy_mwh,
            discharged_energy_mwh,
            average_dod,
            average_c_rate,
        ).marginal_cost_per_mwh