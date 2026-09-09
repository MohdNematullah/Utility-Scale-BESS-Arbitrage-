"""
optimization/degradation_optimizer.py
=====================================

Integrates battery degradation cost into optimization results.
"""

from dataclasses import dataclass

import pyomo.environ as pyo

from battery.degradation_cost import BatteryDegradationCostModel


@dataclass(slots=True)
class OptimizationEconomics:
    gross_revenue_usd: float
    degradation_cost_usd: float
    net_revenue_usd: float

    throughput_mwh: float
    equivalent_full_cycles: float
    remaining_soh: float


class DegradationAwareOptimizer:

    def __init__(self):
        self.cost_model = BatteryDegradationCostModel()

    # ---------------------------------------------------------
    # Revenue from solved Pyomo model
    # ---------------------------------------------------------

    def gross_revenue(self, model) -> float:

        revenue = 0.0

        for t in model.T:

            revenue += (
                pyo.value(model.discharge_power[t])
                - pyo.value(model.charge_power[t])
            ) * pyo.value(model.price[t])

        return float(revenue)

    # ---------------------------------------------------------
    # Throughput
    # ---------------------------------------------------------

    def throughput(self, model):

        charge = sum(
            pyo.value(model.charge_power[t])
            for t in model.T
        )

        discharge = sum(
            pyo.value(model.discharge_power[t])
            for t in model.T
        )

        return float(charge), float(discharge)

    # ---------------------------------------------------------
    # Evaluate Economics
    # ---------------------------------------------------------

    def evaluate(
        self,
        model,
        initial_soh: float = 1.0,
        calendar_days: float = 1,
        average_soc: float = 0.5,
        average_dod: float = 0.8,
        average_c_rate: float = 1.0,
        temperature_c: float = 25,
    ) -> OptimizationEconomics:

        charge, discharge = self.throughput(model)

        degradation = self.cost_model.evaluate(
            initial_soh=initial_soh,
            calendar_days=calendar_days,
            average_soc=average_soc,
            temperature_c=temperature_c,
            charged_energy_mwh=charge,
            discharged_energy_mwh=discharge,
            average_dod=average_dod,
            average_c_rate=average_c_rate,
        )

        gross = self.gross_revenue(model)

        net = gross - degradation.degradation_cost_usd

        throughput = degradation.throughput_mwh

        capacity = self.cost_model.config.chemistry.nominal_capacity_mwh

        efc = throughput / (2 * capacity)

        return OptimizationEconomics(
            gross_revenue_usd=gross,
            degradation_cost_usd=degradation.degradation_cost_usd,
            net_revenue_usd=net,

            throughput_mwh=throughput,
            equivalent_full_cycles=efc,

            remaining_soh=degradation.remaining_soh,
        )