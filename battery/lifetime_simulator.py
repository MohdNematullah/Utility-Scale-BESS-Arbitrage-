"""
battery/lifetime_simulator.py
=============================

Battery Lifetime & Replacement Simulator
----------------------------------------

Simulates long-term degradation of a utility-scale battery using:

1. Calendar ageing.
2. Cycle ageing.
3. Rainflow / EFC ageing.
4. State-of-health evolution.
5. Replacement decision.
6. Lifetime economics.


"""

from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd

from battery.ageing_engine import BatteryAgeingEngine
from battery.config import DEFAULT_BATTERY_CONFIG


# ---------------------------------------------------------------------
# One simulation period
# ---------------------------------------------------------------------

@dataclass(slots=True)
class LifetimePeriod:

    year: int

    calendar_days: float

    annual_efc: float

    soh_start: float
    soh_end: float

    capacity_start_mwh: float
    capacity_end_mwh: float

    degradation_loss: float

    replacement_required: bool

    degradation_cost_usd: float


# ---------------------------------------------------------------------
# Overall simulation result
# ---------------------------------------------------------------------

@dataclass(slots=True)
class LifetimeSimulationResult:

    timeline: pd.DataFrame

    replacement_year: int | None

    total_degradation_cost_usd: float

    total_replacement_cost_usd: float

    years_simulated: int

    final_soh: float


# ---------------------------------------------------------------------
# Lifetime Simulator
# ---------------------------------------------------------------------

class BatteryLifetimeSimulator:

    def __init__(self, config=DEFAULT_BATTERY_CONFIG):

        self.config = config
        self.engine = BatteryAgeingEngine(config)

    # --------------------------------------------------------------
    # Simulate battery lifetime
    # --------------------------------------------------------------

    def simulate(
        self,
        years: int = 20,
        annual_efc: float = 300,
        average_soc: float = 0.50,
        average_dod: float = 0.80,
        average_c_rate: float = 1.0,
        temperature_c: float = 25.0,
    ) -> LifetimeSimulationResult:

        soh = 1.0

        capacity = self.config.chemistry.nominal_capacity_mwh

        timeline = []

        replacement_year = None

        degradation_cost = 0.0

        replacement_cost = 0.0

        for year in range(1, years + 1):

            charged = annual_efc * capacity
            discharged = annual_efc * capacity

            result = self.engine.evaluate(
                initial_soh=soh,
                calendar_days=365,
                average_soc=average_soc,
                temperature_c=temperature_c,
                charged_energy_mwh=charged,
                discharged_energy_mwh=discharged,
                average_dod=average_dod,
                average_c_rate=average_c_rate,
            )

            timeline.append(
                LifetimePeriod(
                    year=year,
                    calendar_days=365,
                    annual_efc=annual_efc,
                    soh_start=soh,
                    soh_end=result.remaining_soh,
                    capacity_start_mwh=soh * capacity,
                    capacity_end_mwh=result.remaining_capacity_mwh,
                    degradation_loss=result.total_capacity_loss_fraction,
                    replacement_required=result.replacement_required,
                    degradation_cost_usd=result.degradation_cost_usd,
                )
            )

            soh = result.remaining_soh

            degradation_cost += result.degradation_cost_usd

            if result.replacement_required and replacement_year is None:
                replacement_year = year

                replacement_cost = (
                    capacity
                    * self.config.replacement.replacement_cost_per_mwh
                    * (1 - self.config.replacement.salvage_fraction)
                )

                break

        dataframe = pd.DataFrame(
            [asdict(item) for item in timeline]
        )

        return LifetimeSimulationResult(
            timeline=dataframe,
            replacement_year=replacement_year,
            total_degradation_cost_usd=round(degradation_cost, 2),
            total_replacement_cost_usd=round(replacement_cost, 2),
            years_simulated=len(dataframe),
            final_soh=round(soh, 5),
        )

    # --------------------------------------------------------------
    # Export simulation
    # --------------------------------------------------------------

    def export(
        self,
        result: LifetimeSimulationResult,
    ) -> Path:

        directory = self.config.results_directory
        directory.mkdir(parents=True, exist_ok=True)

        path = directory / "battery_lifetime_simulation.csv"

        result.timeline.round(6).to_csv(path, index=False)

        return path