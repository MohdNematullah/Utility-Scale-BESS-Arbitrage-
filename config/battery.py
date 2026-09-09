from dataclasses import dataclass

@dataclass(frozen=True)
class BatteryConfig:
    """
    Battery specification used throughout the project.
    """

    # Battery size
    capacity_mwh: float = 1.0
    max_charge_mw: float = 0.5
    max_discharge_mw: float = 0.5

    # Efficiency
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95

    # SOC limits
    soc_min: float = 0.10
    soc_max: float = 0.90
    initial_soc: float = 0.50

    # Health
    initial_soh: float = 1.00
    end_of_life_soh: float = 0.80

    # Economics
    replacement_cost_usd: float = 250000

    # Simulation timestep
    timestep_hours: float = 1.0