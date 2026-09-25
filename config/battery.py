from dataclasses import dataclass

@dataclass(frozen=True)
class BatteryConfig:
    """
    Battery specification used throughout the project.
    """

    # Battery size
    capacity__mwh: float = 1.0
    max__charge__mw: float = 0.5
    max__discharge__mw: float = 0.5

    # Efficiency
    charge__efficiency: float = 0.95
    discharge__efficiency: float = 0.95

    # SOC limits
    soc__min: float = 0.10
    soc__max: float = 0.90
    initial__soc: float = 0.50

    # Health
    initial__soh: float = 1.00
    end__of__life__soh: float = 0.80

    # Economics
    replacement__cost__usd: float = 250000

    # Simulation timestep
    timestep__hours: float = 1.0