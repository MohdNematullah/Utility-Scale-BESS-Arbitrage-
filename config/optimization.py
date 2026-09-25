from dataclasses import dataclass

@dataclass(frozen=True)
class OptimizationConfig:
    """
    Rolling horizon optimization settings.
    """

    solver__name: str = "gurobi"

    optimization__window__hours: int = 48
    execution__window__hours: int = 24

    terminal__soc__mode: str = "hard"

    include__degradation__cost: bool = True

    mip__gap: float = 1e-4
    time__limit__seconds: int = 300