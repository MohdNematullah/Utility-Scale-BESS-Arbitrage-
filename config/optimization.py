from dataclasses import dataclass

@dataclass(frozen=True)
class OptimizationConfig:
    """
    Rolling horizon optimization settings.
    """

    solver_name: str = "gurobi"

    optimization_window_hours: int = 48
    execution_window_hours: int = 24

    terminal_soc_mode: str = "hard"

    include_degradation_cost: bool = True

    mip_gap: float = 1e-4
    time_limit_seconds: int = 300