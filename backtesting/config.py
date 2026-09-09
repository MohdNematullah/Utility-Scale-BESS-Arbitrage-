from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class BacktestConfig:
    """
    Rolling Horizon Backtesting configuration.

    Research-grade configuration used by Part 8.
    """

    # ---------------------------------------------------------
    # Rolling Horizon Parameters
    # ---------------------------------------------------------

    forecast_horizon_hours: int = 48

    implementation_horizon_hours: int = 24

    rolling_step_hours: int = 24

    # ---------------------------------------------------------
    # Battery Initial State
    # ---------------------------------------------------------

    initial_soh: float = 1.0

    reference_temperature_c: float = 25.0

    default_c_rate: float = 1.0

    # ---------------------------------------------------------
    # Export Settings
    # ---------------------------------------------------------

    export_directory: Path = field(
        default_factory=lambda: Path("backtesting/results")
    )


DEFAULT_BACKTEST_CONFIG = BacktestConfig()