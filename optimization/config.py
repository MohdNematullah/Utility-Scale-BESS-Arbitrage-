
"""
config.py
=========

Research-grade optimization configuration for .

Used by:
    • optimization/model_builder.py
    • optimization/solver.py
    • battery/state.py
    • battery/aging.py
    • backtesting/rolling_runner.py

All optimization experiments use this configuration.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
import pandas as pd


# ============================================================
# Battery Configuration
# ============================================================

@dataclass(slots=True)
class BatteryConfig:
    """
    Lithium-ion battery configuration.
    Units follow electricity market convention.
    """

    battery_name: str = "ERCOT_LiIon_Grid_Battery"

    # Energy capacity (MWh)
    energy_capacity_mwh: float = 100.0

    # Charge/Discharge Power (MW)
    max_charge_power_mw: float = 50.0
    max_discharge_power_mw: float = 50.0

    # Efficiencies
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95

    # SOC limits
    soc_min_fraction: float = 0.10
    soc_max_fraction: float = 0.90

    # Initial / terminal SOC
    initial_soc_fraction: float = 0.50
    terminal_soc_fraction: float = 0.50

    # Time resolution
    timestep_hours: float = 1.0

    # Battery ageing placeholder
    degradation_enabled: bool = False

    # Capacity fade placeholder
    remaining_capacity_fraction: float = 1.0

    @property
    def soc_min_mwh(self):

        return (
            self.energy_capacity_mwh
            * self.soc_min_fraction
        )

    @property
    def soc_max_mwh(self):

        return (
            self.energy_capacity_mwh
            * self.soc_max_fraction
        )

    @property
    def initial_soc_mwh(self):

        return (
            self.energy_capacity_mwh
            * self.initial_soc_fraction
        )

    @property
    def terminal_soc_mwh(self):

        return (
            self.energy_capacity_mwh
            * self.terminal_soc_fraction
        )

    @property
    def round_trip_efficiency(self):

        return (
            self.charge_efficiency
            * self.discharge_efficiency
        )


# ============================================================
# Solver Configuration
# ============================================================

@dataclass(slots=True)
class SolverConfig:
    """
    Optimization solver settings.
    """

    solver_name: str = "highs"

    mip_gap: float = 0.0

    time_limit_seconds: int = 60

    tee: bool = False

    symbolic_solver_labels: bool = False

    warm_start: bool = False


# ============================================================
# Forecast Configuration
# ============================================================

@dataclass(slots=True)
class ForecastConfig:
    """
    Forecast settings used by optimization.
    """

    forecast_horizon_hours: int = 24

    recursive: bool = True

    execute_first_hours: int = 24

    retrain_every_window: bool = False

    forecast_model: str = "xgboost"

    keep_history_hours: int = 168


# ============================================================
# Experiment Configuration
# ============================================================

@dataclass(slots=True)
class ExperimentConfig:
    """
    Configuration for thesis experiments.
    """

    experiment_name: str = (
        "Rolling_Horizon_Battery_Arbitrage"
    )

    market_name: str = "ERCOT_DAM"

    optimization_horizon_hours: int = 24

    evaluation_horizon_hours: int = 24

    rolling_window_hours: int = 24

    random_seed: int = 42

    export_results: bool = True

    export_directory: str = (
        "optimization/results"
    )


# ============================================================
# Complete Optimization Configuration
# ============================================================

# ============================================================
# Complete Optimization Configuration
# ============================================================

@dataclass(slots=True)
class OptimizationConfig:
    """
    Master configuration used by the optimization engine.
    Compatible with Python 3.14 dataclasses.
    """

    battery: BatteryConfig = field(
        default_factory=BatteryConfig
    )

    solver: SolverConfig = field(
        default_factory=SolverConfig
    )

    forecast: ForecastConfig = field(
        default_factory=ForecastConfig
    )

    experiment: ExperimentConfig = field(
        default_factory=ExperimentConfig
    )

    # --------------------------------------------------------
    # Validate configuration
    # --------------------------------------------------------
    def validate(self):

        b = self.battery

        assert b.energy_capacity_mwh > 0
        assert b.max_charge_power_mw > 0
        assert b.max_discharge_power_mw > 0

        assert 0 < b.charge_efficiency <= 1
        assert 0 < b.discharge_efficiency <= 1

        assert (
            0 <= b.soc_min_fraction
            < b.soc_max_fraction
            <= 1
        )

        assert (
            b.soc_min_mwh
            <= b.initial_soc_mwh
            <= b.soc_max_mwh
        )

        assert (
            b.soc_min_mwh
            <= b.terminal_soc_mwh
            <= b.soc_max_mwh
        )

        return True

    # --------------------------------------------------------
    # Convert configuration to DataFrame
    # --------------------------------------------------------
    def to_dataframe(self):

        rows = []

        for section_name, section in [
            ("Battery", self.battery),
            ("Forecast", self.forecast),
            ("Solver", self.solver),
            ("Experiment", self.experiment),
        ]:

            for key, value in asdict(section).items():

                rows.append(
                    {
                        "section": section_name,
                        "parameter": key,
                        "value": value,
                    }
                )

        return pd.DataFrame(rows)

    # --------------------------------------------------------
    # Export configuration
    # --------------------------------------------------------
    def export(
        self,
        output_file="optimization/config_preview.csv",
    ):

        output_file = Path(output_file)

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.to_dataframe().to_csv(
            output_file,
            index=False,
        )

        return output_file


# ============================================================
# Default configuration
# ============================================================

DEFAULT_CONFIG = OptimizationConfig()

DEFAULT_CONFIG.validate()