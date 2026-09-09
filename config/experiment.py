from dataclasses import dataclass

@dataclass(frozen=True)
class ExperimentConfig:
    """
    Controls which experiment is executed.
    """

    experiment_name: str = "forecast_aware_arbitrage"

    forecast_model: str = "xgboost"

    enable_dynamic_ageing: bool = True

    enable_forecast_realism: bool = True

    monte_carlo_runs: int = 1000

    save_results: bool = True