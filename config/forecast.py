from dataclasses import dataclass

@dataclass(frozen=True)
class ForecastConfig:
    """
    Forecasting configuration.
    """

    target_column: str = "price"

    forecast_horizon: int = 48
    execution_horizon: int = 24

    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15

    random_state: int = 42

    lag_hours = (1, 2, 3, 6, 12, 24, 48, 72, 168)
    rolling_windows = (6, 12, 24, 48, 168)