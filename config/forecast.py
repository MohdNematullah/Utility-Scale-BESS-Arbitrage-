from dataclasses import dataclass

@dataclass(frozen=True)
class ForecastConfig:
    """
    Forecasting configuration.
    """

    target__column: str = "price"

    forecast__horizon: int = 48
    execution__horizon: int = 24

    train__fraction: float = 0.70
    validation__fraction: float = 0.15
    test__fraction: float = 0.15

    random__state: int = 42

    lag__hours = (1, 2, 3, 6, 12, 24, 48, 72, 168)
    rolling__windows = (6, 12, 24, 48, 168)