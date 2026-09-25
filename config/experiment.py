from dataclasses import dataclass

@dataclass(frozen=True)
class ExperimentConfig:
    """
    Controls which experiment is executed.
    """

    experiment__name: str = "forecast__aware__arbitrage"

    forecast__model: str = "xgboost"

    enable__dynamic__ageing: bool = True

    enable__forecast__realism: bool = True

    monte__carlo__runs: int = 1000

    save__results: bool = True