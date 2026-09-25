"""
data package

Battery Trading Arbitrage ()

Contains data loading, validation, and dataset utilities.
This file should NOT execute any code when imported.
"""

from .loader import MarketDataLoader
from .validator import MarketDataValidator, ValidationReport

__all__ = [
    "MarketDataLoader",
    "MarketDataValidator",
    "ValidationReport",
]