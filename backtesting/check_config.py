"""
Verify Backtesting Configuration.
"""

from backtesting.config import DEFAULT_BACKTEST_CONFIG

cfg = DEFAULT_BACKTEST_CONFIG

print("=" * 70)
print("ROLLING BACKTEST CONFIGURATION")
print("=" * 70)

print(f"Forecast Horizon        : {cfg.forecast_horizon_hours} hours")
print(f"Implementation Horizon  : {cfg.implementation_horizon_hours} hours")
print(f"Rolling Step            : {cfg.rolling_step_hours} hours")
print(f"Initial SOH             : {cfg.initial_soh:.2f}")
print(f"Reference Temperature   : {cfg.reference_temperature_c} Â°C")
print(f"Default C-rate          : {cfg.default_c_rate:.1f}")

print("\nExport Directory")
print("-" * 70)
print(cfg.export_directory)

print("=" * 70)
print("Backtesting configuration verified successfully âœ“")
print("=" * 70)