"""
check_results.py

Research-grade optimization integration check.
"""

from pathlib import Path

from data.loader import MarketDataLoader
from features.engineering import FeatureEngineer

from forecasting.models import XGBoostForecaster
from forecasting.recursive_engine import RecursiveForecastEngine

from optimization.model_builder import BatteryOptimizationModelBuilder
from optimization.solver import BatteryOptimizationSolver
from optimization.results import BatteryResultsProcessor
from optimization.degradation_optimizer import DegradationAwareOptimizer

print("=" * 70)
print("OPTIMIZATION RESULTS CHECK")
print("=" * 70)

# ----------------------------------------------------
# Load engineered feature matrix
# ----------------------------------------------------

loader = MarketDataLoader()
engineer = FeatureEngineer()

market = loader.load_csv("data/raw/ercot_prices.csv")
features = engineer.transform(market)

# ----------------------------------------------------
# Load trained XGBoost model
# ----------------------------------------------------

MODEL_PATH = Path("forecasting/saved_models/xgboost_dayahead.pkl")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Missing trained model: {MODEL_PATH}\n"
        "Run: python -m forecasting.check_trainer"
    )

forecaster = XGBoostForecaster()
forecaster.load(MODEL_PATH)

print("Forecast model loaded ✓")

# ------------------------------------------------------------
# Generate 24-hour recursive forecast
# ------------------------------------------------------------

engine = RecursiveForecastEngine(forecaster)

forecast_df = engine.forecast(
    feature_dataframe=features,
    horizon=24,
)

print("24-hour forecast generated ✓")

# ----------------------------------------------------
# Build optimization model
# ----------------------------------------------------

builder = BatteryOptimizationModelBuilder()
model = builder.build(forecast_df)

print("Optimization model built ✓")

# ----------------------------------------------------
# Solve optimization model
# ----------------------------------------------------

solver = BatteryOptimizationSolver()
solver_summary = solver.solve(model)

print("Solver             :", solver_summary.solver_name)
print("Termination        :", solver_summary.termination_condition)
print("Objective Value    :", round(solver_summary.objective_value, 3))
print("Solve Time (sec)   :", round(solver_summary.solve_time_seconds, 4))

# ----------------------------------------------------
# Process results
# ----------------------------------------------------

processor = BatteryResultsProcessor()
dispatch = processor.dispatch_schedule(model)

battery_summary = processor.battery_summary(
    model,
    dispatch,
)

processor.export(model)

optimizer = DegradationAwareOptimizer()

economics = optimizer.evaluate(
    model,
    initial_soh=1.0,
    calendar_days=1,
    average_soc=0.50,
    average_dod=0.80,
    average_c_rate=1.0,
    temperature_c=25.0,
)

print("-" * 70)
print("DEGRADATION-AWARE ECONOMICS")
print("-" * 70)
print(f"Gross Revenue (USD)      : {economics.gross_revenue_usd:.2f}")
print(f"Degradation Cost (USD)   : {economics.degradation_cost_usd:.2f}")
print(f"Net Revenue (USD)        : {economics.net_revenue_usd:.2f}")
print(f"Throughput (MWh)         : {economics.throughput_mwh:.2f}")
print(f"Equivalent Full Cycles   : {economics.equivalent_full_cycles:.3f}")
print(f"Remaining SOH            : {economics.remaining_soh:.5f}")

print("-" * 70)
print("Dispatch Preview")
print(dispatch.head())

print("-" * 70)
print("Battery Summary")
print(battery_summary)

print("-" * 70)
print("Saved to optimization/results/")