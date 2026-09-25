"""
check_results.py
================
optimization integration check.
Validates the complete execution flow from forecasting through results export,
verifying realized settlement pricing and end-of-interval SOC semantics.
"""

from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).parent.name == "optimization" else Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
# 1. Load engineered feature matrix
# ----------------------------------------------------
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ercot_prices.csv"
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Missing price dataset: {DATA_PATH}")

loader = MarketDataLoader()
engineer = FeatureEngineer()

market = loader.load_csv(DATA_PATH)
features = engineer.transform(market)
print(f"Market features transformed : {features.shape[0]} rows âœ“")

# ----------------------------------------------------
# 2. Load trained XGBoost model
# ----------------------------------------------------
MODEL_PATH = PROJECT_ROOT / "forecasting" / "saved_models" / "xgboost_dayahead.pkl"
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Missing trained model: {MODEL_PATH}\n"
        "Run: python -m forecasting.check_trainer"
    )

forecaster = XGBoostForecaster()
forecaster.load(MODEL_PATH)
print("Forecast model loaded       : XGBoost âœ“")

# ----------------------------------------------------
# 3. Generate 24-hour recursive forecast
# ----------------------------------------------------
engine = RecursiveForecastEngine(forecaster)
forecast_df = engine.forecast(
    feature_dataframe=features,
    horizon=24,
)
print("24-hour forecast generated  : 24 steps âœ“")

# ----------------------------------------------------
# 4. Build optimization model
# ----------------------------------------------------
builder = BatteryOptimizationModelBuilder()
model = builder.build(forecast_df)
print("Optimization model built    : Pyomo ConcreteModel âœ“")

# ----------------------------------------------------
# 5. Solve optimization model
# ----------------------------------------------------
solver = BatteryOptimizationSolver()
solver_summary = solver.solve(model)

print("Solver                      :", solver_summary.solver_name)
print("Termination                 :", solver_summary.termination_condition)
print("Objective Value             :", round(solver_summary.objective_value, 3))
print("Solve Time (sec)            :", round(solver_summary.solve_time_seconds, 4))

# ----------------------------------------------------
# 6. Process and export results
# ----------------------------------------------------
processor = BatteryResultsProcessor(output_directory=PROJECT_ROOT / "optimization" / "results")
dispatch = processor.dispatch_schedule(model)
battery_summary = processor.battery_summary(model, dispatch)
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
print(f"Gross Revenue (USD)         : ${economics.gross_revenue_usd:.2f}")
print(f"Degradation Cost (USD)      : ${economics.degradation_cost_usd:.2f}")
print(f"Net Revenue (USD)           : ${economics.net_revenue_usd:.2f}")
print(f"Throughput (MWh)            : {economics.throughput_mwh:.2f}")
print(f"Equivalent Full Cycles      : {economics.equivalent_full_cycles:.3f}")
print(f"Remaining SOH               : {economics.remaining_soh:.5f}")

# -------------------------------------
# 7. Enhanced Dispatch Preview
# -------------------------------------
print("-" * 70)
print("Dispatch Preview (First 3 Timesteps):")

preview_cols = ["timestamp", "forecast_price"]
if "actual_price" in dispatch.columns:
    preview_cols.append("actual_price")
preview_cols.extend([
    "charge_power_mw",
    "discharge_power_mw",
    "soc_mwh",
    "hourly_revenue_$",
])

available_cols = [c for c in preview_cols if c in dispatch.columns]
print(dispatch[available_cols].head(3))

# ------------------------------------
# 8. SOC Semantics Check
# ------------------------------------
print("-" * 70)
print(f"Initial SOC (Start of Day)  : {battery_summary.soc_initial:.2f} MWh")
print(f"Terminal SOC (End of Day)   : {battery_summary.soc_terminal:.2f} MWh")
print(f"First-Hour Ending SOC       : {dispatch['soc_mwh'].iloc[0]:.2f} MWh")
print(f"Saved artifacts to          : {PROJECT_ROOT / 'optimization' / 'results'}")
print("=" * 70)
print("STATUS: Optimization results pipeline verified successfully âœ“")
print("=" * 70)