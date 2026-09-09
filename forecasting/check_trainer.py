"""
check_trainer.py

Research-grade trainer verification script for .

Verifies:
1. Chronological train/validation/test split.
2. Walk-forward fold generation.
3. XGBoost model training and saving.
4. Random Forest model training and saving.
5. Export of walk-forward summary.
"""

from pathlib import Path

from data.loader import MarketDataLoader
from features.engineering import FeatureEngineer
from forecasting.trainer import ForecastTrainer


# ==========================================================
# Initialize pipeline
# ==========================================================

loader = MarketDataLoader()
engineer = FeatureEngineer()
trainer = ForecastTrainer()

# Create output directory
Path("forecasting").mkdir(exist_ok=True)

# ==========================================================
# Load Dataset
# ==========================================================

market_df = loader.load_csv("data/raw/ercot_prices.csv")
feature_df = engineer.transform(market_df)

# ==========================================================
# Chronological Split
# ==========================================================

split = trainer.chronological_split(feature_df)

print("=" * 70)
print("CHRONOLOGICAL SPLIT")
print("=" * 70)

print(f"Train Rows      : {len(split.train)}")
print(f"Validation Rows : {len(split.validation)}")
print(f"Test Rows       : {len(split.test)}")

# ==========================================================
# Walk-Forward Fold Summary
# ==========================================================

summary = trainer.fold_summary(feature_df)

print("\nWALK-FORWARD SUMMARY")
print("=" * 70)
print(summary.head())

summary_path = trainer.export_fold_summary(feature_df)

print("\nSaved:")
print(summary_path)

# ==========================================================
# Train XGBoost
# ==========================================================

print("\nTraining XGBoost...")

xgb_model = trainer.train(
    split.train,
    model_type="xgboost",
)

xgb_path = trainer.save_model(
    xgb_model,
    "xgboost_dayahead.pkl",
)

print(f"Model saved to: {xgb_path}")

# ==========================================================
# Train Random Forest
# ==========================================================

print("\nTraining Random Forest...")

rf_model = trainer.train(
    split.train,
    model_type="random_forest",
)

rf_path = trainer.save_model(
    rf_model,
    "random_forest_dayahead.pkl",
)

print(f"Model saved to: {rf_path}")

# ==========================================================
# Completion
# ==========================================================

print("\nTrainer check completed successfully.")