"""
forecasting/trainer.py
======================

Forecasting trainer and walk-forward validation engine.

Implements
----------
1. Chronological train/validation/test split.
2. Expanding-window walk-forward validation.
3. Daily retraining.
4. Model persistence.
5. Fold summary generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

import joblib
import numpy as np
import pandas as pd

from forecasting.models import (
    RandomForestForecaster,
    XGBoostForecaster,
)


# ==========================================================
# Configuration
# ==========================================================

@dataclass(slots=True)
class TrainerConfig:
    """Configuration for forecasting trainer."""

    target_column: str = "price"
    train_fraction: float = 0.80
    validation_fraction: float = 0.10
    retrain_frequency_hours: int = 24

    model_directory: Path = field(
        default_factory=lambda: Path("forecasting/saved_models")
    )
    xgboost_filename: str = "xgboost_dayahead.pkl"
    random_forest_filename: str = "random_forest_dayahead.pkl"


# ==========================================================
# Dataset Split
# ==========================================================

@dataclass(slots=True)
class SplitResult:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


# ==========================================================
# Forecast Trainer
# ==========================================================

class ForecastTrainer:
    """
    Chronological trainer with walk-forward retraining.
    """

    def __init__(self, config: TrainerConfig | None = None) -> None:
        self.config = config or TrainerConfig()
        self.config.model_directory.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------
    # Chronological Split
    # ------------------------------------------------------

    def chronological_split(self, dataframe: pd.DataFrame) -> SplitResult:
        df = dataframe.sort_index().copy()
        n = len(df)

        train_end = int(n * self.config.train_fraction)
        validation_end = int(train_end + n * self.config.validation_fraction)

        return SplitResult(
            train=df.iloc[:train_end].copy(),
            validation=df.iloc[train_end:validation_end].copy(),
            test=df.iloc[validation_end:].copy(),
        )

    # ------------------------------------------------------
    # Feature / Target Split
    # ------------------------------------------------------

    def prepare_xy(self, dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        X = dataframe.drop(columns=[self.config.target_column])
        y = dataframe[self.config.target_column]
        return X, y

    # ------------------------------------------------------
    # Train Model
    # ------------------------------------------------------

    def train(
        self,
        dataframe: pd.DataFrame,
        model_type: str = "xgboost",
    ) -> Any:
        m_type = model_type.lower()
        if m_type == "xgboost":
            model = XGBoostForecaster()
            filename = self.config.xgboost_filename
        elif m_type in ("random_forest", "randomforest"):
            model = RandomForestForecaster()
            filename = self.config.random_forest_filename
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        model.fit(dataframe, target_column=self.config.target_column)
        path = self.save_model(model, filename)
        print(f"Model saved to: {path}")

        return model

    # ------------------------------------------------------
    # Save Model
    # ------------------------------------------------------

    def save_model(self, model: Any, filename: str) -> Path:
        save_dir = Path(self.config.model_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / filename

        payload = {
            "model": getattr(model, "model", model),
            "features": getattr(model, "features", []),
            "target_column": self.config.target_column,
        }
        joblib.dump(payload, save_path)

        return save_path

    # ------------------------------------------------------
    # Load Model
    # ------------------------------------------------------

    def load_model(self, model_type: str = "xgboost") -> Any:
        m_type = model_type.lower()

        if m_type == "xgboost":
            model = XGBoostForecaster()
            model_path = self.config.model_directory / self.config.xgboost_filename
        elif m_type in ("random_forest", "randomforest"):
            model = RandomForestForecaster()
            model_path = self.config.model_directory / self.config.random_forest_filename
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        loaded_data = joblib.load(model_path)
        if isinstance(loaded_data, dict) and "model" in loaded_data:
            model.model = loaded_data["model"]
            if hasattr(model, "features") and "features" in loaded_data:
                model.features = loaded_data["features"]
        else:
            model.model = loaded_data

        return model

    # ------------------------------------------------------
    # Walk-Forward Generator
    # ------------------------------------------------------

    def walk_forward_generator(
        self,
        dataframe: pd.DataFrame,
        horizon: int = 24,
        step: int = 24,
    ) -> Generator[tuple[pd.DataFrame, pd.DataFrame], None, None]:
        split = self.chronological_split(dataframe)
        train_df = pd.concat([split.train, split.validation])
        test_df = split.test

        total = len(test_df)
        for start in range(0, total - horizon + 1, step):
            train_until = test_df.iloc[:start]
            current_train = pd.concat([train_df, train_until])
            current_test = test_df.iloc[start:start + horizon]

            yield (current_train.copy(), current_test.copy())

    # ------------------------------------------------------
    # Walk-Forward Training
    # ------------------------------------------------------

    def walk_forward_train(
        self,
        dataframe: pd.DataFrame,
        model_type: str = "xgboost",
        horizon: int = 24,
    ) -> pd.DataFrame:
        predictions = []

        for fold, (train_df, test_df) in enumerate(
            self.walk_forward_generator(
                dataframe,
                horizon=horizon,
                step=self.config.retrain_frequency_hours,
            ),
            start=1,
        ):
            model = self.train(train_df, model_type=model_type)
            forecast = model.predict(test_df)

            # Defensive extraction of prediction values
            if hasattr(forecast, "prediction"):
                pred_vals = forecast.prediction
            elif isinstance(forecast, pd.DataFrame) and "prediction" in forecast.columns:
                pred_vals = forecast["prediction"].values
            elif isinstance(forecast, (pd.Series, np.ndarray)):
                pred_vals = np.asarray(forecast)
            else:
                pred_vals = forecast

            result = pd.DataFrame({
                "timestamp": test_df.index,
                "actual": test_df[self.config.target_column].values,
                "prediction": pred_vals,
                "fold": fold,
                "model": model.__class__.__name__,
            })
            predictions.append(result)

        if not predictions:
            return pd.DataFrame(columns=["timestamp", "actual", "prediction", "fold", "model"])

        return pd.concat(predictions, ignore_index=True)

    # ------------------------------------------------------
    # Fold Summary
    # ------------------------------------------------------

    def fold_summary(
        self,
        dataframe: pd.DataFrame,
        horizon: int = 24,
    ) -> pd.DataFrame:
        summary = []

        for fold, (train_df, test_df) in enumerate(
            self.walk_forward_generator(
                dataframe,
                horizon=horizon,
                step=self.config.retrain_frequency_hours,
            ),
            start=1,
        ):
            summary.append({
                "fold": fold,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "train_start": train_df.index.min(),
                "train_end": train_df.index.max(),
                "test_start": test_df.index.min(),
                "test_end": test_df.index.max(),
            })

        return pd.DataFrame(summary)

    # ------------------------------------------------------
    # Export Walk-Forward Summary
    # ------------------------------------------------------

    def export_fold_summary(
        self,
        dataframe: pd.DataFrame,
        output_path: str | Path = "forecasting/walk_forward_summary.csv",
        horizon: int = 24,
    ) -> Path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        summary = self.fold_summary(dataframe, horizon=horizon)
        summary.to_csv(out_p, index=False)

        return out_p