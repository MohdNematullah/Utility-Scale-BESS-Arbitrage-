"""
trainer.py
==========

forecasting trainer for .

Implements
----------
1. Chronological train/validation/test split.
2. Expanding-window walk-forward validation.
3. Daily retraining.
4. Model persistence.
5. Fold summary generation.

Author:  """

from dataclasses import dataclass, field
from pathlib import Path
import joblib
import pandas as pd

from forecasting.models import (
    XGBoostForecaster,
    RandomForestForecaster,
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

    def __init__(self, config: TrainerConfig = TrainerConfig()):
        self.config = config

        self.config.model_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ------------------------------------------------------
    # Chronological Split
    # ------------------------------------------------------

    def chronological_split(
        self,
        dataframe: pd.DataFrame,
    ) -> SplitResult:

        dataframe = dataframe.sort_index().copy()

        n = len(dataframe)

        train_end = int(n * self.config.train_fraction)
        validation_end = int(
            train_end + n * self.config.validation_fraction
        )

        return SplitResult(
            train=dataframe.iloc[:train_end].copy(),
            validation=dataframe.iloc[
                train_end:validation_end
            ].copy(),
            test=dataframe.iloc[
                validation_end:
            ].copy(),
        )

    # ------------------------------------------------------
    # Feature / Target Split
    # ------------------------------------------------------

    def prepare_xy(
        self,
        dataframe: pd.DataFrame,
    ):

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
    ):
        if model_type == "xgboost":
            model = XGBoostForecaster()
            filename = self.config.xgboost_filename

        elif model_type == "random_forest":
            model = RandomForestForecaster()
            filename = self.config.random_forest_filename

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        model.fit(
            dataframe,
            target_column=self.config.target_column,
        )

        path = self.save_model(model, filename)

        print(f"Model saved to: {path}")

        return model

    # ------------------------------------------------------
    # Save Model
    # ------------------------------------------------------

    def save_model(self, model, filename: str) -> Path:
        save_dir = Path(self.config.model_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        save_path = save_dir / filename

        joblib.dump(
            {
                "model": model.model,
                "features": model.features,
                "target_column": self.config.target_column,
            },
            save_path,
        )

        return save_path
    
    # ------------------------------------------------------
    # Load Model
    # ------------------------------------------------------

    def load_model(
        self,
        model_type: str = "xgboost",
    ):

        model_type = model_type.lower()

        if model_type == "xgboost":

            model = XGBoostForecaster()

            model_path = (
                self.config.model_directory
                / self.config.xgboost_filename
            )

        elif model_type == "random_forest":

            model = RandomForestForecaster()

            model_path = (
                self.config.model_directory
                / self.config.random_forest_filename
            )

        else:
            raise ValueError(model_type)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        model.model = joblib.load(model_path)

        return model

    # ------------------------------------------------------
    # Walk-Forward Generator
    # ------------------------------------------------------

    def walk_forward_generator(
        self,
        dataframe: pd.DataFrame,
        horizon: int = 24,
        step: int = 24,
    ):
        """
        Expanding-window walk-forward folds.

        Yields
        ------
        train_dataframe, test_dataframe
        """

        split = self.chronological_split(dataframe)

        train_df = pd.concat(
            [split.train, split.validation]
        )

        test_df = split.test

        total = len(test_df)

        for start in range(
            0,
            total - horizon + 1,
            step,
        ):

            train_until = test_df.iloc[:start]

            current_train = pd.concat(
                [train_df, train_until]
            )

            current_test = test_df.iloc[
                start:start + horizon
            ]

            yield (
                current_train.copy(),
                current_test.copy(),
            )

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

        for fold, (
            train_df,
            test_df,
        ) in enumerate(
            self.walk_forward_generator(
                dataframe,
                horizon=horizon,
                step=self.config.retrain_frequency_hours,
            ),
            start=1,
        ):

            model = self.train(
                train_df,
                model_type=model_type,
            )

            forecast = model.predict(test_df)

            result = pd.DataFrame({
                "timestamp": test_df.index,
                "actual": test_df[
                    self.config.target_column
                ].values,
                "prediction": forecast.prediction,
                "fold": fold,
                "model": model.__class__.__name__,
            })

            predictions.append(result)

        return pd.concat(
            predictions,
            ignore_index=True,
        )

    # ------------------------------------------------------
    # Fold Summary
    # ------------------------------------------------------

    def fold_summary(
        self,
        dataframe: pd.DataFrame,
        horizon: int = 24,
    ) -> pd.DataFrame:

        summary = []

        for fold, (
            train_df,
            test_df,
        ) in enumerate(
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
        output_path: str | Path = (
            "forecasting/walk_forward_summary.csv"
        ),
        horizon: int = 24,
    ) -> Path:

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        summary = self.fold_summary(
            dataframe,
            horizon=horizon,
        )

        summary.to_csv(
            output_path,
            index=False,
        )

        return output_path