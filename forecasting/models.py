"""
models.py
=========

forecasting models for .

Features
--------
- XGBoost & Random Forest wrappers
- Feature name persistence
- Joblib serialization
- Safe loading
- Prediction validation
"""

from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


# ==========================================================
# Prediction Container
# ==========================================================

@dataclass(slots=True)
class ModelPrediction:
    model_name: str
    prediction: np.ndarray


# ==========================================================
# Base Model
# ==========================================================

class BaseMLForecaster:

    def __init__(self):
        self.model = None
        self.features: list[str] = []

    # ------------------------------------------------------
    # Train
    # ------------------------------------------------------
    def fit(self, dataframe: pd.DataFrame, target_column="price"):

        self.features = [
            c for c in dataframe.columns
            if c != target_column
        ]

        X = dataframe[self.features]
        y = dataframe[target_column]

        self.model.fit(X, y)

        # Recover feature names from sklearn/XGBoost if available.
        if hasattr(self.model, "feature_names_in_"):
            self.features = list(self.model.feature_names_in_)

        return self

    # ------------------------------------------------------
    # Predict
    # ------------------------------------------------------
    def predict(self, dataframe: pd.DataFrame):

        missing = [
            c for c in self.features
            if c not in dataframe.columns
        ]

        if missing:
            raise ValueError(
                f"Missing prediction features: {missing}"
            )

        X = dataframe[self.features]

        prediction = self.model.predict(X)

        return ModelPrediction(
            model_name=self.model.__class__.__name__,
            prediction=np.asarray(prediction, dtype=float),
        )

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------
    def save(self, path):
        """
        Save trained model together with feature metadata.
        """
        from pathlib import Path
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            {
                "model": self.model,
                "features": self.features,
                "model_name": self.__class__.__name__,
            },
            path,
        )

        return path

    # ------------------------------------------------------
    # Load
    # ------------------------------------------------------
    from pathlib import Path
    import joblib

    def load(self, path):
        """
        Loads a trained forecaster from disk.
        """

        path = Path(path)

        payload = joblib.load(path)

        # New format
        if isinstance(payload, dict):
            self.model = payload["model"]
            self.features = payload["features"]
            self.target_column = payload.get("target_column", "price")
        else:
            # Backward compatibility
            self.model = payload

        return self


# ==========================================================
# XGBoost
# ==========================================================

class XGBoostForecaster(BaseMLForecaster):

    def __init__(self):

        super().__init__()

        self.model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )


# ==========================================================
# Random Forest
# ==========================================================

class RandomForestForecaster(BaseMLForecaster):

    def __init__(self):

        super().__init__()

        self.model = RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )