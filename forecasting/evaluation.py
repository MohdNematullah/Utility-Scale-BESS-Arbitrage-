
"""
evaluation.py
=============

Forecast Evaluation Framework
for .

Evaluates recursive forecasts using statistical
metrics and horizon-wise diagnostics.
"""

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(slots=True)
class EvaluationReport:
    model: str
    horizon: int
    mae: float
    rmse: float
    smape: float
    r2: float
    bias: float
    directional_accuracy: float
    relative_mae: float


class ForecastEvaluator:

    def __init__(self):
        pass

    # -------------------------------
    # Validation
    # -------------------------------

    def validate(
            self,
            dataframe: pd.DataFrame,
    ):

        required = {"prediction", "actual"}

        missing = required.difference(dataframe.columns)

        if missing:
            raise ValueError(
                f"Missing columns: {missing}"
            )

        dataframe = dataframe.dropna(
            subset=["prediction", "actual"]
        ).copy()

        if dataframe.empty:
            raise ValueError(
                "No overlapping forecast and actual timestamps found."
            )

        return dataframe

    # -------------------------------
    # MAE
    # -------------------------------
    def mae(self, y_true, y_pred):

        return float(
            np.mean(np.abs(y_true - y_pred))
        )

    # -------------------------------
    # RMSE
    # -------------------------------
    def rmse(self, y_true, y_pred):

        return float(
            np.sqrt(np.mean((y_true - y_pred) ** 2))
        )

    # -------------------------------
    # SMAPE
    # Suitable for electricity prices
    # -------------------------------
    def smape(self, y_true, y_pred):

        denominator = (
            np.abs(y_true) + np.abs(y_pred)
        )

        denominator = np.where(
            denominator == 0,
            1e-8,
            denominator,
        )

        value = np.mean(
            2 * np.abs(y_true - y_pred)
            / denominator
        )

        return float(value * 100)

    # -------------------------------
    # RÂ²
    # -------------------------------
    def r2(self, y_true, y_pred):

        ss_res = np.sum((y_true - y_pred) ** 2)

        ss_tot = np.sum(
            (y_true - np.mean(y_true)) ** 2
        )

        if ss_tot == 0:
            return 0.0

        return float(
            1 - ss_res / ss_tot
        )

    # -------------------------------
    # Mean Bias Error
    # -------------------------------
    def bias(self, y_true, y_pred):

        return float(
            np.mean(y_pred - y_true)
        )

    # -------------------------------
    # Directional Accuracy
    # -------------------------------
    def directional_accuracy(
        self,
        y_true,
        y_pred,
    ):

        true_direction = np.sign(
            np.diff(y_true)
        )

        pred_direction = np.sign(
            np.diff(y_pred)
        )

        accuracy = (
            true_direction == pred_direction
        ).mean()

        return float(accuracy * 100)


    # -------------------------------
    # Relative MAE against persistence
    # -------------------------------
    def relative_mae(
        self,
        dataframe: pd.DataFrame,
    ):

        naive = dataframe["actual"].shift(1)

        valid = dataframe.copy()

        valid["naive"] = naive

        valid = valid.dropna()

        mae_model = self.mae(
            valid.actual.values,
            valid.prediction.values,
        )

        mae_naive = self.mae(
            valid.actual.values,
            valid.naive.values,
        )

        if mae_naive == 0:
            return 0.0

        return float(mae_model / mae_naive)


    # -------------------------------
    # Global Evaluation
    # -------------------------------
    def evaluate(
        self,
        forecast_dataframe: pd.DataFrame,
        model_name="RecursiveForecast",
        horizon=24,
    ) -> EvaluationReport:

        df = self.validate(forecast_dataframe)

        y_true = df.actual.values
        y_pred = df.prediction.values

        return EvaluationReport(
            model=model_name,
            horizon=horizon,
            mae=self.mae(y_true, y_pred),
            rmse=self.rmse(y_true, y_pred),
            smape=self.smape(y_true, y_pred),
            r2=self.r2(y_true, y_pred),
            bias=self.bias(y_true, y_pred),
            directional_accuracy=self.directional_accuracy(
                y_true,
                y_pred,
            ),
            relative_mae=self.relative_mae(df),
        )


    # -------------------------------
    # Metrics for each forecast hour
    # -------------------------------
    def horizon_metrics(
        self,
        dataframe: pd.DataFrame,
    ):

        df = self.validate(dataframe)

        metrics = []

        for step in sorted(df.step.unique()):

            hour_df = df[df.step == step]

            y_true = hour_df.actual.values
            y_pred = hour_df.prediction.values

            metrics.append(
                {
                    "step": step,
                    "mae": self.mae(y_true, y_pred),
                    "rmse": self.rmse(y_true, y_pred),
                    "smape": self.smape(y_true, y_pred),
                    "bias": self.bias(y_true, y_pred),
                }
            )

        return pd.DataFrame(metrics)


    # -------------------------------
    # Daily Evaluation
    # -------------------------------
    def daily_summary(
        self,
        dataframe: pd.DataFrame,
    ):

        df = self.validate(dataframe)

        df["date"] = (
            pd.to_datetime(df.timestamp)
            .dt.date
        )

        summary = (
            df.groupby("date")
            .apply(
                lambda x: pd.Series(
                    {
                        "MAE": self.mae(
                            x.actual.values,
                            x.prediction.values,
                        ),
                        "RMSE": self.rmse(
                            x.actual.values,
                            x.prediction.values,
                        ),
                        "SMAPE": self.smape(
                            x.actual.values,
                            x.prediction.values,
                        ),
                        "Bias": self.bias(
                            x.actual.values,
                            x.prediction.values,
                        ),
                    }
                )
            )
            .reset_index()
        )

        return summary

    # ---------------------------------------------
    # Export evaluation CSVs
    # ---------------------------------------------
    def export(
            self,
            dataframe,
            output_directory="forecasting/results",
    ):

        output_directory = Path(output_directory)

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        report = self.evaluate(dataframe)

        horizon = self.horizon_metrics(dataframe)

        daily = self.daily_summary(dataframe)

        # Dataclass with slots=True
        report_df = pd.DataFrame([asdict(report)])

        report_df.to_csv(
            output_directory / "global_metrics.csv",
            index=False,
        )

        horizon.to_csv(
            output_directory / "hourly_metrics.csv",
            index=False,
        )

        daily.to_csv(
            output_directory / "daily_metrics.csv",
            index=False,
        )

        return report