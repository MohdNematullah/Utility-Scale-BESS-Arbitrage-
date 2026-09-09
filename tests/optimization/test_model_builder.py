import pandas as pd
import pyomo.environ as pyo

from optimization.model_builder import (
    BatteryOptimizationModelBuilder,
)


def load_forecast():
    return pd.read_csv(
        "forecasting/forecast_24h.csv"
    )


def build_model():
    builder = BatteryOptimizationModelBuilder()
    return builder.build(load_forecast())


def test_model_creation():
    model = build_model()

    assert isinstance(
        model,
        pyo.ConcreteModel,
    )


def test_horizon():
    model = build_model()

    assert len(model.T) == 24


def test_variable_blocks():
    model = build_model()

    assert len(model.soc) == 24
    assert len(model.charge_power) == 24
    assert len(model.discharge_power) == 24


def test_constraint_blocks():
    model = build_model()

    assert len(
        model.component_map(
            pyo.Constraint,
            active=True,
        )
    ) == 8


def test_summary():
    model = build_model()

    assert model.summary.horizon == 24
    assert model.summary.variables == 72
    assert model.summary.constraint_blocks == 8