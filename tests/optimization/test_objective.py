
from pyomo.environ import (
    ConcreteModel,
    RangeSet,
    Param,
    Var,
    NonNegativeReals,
)

from optimization.objective import (
    attach_objective,
)


def build_model():

    model = ConcreteModel()

    model.T = RangeSet(0, 23)

    model.delta_t = Param(initialize=1)

    model.price = Param(
        model.T,
        initialize={
            t: 50
            for t in range(24)
        },
    )

    model.forecast_price = Param(
        model.T,
        initialize={
            t: 50
            for t in range(24)
        },
    )

    model.degradation_cost = Param(
        model.T,
        initialize={
            t: 0
            for t in range(24)
        },
    )

    model.charge_power = Var(
        model.T,
        domain=NonNegativeReals,
    )

    model.discharge_power = Var(
        model.T,
        domain=NonNegativeReals,
    )

    return model


def test_energy_objective():

    model = build_model()

    attach_objective(
        model,
        "energy_arbitrage",
    )

    assert model.objective is not None


def test_forecast_objective():

    model = build_model()

    attach_objective(
        model,
        "forecast_value",
    )

    assert model.objective is not None


def test_degradation_objective():

    model = build_model()

    attach_objective(
        model,
        "degradation_penalty",
    )

    assert model.objective is not None