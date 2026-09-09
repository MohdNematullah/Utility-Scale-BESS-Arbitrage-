
from pyomo.environ import (
    ConcreteModel,
    RangeSet,
    Param,
    Var,
    NonNegativeReals,
)

from optimization.constraints import attach_constraints


def build_model():

    model = ConcreteModel()

    model.T = RangeSet(0, 23)

    model.delta_t = Param(initialize=1)

    model.initial_soc = Param(initialize=50)

    model.terminal_soc = Param(initialize=50)

    model.soc_min = Param(initialize=10)

    model.soc_max = Param(initialize=90)

    model.max_charge_power = Param(initialize=50)

    model.max_discharge_power = Param(initialize=50)

    model.charge_efficiency = Param(initialize=0.95)

    model.discharge_efficiency = Param(initialize=0.95)

    model.remaining_capacity_fraction = Param(initialize=1.0)

    model.soc = Var(
        model.T,
        domain=NonNegativeReals,
    )

    model.charge_power = Var(
        model.T,
        domain=NonNegativeReals,
    )

    model.discharge_power = Var(
        model.T,
        domain=NonNegativeReals,
    )

    return attach_constraints(model)


def test_constraints_created():

    model = build_model()

    assert model.initial_soc_constraint is not None
    assert model.soc_dynamics_constraint is not None
    assert model.soc_min_constraint is not None
    assert model.soc_max_constraint is not None
    assert model.charge_limit_constraint is not None
    assert model.discharge_limit_constraint is not None
    assert model.terminal_soc_constraint is not None
    assert model.operation_constraint is not None


def test_soc_constraints():

    model = build_model()

    assert len(model.soc_min_constraint) == 24
    assert len(model.soc_max_constraint) == 24


def test_charge_constraints():

    model = build_model()

    assert len(model.charge_limit_constraint) == 24


def test_discharge_constraints():

    model = build_model()

    assert len(model.discharge_limit_constraint) == 24


def test_operation_constraints():

    model = build_model()

    assert len(model.operation_constraint) == 24