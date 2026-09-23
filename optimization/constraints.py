"""
constraints.py
==============

battery operational constraints.

Implements:
• End-of-interval SOC dynamics
• Charge/discharge power limits
• SOC operating boundaries
• Initial SOC continuity
• Terminal SOC targets
• LP-compatible non-simultaneous charge/discharge constraints
"""

from pyomo.environ import Constraint


# ============================================================
# Initial SOC Constraint
# ============================================================

def initial_soc_constraint(model):
    """Stored energy at the end of the first hour."""
    first = model.T.first()
    return Constraint(
        expr=model.soc[first]
        == model.initial_soc
        + (
            model.charge_efficiency
            * model.charge_power[first]
            - model.discharge_power[first]
            / model.discharge_efficiency
        ) * model.delta_t
    )


# ============================================================
# State-of-Charge Dynamics
# ============================================================

def soc_dynamics_constraint(model):
    """Stored energy at the end of each later hour."""
    first = model.T.first()

    def rule(m, t):
        if t == first:
            return Constraint.Skip

        previous = t - 1
        return (
            m.soc[t]
            == m.soc[previous]
            + (
                m.charge_efficiency * m.charge_power[t]
                - m.discharge_power[t] / m.discharge_efficiency
            ) * m.delta_t
        )

    return Constraint(model.T, rule=rule)


# ============================================================
# SOC Minimum
# ============================================================

def soc_min_constraint(model):
    return Constraint(
        model.T,
        rule=lambda m, t: m.soc[t] >= m.soc_min,
    )


# ============================================================
# SOC Maximum
# ============================================================

def soc_max_constraint(model):
    return Constraint(
        model.T,
        rule=lambda m, t: m.soc[t] <= m.soc_max,
    )


# ============================================================
# Charge Power Limit
# ============================================================

def charge_power_limit(model):
    return Constraint(
        model.T,
        rule=lambda m, t: (
            m.charge_power[t]
            <= m.max_charge_power
            * m.remaining_capacity_fraction
        ),
    )


# ============================================================
# Discharge Power Limit
# ============================================================

def discharge_power_limit(model):
    return Constraint(
        model.T,
        rule=lambda m, t: (
            m.discharge_power[t]
            <= m.max_discharge_power
            * m.remaining_capacity_fraction
        ),
    )


# ============================================================
# Terminal SOC
# ============================================================

def terminal_soc_constraint(model):
    """
    End horizon with desired SOC evaluated after the final action.
    """
    last = model.T.last()
    return Constraint(
        expr=model.soc[last] == model.terminal_soc
    )


# ============================================================
# LP-Compatible No Simultaneous Charge/Discharge
# ============================================================

def relaxed_operation_constraint(model):
    """
    LP relaxation: charge + discharge <= max_power.
    """
    limit = max(
        float(model.max_charge_power.value),
        float(model.max_discharge_power.value),
    )
    return Constraint(
        model.T,
        rule=lambda m, t: (
            m.charge_power[t] + m.discharge_power[t] <= limit
        ),
    )


# ============================================================
# Attach All Constraints
# ============================================================

def attach_constraints(model):
    """
    Attach every battery constraint to the Pyomo model instance.
    """
    model.initial_soc_constraint = initial_soc_constraint(model)
    model.soc_dynamics_constraint = soc_dynamics_constraint(model)
    model.soc_min_constraint = soc_min_constraint(model)
    model.soc_max_constraint = soc_max_constraint(model)
    model.charge_limit_constraint = charge_power_limit(model)
    model.discharge_limit_constraint = discharge_power_limit(model)
    model.terminal_soc_constraint = terminal_soc_constraint(model)
    model.operation_constraint = relaxed_operation_constraint(model)

    return model