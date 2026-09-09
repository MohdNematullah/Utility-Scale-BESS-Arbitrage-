"""
Unit tests for degradation cost model.
"""

import pytest

from battery.degradation_cost import BatteryDegradationCostModel

model = BatteryDegradationCostModel()


def sample():
    return model.evaluate(
        initial_soh=1.0,
        calendar_days=365,
        average_soc=0.5,
        temperature_c=25,
        charged_energy_mwh=120,
        discharged_energy_mwh=120,
        average_dod=0.8,
        average_c_rate=1.0,
    )


def test_replacement_value():
    assert model.battery_replacement_value > 0


def test_salvage_value():
    assert model.salvage_value > 0
    assert model.salvage_value < model.battery_replacement_value


def test_degradation_cost_positive():
    result = sample()
    assert result.degradation_cost_usd > 0


def test_marginal_cost_positive():
    result = sample()
    assert result.marginal_cost_per_mwh > 0


def test_remaining_capacity():
    result = sample()
    assert result.remaining_capacity_mwh == pytest.approx(
        result.remaining_soh * 100
    )


def test_total_loss_sum():
    result = sample()
    assert result.total_loss_fraction == pytest.approx(
        result.calendar_loss_fraction
        + result.cycle_loss_fraction
    )


def test_zero_throughput_cost():
    result = model.evaluate(
        initial_soh=1.0,
        calendar_days=0,
        average_soc=0.5,
        temperature_c=25,
        charged_energy_mwh=0,
        discharged_energy_mwh=0,
        average_dod=0,
        average_c_rate=0,
    )

    assert result.marginal_cost_per_mwh == 0