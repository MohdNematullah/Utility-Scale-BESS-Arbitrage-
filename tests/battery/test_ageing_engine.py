"""
Unit tests for Combined Battery Ageing Engine.
"""

import pytest

from battery.ageing_engine import BatteryAgeingEngine

engine = BatteryAgeingEngine()

def test_calendar_only_degradation():
    result = engine.evaluate(
        initial_soh=1.0,
        calendar_days=365,
        average_soc=0.50,
        temperature_c=25.0,
        charged_energy_mwh=0.0,
        discharged_energy_mwh=0.0,
        average_dod=0.0,
        average_c_rate=0.0,
    )

    assert result.calendar_loss_fraction > 0
    assert result.cycle_loss_fraction == 0
    assert result.remaining_soh < 1.0


def test_cycle_only_degradation():
    result = engine.evaluate(
        initial_soh=1.0,
        calendar_days=0,
        average_soc=0.50,
        temperature_c=25.0,
        charged_energy_mwh=100,
        discharged_energy_mwh=100,
        average_dod=0.80,
        average_c_rate=1.0,
    )

    assert result.calendar_loss_fraction == 0
    assert result.cycle_loss_fraction > 0
    assert result.remaining_soh < 1.0


def test_combined_loss():
    result = engine.evaluate(
        initial_soh=1.0,
        calendar_days=365,
        average_soc=0.50,
        temperature_c=25.0,
        charged_energy_mwh=100,
        discharged_energy_mwh=100,
        average_dod=0.80,
        average_c_rate=1.0,
    )

    assert result.total_capacity_loss_fraction == pytest.approx(
        result.calendar_loss_fraction + result.cycle_loss_fraction
    )


def test_capacity_remaining():
    result = engine.evaluate(
        initial_soh=1.0,
        calendar_days=365,
        average_soc=0.50,
        temperature_c=25.0,
        charged_energy_mwh=100,
        discharged_energy_mwh=100,
        average_dod=0.80,
        average_c_rate=1.0,
    )

    assert result.remaining_capacity_mwh == pytest.approx(
        result.remaining_soh * 100.0
    )


def test_replacement_trigger():
    result = engine.evaluate(
        initial_soh=0.79,
        calendar_days=0,
        average_soc=0.50,
        temperature_c=25.0,
        charged_energy_mwh=0,
        discharged_energy_mwh=0,
        average_dod=0,
        average_c_rate=0,
    )

    assert result.replacement_required is True


def test_degradation_cost_positive():
    result = engine.evaluate(
        initial_soh=1.0,
        calendar_days=365,
        average_soc=0.50,
        temperature_c=25.0,
        charged_energy_mwh=100,
        discharged_energy_mwh=100,
        average_dod=0.80,
        average_c_rate=1.0,
    )

    assert result.degradation_cost_usd > 0


def test_remaining_years_positive():
    result = engine.evaluate(
        initial_soh=0.98,
        calendar_days=365,
        average_soc=0.50,
        temperature_c=25.0,
        charged_energy_mwh=100,
        discharged_energy_mwh=100,
        average_dod=0.80,
        average_c_rate=1.0,
    )

    assert result.remaining_useful_years > 0