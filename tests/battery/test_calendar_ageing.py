import pytest

from battery.calendar_ageing import CalendarAgeingModel


model = CalendarAgeingModel()


def test_temperature_factor_reference():
    assert model.temperature_factor(25.0) == pytest.approx(1.0, rel=1e-6)


def test_temperature_factor_hotter():
    assert model.temperature_factor(45.0) > 1.0


def test_soc_factor_midpoint():
    assert model.soc_factor(0.50) == pytest.approx(1.0)


def test_soc_factor_high_soc():
    assert model.soc_factor(0.90) > model.soc_factor(0.50)


def test_capacity_loss_positive():
    loss = model.capacity_loss(
        days=365,
        average_soc=0.50,
        temperature_c=25.0,
    )
    assert loss > 0
    assert loss < 1


def test_soh_update():
    result = model.update_soh(
        initial_soh=1.0,
        days=365,
        average_soc=0.50,
        temperature_c=25.0,
    )

    assert result.remaining_soh < 1.0
    assert result.remaining_capacity_mwh > 0


def test_end_of_life_floor():
    result = model.update_soh(
        initial_soh=0.72,
        days=10000,
        average_soc=1.0,
        temperature_c=60.0,
    )

    assert result.remaining_soh >= model.config.ageing.end_of_life_soh