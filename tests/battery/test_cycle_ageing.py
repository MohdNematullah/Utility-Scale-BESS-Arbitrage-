from battery.cycle_ageing import CycleAgeingModel

model = CycleAgeingModel()


def test_equivalent_full_cycles():

    efc = model.equivalent_full_cycles(100, 100)

    assert efc == 1.0


def test_half_cycle():

    efc = model.equivalent_full_cycles(50, 50)

    assert efc == 0.5


def test_dod_factor_reference():

    assert model.dod_factor(0.80) == 1.0


def test_dod_factor_high():

    assert model.dod_factor(1.0) > 1.0


def test_temperature_factor_reference():

    assert round(model.temperature_factor(25), 4) == 1.0


def test_temperature_factor_hotter():

    assert model.temperature_factor(45) > 1.0


def test_capacity_loss_positive():

    loss = model.capacity_loss(
        charged_energy_mwh=100,
        discharged_energy_mwh=100,
        average_dod=0.80,
        average_c_rate=1.0,
        temperature_c=25,
    )

    assert loss > 0


def test_update_soh():

    result = model.update_soh(
        initial_soh=1.0,
        charged_energy_mwh=100,
        discharged_energy_mwh=100,
        average_dod=0.80,
        average_c_rate=1.0,
        temperature_c=25,
    )

    assert result.remaining_soh < 1.0
    assert result.remaining_capacity_mwh < 100.0