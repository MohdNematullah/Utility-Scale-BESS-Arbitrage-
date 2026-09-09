from battery.config import DEFAULT_BATTERY_CONFIG


def test_capacity():
    assert DEFAULT_BATTERY_CONFIG.chemistry.nominal_capacity_mwh == 100.0


def test_cycles():
    assert DEFAULT_BATTERY_CONFIG.chemistry.nominal_cycles > 1000


def test_efficiency():
    assert 0.85 < DEFAULT_BATTERY_CONFIG.chemistry.nominal_round_trip_efficiency <= 1.0


def test_soh_limits():
    assert DEFAULT_BATTERY_CONFIG.ageing.minimum_soh > DEFAULT_BATTERY_CONFIG.ageing.end_of_life_soh


def test_replacement_trigger():
    assert DEFAULT_BATTERY_CONFIG.replacement.replacement_trigger_soh == 0.80