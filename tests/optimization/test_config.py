
from optimization.config import DEFAULT_CONFIG


def test_validation():

    assert DEFAULT_CONFIG.validate() is True


def test_soc_limits():

    battery = DEFAULT_CONFIG.battery

    assert battery.soc_min_mwh == 10.0
    assert battery.soc_max_mwh == 90.0


def test_round_trip_efficiency():

    battery = DEFAULT_CONFIG.battery

    assert round(
        battery.round_trip_efficiency,
        4,
    ) == 0.9025


def test_initial_soc():

    battery = DEFAULT_CONFIG.battery

    assert battery.initial_soc_mwh == 50.0


def test_dataframe():

    df = DEFAULT_CONFIG.to_dataframe()

    assert len(df) > 10