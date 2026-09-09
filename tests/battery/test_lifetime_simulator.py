"""
Unit tests for Battery Lifetime Simulator.
"""

from battery.lifetime_simulator import BatteryLifetimeSimulator

simulator = BatteryLifetimeSimulator()


def test_dataframe_created():

    result = simulator.simulate(years=10)

    assert len(result.timeline) == result.years_simulated


def test_soh_decreases():

    result = simulator.simulate(years=5)

    assert result.final_soh < 1.0


def test_capacity_decreases():

    result = simulator.simulate(years=5)

    first = result.timeline.iloc[0].capacity_start_mwh
    last = result.timeline.iloc[-1].capacity_end_mwh

    assert last < first


def test_replacement_year_optional():

    result = simulator.simulate(
        years=20,
        annual_efc=600,
    )

    if result.replacement_year is not None:
        assert result.replacement_year <= 20


def test_degradation_cost_positive():

    result = simulator.simulate(years=10)

    assert result.total_degradation_cost_usd > 0


def test_export(tmp_path):

    result = simulator.simulate(years=5)

    simulator.config.results_directory = tmp_path

    path = simulator.export(result)

    assert path.exists()