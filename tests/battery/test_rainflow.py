"""
Unit tests for ASTM E1049 Rainflow Cycle Counter.
"""

import pytest

from battery.rainflow import RainflowCycleCounter

counter = RainflowCycleCounter()


SOC_TRACE = [
    0.50,
    0.80,
    0.30,
    0.90,
    0.40,
    0.70,
    0.20,
    0.50,
]


def test_reversal_detection():

    reversals = list(counter.reversals(SOC_TRACE))

    assert len(reversals) >= 5

    assert reversals[0][1] == 0.50
    assert reversals[-1][1] == 0.50


def test_cycle_extraction():

    cycles = counter.extract_cycles(SOC_TRACE)

    assert len(cycles) > 0


def test_cycle_depth_positive():

    cycles = counter.extract_cycles(SOC_TRACE)

    assert all(c.depth_of_discharge >= 0 for c in cycles)


def test_cycle_mean_bounds():

    cycles = counter.extract_cycles(SOC_TRACE)

    assert all(0 <= c.mean_soc <= 1 for c in cycles)


def test_half_and_full_cycles():

    summary = counter.summarize(SOC_TRACE)

    assert summary.full_cycles >= 0
    assert summary.half_cycles >= 0


def test_equivalent_full_cycles():

    summary = counter.summarize(SOC_TRACE)

    assert summary.equivalent_full_cycles > 0


def test_energy_throughput():

    summary = counter.summarize(SOC_TRACE)

    assert summary.total_energy_throughput_mwh > 0


def test_single_charge_discharge_cycle():

    soc = [0.50, 1.00, 0.00, 0.50]

    summary = counter.summarize(soc)

    assert summary.equivalent_full_cycles == pytest.approx(1.0, abs=0.2)