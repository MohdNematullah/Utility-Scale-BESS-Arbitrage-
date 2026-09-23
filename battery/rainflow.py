"""
battery/rainflow.py
===================

ASTM E1049 Rainflow Cycle Counting implementation.

implementation for .

Outputs:
- Half cycles
- Full cycles
- Cycle ranges
- Mean SOC
- Equivalent Full Cycles (EFC)
- Energy throughput
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

from battery.config import DEFAULT_BATTERY_CONFIG


# ---------------------------------------------------------------------
# Cycle Dataclass
# ---------------------------------------------------------------------

@dataclass(slots=True)
class RainflowCycle:
    start_index: int
    end_index: int

    depth_of_discharge: float
    mean_soc: float

    count: float          # 0.5 or 1.0

    energy_mwh: float


# ---------------------------------------------------------------------
# Summary Dataclass
# ---------------------------------------------------------------------

@dataclass(slots=True)
class RainflowSummary:
    cycles: list[RainflowCycle]

    full_cycles: float
    half_cycles: float

    equivalent_full_cycles: float
    total_energy_throughput_mwh: float


# ---------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------

class RainflowCycleCounter:

    def __init__(self, config=DEFAULT_BATTERY_CONFIG):
        self.config = config

    # ------------------------------------------------------------
    # Reversal Detection
    # ------------------------------------------------------------

    @staticmethod
    def reversals(series: Iterable[float]):
        values = list(series)

        if len(values) < 2:
            return

        yield (0, values[0])

        for i in range(1, len(values) - 1):

            prev_ = values[i - 1]
            cur_ = values[i]
            next_ = values[i + 1]

            if (cur_ >= prev_ and cur_ > next_) or (
                cur_ <= prev_ and cur_ < next_
            ):
                yield (i, cur_)

        yield (len(values) - 1, values[-1])

    # ------------------------------------------------------------
    # ASTM E1049 Three-Point Rainflow
    # ------------------------------------------------------------

    def extract_cycles(
        self,
        soc_series: Iterable[float],
    ) -> list[RainflowCycle]:

        points = deque()

        cycles: list[RainflowCycle] = []

        def make_cycle(p1, p2, count):

            i1, s1 = p1
            i2, s2 = p2

            dod = abs(s2 - s1)
            mean_soc = 0.5 * (s1 + s2)

            energy = (
                dod
                * self.config.chemistry.nominal_capacity_mwh
            )

            return RainflowCycle(
                start_index=i1,
                end_index=i2,
                depth_of_discharge=dod,
                mean_soc=mean_soc,
                count=count,
                energy_mwh=energy,
            )

        for point in self.reversals(soc_series):

            points.append(point)

            while len(points) >= 3:

                x1, x2, x3 = (
                    points[-3][1],
                    points[-2][1],
                    points[-1][1],
                )

                X = abs(x3 - x2)
                Y = abs(x2 - x1)

                if X < Y:
                    break

                if len(points) == 3:
                    cycles.append(make_cycle(points[0], points[1], 0.5))
                    points.popleft()

                else:
                    cycles.append(make_cycle(points[-3], points[-2], 1.0))

                    last = points.pop()
                    points.pop()
                    points.pop()
                    points.append(last)

        while len(points) > 1:
            cycles.append(make_cycle(points[0], points[1], 0.5))
            points.popleft()

        return cycles

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    def summarize(
            self,
            soc_series: Iterable[float],
    ) -> RainflowSummary:
        """
        Summarize rainflow counting results from a SOC trajectory.
        """

        cycles = self.extract_cycles(soc_series)

        full_cycles = sum(
            cycle.count for cycle in cycles
            if cycle.count == 1.0
        )

        half_cycles = sum(
            cycle.count for cycle in cycles
            if cycle.count == 0.5
        )

        equivalent_full_cycles = sum(
            cycle.depth_of_discharge * cycle.count
            for cycle in cycles
        )

        throughput = sum(
            cycle.energy_mwh * cycle.count * 2.0
            for cycle in cycles
        )

        return RainflowSummary(
            cycles=cycles,
            full_cycles=full_cycles,
            half_cycles=half_cycles,
            equivalent_full_cycles=equivalent_full_cycles,
            total_energy_throughput_mwh=throughput,
        )