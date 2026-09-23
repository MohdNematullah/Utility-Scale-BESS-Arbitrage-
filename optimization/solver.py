"""
solver.py
=========

HiGHS solver interface for .

Solves rolling-horizon battery arbitrage optimization model.

Outputs
-------
- Dispatch schedule
- Revenue summary
- Solver metadata
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from time import perf_counter

import pandas as pd
import pyomo.environ as pyo

# ==========================================================
# Solver Summary
# ==========================================================

@dataclass(slots=True)
class SolverSummary:
    solver_name: str
    termination_condition: str
    objective_value: float
    revenue: float
    solve_time_seconds: float
    horizon: int
    feasible: bool

# ==========================================================
# Battery Solver
# ==========================================================

class BatteryOptimizationSolver:

    def __init__(
        self,
        solver_name: str = "appsi_highs",
        tee: bool = False,
        time_limit: int = 60,
        mip_gap: float = 0.0,
    ):
        self.solver_name = solver_name
        self.tee = tee
        self.time_limit = time_limit
        self.mip_gap = mip_gap

        self.solver = self._create_solver()

    # ------------------------------------------------------
    # Create HiGHS Solver
    # ------------------------------------------------------

    def _create_solver(self):

        solver = pyo.SolverFactory(self.solver_name)

        if solver is None:
            raise RuntimeError(
                f"Solver '{self.solver_name}' not found."
            )

        if not solver.available():
            raise RuntimeError(
                "HiGHS solver is unavailable.\n"
                "Install using:\n"
                "pip install highspy"
            )

        # HiGHS options
        solver.options["time_limit"] = self.time_limit
        solver.options["mip_gap"] = self.mip_gap

        return solver

    # ------------------------------------------------------
    # Solve Model
    # ------------------------------------------------------

    def solve(self, model) -> SolverSummary:
        """
        Solve the battery optimization model and return a standardized summary.
        Compatible with Pyomo + HiGHS (appsi_highs).
        """

        start = perf_counter()

        results = self.solver.solve(
            model,
            tee=self.tee,
        )

        elapsed = perf_counter() - start

        # --------------------------------------------------
        # Read solver information (Pyomo API)
        # --------------------------------------------------
        solver_status = str(results.solver.status)
        termination = str(results.solver.termination_condition)

        # --------------------------------------------------
        # Verify optimal solution
        # --------------------------------------------------
        if not pyo.check_optimal_termination(results):
            raise RuntimeError(
                f"Optimization failed.\n"
                f"Status: {solver_status}\n"
                f"Termination: {termination}"
            )

        # --------------------------------------------------
        # Objective value
        # --------------------------------------------------
        objective_value = float(pyo.value(model.objective))

        return SolverSummary(
            solver_name=self.solver_name,
            termination_condition=termination,
            objective_value=objective_value,
            revenue=objective_value,
            solve_time_seconds=round(elapsed, 4),
            horizon=len(model.T),
            feasible=True,
        )

    # ------------------------------------------------------
    # Extract Dispatch Schedule
    # ------------------------------------------------------

    def dispatch_schedule(self, model) -> pd.DataFrame:

        forecast = model.forecast_dataframe.copy()

        dataframe = pd.DataFrame({
            "timestamp": forecast["timestamp"],
            "forecast_price": forecast["prediction"],
            "charge_power_mw": [
                pyo.value(model.charge_power[t])
                for t in model.T
            ],
            "discharge_power_mw": [
                pyo.value(model.discharge_power[t])
                for t in model.T
            ],
            "soc_mwh": [
                pyo.value(model.soc[t])
                for t in model.T
            ],
        })

        dataframe["net_power_mw"] = (
            dataframe["discharge_power_mw"]
            - dataframe["charge_power_mw"]
        )

        dataframe["hourly_revenue_$"] = (
            dataframe["forecast_price"]
            * dataframe["net_power_mw"]
        )

        dataframe["cumulative_revenue_$"] = (
            dataframe["hourly_revenue_$"].cumsum()
        )

        return dataframe.round(4)

    # ------------------------------------------------------
    # Export Results
    # ------------------------------------------------------

    def export_results(
        self,
        model,
        summary: SolverSummary,
        output_dir: str = "optimization/results",
    ):

        output_path = Path(output_dir)
        output_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        dispatch = self.dispatch_schedule(model)

        dispatch_file = output_path / "dispatch_schedule.csv"
        dispatch.to_csv(dispatch_file, index=False)

        summary_file = output_path / "solver_summary.csv"

        pd.DataFrame([asdict(summary)]).to_csv(
            summary_file,
            index=False,
        )

        return dispatch_file, summary_file

    # ------------------------------------------------------
    # Revenue Helper
    # ------------------------------------------------------

    def total_revenue(self, model):

        return float(
            pyo.value(model.objective)
        )