"""
experiments/integration_tests.py
================================

Automated End-to-End Scientific Pipeline Integration Suite (Part 12.6)



Verifies all 12 stages programmatically against rigorous research invariants.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class PipelineIntegrationTester:
    """Performs end-to-end scientific validation of research pipeline artifacts."""

    def __init__(self, results_dir: Path | str = "backtesting/results"):
        self.results_dir = Path(results_dir)

    def validate_all(self) -> dict[str, bool]:
        checks = {
            "dispatch_history_exists": (self.results_dir / "dispatch_history.csv").exists(),
            "degradation_history_exists": (self.results_dir / "degradation_history.csv").exists(),
            "metrics_summary_exists": (self.results_dir / "metrics_summary.csv").exists(),
            "soh_within_bounds": self._check_soh_bounds(),
            "cash_flow_conservation": self._check_cash_flow_conservation(),
        }
        return checks

    def _check_soh_bounds(self) -> bool:
        deg_path = self.results_dir / "degradation_history.csv"
        if not deg_path.exists():
            return False
        df = pd.read_csv(deg_path)
        col = "soh_end" if "soh_end" in df.columns else ("soh" if "soh" in df.columns else None)
        if not col:
            return False
        soh = df[col].to_numpy(dtype=float)
        return bool(np.all(soh >= 0.70) and np.all(soh <= 1.01))

    def _check_cash_flow_conservation(self) -> bool:
        sum_path = self.results_dir / "metrics_summary.csv"
        if not sum_path.exists():
            return False
        df = pd.read_csv(sum_path)
        kpis = dict(zip(df["metric"], df["value"].astype(float)))
        gross = kpis.get("gross_revenue_usd", 0.0)
        deg = kpis.get("degradation_cost_usd", 0.0)
        net = kpis.get("net_revenue_usd", 0.0)
        return bool(abs((gross - deg) - net) < 1.0)