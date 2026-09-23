"""
tests/experiments/test_experiments.py
=====================================

Unit & Integration Test Suite for Experiments & Reproducibility Suite.
"""

from pathlib import Path
import pytest
import pandas as pd

from experiments.artifact_manifest import ArtifactManifestGenerator
from experiments.environment_snapshot import EnvironmentSnapshotter
from experiments.experiment_suite import ExperimentSuiteRunner
from experiments.final_research_report import FinalResearchReportBuilder
from experiments.reproducibility import ReproducibilityEngine
from experiments.runtime_benchmark import RuntimeBenchmarker


class TestExperimentsSuite:
    def test_28_scenario_generation(self, tmp_path):
        runner = ExperimentSuiteRunner(output_dir=tmp_path / "exp")
        df = runner.run_all_scenarios()
        assert len(df) == 28
        assert "net_ebitda_usd" in df.columns
        assert (df["net_ebitda_usd"] > 0).all()
        assert (df["final_soh"] >= 0.90).all()

    def test_reproducibility_engine(self, tmp_path):
        repro = ReproducibilityEngine(output_dir=tmp_path / "repro", seed=42)
        p_seed = repro.record_seed_state()
        p_cfg = repro.record_config_hash({"power": 50.0})
        assert p_seed.exists()
        assert p_cfg.exists()

    def test_runtime_benchmark_export(self, tmp_path):
        bm = RuntimeBenchmarker(output_dir=tmp_path / "bench")
        c_p, j_p, f_p = bm.export_benchmark_reports()
        assert c_p.exists()
        assert j_p.exists()
        assert f_p.exists()

    def test_environment_snapshot(self, tmp_path):
        snap = EnvironmentSnapshotter(output_dir=tmp_path / "env")
        j_p, t_p = snap.capture_snapshot()
        assert j_p.exists()
        assert t_p.exists()

    def test_final_report_builder(self, tmp_path):
        runner = ExperimentSuiteRunner(output_dir=tmp_path / "exp")
        df = runner.run_all_scenarios()
        builder = FinalResearchReportBuilder(output_dir=tmp_path / "report")
        j_p, c_p, x_p, m_p = builder.build_complete__package(
            df, {"gross_revenue_usd": 4982570.0, "net_operating_profit_usd": 4350206.0}
        )
        assert j_p.exists()
        assert c_p.exists()
        assert x_p.exists()
        assert m_p.exists()

    def test_artifact_manifest_generation(self, tmp_path):
        (tmp_path / "dummy.csv").write_text("a,b\n1,2", encoding="utf-8")
        gen = ArtifactManifestGenerator(root_dir=tmp_path, output_dir=tmp_path / "manifest")
        c_p, j_p = gen.scan_and_generate()
        assert c_p.exists()
        assert j_p.exists()