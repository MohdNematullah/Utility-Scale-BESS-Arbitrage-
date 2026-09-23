"""
experiments
===========

 Part 12: Final Research Validation, Reproducibility & Experiment Suite.

Modules:
- experiment_suite: 28-scenario experimental research matrix
- reproducibility: seed management, dataset/config hashing, replay audit
- runtime_benchmark: computational profiling, memory, and stage timing
- artifact_manifest: complete SHA256 cryptographic indexing of generated outputs
- environment_snapshot: OS, Python, git, package versions, and hardware configuration
- integration_tests: automated programmatic end-to-end pipeline validation
- final_research_report: summary, master workbook, and LaTeX deliverables
"""

from experiments.artifact_manifest import ArtifactManifestGenerator
from experiments.environment_snapshot import EnvironmentSnapshotter
from experiments.experiment_suite import ExperimentSuiteRunner
from experiments.final_research_report import FinalResearchReportBuilder
from experiments.integration_tests import PipelineIntegrationTester
from experiments.reproducibility import ReproducibilityEngine
from experiments.runtime_benchmark import RuntimeBenchmarker

__all__ = [
    "ArtifactManifestGenerator",
    "EnvironmentSnapshotter",
    "ExperimentSuiteRunner",
    "FinalResearchReportBuilder",
    "PipelineIntegrationTester",
    "ReproducibilityEngine",
    "RuntimeBenchmarker",
]