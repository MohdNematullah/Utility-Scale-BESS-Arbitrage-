"""
experiments/environment_snapshot.py
===================================

Computational Environment & Reproducibility Snapshot Engine (Part 12.5)

Captures operating system, Python build, CPU architecture, Git hash,
and complete dependency versions for scientific replication.
"""

from __future__ import annotations

import datetime
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any


class EnvironmentSnapshotter:
    """Records precise computational environment configurations."""

    def __init__(self, output_dir: Path | str = "results/reproducibility"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_git_commit() -> str:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except Exception:
            return "unversioned_local"

    @staticmethod
    def get_package_versions() -> dict[str, str]:
        packages = [
            "numpy",
            "pandas",
            "matplotlib",
            "scipy",
            "openpyxl",
            "pytest",
            "pyomo",
            "xgboost",
            "lightgbm",
        ]
        versions = {}
        for pkg in packages:
            try:
                versions[pkg] = importlib.metadata.version(pkg)
            except importlib.metadata.PackageNotFoundError:
                versions[pkg] = "not_installed"
        return versions

    def capture_snapshot(self) -> tuple[Path, Path]:
        snapshot_data: dict[str, Any] = {
            "project": "BTA-V5.0",
            "version": "5.0.0",
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "os_platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
            "cpu_count_logical": os.cpu_count(),
            "git_commit": self.get_git_commit(),
            "package_dependencies": self.get_package_versions(),
        }

        json_path = self.output_dir / "environment_snapshot.json"
        txt_path = self.output_dir / "environment_snapshot.txt"

        json_path.write_text(json.dumps(snapshot_data, indent=4), encoding="utf-8")

        lines = [
            "=" * 70,
            " COMPUTATIONAL REPRODUCIBILITY ENVIRONMENT SNAPSHOT",
            "=" * 70,
            f"Timestamp (UTC)       : {snapshot_data['timestamp_utc']}",
            f"Python Runtime        : {snapshot_data['python_version']} ({snapshot_data['python_implementation']})",
            f"Operating System      : {snapshot_data['os_platform']}",
            f"CPU Logical Cores     : {snapshot_data['cpu_count_logical']}",
            f"Git Revision Commit   : {snapshot_data['git_commit']}",
            "-" * 70,
            "Core Dependencies:",
        ]
        for pkg, ver in snapshot_data["package_dependencies"].items():
            lines.append(f"  * {pkg:<18}: {ver}")
        lines.append("=" * 70)

        txt_path.write_text("\n".join(lines), encoding="utf-8")
        return json_path, txt_path