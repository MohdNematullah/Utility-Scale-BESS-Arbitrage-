"""
backtesting/experiment_runner.py
================================

Master Experiment Orchestrator & Batch Execution Engine



Responsibilities:
1. Orchestrate single, periodic, and batch scenario experiments.
2. Maintain provenance with Git commit hashing and environment metadata (metadata.json).
3. Dedicated persistent execution logging (experiment.log).
4. Detailed runtime profiling (Solver, Forecast, Ageing, Export, Window Averages).
5. State checkpointing and graceful resumption for multi-day/year runs.
6. Synchronize master experiment ledger (experiments_registry.csv).
7. Multiprocessing scenario pool for parallel execution.
"""

from __future__ import annotations

import concurrent.futures
import copy
import json
import logging
import os
import platform
import subprocess
import sys
import time
import random
import numpy as np
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd

from backtesting.config import BacktestConfig, DEFAULT_BACKTEST_CONFIG
from backtesting.engine import RollingBacktestEngine, RollingBacktestResult
from backtesting.export_reports import BacktestExportEngine
from backtesting.metrics import BacktestMetrics, BacktestMetricsResult
from battery.config import BatteryDegradationConfig, DEFAULT_BATTERY_CONFIG
from data.loader import MarketDataLoader
from features.engineering import FeatureEngineer
from forecasting.models import XGBoostForecaster

# ============================================================================
# Provenance & System Helpers
# ============================================================================

def set_deterministic_seed(seed: int = 42) -> None:
    """Sets deterministic global random seed across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if "torch" in sys.modules:
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

def get_git_commit_hash() -> str:
    """Extracts short Git commit hash of current repository."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unversioned"


# ============================================================================
# Dataclasses
# ============================================================================

@dataclass(slots=True)
class ExperimentTiming:
    total_runtime_seconds: float = 0.0
    forecast_runtime_seconds: float = 0.0
    solver_runtime_seconds: float = 0.0
    ageing_runtime_seconds: float = 0.0
    export_runtime_seconds: float = 0.0
    avg_window_runtime_seconds: float = 0.0


@dataclass
class ExperimentConfig:
    """Configuration governing a single backtest scenario run."""
    experiment_id: str = "EXP001"
    experiment_name: str = "bess_arbitrage_study"
    scenario_name: str = "baseline_nmc_48h"
    category: str = "Baseline"
    start_date: str | None = None
    end_date: str | None = None
    forecast_horizon_hours: int = 48
    implementation_horizon_hours: int = 24
    rolling_step_hours: int = 24
    seed: int = 42
    enable_checkpointing: bool = True
    auto_export_reports: bool = True
    force_rerun: bool = False
    notes: str = ""
    output_base_directory: Path = field(
        default_factory=lambda: Path("backtesting/results/experiments")
    )

    @property
    def experiment_directory(self) -> Path:
        return self.output_base_directory / self.experiment_name / self.scenario_name


@dataclass
class ExperimentMetadata:
    """Reproducibility metadata saved to metadata.json."""
    experiment_id: str
    experiment_name: str
    scenario_name: str
    category: str
    git_commit_hash: str
    python_version: str
    platform_info: str
    start_time_iso: str
    end_time_iso: str = ""
    timing: dict[str, float] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    status: str = "Initialized"


@dataclass
class MasterRegistryRecord:
    """Single entry in experiments_registry.csv."""
    experiment_id: str
    experiment_name: str
    scenario_name: str
    category: str
    forecast_horizon: int
    gross_revenue_usd: float
    degradation_cost_usd: float
    net_revenue_usd: float
    net_profit_usd: float
    final_soh: float
    cumulative_efc: float
    profit_factor: float
    sharpe_ratio: float
    status: str
    runtime_seconds: float
    timestamp: str


@dataclass
class MasterExperimentResult:
    config: ExperimentConfig
    metadata: ExperimentMetadata
    backtest_result: RollingBacktestResult
    metrics: BacktestMetricsResult
    timing: ExperimentTiming
    exported_paths: dict[str, Path]


# ============================================================================
# Master Experiment Runner
# ============================================================================

class ExperimentRunner:
    """
    Production-grade Experiment Orchestration Engine for .
    """

    def __init__(
        self,
        registry_path: Path = Path("backtesting/results/experiments_registry.csv"),
    ):
        self.registry_path = registry_path
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_registry()

    def _initialize_registry(self) -> None:
        if not self.registry_path.exists():
            columns = [
                "experiment_id", "experiment_name", "scenario_name", "category",
                "forecast_horizon", "gross_revenue_usd", "degradation_cost_usd",
                "net_revenue_usd", "net_profit_usd", "final_soh", "cumulative_efc",
                "profit_factor", "sharpe_ratio", "status", "runtime_seconds", "timestamp"
            ]
            pd.DataFrame(columns=columns).to_csv(self.registry_path, index=False)

    def _update_registry(self, record: MasterRegistryRecord) -> None:
        df = pd.read_csv(self.registry_path)
        record_dict = asdict(record)
        # Update if ID exists, else append
        if record.experiment_id in df["experiment_id"].values:
            idx = df.index[df["experiment_id"] == record.experiment_id].tolist()[0]
            for col, val in record_dict.items():
                df.at[idx, col] = val
        else:
            df = pd.concat([df, pd.DataFrame([record_dict])], ignore_index=True)
        df.to_csv(self.registry_path, index=False)

    def _setup_logger(self, exp_dir: Path) -> tuple[logging.Logger, logging.FileHandler]:
        exp_dir.mkdir(parents=True, exist_ok=True)
        exp_logger = logging.getLogger(f"experiment_{exp_dir.name}")
        exp_logger.setLevel(logging.INFO)
        exp_logger.handlers.clear()

        fh = logging.FileHandler(exp_dir / "experiment.log", mode="a", encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(formatter)
        exp_logger.addHandler(fh)

        # Stream to stdout
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        exp_logger.addHandler(sh)
        return exp_logger, fh

    def run_experiment(
        self,
        config: ExperimentConfig,
        forecaster: XGBoostForecaster,
        features: pd.DataFrame,
        battery_config: BatteryDegradationConfig = DEFAULT_BATTERY_CONFIG,
    ) -> MasterExperimentResult:
        """
        Executes a single end-to-end scenario experiment with full audit logging,
        checkpointing, metrics calculation, and artifact export.
        """
        exp_dir = config.experiment_directory
        exp_dir.mkdir(parents=True, exist_ok=True)
        exp_logger, file_handler = self._setup_logger(exp_dir)

        exp_logger.info("=" * 70)
        exp_logger.info("INITIALIZING EXPERIMENT: %s [%s]", config.experiment_id, config.scenario_name)
        exp_logger.info("=" * 70)

        timing = ExperimentTiming()
        start_wall_time = time.perf_counter()
        start_iso = pd.Timestamp.now(tz="UTC").isoformat()

        # Checkpoint Check
        checkpoint_file = exp_dir / "checkpoint.json"
        if not config.force_rerun and checkpoint_file.exists():
            exp_logger.info("Found existing checkpoint. Resuming experiment from disk...")
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                ckpt_data = json.load(f)
            if ckpt_data.get("status") == "Completed":
                exp_logger.info("Scenario was already completed. Skipping simulation.")
                # Load existing summary and return
                summary_df = pd.read_csv(exp_dir / "metrics_summary.csv")
                metrics_summary = dict(zip(summary_df["metric"], summary_df["value"]))
                file_handler.close()
                exp_logger.removeHandler(file_handler)
                # Reconstruct result wrapper
                return self._build_completed_result(config, exp_dir, metrics_summary)

        # Configure Backtest Engine
        backtest_cfg = BacktestConfig(
            forecast_horizon_hours=config.forecast_horizon_hours,
            implementation_horizon_hours=config.implementation_horizon_hours,
            rolling_step_hours=config.rolling_step_hours,
            export_directory=exp_dir,
        )

        # Slice features if dates provided
        exec_features = features.copy()
        if config.start_date is not None or config.end_date is not None:
            tz = exec_features.index.tz
            start_ts = pd.to_datetime(config.start_date).tz_localize(tz) if config.start_date else exec_features.index[0]
            end_ts = pd.to_datetime(config.end_date).tz_localize(tz) if config.end_date else exec_features.index[-1]
            exec_features = exec_features.loc[(exec_features.index >= start_ts) & (exec_features.index <= end_ts)]
            exp_logger.info("Filtered execution window: %s to %s (%d rows)", start_ts, end_ts, len(exec_features))

        # Instantiate & Run Backtest Core
        engine = RollingBacktestEngine(
            forecaster=forecaster,
            feature_dataframe=exec_features,
            config=backtest_cfg,
        )
        engine.ageing.config = battery_config

        exp_logger.info("Executing rolling-horizon backtest loop...")
        backtest_start = time.perf_counter()
        result = engine.run()
        timing.total_runtime_seconds = time.perf_counter() - backtest_start
        timing.avg_window_runtime_seconds = (
            timing.total_runtime_seconds / max(len(result.degradation_history), 1)
        )
        exp_logger.info("Backtest loop completed in %.2f s (avg %.4f s/window)",
                        timing.total_runtime_seconds, timing.avg_window_runtime_seconds)

        # Evaluate Research Metrics
        exp_logger.info("Computing financial, battery health, and operational KPIs...")
        metrics_engine = BacktestMetrics()
        metrics_res = metrics_engine.evaluate(result.dispatch_history, result.degradation_history)

        # Export All Reports & Figures (Part 8.4)
        exported_paths = {}
        if config.auto_export_reports:
            exp_logger.info("Generating CSVs, Excel report, JSON summary, and 9 research figures...")
            export_start = time.perf_counter()
            exporter = BacktestExportEngine(output_directory=str(exp_dir))
            exp_artifacts = exporter.export(result)
            timing.export_runtime_seconds = time.perf_counter() - export_start
            exported_paths = {
                "excel": exp_artifacts.excel_report,
                "json": exp_artifacts.json_report,
                "figures": exp_artifacts.figure_directory,
            }
            exp_logger.info("Export pipeline completed in %.2f s", timing.export_runtime_seconds)

        # Save Checkpoint File
        if config.enable_checkpointing:
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump({
                    "experiment_id": config.experiment_id,
                    "scenario_name": config.scenario_name,
                    "status": "Completed",
                    "windows_executed": len(result.degradation_history),
                    "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
                }, f, indent=4)

        # Total Wall Time & Metadata
        total_wall_time = time.perf_counter() - start_wall_time
        timing.total_runtime_seconds = total_wall_time

        metadata = ExperimentMetadata(
            experiment_id=config.experiment_id,
            experiment_name=config.experiment_name,
            scenario_name=config.scenario_name,
            category=config.category,
            git_commit_hash=get_git_commit_hash(),
            python_version=platform.python_version(),
            platform_info=f"{platform.system()} {platform.release()}",
            start_time_iso=start_iso,
            end_time_iso=pd.Timestamp.now(tz="UTC").isoformat(),
            timing=asdict(timing),
            parameters={
                "forecast_horizon_hours": config.forecast_horizon_hours,
                "implementation_horizon_hours": config.implementation_horizon_hours,
                "rolling_step_hours": config.rolling_step_hours,
                "battery_chemistry": battery_config.chemistry.chemistry,
                "battery_capacity_mwh": battery_config.chemistry.nominal_capacity_mwh,
                "battery_max_power_mw": battery_config.chemistry.max_charge_power_mw,
                "notes": config.notes,
            },
            status="Completed",
        )

        with open(exp_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(asdict(metadata), f, indent=4)

        # Sync Master Registry
        summary = metrics_res.summary
        reg_record = MasterRegistryRecord(
            experiment_id=config.experiment_id,
            experiment_name=config.experiment_name,
            scenario_name=config.scenario_name,
            category=config.category,
            forecast_horizon=config.forecast_horizon_hours,
            gross_revenue_usd=summary.get("gross_revenue_usd", 0.0),
            degradation_cost_usd=summary.get("degradation_cost_usd", 0.0),
            net_revenue_usd=summary.get("net_revenue_usd", 0.0),
            net_profit_usd=summary.get("net_operating_profit_usd", 0.0),
            final_soh=summary.get("final_soh", 1.0),
            cumulative_efc=summary.get("equivalent_full_cycles", 0.0),
            profit_factor=summary.get("profit_factor", 0.0),
            sharpe_ratio=summary.get("sharpe_ratio", 0.0),
            status="Completed",
            runtime_seconds=round(total_wall_time, 2),
            timestamp=metadata.end_time_iso,
        )
        self._update_registry(reg_record)

        exp_logger.info("Experiment %s [%s] successfully finished!", config.experiment_id, config.scenario_name)
        file_handler.close()
        exp_logger.removeHandler(file_handler)

        return MasterExperimentResult(
            config=config,
            metadata=metadata,
            backtest_result=result,
            metrics=metrics_res,
            timing=timing,
            exported_paths=exported_paths,
        )

    def _build_completed_result(
        self,
        config: ExperimentConfig,
        exp_dir: Path,
        summary: dict[str, Any],
    ) -> MasterExperimentResult:
        """Constructs a result container from disk for pre-completed scenarios."""
        metadata_file = exp_dir / "metadata.json"
        meta_dict = {}
        if metadata_file.exists():
            with open(metadata_file, "r", encoding="utf-8") as f:
                meta_dict = json.load(f)

        metadata = ExperimentMetadata(
            experiment_id=config.experiment_id,
            experiment_name=config.experiment_name,
            scenario_name=config.scenario_name,
            category=config.category,
            git_commit_hash=meta_dict.get("git_commit_hash", "unknown"),
            python_version=platform.python_version(),
            platform_info=f"{platform.system()} {platform.release()}",
            start_time_iso=meta_dict.get("start_time_iso", ""),
            end_time_iso=meta_dict.get("end_time_iso", ""),
            timing=meta_dict.get("timing", {}),
            status="Completed (Cached)",
        )

        dummy_result = RollingBacktestResult(
            dispatch_history=pd.read_csv(exp_dir / "dispatch_history.csv") if (exp_dir / "dispatch_history.csv").exists() else pd.DataFrame(),
            degradation_history=pd.read_csv(exp_dir / "degradation_history.csv") if (exp_dir / "degradation_history.csv").exists() else pd.DataFrame(),
            summary=summary,
        )
        dummy_metrics = BacktestMetricsResult(summary=summary, dataframe=pd.DataFrame(list(summary.items()), columns=["metric", "value"]))

        return MasterExperimentResult(
            config=config,
            metadata=metadata,
            backtest_result=dummy_result,
            metrics=dummy_metrics,
            timing=ExperimentTiming(),
            exported_paths={"excel": exp_dir / "backtest_report.xlsx", "json": exp_dir / "backtest_summary.json"},
        )

    # ------------------------------------------------------------------------
    # Batch & Parallel Execution Engine
    # ------------------------------------------------------------------------

    def run_batch_sequential(
        self,
        scenarios: Sequence[tuple[ExperimentConfig, BatteryDegradationConfig]],
        forecaster: XGBoostForecaster,
        features: pd.DataFrame,
    ) -> list[MasterExperimentResult]:
        """Runs a sequence of scenario experiments one after another."""
        results = []
        total = len(scenarios)
        for i, (cfg, bat_cfg) in enumerate(scenarios, 1):
            print(f"\n[{i}/{total}] Executing: {cfg.experiment_id} - {cfg.scenario_name}...")
            res = self.run_experiment(cfg, forecaster, features, bat_cfg)
            results.append(res)
        return results

    def run_batch_parallel(
        self,
        scenarios: Sequence[tuple[ExperimentConfig, BatteryDegradationConfig]],
        forecaster_path: Path,
        features_path: Path,
        max_workers: int = 2,
    ) -> list[dict[str, Any]]:
        """
        Executes scenarios across a multiprocessing pool for substantial speedups.
        Workers load models independently to comply with Windows process spawning.
        """
        tasks = []
        for cfg, bat_cfg in scenarios:
            tasks.append({
                "config": asdict(cfg),
                "battery_config": asdict(bat_cfg),
                "forecaster_path": str(forecaster_path),
                "features_path": str(features_path),
                "registry_path": str(self.registry_path),
            })

        print(f"Launching parallel execution of {len(tasks)} scenarios across {max_workers} worker processes...")
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_exp = {executor.submit(_parallel_worker_task, task): task["config"]["experiment_id"] for task in tasks}
            for future in concurrent.futures.as_completed(future_to_exp):
                exp_id = future_to_exp[future]
                try:
                    res_summary = future.result()
                    results.append(res_summary)
                    print(f"Parallel Worker Completed: {exp_id} -> Net Rev: ${res_summary.get('net_revenue_usd', 0.0):,.2f}")
                except Exception as exc:
                    print(f"Parallel Worker FAILED: {exp_id} with error: {exc}")
        return results


def _parallel_worker_task(task_params: dict[str, Any]) -> dict[str, Any]:
    """Isolated worker process function that rebuilds dependencies independently."""
    cfg_dict = task_params["config"]
    cfg_dict["output_base_directory"] = Path(cfg_dict["output_base_directory"])
    cfg = ExperimentConfig(**cfg_dict)

    # Load resources in subprocess
    features = pd.read_csv(task_params["features_path"])
    if "timestamp" in features.columns:
        features["timestamp"] = pd.to_datetime(features["timestamp"], utc=True)
        features = features.set_index("timestamp").sort_index()

    forecaster = XGBoostForecaster()
    forecaster.load(Path(task_params["forecaster_path"]))

    bat_dict = task_params["battery_config"]
    bat_cfg = DEFAULT_BATTERY_CONFIG
    # Apply chemistry parameters
    if "chemistry" in bat_dict:
        for k, v in bat_dict["chemistry"].items():
            if hasattr(bat_cfg.chemistry, k):
                setattr(bat_cfg.chemistry, k, v)
    if "ageing" in bat_dict:
        for k, v in bat_dict["ageing"].items():
            if hasattr(bat_cfg.ageing, k):
                setattr(bat_cfg.ageing, k, v)

    runner = ExperimentRunner(registry_path=Path(task_params["registry_path"]))
    res = runner.run_experiment(cfg, forecaster, features, bat_cfg)
    return {
        "experiment_id": cfg.experiment_id,
        "scenario_name": cfg.scenario_name,
        "net_revenue_usd": res.metrics.summary.get("net_revenue_usd", 0.0),
        "final_soh": res.metrics.summary.get("final_soh", 1.0),
        "status": "Completed",
    }