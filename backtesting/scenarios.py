"""
backtesting/scenarios.py
========================

Predefined Research Scenarios Library for 

Implements 28 structured research scenarios spanning:
1. Look-Ahead Forecast Horizons (12h, 24h, 36h, 48h, 72h)
2. Forecasting Model Configurations (XGBoost, Persistence, Moving Average, Perfect Foresight)
3. Battery Chemistries (NMC, LFP, LTO)
4. Thermal Stress & Sensitivity (15Â°C, 25Â°C, 35Â°C, 45Â°C)
5. Power & Energy Sizing / Duration (25MW/50MWh, 50MW/100MWh, 50MW/200MWh, 100MW/200MWh)
6. AC-AC Round-Trip Efficiencies (85%, 90.25%, 92%, 95%)
7. Degradation Model Formulations (Full Rainflow, Calendar Only, Linear Throughput, Zero Wear)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Sequence

from backtesting.experiment_runner import ExperimentConfig
from battery.config import (
    BatteryDegradationConfig,
    DEFAULT_BATTERY_CONFIG,
    BatteryChemistry,
    AgeingParameters,
    ReplacementParameters,
    OperatingParameters,
)


class ScenarioCategory(str, Enum):
    HORIZON = "Forecast_Horizon"
    MODEL = "Forecast_Model"
    CHEMISTRY = "Battery_Chemistry"
    TEMPERATURE = "Thermal_Sensitivity"
    SIZING = "System_Sizing"
    EFFICIENCY = "Efficiency_Sensitivity"
    DEGRADATION = "Degradation_Modeling"


@dataclass
class ScenarioDefinition:
    """Complete specification of an experimental scenario."""
    scenario_id: str
    scenario_name: str
    category: ScenarioCategory
    description: str
    experiment_config: ExperimentConfig
    battery_modifier: Callable[[BatteryDegradationConfig], BatteryDegradationConfig] | None = None
    custom_parameters: dict[str, Any] = field(default_factory=dict)

    def instantiate_configs(
        self,
        base_exp_name: str = "_scenarios",
    ) -> tuple[ExperimentConfig, BatteryDegradationConfig]:
        """Generates ready-to-run configurations for ExperimentRunner."""
        exp_cfg = copy.deepcopy(self.experiment_config)
        exp_cfg.experiment_name = base_exp_name
        exp_cfg.experiment_id = self.scenario_id
        exp_cfg.scenario_name = self.scenario_name
        exp_cfg.category = self.category.value

        bat_cfg = copy.deepcopy(DEFAULT_BATTERY_CONFIG)
        if self.battery_modifier is not None:
            bat_cfg = self.battery_modifier(bat_cfg)

        return exp_cfg, bat_cfg


# ============================================================================
# Battery Modifier Helpers
# ============================================================================

def _mod_lfp(cfg: BatteryDegradationConfig) -> BatteryDegradationConfig:
    cfg.chemistry.chemistry = "Lithium-Ion LFP"
    cfg.chemistry.nominal_cycle_life = 7000
    cfg.ageing.calendar_loss_per_year = 0.010       # LFP lower calendar fade (1.0%/yr)
    cfg.ageing.cycle_loss_per_efc = 0.000028       # 0.20 fade / 7000 EFC
    cfg.replacement.replacement_cost_per_mwh = 130000.0  # LFP pack cost ($130/kWh)
    return cfg


def _mod_lto(cfg: BatteryDegradationConfig) -> BatteryDegradationConfig:
    cfg.chemistry.chemistry = "Lithium Titanate (LTO)"
    cfg.chemistry.nominal_cycle_life = 15000
    cfg.ageing.calendar_loss_per_year = 0.005       # LTO minimal calendar fade (0.5%/yr)
    cfg.ageing.cycle_loss_per_efc = 0.000013       # 0.20 fade / 15000 EFC
    cfg.chemistry.round_trip_efficiency = 0.88
    cfg.chemistry.charge_efficiency = 0.938
    cfg.chemistry.discharge_efficiency = 0.938
    cfg.replacement.replacement_cost_per_mwh = 260000.0  # LTO premium cost ($260/kWh)
    return cfg


def _mod_temp(temp_c: float) -> Callable[[BatteryDegradationConfig], BatteryDegradationConfig]:
    def modifier(cfg: BatteryDegradationConfig) -> BatteryDegradationConfig:
        cfg.ageing.reference_temperature_c = temp_c
        return cfg
    return modifier


def _mod_sizing(power_mw: float, capacity_mwh: float) -> Callable[[BatteryDegradationConfig], BatteryDegradationConfig]:
    def modifier(cfg: BatteryDegradationConfig) -> BatteryDegradationConfig:
        cfg.chemistry.nominal_capacity_mwh = capacity_mwh
        cfg.chemistry.max_charge_power_mw = power_mw
        cfg.chemistry.max_discharge_power_mw = power_mw
        return cfg
    return modifier


def _mod_efficiency(rte: float) -> Callable[[BatteryDegradationConfig], BatteryDegradationConfig]:
    def modifier(cfg: BatteryDegradationConfig) -> BatteryDegradationConfig:
        one_way = rte ** 0.5
        cfg.chemistry.round_trip_efficiency = rte
        cfg.chemistry.charge_efficiency = one_way
        cfg.chemistry.discharge_efficiency = one_way
        return cfg
    return modifier


def _mod_degradation_mode(mode: str) -> Callable[[BatteryDegradationConfig], BatteryDegradationConfig]:
    def modifier(cfg: BatteryDegradationConfig) -> BatteryDegradationConfig:
        if mode == "calendar_only":
            cfg.ageing.cycle_loss_per_efc = 0.0
        elif mode == "zero_wear":
            cfg.ageing.calendar_loss_per_year = 0.0
            cfg.ageing.cycle_loss_per_efc = 0.0
        return cfg
    return modifier


# ============================================================================
# Predefined Scenario Registry
# ============================================================================

_SCENARIOS_LIST: list[ScenarioDefinition] = [

    # ------------------------------------------------------------------------
    # Category 1: Look-Ahead Forecast Horizons (5 Scenarios)
    # ------------------------------------------------------------------------
    ScenarioDefinition(
        scenario_id="SCN_HORIZON_12H",
        scenario_name="horizon_12h_lookahead",
        category=ScenarioCategory.HORIZON,
        description="Short 12-hour look-ahead horizon evaluating near-term dispatch myopic limits.",
        experiment_config=ExperimentConfig(forecast_horizon_hours=12, implementation_horizon_hours=12, rolling_step_hours=12),
    ),
    ScenarioDefinition(
        scenario_id="SCN_HORIZON_24H",
        scenario_name="horizon_24h_dayahead",
        category=ScenarioCategory.HORIZON,
        description="Standard 24-hour day-ahead market settlement horizon.",
        experiment_config=ExperimentConfig(forecast_horizon_hours=24, implementation_horizon_hours=24, rolling_step_hours=24),
    ),
    ScenarioDefinition(
        scenario_id="SCN_HORIZON_36H",
        scenario_name="horizon_36h_intermediate",
        category=ScenarioCategory.HORIZON,
        description="Intermediate 36-hour look-ahead bridging Day-Ahead and next-morning peaks.",
        experiment_config=ExperimentConfig(forecast_horizon_hours=36, implementation_horizon_hours=24, rolling_step_hours=24),
    ),
    ScenarioDefinition(
        scenario_id="SCN_HORIZON_48H",
        scenario_name="horizon_48h_baseline",
        category=ScenarioCategory.HORIZON,
        description="Baseline: 48-hour rolling forecast look-ahead with 24-hour execution.",
        experiment_config=ExperimentConfig(forecast_horizon_hours=48, implementation_horizon_hours=24, rolling_step_hours=24),
    ),
    ScenarioDefinition(
        scenario_id="SCN_HORIZON_72H",
        scenario_name="horizon_72h_weekend",
        category=ScenarioCategory.HORIZON,
        description="Extended 72-hour look-ahead spanning multi-day weekend price volatility.",
        experiment_config=ExperimentConfig(forecast_horizon_hours=72, implementation_horizon_hours=24, rolling_step_hours=24),
    ),

    # ------------------------------------------------------------------------
    # Category 2: Forecasting Model Configurations (4 Scenarios)
    # ------------------------------------------------------------------------
    ScenarioDefinition(
        scenario_id="SCN_MODEL_XGBOOST",
        scenario_name="model_xgboost_recursive",
        category=ScenarioCategory.MODEL,
        description="Main Machine Learning Model: Recursive multi-step XGBoost forecaster.",
        experiment_config=ExperimentConfig(forecast_horizon_hours=48, implementation_horizon_hours=24, rolling_step_hours=24),
        custom_parameters={"forecaster_type": "XGBoost"},
    ),
    ScenarioDefinition(
        scenario_id="SCN_MODEL_PERSISTENCE",
        scenario_name="model_persistence_naive",
        category=ScenarioCategory.MODEL,
        description="Statistical Benchmark: 24h diurnal persistence naive baseline.",
        experiment_config=ExperimentConfig(forecast_horizon_hours=48, implementation_horizon_hours=24, rolling_step_hours=24),
        custom_parameters={"forecaster_type": "Persistence"},
    ),
    ScenarioDefinition(
        scenario_id="SCN_MODEL_MOVING_AVG",
        scenario_name="model_moving_average",
        category=ScenarioCategory.MODEL,
        description="Statistical Benchmark: 7-day rolling diurnal moving average.",
        experiment_config=ExperimentConfig(forecast_horizon_hours=48, implementation_horizon_hours=24, rolling_step_hours=24),
        custom_parameters={"forecaster_type": "MovingAverage"},
    ),
    ScenarioDefinition(
        scenario_id="SCN_MODEL_PERFECT_FORESIGHT",
        scenario_name="model_perfect_foresight",
        category=ScenarioCategory.MODEL,
        description="Upper Economic Bound: Clairvoyant perfect price foresight (0% forecast error).",
        experiment_config=ExperimentConfig(forecast_horizon_hours=48, implementation_horizon_hours=24, rolling_step_hours=24),
        custom_parameters={"forecaster_type": "PerfectForesight"},
    ),

    # ------------------------------------------------------------------------
    # Category 3: Battery Chemistry Alternatives (3 Scenarios)
    # ------------------------------------------------------------------------
    ScenarioDefinition(
        scenario_id="SCN_CHEM_NMC",
        scenario_name="chem_nmc_baseline",
        category=ScenarioCategory.CHEMISTRY,
        description="Baseline Chemistry: Lithium Nickel Manganese Cobalt (NMC), 4000 cycles life.",
        experiment_config=ExperimentConfig(),
        battery_modifier=lambda cfg: cfg,
    ),
    ScenarioDefinition(
        scenario_id="SCN_CHEM_LFP",
        scenario_name="chem_lfp_stationary",
        category=ScenarioCategory.CHEMISTRY,
        description="Stationary Standard: Lithium Iron Phosphate (LFP), 7000 cycles, lower calendar decay.",
        experiment_config=ExperimentConfig(),
        battery_modifier=_mod_lfp,
    ),
    ScenarioDefinition(
        scenario_id="SCN_CHEM_LTO",
        scenario_name="chem_lto_heavy_cycle",
        category=ScenarioCategory.CHEMISTRY,
        description="Heavy Duty: Lithium Titanate (LTO), 15000 cycles life, premium capital cost.",
        experiment_config=ExperimentConfig(),
        battery_modifier=_mod_lto,
    ),

    # ------------------------------------------------------------------------
    # Category 4: Thermal Stress & Operating Environment (4 Scenarios)
    # ------------------------------------------------------------------------
    ScenarioDefinition(
        scenario_id="SCN_TEMP_15C",
        scenario_name="temp_15c_subcooled",
        category=ScenarioCategory.TEMPERATURE,
        description="Chilled/Subcooled HVAC environment at 15Â°C (retards calendar SEI growth).",
        experiment_config=ExperimentConfig(),
        battery_modifier=_mod_temp(15.0),
    ),
    ScenarioDefinition(
        scenario_id="SCN_TEMP_25C",
        scenario_name="temp_25c_reference",
        category=ScenarioCategory.TEMPERATURE,
        description="Standard reference room temperature at 25Â°C.",
        experiment_config=ExperimentConfig(),
        battery_modifier=_mod_temp(25.0),
    ),
    ScenarioDefinition(
        scenario_id="SCN_TEMP_35C",
        scenario_name="temp_35c_elevated",
        category=ScenarioCategory.TEMPERATURE,
        description="Elevated operational temperature at 35Â°C (Arrhenius calendar acceleration).",
        experiment_config=ExperimentConfig(),
        battery_modifier=_mod_temp(35.0),
    ),
    ScenarioDefinition(
        scenario_id="SCN_TEMP_45C",
        scenario_name="temp_45c_severe_stress",
        category=ScenarioCategory.TEMPERATURE,
        description="Severe thermal stress at 45Â°C (cooling failure / harsh desert climate).",
        experiment_config=ExperimentConfig(),
        battery_modifier=_mod_temp(45.0),
    ),

    # ------------------------------------------------------------------------
    # Category 5: System Sizing & Duration (4 Scenarios)
    # ------------------------------------------------------------------------
    ScenarioDefinition(
        scenario_id="SCN_SIZE_25MW_50MWH",
        scenario_name="size_25mw_50mwh_2h",
        category=ScenarioCategory.SIZING,
        description="Small Utility: 25 MW / 50 MWh (2-hour duration battery).",
        experiment_config=ExperimentConfig(),
        battery_modifier=_mod_sizing(25.0, 50.0),
    ),
    ScenarioDefinition(
        scenario_id="SCN_SIZE_50MW_100MWH",
        scenario_name="size_50mw_100mwh_2h",
        category=ScenarioCategory.SIZING,
        description="Baseline Utility: 50 MW / 100 MWh (2-hour duration battery).",
        experiment_config=ExperimentConfig(),
        battery_modifier=_mod_sizing(50.0, 100.0),
    ),
    ScenarioDefinition(
        scenario_id="SCN_SIZE_50MW_200MWH",
        scenario_name="size_50mw_200mwh_4h",
        category=ScenarioCategory.SIZING,
        description="Long Duration Utility: 50 MW / 200 MWh (4-hour duration battery).",
        experiment_config=ExperimentConfig(),
        battery_modifier=_mod_sizing(50.0, 200.0),
    ),
    ScenarioDefinition(
        scenario_id="SCN_SIZE_100MW_200MWH",
        scenario_name="size_100mw_200mwh_2h",
        category=ScenarioCategory.SIZING,
        description="Large Scale Utility: 100 MW / 200 MWh (2-hour duration battery).",
        experiment_config=ExperimentConfig(),
        battery_modifier=_mod_sizing(100.0, 200.0),
    ),

    # ------------------------------------------------------------------------
    # Category 6: AC-AC Round-Trip Efficiency (4 Scenarios)
    # ------------------------------------------------------------------------
    ScenarioDefinition(
        scenario_id="SCN_EFF_85PCT",
        scenario_name="eff_85pct_aged_inverters",
        category=ScenarioCategory.EFFICIENCY,
        description="Low Round-Trip Efficiency (85.0% RTE) representing aging balance of plant.",
        experiment_config=ExperimentConfig(),
        battery_modifier=_mod_efficiency(0.85),
    ),
    ScenarioDefinition(
        scenario_id="SCN_EFF_90PCT",
        scenario_name="eff_90pct_baseline",
        category=ScenarioCategory.EFFICIENCY,
        description="Baseline Modern Round-Trip Efficiency (90.25% RTE).",
        experiment_config=ExperimentConfig(),
        battery_modifier=_mod_efficiency(0.9025),
    ),
    ScenarioDefinition(
        scenario_id="SCN_EFF_92PCT",
        scenario_name="eff_92pct_high_spec",
        category=ScenarioCategory.EFFICIENCY,
        description="High Specification Round-Trip Efficiency (92.0% RTE).",
        experiment_config=ExperimentConfig(),
        battery_modifier=_mod_efficiency(0.92),
    ),
    ScenarioDefinition(
        scenario_id="SCN_EFF_95PCT",
        scenario_name="eff_95pct_nextgen",
        category=ScenarioCategory.EFFICIENCY,
        description="Next-Generation Solid-State / SiC Inverter Efficiency (95.0% RTE).",
        experiment_config=ExperimentConfig(),
        battery_modifier=_mod_efficiency(0.95),
    ),

    # ------------------------------------------------------------------------
    # Category 7: Degradation Model Sensitivity (4 Scenarios)
    # ------------------------------------------------------------------------
    ScenarioDefinition(
        scenario_id="SCN_DEG_CALENDAR_ONLY",
        scenario_name="deg_calendar_only",
        category=ScenarioCategory.DEGRADATION,
        description="Calendar Aging Only: Evaluates asset depreciation assuming zero cycling fatigue.",
        experiment_config=ExperimentConfig(),
        battery_modifier=_mod_degradation_mode("calendar_only"),
    ),
    ScenarioDefinition(
        scenario_id="SCN_DEG_RAINFLOW",
        scenario_name="deg_calendar_rainflow_baseline",
        category=ScenarioCategory.DEGRADATION,
        description="Full Research Model: Combined Arrhenius calendar + ASTM E1049 Rainflow cycle wear.",
        experiment_config=ExperimentConfig(),
        battery_modifier=lambda cfg: cfg,
    ),
    ScenarioDefinition(
        scenario_id="SCN_DEG_ZERO_WEAR",
        scenario_name="deg_zero_wear_unconstrained",
        category=ScenarioCategory.DEGRADATION,
        description="Ideal Storage Benchmark: Zero degradation costs (unconstrained merchant arbitrage).",
        experiment_config=ExperimentConfig(),
        battery_modifier=_mod_degradation_mode("zero_wear"),
    ),
    ScenarioDefinition(
        scenario_id="SCN_DEG_HIGH_WEAR_HURDLE",
        scenario_name="deg_high_wear_penalty",
        category=ScenarioCategory.DEGRADATION,
        description="High Hurdle Dispatch: Evaluates aggressive battery preservation ($25/MWh hurdle).",
        experiment_config=ExperimentConfig(),
        custom_parameters={"marginal_wear_hurdle": 25.0},
    ),
]

_SCENARIOS_MAP: dict[str, ScenarioDefinition] = {s.scenario_id: s for s in _SCENARIOS_LIST}


# ============================================================================
# Public Registry Query API
# ============================================================================

def list_scenarios() -> list[ScenarioDefinition]:
    """Returns the complete list of all 28 predefined research scenarios."""
    return list(_SCENARIOS_LIST)


def get_scenario(scenario_id: str) -> ScenarioDefinition:
    """Fetches a specific scenario by ID (case-insensitive)."""
    clean_id = scenario_id.strip().upper()
    if clean_id not in _SCENARIOS_MAP:
        # Search by scenario_name
        for s in _SCENARIOS_LIST:
            if s.scenario_name.lower() == scenario_id.strip().lower():
                return s
        available = ", ".join(_SCENARIOS_MAP.keys())
        raise KeyError(f"Scenario '{scenario_id}' not found. Available IDs:\n{available}")
    return _SCENARIOS_MAP[clean_id]


def get_scenarios_by_category(category: ScenarioCategory | str) -> list[ScenarioDefinition]:
    """Filters scenarios by category."""
    cat_val = category.value if isinstance(category, ScenarioCategory) else category
    return [s for s in _SCENARIOS_LIST if s.category.value.lower() == cat_val.lower()]


def build_scenario_configs(
    scenario_ids: Sequence[str] | None = None,
    base_experiment_name: str = "_scenarios",
) -> list[tuple[ExperimentConfig, BatteryDegradationConfig]]:
    """Builds runnable configuration tuples for ExperimentRunner."""
    target_scenarios = (
        [get_scenario(sid) for sid in scenario_ids]
        if scenario_ids is not None
        else _SCENARIOS_LIST
    )
    return [s.instantiate_configs(base_experiment_name) for s in target_scenarios]