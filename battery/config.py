"""
battery/config.py
=================

Battery Degradation, System & Operating Configuration for .

Defines physical, electrochemical, operational, and replacement economic
parameters calibrated for utility-scale BESS rolling-horizon arbitrage studies.

References:
- NREL Battery Lifetime Models (NMC/LFP)
- NREL Annual Technology Baseline (ATB) for BESS O&M Benchmarks
- IEEE 1188 Recommended Practice
- Xu et al. (2021) Li-ion Degradation Modeling
- Standard Stationary BESS 80% Operational / 70% Terminal EOL Framework
"""

from dataclasses import dataclass, field
from pathlib import Path


# ==========================================================
# Cell Ageing Parameters
# ==========================================================

@dataclass(slots=True)
class AgeingParameters:
    """
    Physical calendar and cycle degradation parameters for Lithium-Ion NMC cells.
    """

    # ------------------------------------------------------
    # Calendar Ageing
    # ------------------------------------------------------

    # Nominal annual capacity fade under reference conditions (1.5% / year)
    calendar_loss_per_year: float = 0.015

    # Arrhenius reference temperature
    reference_temperature_c: float = 25.0

    # Activation energy for electrolyte/SEI degradation (J/mol)
    activation_energy_j_per_mol: float = 31500.0

    # Non-linear SOC stress multiplier around nominal storage SOC (50%)
    soc_ageing_factor: float = 0.50

    # ------------------------------------------------------
    # Cycle Ageing
    # ------------------------------------------------------

    # Capacity loss per Equivalent Full Cycle (EFC)
    # Calibrated to nominal cycle life: 0.20 usable fade / 4000 EFC = 0.00005 per EFC
    cycle_loss_per_efc: float = 0.00005

    # Baseline Depth of Discharge (DoD) for normalization
    reference_depth_of_discharge: float = 0.80

    # Non-linear DoD stress exponent
    dod_ageing_exponent: float = 1.35

    # Dynamic C-rate stress sensitivity
    c_rate_ageing_factor: float = 0.15

    # Thermal acceleration coefficient during cycling
    cycle_temperature_factor: float = 0.010

    # ------------------------------------------------------
    # SOH Thresholds (80% Operational Minimum / 70% Terminal EOL)
    # ------------------------------------------------------

    # Minimum warrantied operational State-of-Health (triggers augment/service)
    minimum_soh: float = 0.80

    # Terminal utility-scale End-of-Life decommissioning/scrap threshold
    end_of_life_soh: float = 0.70


# ==========================================================
# Battery Replacement Parameters
# ==========================================================

@dataclass(slots=True)
class ReplacementParameters:
    """
    Economic parameters governing asset depreciation and pack replacement.
    """

    # Turnkey capital replacement cost per MWh capacity
    replacement_cost_per_mwh: float = 150000.0

    # Residual salvage value of decommissioned pack (10%)
    salvage_fraction: float = 0.10

    # Annual inflation escalation factor
    inflation_rate: float = 0.02

    # SOH threshold that triggers asset augmentation or replacement
    replacement_trigger_soh: float = 0.80


# ==========================================================
# Operating & Maintenance (O&M) Parameters
# ==========================================================

@dataclass(slots=True)
class OperatingParameters:
    """
    Fixed and Variable non-wear Operating & Maintenance assumptions.
    Benchmarks sourced from NREL Annual Technology Baseline (ATB).
    """

    # Fixed O&M in $/MW-year (insurance, scheduled inspections, balance of plant)
    fixed_om_per_mw_year: float = 7500.0

    # Variable non-wear O&M in $/MWh throughput (inverter servicing, auxiliary loads)
    variable_om_per_mwh: float = 0.50


# ==========================================================
# Battery Chemistry & Operating Limits
# ==========================================================

@dataclass(slots=True)
class BatteryChemistry:
    """
    System-level electrical and power constraints for the BESS asset.
    """

    chemistry: str = "Lithium-Ion NMC"

    # Nameplate energy rating
    nominal_capacity_mwh: float = 100.0

    # Pack DC bus operating voltage
    nominal_voltage_v: float = 1000.0

    # Inverter interconnect limits (C-rate ~ 0.38C)
    max_charge_power_mw: float = 38.0
    max_discharge_power_mw: float = 38.0

    # Safe operating SOC range
    minimum_soc_fraction: float = 0.10
    maximum_soc_fraction: float = 0.90
    initial_soc_fraction: float = 0.50
    terminal_soc_fraction: float = 0.50

    # One-way conversion efficiencies
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95

    # Combined round-trip AC-AC efficiency
    round_trip_efficiency: float = 0.9025

    # Nominal design full cycles until operational threshold is reached
    nominal_cycle_life: int = 4000

    # ------------------------------------------------------
    # Compatibility Properties & Aliases
    # ------------------------------------------------------

    @property
    def capacity_mwh(self) -> float:
        """Backward compatibility alias for nominal_capacity_mwh."""
        return self.nominal_capacity_mwh

    @capacity_mwh.setter
    def capacity_mwh(self, val: float) -> None:
        self.nominal_capacity_mwh = float(val)

    @property
    def nominal_cycles(self) -> int:
        """Alias for nominal_cycle_life."""
        return self.nominal_cycle_life

    @nominal_cycles.setter
    def nominal_cycles(self, val: int) -> None:
        self.nominal_cycle_life = int(val)

    @property
    def nominal_round_trip_efficiency(self) -> float:
        """Alias for round_trip_efficiency."""
        return self.round_trip_efficiency

    @nominal_round_trip_efficiency.setter
    def nominal_round_trip_efficiency(self, val: float) -> None:
        self.round_trip_efficiency = float(val)


# ==========================================================
# Master Battery Configuration
# ==========================================================

@dataclass(slots=True)
class BatteryDegradationConfig:
    """
    Master configuration combining chemistry, ageing physics, replacement economics,
    and operational OPEX parameters.
    """

    chemistry: BatteryChemistry = field(default_factory=BatteryChemistry)
    ageing: AgeingParameters = field(default_factory=AgeingParameters)
    replacement: ReplacementParameters = field(default_factory=ReplacementParameters)
    operating: OperatingParameters = field(default_factory=OperatingParameters)

    results_directory: Path = field(
        default_factory=lambda: Path("battery/results")
    )

    @property
    def usable_capacity_fraction(self) -> float:
        """
        Calculates total fraction of capacity consumable before EOL.
        """
        return max(0.0, 1.0 - self.ageing.end_of_life_soh)

    @property
    def usable_replacement_value(self) -> float:
        """
        Total asset replacement value net of salvage value.
        """
        gross_value = (
            self.chemistry.nominal_capacity_mwh
            * self.replacement.replacement_cost_per_mwh
        )
        return gross_value * (1.0 - self.replacement.salvage_fraction)


# ==========================================================
# Global Default Instance
# ==========================================================

DEFAULT_BATTERY_CONFIG = BatteryDegradationConfig()