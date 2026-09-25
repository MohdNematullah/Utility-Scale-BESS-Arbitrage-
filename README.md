
# ⚡ Utility-Scale Battery Energy Storage System (BESS) Arbitrage & Electrochemical Degradation Research Platform

<div align="center">

### AI-Powered Electricity Price Forecasting • Mixed-Integer Battery Dispatch Optimization • Battery Aging Analytics • Rolling Horizon Backtesting • Interactive Dashboard

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Pyomo-MILP_Optimization-green" alt="Pyomo">
  <img src="https://img.shields.io/badge/Solver-HiGHS-purple" alt="HiGHS">
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/XGBoost-Forecasting-orange" alt="XGBoost">
  <img src="https://img.shields.io/badge/LightGBM-Gradient_Boosting-yellowgreen" alt="LightGBM">
  <img src="https://img.shields.io/badge/Plotly-Interactive_Visualizations-3F4F75?logo=plotly" alt="Plotly">
  <img src="https://img.shields.io/badge/License-MIT-success" alt="License">
</p>

<div align="center">

**🌐 Live Dashboard**

**Streamlit Cloud:** [utility-scale-bess-arbitrage.streamlit.app](https://utility-scale-bess-arbitrage.streamlit.app/)

</div>
</div>

---

## Overview

This project is an institutional-grade simulation, optimization, and research analytics platform for **utility-scale Battery Energy Storage Systems (BESS)** participating in deregulated wholesale electricity markets.

The platform combines machine learning, Mixed-Integer Linear Programming (MILP) battery dispatch optimization with mutually exclusive charging and discharging operating modes, physical electrochemical degradation kinetics, financial risk analytics, and interactive web visualization into an end-to-end reproducible research workflow.

Rather than assuming perfect foresight of future prices, the platform couples multi-step recursive price forecasting with rolling-horizon dispatch optimization under battery health degradation, operational cycling limits, and market price uncertainty.

It covers the complete lifecycle of a grid-scale battery storage asset:
- Wholesale electricity market ingestion and data validation
- Multi-lag temporal feature engineering
- Multi-step recursive price forecasting (XGBoost / LightGBM)
- Rolling-horizon Mixed-Integer Linear Programming (MILP) dispatch optimization
- ASTM E1049 Rainflow cycle counting and Arrhenius calendar degradation modeling
- Closed-loop chronological backtesting with dynamic State of Health (SOH) feedback
- Financial valuation (Gross revenue, degradation wear cost, O&M OPEX, EBITDA)
- Downside risk profiling (Historical VaR/CVaR, Sharpe, drawdown)
- 28-scenario sensitivity matrix and multi-objective Pareto frontier extraction
- Publication-quality graphics export (PNG, PDF, SVG, TIFF) and interactive Streamlit portal

---

# Table of Contents

- [Overview](#overview)
- [Project Highlights](#project-highlights)
- [Problem Statement](#problem-statement)
- [Solution Architecture](#solution-architecture)
- [Platform Architecture](#platform-architecture)
- [Core Modules](#core-modules)
- [12-Stage Research Workflow](#12-stage-research-workflow)
- [Mathematical Framework](#mathematical-framework)
- [Baseline Research Results](#baseline-research-results)
- [28-Scenario Experimental Matrix](#28-scenario-experimental-matrix)
- [Installation Guide](#installation-guide)
- [Quick Start](#quick-start)
- [Command Line Reference](#command-line-reference)
- [Testing & Validation](#testing--validation)
- [Changelog](#changelog)

---

# Project Highlights

| Feature | Description |
|:---|:---|
| **Price Forecasting** | Multi-step recursive forecasting using XGBoost and LightGBM across 12h, 24h, 36h, 48h, and 72h horizons. |
| **Rolling MILP Optimization** | Mixed-Integer Linear Programming (MILP) battery dispatch optimization with mutually exclusive charging and discharging operating modes solved via HiGHS. |
| **Binary Operating Constraints** | Rigorous binary status variables enforce zero simultaneous charging and discharging ($u_{chg} + u_{dis} \le 1$). |
| **Battery Aging Kinetics** | Cycle fatigue via ASTM E1049-85 Rainflow cycle counting coupled with temperature-dependent Arrhenius calendar loss. |
| **Deterministic Reproducibility** | Seeded NumPy random generator (`default_rng(seed=42)`) guarantees bitwise reproducible prices, schedules, and metrics. |
| **Closed-Loop Backtest** | 8,760-hour annual chronological execution with continuous State of Health (SOH) and capacity fade updating. |
| **Techno-Economic Valuation** | Gross capture, Rainflow degradation wear cost, fixed/variable O&M, EBITDA operating margin, and Value Capture Ratio (VCR). |
| **Downside Risk Engine** | Daily P&L variance, parametric and historical 95%/99% VaR, Conditional VaR (Expected Shortfall), and annualized Sharpe ratio. |
| **28-Scenario Sensitivity Matrix** | Parametric sweeps over cell chemistry (NMC, LFP, LTO), look-ahead horizons, duration sizing, efficiency, and wear hurdles. |
| **Publication Figure Suite** | 44 publication-ready figures automatically exported across raster and vector formats (600 DPI PNG, PDF, SVG, TIFF). |
| **Interactive Dashboard** | 12-page Streamlit portal with Plotly visualizers, interactive KPI cards, and experiment batch sweep controls. |

---

# Problem Statement

Utility-scale BESS assets capture revenue by absorbing energy during low- or negative-price intervals and delivering energy during peak settlement periods. In practical electricity markets, asset operators face critical interconnected challenges:

1. **Price Uncertainty:** Day-ahead and real-time prices exhibit extreme volatility, fat-tailed spikes, and horizon-dependent forecast degradation.
2. **Electrochemical Fatigue:** Cycling accelerates solid electrolyte interphase (SEI) growth, lithium plating, and active material loss, diminishing pack equity.
3. **Physical Exclusivity:** A single inverter/converter interconnect cannot physically charge and discharge simultaneously during any settlement interval.
4. **Myopic Horizon Traps:** Short optimization horizons induce sub-optimal cycling and leave packs depleted before high-value market events.
5. **Coupled Degradation Economics:** Arbitrage algorithms that ignore degradation wear cycle excessively on low-margin spreads, eroding overall pack life.

This platform resolves these coupled constraints within an integrated, closed-loop simulation framework.

---

## Solution Architecture

The framework coordinates four interconnected engineering subsystems:


```

┌─────────────────────────────────────────────────────────────────────────────┐
│                             1. FORECAST LAYER                               │
│        Recursive Machine Learning Price Prediction (XGBoost / LightGBM)     │
│        Deterministic Fallback Gaussian Perturbation (NumPy default_rng)     │
└──────────────────────────────────────┬──────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           2. OPTIMIZATION LAYER                             │
│        Pyomo MILP Formulation • HiGHS Mixed-Integer Solver                  │
│        Binary Operating Modes: Charging (z=1) vs Discharging (z=0)         │
│        Automated Feasibility Validation: Zero Simultaneous Power Flow       │
└──────────────────────────────────────┬──────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             3. BATTERY LAYER                                │
│        ASTM E1049 Rainflow Half-Cycle Counting & Miner's Cumulative Rule    │
│        Arrhenius Calendar Fade Kinetics • Dynamic SOH Feedback Loop         │
└──────────────────────────────────────┬──────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            4. ANALYTICS LAYER                               │
│        Techno-Economic Waterfall • 95%/99% VaR & CVaR Tail Analytics        │
│        Multi-Scenario Pareto Frontier • 44 Publication Figures Suite        │
└─────────────────────────────────────────────────────────────────────────────┘

```

---

# Core Modules

```text
├── main.py                         # 12-Stage Master CLI Pipeline Orchestrator
├── app.py                          # Multi-Page Streamlit Web Portal
│
├── data/                           # Ingestion, validation, and historical time series
├── features/                       # Temporal lags, rolling moments, and Fourier harmonics
├── forecasting/                    # Recursive XGBoost/LightGBM engines and model artifacts
├── optimization/                   # Pyomo MILP formulations, binary constraints, and HiGHS solver
├── battery/                        # ASTM Rainflow counting, Arrhenius calendar wear, and SOH tracking
├── backtesting/                    # Chronological rolling-horizon engine and telemetry builders
├── analytics/                      # Financial waterfalls, VaR/CVaR risk engine, and KPI evaluation
├── experiments/                    # 28-scenario experimental suite, snapshots, and manifest builders
├── visualization/                  # FigureExporter suite (44 figures across PNG, PDF, SVG, TIFF)
├── pages/                          # Streamlit UI pages (01 through 12)
├── streamlit_utils/                # Data loaders, Plotly visualizers, session state, and themes
└── tests/                          # PyTest unit and integration verification suites

```

---

## 12-Stage Research Workflow

The pipeline executes sequentially through 12 discrete stages:

### Stage 1 — Wholesale Market Data Ingestion

Ingests 8,760 hourly settlement intervals covering a full calendar year. Formats, deduplicates, and structures market time series with consistent timezone-aware DatetimeIndex schemas.

### Stage 2 — Market Dataset Validation

Conducts continuity checks, missingness validation, statistical anomaly bounds, and ensures simultaneous availability of standard price columns (`actual_price` and `price`).

### Stage 3 — Temporal Feature Engineering

Constructs 53 predictive features including 1h/2h/24h/48h/168h lags, rolling statistical windows (mean, standard deviation, min, max), calendar categoricals, and diurnal/seasonal Fourier harmonics.

### Stage 4 — Multi-Step Recursive Price Forecasting

Stage 4 generates look-ahead electricity price forecasts using the configured forecasting model. When a trained forecasting model is unavailable, the pipeline falls back to a deterministic Gaussian perturbation generated from a seeded NumPy random generator, ensuring reproducible experiments across repeated executions.

### Reproducible Research Pipeline

The BTA-V5.0 pipeline is deterministic and reproducible:

* A dedicated NumPy random generator (`default_rng(seed=42)`) is initialized once during pipeline creation.
* Synthetic market prices and fallback forecast perturbations use the same seeded generator.
* Identical inputs produce identical forecast prices, dispatch schedules, EBITDA, Sharpe ratio, SOH trajectory, and exported reports.
* This enables reproducible experiments suitable for research publications and benchmarking.

### Stage 5 — Rolling-Horizon Mixed-Integer Dispatch Optimization

* Solves a utility-scale Battery Energy Storage System dispatch problem using Pyomo MILP.
* Binary operating variables enforce mutually exclusive charging and discharging.
* HiGHS solves the resulting mixed-integer optimization problem.
* Produces an economically optimal and physically feasible dispatch trajectory.

### Binary Dispatch Validation

After every optimization run, the pipeline automatically validates dispatch feasibility. The validation checks every hourly interval and confirms that:

* Charging power and discharging power are never simultaneously positive ($\min(P_t^{chg}, P_t^{dis}) \le 10^{-6}$).
* Idle hours correctly allow both powers to be zero.
* A runtime exception is raised if any simultaneous charging/discharging timestep is detected.
* This validation guarantees compliance with the binary operating-mode constraint introduced in the MILP formulation.

### Stage 6 — Electrochemical Battery Ageing Accounting

Applies ASTM E1049 Rainflow cycle counting on the State of Charge (SOC) profile to quantify cycle fatigue via Miner's Rule, and evaluates thermal Arrhenius calendar loss to update State of Health (SOH) and degradation wear costs.

### Stage 7 — Rolling-Horizon Backtest Consolidation

Consolidates sequential execution horizons into a continuous 8,760-hour backtest record, exporting `dispatch_history.csv` and `degradation_history.csv` to disk.

### Stage 8 — Techno-Economic Arbitrage Performance

Evaluates financial waterfalls: gross market revenue, Rainflow degradation wear, fixed/variable O&M expenses, net EBITDA, unit margin per cycled MWh, and revenue per installed kW-year.

### Stage 9 — Forecast Realism & Value Capture Quality

Computes forecast accuracy (MAE, RMSE, MAPE) and evaluates the Value Capture Ratio (VCR), quantifying revenue realization against a theoretical clairvoyant perfect-foresight benchmark.

### Stage 10 — Downside Risk & Tail Analytics

Extracts daily P&L distributions to compute historical and parametric Value-at-Risk (95% and 99% VaR), Conditional Value-at-Risk (CVaR / Expected Shortfall), maximum drawdown, and annualized Sharpe ratio.

### Stage 11 — Multi-Scenario Sensitivity & Pareto Frontier

Extracts non-dominated solutions across the revenue-longevity trade-off spectrum via `ComparisonEngine`, outputting ranked catalogs and Pareto frontier tables.

### Stage 12 — Publication Synthesis & Master Reports

Renders the complete 44-figure publication visualization suite across four formats (`PNG`, `PDF`, `SVG`, `TIFF`), compiles the multi-tab Master Excel workbook (`summary.xlsx`), and generates the final chapter report.

---

## Mathematical Framework

## 1. Rolling-Horizon MILP Dispatch Formulation

At each decision step, the optimizer maximizes net operating profit over look-ahead horizon $H$:

$$\max \sum_{t=1}^{H} \left[ \hat{\lambda}_t \left( P_t^{dis} - P_t^{chg} \right) - c_{deg} P_t^{dis} - c_{vOM} \left( P_t^{chg} + P_t^{dis} \right) \right] \Delta t$$

Subject to the following operational and physical constraints:

### Binary Operating Mode Constraint

The dispatch model uses a binary operating variable $z_t \in \{0, 1\}$ to ensure the battery can operate in only one direction during each hour:

$$P_t^{chg} \le P_{max}^{chg} z_t \quad \forall t \in \{1, \dots, H\}$$

$$P_t^{dis} \le P_{max}^{dis} (1 - z_t) \quad \forall t \in \{1, \dots, H\}$$

$$z_t \in \{0, 1\} \quad \forall t \in \{1, \dots, H\}$$

Where:

* $z_t = 1 \implies$ charging is permitted and discharging is forced to zero ($P_t^{dis} = 0$).
* $z_t = 0 \implies$ discharging is permitted and charging is forced to zero ($P_t^{chg} = 0$).
* Both powers are zero when the battery is idle ($P_t^{chg} = 0, P_t^{dis} = 0$).

This converts the dispatch model into a Mixed-Integer Linear Program (MILP) while eliminating non-physical simultaneous charging and discharging states.

### State of Energy (SOE) Dynamics

$$E_t = E_{t-1} + \left( \eta_{chg} P_t^{chg} - \frac{P_t^{dis}}{\eta_{dis}} \right) \Delta t \quad \forall t \in \{1, \dots, H\}$$

### State of Charge (SOC) Bounds

$$SOC_t = \frac{E_t}{E_{nom} \cdot SOH_t}$$

$$SOC_{min} \le SOC_t \le SOC_{max} \quad \forall t \in \{1, \dots, H\}$$

### Terminal Energy Constraint

$$E_H \ge E_{target}$$

---

## 2. Electrochemical Degradation Kinetics

Total capacity loss combines cycling fatigue and calendar aging:

$$SOH_t = 1.0 - \left( D_{cycle, t} + D_{calendar, t} \right)$$

### ASTM E1049 Rainflow Cycle Fatigue

Charge-discharge cycles are decomposed into discrete stress reversals using ASTM E1049 Rainflow cycle counting, with damage accumulated via Miner's Rule:

$$D_{cycle} = \sum_{i=1}^{K} \frac{n_i}{N_f(DoD_i)}$$

Where cycle life $N_f$ follows a power-law relationship:

$$N_f(DoD) = a \cdot DoD^{-b}$$

### Arrhenius Calendar Aging Kinetics

Storage aging accounts for cell temperature $T$ and average State of Charge $\overline{SOC}$:

$$D_{calendar}(t) = k_{cal} \cdot \exp\left( -\frac{E_a}{R \cdot T} \right) \cdot \exp\left( k_{soc} \overline{SOC} \right) \cdot t^z$$

---

## 3. Solver Implementation

HiGHS solves the Mixed-Integer Linear Programming (MILP) dispatch model, including binary operating variables that enforce mutually exclusive battery operating modes. GLPK and CBC are fully supported as alternative solvers.

---

##  Baseline Research Results

Empirical results from the full annual 8,760-hour backtest for a **50 MW / 100 MWh NMC storage system** (90.25% AC-AC RTE, 48h horizon, 24h step):

```text
==================================================================================
                 BTA-V5.0 RESEARCH PIPELINE SUMMARY
==================================================================================
  * Data Ingestion                             : [PASS] (  0.07s)
  * Data Validation                            : [PASS] (  0.00s)
  * Feature Engineering                        : [PASS] (  0.09s)
  * Price Forecasting                          : [PASS] (  2.45s)
  * Mixed-Integer Dispatch Optimization        : [PASS] (  0.79s)
  * Battery Ageing                             : [PASS] (  0.01s)
  * Backtest Consolidation                     : [PASS] (  0.64s)
  * Arbitrage Economics                        : [PASS] (  0.00s)
  * Forecast Realism                           : [PASS] (  5.00s)
  * Risk Analytics                             : [PASS] (  0.00s)
  * Sensitivity & Scenarios                    : [PASS] (  0.05s)
  * Publication Reports & Figure Suite         : [PASS] (257.09s)
----------------------------------------------------------------------------------
  Gross Arbitrage Revenue      : $2,713,544.75
  Cell Degradation Wear Cost   : -$ 259,557.16
  Net Operating Profit (EBITDA): $2,075,380.58
  Final State of Health (SOH)  :        98.16%
  Forecast MAE                 :        $2.04/MWh
  Value Capture Ratio (VCR)    :        66.67%
  Asset Sharpe Ratio (Rf=4.0%) :        4.471
  Optimization Solver          : HiGHS (MILP)
  Random Seed                  : 42
  Total Execution Runtime      : 00:04:26 (266.25s)
  Python Runtime               : 3.14.2 on Windows
  Artifacts Saved To           : results/
==================================================================================

```

### Key Performance Indices

| Metric Dimension | Benchmark Value | Research Interpretation |
| --- | --- | --- |
| **Gross Arbitrage Revenue** | **$2,713,544.75** | Captured market spread over 8,760 hourly settlement intervals. |
| **Degradation Wear Cost** | **-$259,557.16** | Monetized ASTM Rainflow fatigue and Arrhenius calendar wear. |
| **Fixed O&M OPEX** | **$359,589.04** | Balance of plant maintenance and administrative overhead. |
| **Variable O&M OPEX** | **$19,017.97** | Throughput-dependent inverter/transformer operating cost. |
| **Net Operating EBITDA** | **$2,075,380.58** | Operating profit after full wear cost and OPEX deduction. |
| **Final State of Health (SOH)** | **98.16%** | Total annual capacity fade restricted to 1.84% under wear penalties. |
| **Value Capture Ratio (VCR)** | **66.67%** | Arbitrage realization relative to a theoretical perfect foresight ceiling. |
| **Forecast Error (MAE)** | **$2.04/MWh** | Multi-step look-ahead forecast error across the 48h horizon. |
| **Annualized Sharpe Ratio** | **4.471** | Risk-adjusted return profile evaluated against a 4.0% risk-free rate. |
| **Daily 95% Value-at-Risk** | **-$12,089.65/day** | Downside market exposure at the 95th percentile confidence tail. |

---

##  28-Scenario Experimental Matrix

The platform includes 28 predefined operating scenarios across seven core sensitivity categories:

| Category | Count | Focus Parameters Tested |
| --- | --- | --- |
| **Look-Ahead Horizon** | 5 | 12h, 24h, 36h, 48h (Baseline), and 72h rolling windows. |
| **Forecasting Model** | 4 | Recursive XGBoost, Persistence naive, Moving Average, and Perfect Foresight. |
| **Battery Chemistry** | 3 | NMC (Baseline, 4,000 cycles), LFP (Stationary, 7,000 cycles), and LTO (15,000 cycles). |
| **Thermal Environment** | 4 | Subcooled 15°C, Reference 25°C, Elevated 35°C, and Harsh Desert 45°C. |
| **Asset Sizing & Duration** | 4 | 25MW/50MWh (2h), 50MW/100MWh (2h), 50MW/200MWh (4h), and 100MW/200MWh (2h). |
| **Round-Trip Efficiency** | 4 | 85.0% (Aging inverters), 90.25% (Baseline), 92.0%, and 95.0% (Next-gen). |
| **Degradation Modeling** | 4 | Calendar-only, ASTM Rainflow combined, Zero-wear unconstrained, and $25/MWh high hurdle. |

---

## Installation Guide

### Prerequisites

* Python 3.11, 3.12, 3.13, or 3.14
* HiGHS optimization solver (or GLPK / CBC)

### 1. Clone the Repository

```bash
git clone [https://github.com/MohdNematullah/utility-scale-bess-arbitrage.git](https://github.com/MohdNematullah/utility-scale-bess-arbitrage.git)
cd utility-scale-bess-arbitrage

```

### 2. Set Up Virtual Environment

```powershell
# Windows
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt

```

### 4. Solver Setup

HiGHS is installed automatically with Python packages (`highspy`). For system-level solver verification:

```powershell
# Windows (Winget)
winget install GLPK.GLPK

# Ubuntu / Debian
sudo apt update && sudo apt install -y glpk-utils

```

---

##  Quick Start

### Execute Full 12-Stage Production Pipeline

Generates a publication-ready dispatch schedule, EBITDA, degradation cost, SOH trajectory, Value Capture Ratio (VCR), Sharpe ratio, VaR/CVaR risk metrics, Pareto frontier, and reproducible research artifacts:

```bash
python main.py --run

```

### Fast Verification Mode (7-Day Sample Run)

```bash
python main.py --run --fast

```

### Run Streamlit Dashboard

```bash
python -m streamlit run app.py

```

Open [http://localhost:8501](http://localhost:8501?utm_source=gemini) in your browser.

---

## Command Line Reference

```bash
# Complete 12-stage analytical pipeline execution
python main.py --run

# Fast verification run (truncated horizon)
python main.py --run --fast

# Rolling-horizon backtest only (Stages 1 through 7)
python main.py --backtest

# Techno-economic and risk metrics evaluation (Stages 8 through 10)
python main.py --metrics

# Generate LaTeX tables, Markdown, and Master Excel workbook
python main.py --reports

# Export all 44 figures across PNG, PDF, SVG, and TIFF
python main.py --figures

# Generate frontend dashboard data feeds
python main.py --dashboard

# Run all 28 sensitivity scenarios
python main.py --experiment ALL

# Run scenario comparison and Pareto frontier extraction
python main.py --compare

# Safely purge previous checkpoints and artifacts
python main.py --clean

# Resume execution from last checkpoint
python main.py --resume

```

---

## Testing & Validation

Execute the complete test suite:

```bash
pytest -v

```

Run domain-specific test suites:

```bash
# Mixed-Integer dispatch optimization & constraint validation
pytest tests/optimization/ -v

# ASTM Rainflow cycle counting & Arrhenius degradation
pytest tests/battery/ -v

# Recursive forecasting models & error metrics
pytest tests/forecasting/ -v

# Backtest engine & experiment runner
pytest tests/backtesting/ -v

# Publication figure exporter and visualizers
pytest tests/visualization/ -v

```

---

## Changelog

### Binary MILP Dispatch Update

* **Optimization Improvements:** Replaced relaxed charging/discharging constraint with a binary operating-mode MILP formulation.
* **Mutual Exclusivity:** Enforced strict non-simultaneous charging and discharging via binary status variables ($u_{chg} + u_{dis} \le 1$).
* **Automated Validation:** Added post-dispatch verification raising runtime exceptions if concurrent charge/discharge intervals occur.
* **Deterministic Reproducibility:** Integrated dedicated NumPy random number generation (`default_rng(seed=42)`) for exact seed reproduction.
* **Solver Architecture:** Standardized on HiGHS Mixed-Integer Linear Programming solver with zero gap tolerance.

---

## Author

**Mohd Nematullah**

*Mechanical Engineer | Energy Storage Analytics | Mathematical Optimization*

* GitHub: [@MohdNematullah](https://github.com/MohdNematullah)
* LinkedIn: [Mohammed Nematullah](https://www.google.com/search?q=https://www.linkedin.com/in/mohammed-nematullah-573a18249/&utm_source=gemini)

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](https://www.google.com/search?q=LICENSE&utm_source=gemini) file for details.

