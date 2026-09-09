# Utility-Scale BESS Arbitrage & Electrochemical Degradation Analytics Platform

> 🚀 **Live Interactive Web Portal:** [utility-scale-bess-arbitrage.streamlit.app](https://utility-scale-bess-arbitrage.streamlit.app/)

An institutional-grade techno-economic simulation, mathematical programming, and quantitative backtesting framework designed for utility-scale Battery Energy Storage Systems (BESS) operating in wholesale electricity markets.

The platform integrates multi-step recursive machine learning price forecasting (XGBoost/LightGBM) with rolling-horizon mixed-integer/linear dispatch optimization (Pyomo) and physical electrochemical degradation accounting based on ASTM E1049-85 Rainflow cycle counting and Arrhenius thermal-stress kinetics.

---

## Executive Summary & Research Framework

### Primary Research Problem

> **How do realistic multi-step electricity price forecasts and dynamic battery ageing affect rolling-horizon arbitrage value in utility-scale Battery Energy Storage Systems (BESS)?**

Wholesale merchant battery revenue is governed by an inherent trade-off between maximizing energy arbitrage spread capture and mitigating irreversible capacity loss. Classical asset valuation models frequently adopt one of two flawed extremes:

1. **Unrealistic Perfect Foresight:** Assuming zero forecast error over the planning horizon, yielding unachievable theoretical revenue upper bounds.
2. **Ageing-Blind Linear Dispatch:** Treating cell degradation as a fixed sunk capital expenditure or uniform amortized accounting charge rather than an active marginal opportunity cost ($/MWh throughput).

This platform provides a closed-loop simulation environment that couples non-linear, multi-step price prediction error propagation directly with physical cell fatigue mechanics.

```
+-------------------------------------------------------------------------------------------------------+
|                                    CLOSED-LOOP SIMULATION DYNAMICS                                    |
|                                                                                                       |
|   [ Wholesale Market Data ] ──► [ Recursive ML Forecaster ]                                           |
|                                            │                                                          |
|                                     Price Trajectory                                                  |
|                                            ▼                                                          |
|   [ Battery State (SOH, EFC) ] ──► [ Rolling Pyomo Dispatch ] ──► [ Power Setpoints (Pchg, Pdis) ]    |
|                                            ▲                               │                          |
|                                            │ Wear Hurdle ($/MWh)           ▼                          |
|                                 [ Degradation Feedback ] ◄─── [ Physical Cell Stress Engine ]         |
|                                 (Rainflow + Arrhenius)        (ASTM E1049-85 Cycle + Temp Fade)       |
+-------------------------------------------------------------------------------------------------------+

```

### Key Empirical Findings (50 MW / 100 MWh Reference Asset, 8,400 Hours)

* **Forecast Horizon Error Accumulation:** As look-ahead horizons expand from 12h to 48h, recursive multi-step forecasting error compounds ($\text{MAE} = \$2.04/\text{MWh}$ at 48h). This error propagation misallocates storage inventory across secondary price peaks, bounding realized value capture at **85.5%** of the theoretical perfect-foresight ceiling while delivering a **+27.3%** margin above persistence baselines.
* **Marginal Wear Penalization & Pack Preservation:** Introducing an explicit electrochemical wear penalty ($c_{\text{deg}} = \$10.00/\text{MWh}$) in the optimization objective eliminates low-margin churn. Sacrificing $\$137.8\text{k}$ in gross revenue avoids $\$431.4\text{k}$ in physical battery wear, increasing net operating profit by **$+\$293.6\text{k}$** while keeping annual capacity loss at **1.80%** ($\text{SOH} = 98.20\%$, $190.2\text{ EFC}$).
* **Thermal Compounding:** Elevated operating temperatures (35°C–45°C) increase Arrhenius calendar degradation by up to **2.21×**, requiring active thermal management to prevent premature augmentation.

---

## Mathematical Formulations

### 1. Rolling-Horizon Dispatch Optimization (Pyomo LP/MILP)

At each rolling decision epoch $k$, the optimization agent solves an open-loop scheduling problem over look-ahead horizon $H \in \{12, 24, 36, 48, 72\}$ hours discretized at $\Delta t = 1.0\text{ h}$, with execution step $S \le H$.

#### Objective Function

Maximize net operational profit across the horizon window:

$$\max_{\mathbf{P}^{\text{dis}}, \mathbf{P}^{\text{chg}}} \mathcal{J} = \sum_{t=1}^{H} \left[ \hat{\lambda}_t \cdot \left( P_t^{\text{dis}} - P_t^{\text{chg}} \right) - c_{\text{deg}} \cdot P_t^{\text{dis}} - c_{\text{vOM}} \cdot \left( P_t^{\text{dis}} + P_t^{\text{chg}} \right) \right] \Delta t$$

Where:

* $\hat{\lambda}_t$: Forecasted wholesale settlement price at interval $t$ ($/MWh).
* $P_t^{\text{dis}}, P_t^{\text{chg}}$: Continuous discharge and charge power dispatched at interval $t$ (MW).
* $c_{\text{deg}}$: Marginal cell degradation wear penalty ($/MWh).
* $c_{\text{vOM}}$: Variable Operations & Maintenance cost ($/MWh throughput).
* $\Delta t$: Interval time step ($\Delta t = 1.0\text{ hour}$).

#### System Constraints

**State of Energy (SOE) Storage Dynamics:**


$$E_t = E_{t-1} + \left( \eta_{\text{chg}} \cdot P_t^{\text{chg}} - \frac{1}{\eta_{\text{dis}}} \cdot P_t^{\text{dis}} \right) \Delta t, \quad \forall t \in \{1, \dots, H\}$$

$$\text{SOC}_t = \frac{E_t}{E_{\text{nom}} \cdot \text{SOH}_k}$$

$$\text{SOC}_{\min} \le \text{SOC}_t \le \text{SOC}_{\max}, \quad \forall t \in \{1, \dots, H\}$$

Where:

* $E_{\text{nom}}$: Nameplate energy storage capacity (MWh).
* $\text{SOH}_k$: Battery State of Health at rolling epoch $k \in (0, 1]$.
* $\eta_{\text{chg}}, \eta_{\text{dis}}$: One-way charge and discharge conversion efficiencies, satisfying $\eta_{\text{RTE}} = \eta_{\text{chg}} \cdot \eta_{\text{dis}}$.

**Interconnection & Inverter Bounds:**


$$0 \le P_t^{\text{chg}} \le P_{\max} \cdot u_t^{\text{chg}}, \quad \forall t \in \{1, \dots, H\}$$

$$0 \le P_t^{\text{dis}} \le P_{\max} \cdot u_t^{\text{dis}}, \quad \forall t \in \{1, \dots, H\}$$

$$u_t^{\text{chg}} + u_t^{\text{dis}} \le 1, \quad u_t^{\text{chg}}, u_t^{\text{dis}} \in \{0, 1\}$$

*(Note: For convex linear relaxations where prices $\hat{\lambda}_t > 0$ and round-trip efficiency $\eta_{\text{RTE}} < 1$, binary complementarity variables $u_t$ can be relaxed to continuous $[0, 1]$ bounds without risk of simultaneous charging and discharging).*

**Terminal Horizon Boundary Condition:**


$$E_H = \text{SOC}_{\text{target}} \cdot E_{\text{nom}} \cdot \text{SOH}_k$$

This terminal condition prevents the optimizer from artificially depleting the storage inventory at the end of every look-ahead horizon.

---

### 2. Physical Battery Ageing Engine

Cell health is tracked via total capacity fade ($1.0 - \text{SOH}$), split into mechanical cycle fatigue and chemical calendar loss:

$$\text{SOH}_k = 1.0 - \left( D_{\text{cycle}, k} + D_{\text{cal}, k} \right)$$

#### Cycle Fatigue (ASTM E1049-85 Rainflow Counting)

The continuous State of Charge trajectory $\mathbf{SOC} = \{\text{SOC}_1, \dots, \text{SOC}_N\}$ is filtered for local turning points (peaks and valleys). The ASTM E1049-85 Rainflow Counting algorithm processes this sequence to extract discrete stress events characterized by range $\Delta \text{SOC}_i$ and cycle mean $\overline{\text{SOC}}_i$:

```
SOC(t)
 ^       Peak 1
 |        /\
 |       /  \    Peak 2
 |  /\  /    \    /\
 | /  \/      \  /  \
 |/  Valley 1  \/    \
 +-------------------------> Time

```

Allowable cycles to failure $N_f$ for a cycle with depth of discharge $\text{DoD}_i = \Delta \text{SOC}_i$ follows a power-law fatigue curve adjusted for mean stress:

$$N_f(\text{DoD}_i) = \alpha \cdot (\text{DoD}_i)^{-\beta} \cdot \exp\left( \gamma \cdot (1 - \overline{\text{SOC}}_i) \right)$$

Using Miner's Rule of Linear Damage Accumulation across all $M$ identified cycles:

$$D_{\text{cycle}} = \sum_{i=1}^{M} \frac{n_i}{N_f(\text{DoD}_i)}$$

Where $n_i = 1.0$ for full closed cycles and $n_i = 0.5$ for unclosed half-cycles.

#### Calendar Aging (Arrhenius Kinetics)

Calendar degradation accumulates continuously based on cell temperature and resting state:

$$D_{\text{cal}}(t) = k_{\text{cal}} \cdot \exp\left( - \frac{E_a}{R \cdot T_{\text{cell}}} \right) \cdot \exp\left( k_{\text{soc}} \cdot \overline{\text{SOC}} \right) \cdot t^z$$

Where:

* $E_a$: Activation energy of solid-electrolyte interphase (SEI) passivation ($J/\text{mol}$).
* $R$: Universal gas constant ($8.314\text{ J}/(\text{mol}\cdot\text{K})$).
* $T_{\text{cell}}$: Core cell temperature ($K$).
* $z$: Time exponent reflecting diffusion-limited film growth ($z \approx 0.5$).

#### Equivalent Full Cycles (EFC)

Normalized throughput is tracked as:

$$\text{EFC} = \frac{\sum_{t=1}^T \left( P_t^{\text{chg}} \cdot \eta_{\text{chg}} + P_t^{\text{dis}} \right) \Delta t}{2 \cdot E_{\text{nom}}}$$

#### Financial Cost of Degradation

The capital cost of capacity loss over any operational duration is:

$$C_{\text{deg}} = \Delta \text{SOH} \cdot \text{CAPEX}_{\text{repl}} \cdot E_{\text{nom}}$$

Where $\text{CAPEX}_{\text{repl}} = \$130.00/\text{kWh}$ represents the anticipated pack-level module replacement cost.

---

### 3. Forecast Realism & Value Capture Ratio (VCR)

Recursive multi-step forecasts predict price trajectories step-by-step:

$$\hat{\lambda}_{t+h\vert{}t} = f_\theta\left( \hat{\lambda}_{t+h-1\vert{}t}, \dots, \lambda_t, \mathbf{X}_{t+h} \right), \quad \forall h \in \{1, \dots, H\}$$

#### Error Formulations

$$\text{MAE} = \frac{1}{N} \sum_{i=1}^N \left\vert{} \hat{\lambda}_i - \lambda_i \right\vert{}, \quad \text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (\hat{\lambda}_i - \lambda_i)^2}$$

$$\text{Directional Accuracy (DA)} = \frac{1}{N-1} \sum_{i=2}^N \mathbb{I}\left[ \text{sgn}(\hat{\lambda}_i - \lambda_{i-1}) = \text{sgn}(\lambda_i - \lambda_{i-1}) \right] \times 100\%$$

#### Value Capture Ratio (VCR)

To decouple forecast quality from battery physical sizing, the Value Capture Ratio ($\text{VCR}$) standardizes realized market performance against a theoretical perfect-foresight upper bound:

$$\text{VCR} = \frac{\Pi_{\text{realized}}\left( \mathbf{P}^*(\hat{\lambda}) \mid \lambda \right)}{\Pi_{\text{theoretical}}\left( \mathbf{P}^*(\lambda) \mid \lambda \right)} \times 100\%$$

Where:

* $\mathbf{P}^*(\hat{\lambda})$: Optimal power dispatch schedule vector derived using forecasted price trajectory $\hat{\lambda}$.
* $\mathbf{P}^*(\lambda)$: Optimal power dispatch schedule vector derived under perfect foresight of actual price trajectory $\lambda$.
* $\Pi(\mathbf{P} \mid \lambda)$: Realized net arbitrage profit when dispatch schedule $\mathbf{P}$ is settled against actual spot market prices $\lambda$.

---

---

### 4. Financial Risk & Multi-Criteria Pareto Dominance

#### Financial Risk Profiling

Daily net earnings series $R_d$ ($d = 1, \dots, 350$) determines downside risk:

$$\text{VaR}_\alpha = - \inf \left\{ r \in \mathbb{R} : F_R(r) \ge 1 - \alpha \right\}$$

$$\text{CVaR}_\alpha = - \mathbb{E}\left[ R \mid R \le -\text{VaR}_\alpha \right]$$

$$\text{Annualized Sharpe Ratio} = \frac{\overline{R}_d - (r_f / 350)}{\sigma_d} \cdot \sqrt{350}$$

Where risk-free rate $r_f = 4.0\%$.

#### Pareto Dominance Engine

A scenario $A$ dominates scenario $B$ ($A \succ B$) if:

$$\left( \text{SOH}_A \ge \text{SOH}_B \land \text{EBITDA}_A \ge \text{EBITDA}_B \right) \land \left( \text{SOH}_A > \text{SOH}_B \lor \text{EBITDA}_A > \text{EBITDA}_B \right)$$

Scenarios without dominators form the non-dominated **Pareto Optimal Frontier**.

#### Composite Performance Ranking

Scenarios are ranked using a multi-criteria index:

$$S_i = w_R \cdot \tilde{\Pi}_i + w_{\text{SOH}} \cdot \widetilde{\text{SOH}}_i + w_{\mathcal{S}} \cdot \tilde{\mathcal{S}}_i$$

Normalized via min-max scaling $\tilde{x} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$, with default weights $w_R = 0.45$ (EBITDA), $w_{\text{SOH}} = 0.35$ (Health), and $w_{\mathcal{S}} = 0.20$ (Sharpe).

---

## The 12-Stage Analytical Architecture

```
Stage 01: Data Ingestion & Sanitization
  ├── 8,400 consecutive hours (50 full weeks; balances weekday/weekend cyclicality)
  └── Truncates warm-up lags (168h) and boundary buffers (192h) to prevent look-ahead bias

Stage 02: Exploratory Market Analytics
  ├── Price distribution diagnostics, skewness, kurtosis, and negative price settlement checks
  └── Sparkline volatility clustering and daily spread distribution profiles

Stage 03: Feature Engineering
  ├── Autoregressive lags: t-1, t-2, t-24, t-48, t-168
  ├── Rolling summary statistics: 6h, 12h, 24h, 168h rolling mean, std, min, max
  └── Fourier calendar harmonics: sin/cos daily (24h) and annual (8,760h) periodicities

Stage 04: Multi-Step Recursive Price Forecasting
  ├── Recursive gradient boosted trees (XGBoost / LightGBM)
  ├── Compounding look-ahead validation across 12h, 24h, 36h, 48h, and 72h horizons
  └── Baselines: Persistence, Day-Ahead Mean, and Perfect Foresight benchmarks

Stage 05: Rolling-Horizon Pyomo Dispatch Optimization
  ├── Formulates linear/mixed-integer programming models with solver interfaces (GLPK, HiGHS)
  ├── Enforces non-simultaneous power flow, round-trip losses, and SOC boundary targets
  └── Evaluates variable degradation wear penalties ($0/MWh to $35/MWh)

Stage 06: Semi-Empirical Electrochemical Ageing Engine
  ├── ASTM E1049-85 Rainflow cycle counting on operational SOC trajectories
  ├── Miner's rule stress aggregation and Arrhenius calendar fade kinetics
  └── Chemistry configurations: NMC, LFP, and LTO

Stage 07: Closed-Loop Rolling Backtesting
  ├── Step-by-step chronological rolling simulation loop (e.g., 48h horizon, 24h execution step)
  └── Dynamic health updates passed between successive rolling execution windows

Stage 08: Techno-Economic & Asset Valuation
  ├── Income waterfall: Gross Revenue, Battery Wear, Fixed O&M, Variable O&M, Net EBITDA
  └── Unit economics: $/kW-year installed, $/MWh cycled, and internal rate of return (IRR)

Stage 09: Downside Risk & Tail Analytics
  ├── Daily P&L distributions, 95% Parametric & Historical Value-at-Risk (VaR)
  └── Conditional Value-at-Risk (CVaR), maximum drawdown (MDD), and rolling Sharpe ratios

Stage 10: Multi-Scenario Sensitivity & Pareto Optimization
  ├── 28 predefined experimental scenarios across 7 operational dimensions
  └── Automated Pareto frontier identification and sensitivity tornado decomposition

Stage 11: Publication Graphics Suite
  └── 44 IEEE-formatted high-DPI figures exported to vector (PDF) and raster (PNG)

Stage 12: Automated Reporting & Telemetry Synchronization
  ├── Structured JSON/CSV telemetry exports for web consumption
  └── Automated LaTeX/Markdown academic summary report compilation

```

---

## 28 Predefined Research Scenarios (Test Matrix)

The platform evaluates system sensitivities across 7 key dimensions:

| Category | Scenario ID | Configuration Name | Horizon ($H$) | Temp ($T_c$) | Wear Hurdle | Chemistry |
| --- | --- | --- | --- | --- | --- | --- |
| **Chemistry** | `chem_nmc_baseline` | 50MW / 100MWh Reference Asset | 48h | 25°C | $10.00/MWh | NMC |
|  | `chem_lfp_stationary` | LFP Stationary High-Cycle Pack | 48h | 25°C | $10.00/MWh | LFP |
|  | `chem_lto_high_cycle` | LTO Extreme Endurance | 48h | 25°C | $10.00/MWh | LTO |
| **Look-Ahead** | `horizon_12h` | Short Look-Ahead Intraday | 12h | 25°C | $10.00/MWh | NMC |
|  | `horizon_24h` | Day-Ahead Standard | 24h | 25°C | $10.00/MWh | NMC |
|  | `horizon_36h` | Multi-Day Extended | 36h | 25°C | $10.00/MWh | NMC |
|  | `horizon_48h` | Two-Day Base Reference | 48h | 25°C | $10.00/MWh | NMC |
|  | `horizon_72h` | Long-Range 3-Day Window | 72h | 25°C | $10.00/MWh | NMC |
| **Thermal** | `thermal_mild_15c` | Liquid Chilled HVAC | 48h | 15°C | $10.00/MWh | NMC |
|  | `thermal_reference_25c` | Standard Ambient Controlled | 48h | 25°C | $10.00/MWh | NMC |
|  | `thermal_elevated_35c` | Sub-Tropical Ambient Stress | 48h | 35°C | $10.00/MWh | NMC |
|  | `thermal_extreme_45c` | Arid Desert Ambient Heat | 48h | 45°C | $10.00/MWh | NMC |
| **Wear Hurdle** | `deg_cost_zero` | Unconstrained Ageing-Blind | 48h | 25°C | $0.00/MWh | NMC |
|  | `deg_cost_low_5` | Low Opportunity Hurdle | 48h | 25°C | $5.00/MWh | NMC |
|  | `deg_cost_nominal_10` | Baseline Wear Hurdle | 48h | 25°C | $10.00/MWh | NMC |
|  | `deg_cost_high_20` | Conservative Cell Protection | 48h | 25°C | $20.00/MWh | NMC |
|  | `deg_cost_ultra_35` | Ultra-Preservation Mode | 48h | 25°C | $35.00/MWh | NMC |
| **Sizing** | `size_25mw_100mwh` | 4-Hour Long Duration (C/4) | 48h | 25°C | $10.00/MWh | NMC |
|  | `size_50mw_100mwh` | 2-Hour Standard Peaker (C/2) | 48h | 25°C | $10.00/MWh | NMC |
|  | `size_100mw_100mwh` | 1-Hour Fast Response (1C) | 48h | 25°C | $10.00/MWh | NMC |
| **Efficiency** | `eff_low_85` | Legacy Sub-Optimal Plant | 48h | 25°C | $10.00/MWh | NMC |
|  | `eff_base_90` | Nominal Modern System | 48h | 25°C | $10.00/MWh | NMC |
|  | `eff_high_95` | Advanced SiC Power Train | 48h | 25°C | $10.00/MWh | NMC |
| **Model** | `fc_persistence` | Naive Persistence Baseline | 48h | 25°C | $10.00/MWh | NMC |
|  | `fc_recursive_ml` | Dynamic Recursive XGBoost | 48h | 25°C | $10.00/MWh | NMC |
|  | `fc_perfect_foresight` | Theoretical Perfect Upper Bound | 48h | 25°C | $10.00/MWh | NMC |
| **Degradation** | `aging_none` | Infinite-Life Model | 48h | 25°C | $0.00/MWh | NMC |
|  | `aging_calendar_only` | Passive Calendar Aging Only | 48h | 25°C | $0.00/MWh | NMC |

---

## Baseline Techno-Economic Audit (8,400 Operating Hours)

Performance metrics for the reference configuration (**50 MW / 100 MWh NMC, 48h Horizon, 24h Step, $10/MWh Wear Hurdle, 25°C**):

```
========================================================================================
SYSTEM OPERATIONAL & FINANCIAL AUDIT REPORT
========================================================================================
Metric Item                              Result Value   Benchmark Baseline / Reference
----------------------------------------------------------------------------------------
Operating Horizon Duration               8,400 Hours    50 Full Calendar Weeks (350 Days)
Gross Wholesale Arbitrage Revenue        $4,982,570     +$632,364 (+14.5% vs Persistence)
Electrochemical Battery Wear Cost        -$253,757      5.09% of Gross Arbitrage
Fixed Operations & Maintenance           -$359,589      $7,191.78 / MW-yr Installed
Variable Operations & Maintenance        -$19,018       $0.50 / MWh Throughput Cycled
Net Operating Profit (EBITDA)            $4,350,206     87.31% Net EBITDA Margin
Final Battery State of Health (SOH)      98.20%         1.80% Cumulative Capacity Fade
Equivalent Full Cycles (EFC)             190.18 EFC     0.543 EFC / Day Utilization
Recursive Forecast Error (MAE)           $2.04 / MWh    Compounding Multi-Step Error
Value Capture Ratio (VCR)                85.5%          Relative to Perfect Foresight
Annualized Asset Sharpe Ratio            3.652          Risk-Free Return rf = 4.0%
95% Daily Parametric VaR                 -$1,482/day    Parametric Tail Probability
95% Daily Conditional VaR (CVaR)         -$2,104/day    Expected Shortfall in Tail
========================================================================================

```

---

## Interactive Streamlit Web Portal

The framework features a dedicated, multipage interactive dashboard accessible both locally and through the cloud deployment:

* 🌐 **Live Cloud Portal:** [utility-scale-bess-arbitrage.streamlit.app](https://utility-scale-bess-arbitrage.streamlit.app/)
* 💻 **Local Execution:**
```powershell
streamlit run app.py

```



```
========================================================================================
STREAMLIT RESEARCH CHAPTERS
========================================================================================
01. Executive Overview          Summary findings, 8 research KPIs, economic waterfall
02. Market Explorer             Time-series pricing, hourly heatmaps, and price spreads
03. Feature Engineering         Autoregressive features, moving averages, cyclical signals
04. Price Forecasting           Multi-step recursive ML vs actuals, error diagnostics
05. Dispatch Optimization       Pyomo dispatch schedules, SOC boundaries, and power flows
06. Battery Ageing              ASTM E1049 Rainflow cycle distributions and SOH fade
07. Rolling Backtesting         Chronological simulation timeline and profit trajectories
08. Techno-Economic Metrics     Unit revenue ($/kW-yr, $/MWh), EFC costs, and margins
09. Risk Analytics              Parametric VaR, Conditional CVaR, and rolling Sharpe
10. Scenario Comparison         28-scenario Pareto frontier, sensitivity tornado plots
11. Publication Figures         High-resolution vector/PNG viewer and figure exporter
12. Experiment Suite Runner     Live interactive batch execution console
========================================================================================

```

---

## Directory Structure

```text
├── app.py                            # Streamlit Multipage Web Application
├── main.py                           # CLI Pipeline Orchestrator (Stages 1–12)
├── pyproject.toml                    # Environment specifications & pytest configuration
├── requirements.txt                  # Production dependencies
├── packages.txt                      # OS-level packages (GLPK solver, OpenGL runtime)
├── streamlit_utils/                  # Shared Streamlit UI components
│   ├── charts.py                     # Plotly chart builders (Pareto, SOH gauge, Waterfall)
│   ├── downloads.py                  # In-memory artifact exporters (Excel/ZIP)
│   ├── loaders.py                    # Caching data readers for pipeline outputs
│   ├── session.py                    # Session state and configuration managers
│   └── theme.py                      # CSS styling and KPI cards
├── pages/                            # Multipage analytical chapters
│   ├── 01_Home.py                    # Executive overview dashboard
│   ├── 02_Data_Explorer.py           # Wholesale price series and distribution analysis
│   ├── 03_Feature_Engineering.py      # Autoregressive lags, rolling statistics, cyclical signals
│   ├── 04_Price_Forecasting.py       # Multi-step recursive ML forecasting (MAE/RMSE/VCR)
│   ├── 05_Dispatch_Optimization.py   # Pyomo rolling dispatch and SOC trajectories
│   ├── 06_Battery_Ageing.py          # ASTM E1049-85 Rainflow & Arrhenius degradation
│   ├── 07_Rolling_Backtesting.py     # Chronological closed-loop execution timeline
│   ├── 08_Backtest_Metrics.py        # Financial, asset health, and operational indices
│   ├── 09_Risk_Analytics.py          # Downside risk, VaR, CVaR, and rolling Sharpe
│   ├── 10_Scenario_Comparison.py     # 28-Scenario Pareto analysis and sensitivity tornado
│   ├── 11_Publication_Figures.py     # Publication graphics browser (44 figures)
│   └── 12_Experiment_Runner.py       # Batch experiment execution console
├── forecasting/                      # Feature extraction and multi-step ML models
├── optimization/                     # Pyomo MILP/LP formulations and solver interfaces
├── battery/                          # Rainflow counting, degradation models, chemistry specs
├── backtesting/                      # Rolling simulation loops, comparison engine, and telemetry
│   ├── comparison.py                 # Pareto frontier extraction and scenario ranking
│   └── dashboard_data.py             # Frontend JSON/CSV telemetry serialization
├── analytics/                        # Financial waterfall, risk calculations, and KPIs
├── visualization/                    # Publication figure generator (44 figures)
├── experiments/                      # Scenario definitions and sweep runners
├── tests/                            # Automated test suite (Pytest)
│   └── backtesting/                  # Unit and integration test coverage
└── results/                          # Output directory for logs, figures, and data feeds
    ├── dashboard/                    # Live telemetry feeds for Streamlit (kpis.json, etc.)
    ├── experiments/                  # Experimental sweep results and scenario matrices
    ├── figures/                      # High-resolution PNG and vector exports
    └── sensitivity_analysis.xlsx     # Cross-category sensitivity workbook

```

---

## Installation & Setup

### Prerequisites

* **Python 3.11, 3.12, or 3.13**
* **Git**
* **Linear/MILP Solver:** [GLPK](https://www.google.com/search?q=https://www.gnu.org/software/glpk/) (recommended default), [HiGHS](https://highs.dev/), or CBC.

### 1. Clone the Repository

```powershell
git clone https://github.com/your-username/bess-arbitrage-platform.git
cd bess-arbitrage-platform

```

### 2. Environment Setup

#### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

```

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

```

### 3. Solver Installation

#### Windows

Install WinGLPK via Winget:

```powershell
winget install -e --id GLPK.GLPK

```

Alternatively, download from [SourceForge](https://sourceforge.net/projects/winglpk/) and add `C:\glpk\glpk-X.XX\w64` to your Windows System `PATH`.

#### Linux (Debian / Ubuntu)

```bash
sudo apt-get update
sudo apt-get install -y glpk-utils libgl1-mesa-glx

```

Verify solver availability:

```bash
glpsol --version

```

---

## Execution Modes & CLI Reference

Execute the pipeline via `main.py`:

```powershell
# 1. Full Academic Pipeline (8,400 hours / 350 days, ~25m)
python main.py --run

# 2. Fast Interactive Mode (7-day test window for quick verification, ~20s)
python main.py --fast

# 3. Batch Scenario Matrix Sweep (Executes all 28 predefined research scenarios)
python main.py --experiments

# 4. Generate Master Publication Figures (Renders 44 high-DPI figures to disk)
python main.py --figures

# 5. Evaluate Multi-Scenario Rankings & Extract Pareto Frontier
python main.py --compare --registry-file results/experiments/scenario_matrix.csv --output-dir results/comparison

```

---

## Automated Test Suite

The test suite validates data ingestion, recursive feature engineering, Pyomo model generation, Rainflow cycle counting, and scenario comparison routines.

Run the test suite with `pytest`:

```powershell
# Run the complete test suite
pytest -v

# Run the experimental framework and comparison engine tests
pytest tests/backtesting/test_experiment_framework.py -v

```

```text
tests/backtesting/test_experiment_framework.py::TestScenariosLibrary::test_total_predefined_scenarios_count PASSED [  4%]
tests/backtesting/test_experiment_framework.py::TestScenariosLibrary::test_all_seven_categories_represented PASSED [  8%]
tests/backtesting/test_experiment_framework.py::TestScenariosLibrary::test_lfp_chemistry_modifiers PASSED        [ 25%]
tests/backtesting/test_experiment_framework.py::TestComparisonEngine::test_pareto_frontier_math PASSED           [ 54%]
tests/backtesting/test_experiment_framework.py::TestComparisonEngine::test_full_evaluation_and_file_exports PASSED [ 62%]
tests/backtesting/test_experiment_framework.py::TestDashboardDataBuilder::test_kpi_payload_structure PASSED      [ 66%]
tests/backtesting/test_experiment_framework.py::TestCLIParser::test_cli_compare_command PASSED                    [100%]

======================================== 24 passed in 8.42s ========================================

```


```

---

## License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for complete details. Free for academic, scientific, and commercial use with appropriate attribution.