# Utility-Scale BESS Arbitrage & Degradation Platform

A research-grade techno-economic simulation framework evaluating utility-scale Battery Energy Storage System (BESS) arbitrage. The platform couples multi-step recursive price forecasting (XGBoost/LightGBM) with rolling-horizon mathematical dispatch optimization (Pyomo) and physical electrochemical degradation modeling (ASTM E1049-85 Rainflow cycle counting + Arrhenius thermal kinetics).

---

## Core Research Focus

> **How do realistic multi-step electricity price forecasts and dynamic battery ageing affect rolling-horizon arbitrage value in utility-scale Battery Energy Storage Systems (BESS)?**

* **Forecast Horizon Trade-off:** Recursive forecast error compounds over extended horizons ($\text{MAE} = \$2.04/\text{MWh}$ at 48h), capping value capture at **85.5%** of the perfect foresight ceiling, while delivering **+27.3%** higher revenue than persistence baselines.
* **Degradation Opportunity Cost:** Enforcing an explicit wear hurdle ($c_{\text{deg}} = \$10/\text{MWh}$) in the dispatch objective suppresses marginal cycling. Incurring $\$253.8\text{k}$ in battery wear saves $\$431.4\text{k}$ in pack replacement equity compared to unconstrained dispatch, delivering **$4.35M Net EBITDA** at **98.20% SOH** retention ($190.2\text{ EFC}$).

---

## Mathematical Formulations

### 1. Rolling-Horizon Dispatch (Pyomo LP/MILP)

At each decision step, the model solves an open-loop optimization over look-ahead horizon $H \in \{12, 24, 36, 48, 72\}$ with execution step $S \le H$:

$$\max_{\mathbf{P}^{\text{dis}}, \mathbf{P}^{\text{chg}}} \sum_{t=1}^{H} \left[ \hat{\lambda}_t \left(P_t^{\text{dis}} - P_t^{\text{chg}}\right) - c_{\text{deg}} P_t^{\text{dis}} - c_{\text{vOM}} \left(P_t^{\text{dis}} + P_t^{\text{chg}}\right) \right] \Delta t$$

$$\text{subject to:}$$

$$E_t = E_{t-1} + \left( \eta_{\text{chg}} P_t^{\text{chg}} - \frac{P_t^{\text{dis}}}{\eta_{\text{dis}}} \right) \Delta t, \quad \forall t \in \{1, \dots, H\}$$

$$\text{SOC}_{\min} \le \frac{E_t}{E_{\text{nom}} \cdot \text{SOH}} \le \text{SOC}_{\max}$$

$$0 \le P_t^{\text{chg}} \le P_{\max}, \quad 0 \le P_t^{\text{dis}} \le P_{\max}$$

$$E_H = \text{SOC}_{\text{target}} \cdot E_{\text{nom}} \cdot \text{SOH}$$

### 2. Electrochemical Degradation Engine

$$\text{SOH} = 1.0 - \left( D_{\text{cycle}} + D_{\text{cal}} \right)$$

* **Cycle Fatigue (ASTM E1049-85 Rainflow + Miner's Rule):**

$$D_{\text{cycle}} = \sum_{i=1}^{M} \frac{n_i}{N_f(\text{DoD}_i)}, \quad N_f(\text{DoD}_i) = \alpha (\text{DoD}_i)^{-\beta} \exp\left(\gamma (1 - \overline{\text{SOC}}_i)\right)$$


* **Calendar Fade (Arrhenius Kinetics):**

$$D_{\text{cal}}(t) = k_{\text{cal}} \exp\left(-\frac{E_a}{R \cdot T_{\text{cell}}}\right) \exp\left(k_{\text{soc}} \cdot \overline{\text{SOC}}\right) t^{0.5}$$


* **Degradation Cost:** $C_{\text{deg}} = \Delta \text{SOH} \cdot \text{CAPEX}_{\text{repl}} \cdot E_{\text{nom}}$ where $\text{CAPEX}_{\text{repl}} = \$130/\text{kWh}$.

### 3. Valuation & Risk Metrics

$$\text{Value Capture Ratio (VCR)} = \frac{\Pi_{\text{realized}}(\hat{\boldsymbol{\lambda}} \mid \boldsymbol{\lambda})}{\Pi_{\text{theoretical}}(\boldsymbol{\lambda} \mid \boldsymbol{\lambda})} \times 100\%$$

$$\text{Annualized Sharpe} = \frac{\overline{R}_d - (r_f / 350)}{\sigma(R_d)} \sqrt{350}, \quad \text{VaR}_{0.95} = -\inf \{r : F_R(r) \ge 0.05\}$$

---

## Benchmark Audit (50 MW / 100 MWh NMC, 48h Horizon, 8,400h)

| Metric | Output Value | Benchmark Context |
| --- | --- | --- |
| **Gross Arbitrage Revenue** | **$4,982,570** | +$632k (+14.5% vs Persistence) |
| **Degradation Wear Cost** | **-$253,757** | 5.09% of gross revenue |
| **Fixed O&M Cost** | **-$359,589** | $7,191.78 / MW-yr installed |
| **Variable O&M Cost** | **-$19,018** | $0.50 / MWh throughput |
| **Net Operating Profit (EBITDA)** | **$4,350,206** | 87.31% net EBITDA margin |
| **Final State of Health (SOH)** | **98.20%** | 1.80% annual capacity fade |
| **Utilization & Cycles** | **190.18 EFC** | 0.543 EFC / day |
| **Forecast Accuracy (MAE)** | **$2.04 / MWh** | 48h multi-step recursive error |
| **Value Capture Ratio (VCR)** | **85.5%** | Standardized to perfect foresight |
| **Annualized Sharpe Ratio** | **3.652** | Risk-free rate $r_f = 4.0\%$ |

---

## System Architecture

```text
├── app.py                         # Multipage Streamlit orchestrator & navigation hub
├── main.py                        # CLI 12-stage pipeline execution engine
├── streamlit_utils/               # Theme, Plotly charts, loaders, and session state
├── pages/                         # 12 Research UI modules (01_Home to 12_Experiment_Runner)
├── forecasting/                   # Recursive XGBoost/LightGBM multi-step models
├── optimization/                  # Pyomo rolling MILP/LP dispatch formulations
├── battery/                       # Rainflow counting, degradation models, chemistry specs
├── backtesting/                   # Closed-loop rolling simulation & telemetry builder
├── analytics/                     # Financial waterfall, risk analytics (VaR/CVaR), and KPIs
├── visualization/                 # 44 high-DPI IEEE publication figures generator
├── experiments/                   # 28-scenario matrix runner and Pareto extractor
├── tests/                         # Comprehensive pytest verification suite
└── results/                       # Generated artifacts, figures, and telemetry JSON/CSVs

```

---

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/your-username/bess-arbitrage-platform.git
cd bess-arbitrage-platform

# Create and activate environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

```

### 2. Solver Setup (GLPK)

* **Windows:** `winget install -e --id GLPK.GLPK`
* **Linux (Ubuntu/Debian):** `sudo apt-get install -y glpk-utils`
* **Verify:** `glpsol --version`

---

## Execution Modes

```bash
# 1. Launch Interactive Streamlit Research Portal
streamlit run app.py

# 2. Run Full 12-Stage Academic Pipeline (8,400h / 350 days, ~25m)
python main.py --run

# 3. Fast Validation Mode (7-day window, ~20s)
python main.py --fast

# 4. Execute 28-Scenario Experimental Sweep
python main.py --experiments

# 5. Export Master Publication Figures (44 figures to results/figures/)
python main.py --figures

# 6. Run Test Suite
pytest -v

```

---

## Experimental Scenarios (28-Run Matrix)

The framework evaluates system sensitivities across 7 key dimensions:

* **Chemistry:** NMC (reference), LFP (high cycle life), LTO (extreme endurance).
* **Look-Ahead Horizon:** 12h, 24h, 36h, 48h, 72h.
* **Thermal Stress:** 15°C (chilled), 25°C (standard), 35°C (sub-tropical), 45°C (desert).
* **Wear Hurdle:** $0/MWh (unconstrained), $5/MWh, $10/MWh (nominal), $20/MWh, $35/MWh.
* **Duration/Sizing:** 1-hour (1C), 2-hour (C/2), 4-hour (C/4).
* **Round-Trip Efficiency:** 85%, 90%, 95%.
* **Forecasting Strategy:** Persistence baseline, Recursive ML, Perfect Foresight.

---

## License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.