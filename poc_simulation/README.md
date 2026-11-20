# Group 3 – Cleanroom Feed-Forward Cooling PoC (Real HIPE 3-Month + LBNL FDD)

This directory contains the fully reproducible Python Proof-of-Concept (PoC) for **Section 2.3** (*Short-Horizon Thermal Load Forecasting Model*) and **Section 3.2** (*24-Hour Python Simulation Proof-of-Concept*) of the Group 3 report ([report.pdf](../report.pdf)), grounded in two real public industrial datasets:

1. **HIPE 3-Month Real Packaging Plant Dataset** (`../datasets/hipe_3m_1min_clean.csv`, KIT Germany, Zenodo `14054902`, Oct 1 2017 – Jan 1 2018, 10 packaging machines + `MainTerminal`).
2. **LBNL FDD Experimental AHU Datasets** (`../datasets/lbnl_fdd/{SZVAV,SZCAV,MZVAV-2-1}.csv`, Figshare `11752740`).

## Active Modules (Step 2 — Real-Data, Zero-Leakage Pipeline)

| Module | File | Role |
| :--- | :--- | :--- |
| Psychrometrics & RC Zone Dynamics | `physics_model.py` | Section 3.1 (Eqs. 3.1–3.18), Table 3.1–3.2, Eq. 3.21 |
| Real HIPE + LBNL Pipeline | `data_pipeline_real.py` | 48 clean production days (31 Train / 7 Val / 10 Test), Train-calibrated SEMI E10 states, zero-leakage noisy MES schedule ($\sigma = 3\text{ min}$ jitter + 2% unlogged tests), bay scaling & LBNL lag extraction |
| 5-Model Forecaster & CQR | `forecaster_real.py` | Learning curve (7→31d), Val grid tuning, 5-model benchmark (Persistence, Ridge Sensor/Equip, LightGBM Sensor/Equip), Active-Window Split Conformal P10/P90 |
| 3-Controller Ablation & PLC Guardrails | `controllers.py` | `Static_PID`, `SensorOnly_MPC`, `EquipState_MPC`, pure-L4 mode, $\pm 20\%$ Plant–Model Mismatch, 2-of-3 drift voting, `<1 s` bumpless fallback, LBNL cooling-valve fault interlock |
| Master Runner & Figure Builder | `run_poc_real.py` | Runs all experiments and exports CSV/JSON results + 4 PDF/PNG figures |
| Ready-to-Paste LaTeX | `sections_2_3_and_3_2.tex` | Complete Sections 2.3 and 3.2 formatted for Overleaf |

*(Legacy v1 synthetic scripts `data_generator.py`, `forecaster.py`, `run_poc.py` remain archived for reference.)*

## How to Run

From the repository root:

```bash
pip install -r requirements.txt
python poc_simulation/run_poc_real.py
```

### Generated Outputs
- **Tables (`results/`):**
  - `real_learning_curve.csv`: Validation RMSE across 7, 15, 23, 31 training days (`h = 15 min`).
  - `real_tuning_trials.csv`: Validation hyperparameter grid search logs.
  - `real_forecast_benchmark.csv`: 5-model multi-horizon (`5–60 min`) metrics on 10 unseen December test days (raw real HIPE kW).
  - `real_closed_loop_summary.csv`: 24-hour closed-loop ablation on peak held-out day `2017-12-06` (Scenarios A2 & B, $f = 10\%$).
  - `real_l4_s4_sweep.csv`: Isolated Pure-L4 chiller saving ($s_4$) across $f \in \{5\%, 10\%, 20\%\}$.
  - `real_plant_mismatch.csv`: $\pm 20\%$ Plant–Model Mismatch test ($C_{\text{eff}} \in [25.28, 37.92]\text{ MJ/K}$).
  - `real_fault_guardrail_trace.csv` & `master_results_real.json`.
- **Vector Figures (`figures/`):**
  - `fig1_forecast_real_hipe.{pdf,png}`
  - `fig2_closed_loop_24h_real.{pdf,png}`
  - `fig3_ablation_mismatch_energy.{pdf,png}`
  - `fig4_lbnl_fault_guardrail.{pdf,png}`
