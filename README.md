# Feed-Forward Cleanroom Cooling Control via Equipment-State Forecasting (ESD/MSL-Constrained MPC)

**Team:** Group 3 — Mentor Pod 1 (HCMUT) · **SEMI TECH Zoomers Challenge @ SEMIEXPO Viet Nam 2026**  
**Code Maintainers (CS Track):** Đinh Tiến (Forecasting AI & 24h PoC — Sec. 2.3 & 3.2) · Hứa Tuệ Minh (Constrained MPC, PLC Guardrails & Operator Dashboard — Sec. 2.4, 4.1 & 4.2)

---

## 1. Overview — What This Repository Proves

Back-end semiconductor packaging cleanrooms (ISO 7 OSAT bays in tropical Ho Chi Minh City) waste large amounts of chiller electricity, electric reheat, and cooling-tower water because **Facilities BMS/PID loops are blind to Manufacturing MES/SECS-GEM states** — forcing static worst-case setpoints (`9.0 °C` MAU coil, `21.0 °C` reheat, `35 ACH` FFU, `12.0 °C` CHW) 24/7 out of fear of ESD (`<40% RH`) or MSL (`>45% RH` under J-STD-033) excursions.

This repository provides a **100% reproducible, zero-target-leakage Python Proof-of-Concept (PoC)** grounded in **two real public industrial datasets**:
1. **HIPE 3-Month Real Power-Electronics Packaging Plant Telemetry** (KIT Germany, Zenodo `14054902`, Oct 1 2017 – Jan 1 2018; 10 real packaging machines + `MainTerminal`).
2. **LBNL FDD Real Experimental AHU Datasets** (FLEXLAB `SZVAV`/`SZCAV` & ASHRAE RP-1312 `MZVAV-2-1`, Figshare `11752740`).

### Key Verified Results (Reproduced Automatically by `run_poc_real.py`)
* **Real-Data Multi-Horizon Forecasting (10 unseen December test days, raw unscaled HIPE kW, zero target leakage):**
  * **Equipment-State LightGBM** ranks **#1 across all 5 horizons (`5, 10, 15, 30, 60 min`)** against Persistence, Ridge (Sensor/Equip), and LightGBM (Sensor-Only).
  * Cuts **Thermal-Shock RMSE (`|dP| > 2 kW`) by 20.6%–30.6%** at `h = 15–60 min` (`4.80 vs 6.04 kW` at `15 min`), shrinks the **Conformal `P90–P10` band by 19.2%–42.6%**, and holds **`94.0%–95.1%` active upper-`P90` coverage** (whereas Sensor-Only under-covers at `85.9%–88.9%`).
* **24-Hour Closed-Loop Cleanroom Simulation (Held-Out Peak Day `2017-12-06`, 1,500 m² HCMC Reference Bay, Table 3.4 aligned within `0.04%`):**
  * **Static PID (`1,421.6 MWh/yr` A2 / `1,069.4 MWh/yr` B):** `4 min` Tier-2 over-temperature excursions (`23.10 °C`).
  * **Sensor-Only MPC:** `3 min` Tier-2 excursions (`23.04 °C`, rising to `19 min` at `f=20%`) — fails the zero-excursion IPMVP gate.
  * **Equipment-State MPC:** **0 min Tier-1 and 0 min Tier-2 (T & RH) violations** (`T_z ∈ [21.58, 22.85] °C`, `RH_max = 43.94%` during `MSL ≥ 3`), saving **20.28% electricity (`288.3 MWh/yr`, `196.3 tCO₂e/yr`) in Scenario A2**, **14.68% (`157.0 MWh/yr`, `106.9 tCO₂e/yr`) in Scenario B**, and **7.05% cooling-tower make-up water (`508 m³/yr` + `853 m³/yr` recoverable MAU condensate)**.
* **Isolated Lever L4 (`s4`, excluding L1–L3):** `3.36%` (`f=5%`), `5.60%` (`f=10%`), `9.06%` (`f=20%`) — validating the workbook prior (`2.5% / 5.0% / 10.0%`).
* **Robustness & Safety Guardrails:** Survives **±20% Plant–Model Mismatch** (`C_eff = 25.28 / 37.92 MJ/K`, `0 min` Tier-1 alarms, `0 min` RH violations) and **3 simultaneous 60-minute faults** (Sensor-1 `+1.8 °C / +6% RH` drift, SECS/GEM comms blackout, and LBNL-derived cooling-valve sticking at 78% authority) with **0 min Tier-1 and 0 min Tier-2 excursions**.

---

## 2. Quickstart — Reproduce Everything in ~2 Minutes (Windows / macOS / Linux)

> **No large downloads required!** The cleaned 1-minute 3-month HIPE CSV (`21 MB`) and the 3 experimental LBNL FDD CSVs (`4.5 MB`) are already included under `datasets/`.

```bash
# 1. Clone the repository
git clone <YOUR_GITHUB_REPO_URL>
cd <REPO_FOLDER>

# 2. Create & activate a Python virtual environment (Python 3.10+ recommended)
python -m venv .venv

# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Windows (CMD):
# .venv\Scripts\activate.bat
# macOS / Linux:
# source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the complete end-to-end PoC (training + 24h MPC ablation + mismatch + fault test + 4 figures)
python poc_simulation/run_poc_real.py
```

**Expected runtime:** ~60–90 seconds on a standard laptop CPU (no GPU needed).  
**Outputs written automatically:**
* All CSV/JSON tables → `poc_simulation/results/`
* All 4 vector PDF + PNG figures → `poc_simulation/figures/`

### Optional: Re-processing HIPE From Raw 5-Second Smart-Meter Archives
The raw 3-month archive (`hipe_cleaned_v1.0.1_geq_2017-10-01_lt_2018-01-01.zip`, `1.33 GB`, MD5 `9de58f4a2b2df3efb0ba29af84b55155`) exceeds GitHub's 100 MB limit and is excluded via `.gitignore`. To reproduce `datasets/hipe_3m_1min_clean.csv` from scratch:
1. Download the 3-month zip from Zenodo Record `14054902` (`https://doi.org/10.5281/zenodo.14054902`) and save as `datasets/hipe/hipe_3months.zip`.
2. Run `python datasets/prepare_hipe_3m.py` (streams directly from `.zip` in ~60 s without unpacking the ~30.6 GB raw CSVs).

---

## 3. Repository Structure

```text
.
├── README.md                                  # This guide
├── requirements.txt                           # Python dependencies (numpy, pandas, scipy, sklearn, lightgbm, matplotlib)
├── .gitignore
├── datasets/
│   ├── hipe_3m_1min_clean.csv                 # Clean 1-min 3-month HIPE telemetry (132,540 rows × 11 meters, 21 MB)
│   ├── hipe_3m_daily_summary.csv              # Per-day production activity & sensor-gap audit (92 days)
│   ├── prepare_hipe_3m.py                     # Streaming 5-sec -> 1-min HIPE 3-month preprocessor & EDA
│   ├── prepare_and_eda.py                     # LBNL lag + initial EDA script
│   ├── eda_hipe_3months.png                   # 3-month production & holiday/drift overview
│   ├── eda_real_datasets.png                  # Individual tool waveforms & LBNL empirical lag/faults
│   ├── hipe/zenodo_meta.json                  # Zenodo 14054902 provenance metadata
│   └── lbnl_fdd/
│       ├── SZVAV.csv                          # LBNL FLEXLAB experimental SZVAV AHU (15,839 min)
│       ├── SZCAV.csv                          # LBNL FLEXLAB experimental SZCAV AHU (21,600 min)
│       ├── MZVAV-2-1.csv                      # ASHRAE RP-1312 experimental MZVAV AHU (21,600 min)
│       └── LBNLDataSynthesisInventory.pdf     # Official LBNL fault dates & descriptions
└── poc_simulation/
    ├── physics_model.py                       # Psychrometrics (Arden Buck), RC thermal-moisture model, chiller/tower (Sec 3.1)
    ├── data_pipeline_real.py                  # 48-day split (31/7/10), SEMI E10 extractor, noisy MES (zero leakage), bay mapper
    ├── forecaster_real.py                     # Learning curve, Val tuning, 5-model benchmark, Active-Window Split CQR (Sec 2.3)
    ├── controllers.py                         # Static_PID, ZoneMPCController, Plant CHW Coordinator, PLC Guardrails (Sec 2.4 & 4.1)
    ├── run_poc_real.py                        # Master experiment orchestrator & 4-figure generator (Sec 3.2)
    ├── sections_2_3_and_3_2.tex               # LaTeX draft for Sections 2.3 & 3.2
    ├── results/
    │   ├── real_learning_curve.csv            # 7 -> 31 train-day validation learning curve
    │   ├── real_tuning_trials.csv             # Validation hyperparameter grid search results
    │   ├── real_forecast_benchmark.csv        # 5-model metrics across h = 5, 10, 15, 30, 60 min (10 unseen test days)
    │   ├── real_feature_importance_h15.csv    # Top split features at h = 15 min
    │   ├── real_closed_loop_summary.csv       # 24h 3-controller ablation summary (Scenarios A2 & B)
    │   ├── real_l4_s4_sweep.csv               # Isolated Pure-L4 chiller saving s4 at f = 5%, 10%, 20%
    │   ├── real_plant_mismatch.csv            # ±20% C_eff Plant–Model Mismatch test
    │   ├── sim_24h_A2_Static_PID.csv          # Minute-by-minute 24h trace (Static PID)
    │   ├── sim_24h_A2_SensorOnly_MPC.csv      # Minute-by-minute 24h trace (Sensor-Only MPC)
    │   ├── sim_24h_A2_EquipState_MPC.csv      # Minute-by-minute 24h trace (Equipment-State MPC)
    │   ├── real_fault_guardrail_trace.csv     # Minute-by-minute 24h trace under 3 simultaneous faults
    │   └── master_results_real.json           # Consolidated JSON snapshot
    └── figures/
        ├── fig1_forecast_real_hipe.{pdf,png}
        ├── fig2_closed_loop_24h_real.{pdf,png}
        ├── fig3_ablation_mismatch_energy.{pdf,png}
        └── fig4_lbnl_fault_guardrail.{pdf,png}
```

---

## 4. Handoff Guide for Hứa Tuệ Minh (Sections 2.4, 4.1 & 4.2)

Everything in `poc_simulation/controllers.py` and `poc_simulation/results/` is structured so you can write **Sections 2.4 & 4.1** and build the **Section 4.2 Operator Monitoring Dashboard** in exact alignment with the live simulation:

### A. Section 2.4 — Control Logic: Two-Level Constrained MPC (`poc_simulation/controllers.py`)
* **Zone-Level Receding-Horizon MPC** (`ZoneMPCController.solve_zone_mpc`, lines `125–240`):
  * Sampling step $\Delta t = 1\text{ min}$, prediction horizon $H = 15\text{ min}$, 3-block move blocking (`u[0:3], u[3:8], u[8:15]`) solved via `L-BFGS-B`.
  * Enforces **asymmetric quantile chance constraints**: rolls out zone temperature under $P_{50}$ (tracking $22.05^\circ\text{C}$), $P_{90}$ (upper barrier inside $23.0^\circ\text{C}$), and $P_{10}$ (lower barrier inside $21.0^\circ\text{C}$), subject to valve slew-rate limit (`≤ 35 kW/min`) and coil capacity $Q_{\text{DCC}} \le UA_{\text{DCC}}(T_z - T_{\text{CHW}})$.
* **Plant-Level Chilled-Water Coordinator** (`ZoneMPCController.compute_plant_setpoint`, lines `71–131`):
  * Updates every `15 min` (with immediate pre-chill override if $T_{\text{CHW,ideal}}$ drops $>0.4\text{ K}$), sizing $T_{\text{CHW,DCC}} \in [11.8, 16.5]^\circ\text{C}$ against the `15–30 min` $P_{90}$ peak demand and recovering standby-chiller headroom as `P90 - P50` narrows.
* **L1–L3 Supervisory Rules** (`run_24h_simulation`, lines `450–480`): Identical between `SensorOnly_MPC` and `EquipState_MPC` for fair ablation (L1 pre-tightens MAU coil to `8.6 °C` `65–75 min ≥ 2τ_w` ahead of `MSL ≥ 3`; L2 bypasses electric reheat when `frac_run ≥ 0.60` & `T_z > 21.35 °C`; L3 turns FFUs down `35 → 30 ACH` in non-RUN with `10 min` pre-restore).

### B. Section 4.1 — PLC Hard Safety Guardrails & Fail-Safe Fallback Matrix (`poc_simulation/controllers.py`)
* **2-of-3 Median Sensor Voting & Drift Isolation** (lines `378–404`): Detects any sensor diverging by `> 0.6 °C` or `> 2.5% RH` from median and isolates it (`60/60 min` isolated during `+1.8 °C / +6% RH` Sensor-1 drift in `fig4`).
* **Bumpless `< 1 s` PID Fallback** (lines `423–449`): Triggers on Tier-1 margin approach (`20.3–23.7 °C`, `36.5–53.5% RH`) or SECS/GEM heartbeat loss; transfers seamlessly to Skogestad SIMC-tuned PID (`kp=55, ki=0.45, kd=180`) while compensating for reheat step change (`60/60 min` safe fallback in `fig4`).
* **LBNL Cooling-Valve Sticking Interlock** (lines `525–539`): Detects valve saturation (`Q_cmd > Q_applied + 5 kW`), forces `T_chw = 11.8 °C` and `35 ACH` (`0 min` Tier-1/Tier-2 violation in `fig4`).

### C. Section 4.2 — Operator Monitoring Dashboard Data Feeds (`poc_simulation/results/`)
You can build a lightweight Streamlit / Dash / HTML interactive dashboard reading directly from:
* **`sim_24h_A2_EquipState_MPC.csv`** (1,440 1-min rows): columns `hour`, `Q_room_true_kW`, `msl_ge3_active`, `RH_tier2_hi`, `T_z`, `RH_z`, `DP_z`, `T_chw_dcc`, `ach_rec`, `P_chiller_kW`, `P_reheat_kW`, `P_fans_kW`, `P_hvac_total_kW`, `makeup_L_min`, `tier1_viol`, `tier2_viol`.
* **`real_fault_guardrail_trace.csv`** (1,440 1-min rows): same columns plus live alarm flags `T_s1_raw`, `RH_s1_raw`, `sensor_fault_detected`, `fallback_active`, `valve_fault_active` — perfect for demonstrating live PLC fault isolation & fallback on slides!

---

## 5. Sections 2.4, 4.1 & 4.2 — Delivered (Hứa Tuệ Minh)

| Deliverable | File |
| :--- | :--- |
| LaTeX for Sec. 2.4 (Two-Level MPC), 4.1 (PLC guardrails), 4.2 (dashboard) | `poc_simulation/sections_2_4_4_1_4_2.tex` |
| Operator dashboard (Sec. 4.2) | `dashboard/app.py` + theme `.streamlit/config.toml` |
| Dashboard figures for the report | `poc_simulation/figures/fig5_dashboard_fault_test.png`, `fig6_dashboard_comparison.png`, `fig7_dashboard_live.png` |
| L4 saving decomposition (CHW-reset COP gain vs. staging credit) | `poc_simulation/analysis_l4_decomposition.py` → `results/real_l4_staging_decomposition.csv` |

### Run the dashboard
```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```
Three views: **Live operation** (24 h replay of any controller), **PLC fault test** (three injected faults with an event log built from the simulator flags), **Controller comparison** (10:12 ChipPress surge, 24 h summary, electricity split). Design follows the ISA-101 high-performance HMI idea: neutral when normal, colour only for abnormal states and always paired with a word.

### Calibration note for Sec. 3.2 / 2.4
`analysis_l4_decomposition.py` shows that the isolated-L4 saving `s4` = CHW-reset COP gain + a parametric standby-staging credit (`standby_margin_frac` in `controllers.py`):

| f | s4 total | CHW-reset COP gain | Staging credit | Workbook prior |
| :-: | :-: | :-: | :-: | :-: |
| 5 % | 3.36 % | 2.26 % | 1.10 % | 2.5 % |
| 10 % | 5.60 % | 3.41 % | 2.20 % | 5.0 % |
| 20 % | 9.06 % | 4.47 % | 4.59 % | 10.0 % |

The match with the workbook prior relies on the credit, which is parameterised with `f`. In the full A2 ablation the credit is 1.80 kW of the 32.92 kW average reduction (≈1.1 pp of the 20.28 % headline). Sec. 2.4 states this explicitly; Sec. 3.2 wording ("tightly validating the prior") may need softening.
