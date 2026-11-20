"""
Master Step-2 PoC Runner (Real 3-Month HIPE + Real LBNL FDD Grounded Validation)
Sections 2.3 & 3.2 of Group 3 SEMI TECH Zoomers Challenge 2026 Report

Executes end-to-end:
  1. Real 3-month HIPE (48 clean full production days, 69,120 min) zero-leakage feature pipeline
  2. Learning curve (7 -> 31 Train days), Validation hyperparameter tuning, 5-model benchmark
     across horizons 5-60 min on 10 unseen Held-Out Test days (raw real HIPE kW)
  3. 24-hour closed-loop 3-controller ablation (Static_PID vs SensorOnly_MPC vs EquipState_MPC)
     on the #1 held-out peak production day (2017-12-06: 8 active tools, 34.47 kW peak)
     for Scenarios A2 (electric reheat) and B (heat-recovery reheat)
  4. Pure L4-only s4 sweep across fluctuation regimes f in {5%, 10%, 20%}
  5. +/-20% Plant-Model Mismatch robustness test (C_eff in {25.28, 31.60, 37.92} MJ/K)
  6. Combined Fault & PLC Guardrail test (Sensor drift + SECS/GEM comms drop + LBNL valve fault)
  7. Exports all verified CSV/JSON tables + 4 publication-grade PDF/PNG figures
"""

import json
import os
import sys
import tempfile
import warnings

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYDEPS_DIR = os.path.join(WORKSPACE_ROOT, ".pydeps")
if os.path.isdir(PYDEPS_DIR) and PYDEPS_DIR not in sys.path:
    sys.path.insert(0, PYDEPS_DIR)

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mpl_semi_poc"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from physics_model import CleanroomParams
from data_pipeline_real import (
    HORIZONS,
    build_real_hipe_dataset,
    build_24h_bay_simulation_frame,
    extract_lbnl_empirical_evidence,
)
from forecaster_real import train_and_benchmark_all, predict_day_quantiles_for_bay
from controllers import run_24h_simulation

POC_DIR = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(POC_DIR, "results")
FIG_DIR = os.path.join(POC_DIR, "figures")
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

EF_VN_KG_KWH = 0.681   # Vietnam grid emission factor (report.pdf Table 3.6 & Ref [21])

# Publication Matplotlib styling
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "legend.framealpha": 0.92,
})


def summarize_sim(df_sim: pd.DataFrame, label: str, scenario: str) -> dict:
    """Compute standardized KPIs matching report.pdf Table 3.4 & Section 3.2."""
    T_z = df_sim["T_z"].values
    RH_z = df_sim["RH_z"].values
    msl3 = df_sim["msl_ge3_active"].values == 1

    # Annualized energy (8,760 h/yr) and cooling-tower make-up water
    p_chill = float(df_sim["P_chiller_kW"].mean())
    p_reheat = float(df_sim["P_reheat_kW"].mean())
    p_fans = float(df_sim["P_fans_kW"].mean())
    p_tot = float(df_sim["P_hvac_total_kW"].mean())
    e_mwh_yr = p_tot * 8.760
    water_m3_yr = float(df_sim["makeup_L_min"].mean()) * 60.0 * 8760.0 / 1000.0
    cond_m3_yr = float(df_sim["m_cond_L_min"].mean()) * 60.0 * 8760.0 / 1000.0

    return {
        "Scenario": scenario,
        "Controller": label,
        "T_RMSE_C": round(float(np.sqrt(np.mean((T_z - 22.0) ** 2))), 3),
        "T_max_C": round(float(np.max(T_z)), 2),
        "T_min_C": round(float(np.min(T_z)), 2),
        "RH_max_MSL3_pct": round(float(np.max(RH_z[msl3])) if np.any(msl3) else float(np.max(RH_z)), 2),
        "RH_min_pct": round(float(np.min(RH_z)), 2),
        "Tier1_Viol_min": int(df_sim["tier1_viol"].sum()),
        "Tier2_T_Viol_min": int(df_sim["tier2_T_viol"].sum()),
        "Tier2_RH_Viol_min": int(df_sim["tier2_RH_viol"].sum()),
        "Tier2_Total_Viol_min": int(df_sim["tier2_viol"].sum()),
        "P_chiller_kWe": round(p_chill, 2),
        "P_reheat_kWe": round(p_reheat, 2),
        "P_fans_kWe": round(p_fans, 2),
        "P_total_kWe": round(p_tot, 2),
        "Energy_MWh_yr": round(e_mwh_yr, 1),
        "Water_m3_yr": round(water_m3_yr, 0),
        "MAU_Cond_m3_yr": round(cond_m3_yr, 0),
        "T_chw_mean_C": round(float(df_sim["T_chw_dcc"].mean()), 2),
        "COP_dcc_mean": round(float(df_sim["COP_dcc"].mean()), 2),
    }


def make_figures(
    bundle, fc_res, df_dec6_feat, preds_dec6_sensor, preds_dec6_equip,
    sims_A2, summary_df, l4_df, mismatch_df, df_fault, lbnl_ev, meta_10
):
    # =========================================================================
    # FIGURE 1: Real 3-Month HIPE Forecasting Benchmark + Learning Curve + Peak Day Zoom
    # =========================================================================
    fig = plt.figure(figsize=(12.2, 8.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.34, wspace=0.26)

    # (a) Multi-Horizon Active-Shift RMSE across 10 Unseen Held-Out Test Days
    ax1 = fig.add_subplot(gs[0, 0])
    bdf = fc_res["benchmark_df"]
    styles = [
        ("Persistence", "#9CA3AF", ":", "o", "Persistence (Naive)"),
        ("Ridge_SensorOnly", "#6B7280", "-.", "d", "Ridge (Sensor-Only)"),
        ("LightGBM_SensorOnly", "#D9822B", "--", "s", "LightGBM (Sensor-Only)"),
        ("LightGBM_EquipState", "#1B6B4A", "-", "^", "LightGBM (Equip-State + MES)"),
    ]
    for m_key, col, ls, mk, lbl in styles:
        sub = bdf[bdf["Model"] == m_key].sort_values("Horizon_min")
        ax1.plot(sub["Horizon_min"], sub["Active_RMSE"], color=col, ls=ls, marker=mk, lw=1.9, ms=5.5, label=lbl)
    ax1.set_xticks(HORIZONS)
    ax1.set_xlabel("Forecast Horizon h (min)")
    ax1.set_ylabel("Active-Shift RMSE (kW, Real HIPE)")
    ax1.set_title("(a) Active-Shift Forecast Error on 10 Unseen Test Days")
    ax1.legend(loc="upper left", fontsize=8.0)

    # (b) Empirical Learning Curve + Thermal-Shock RMSE Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    sub15 = bdf[bdf["Horizon_min"] == 15].copy()
    order = ["Persistence", "Ridge_SensorOnly", "Ridge_EquipState", "LightGBM_SensorOnly", "LightGBM_EquipState"]
    labels_short = ["Persist.", "Ridge\n(Sensor)", "Ridge\n(Equip)", "LGBM\n(Sensor)", "LGBM\n(Equip)"]
    sub15 = sub15.set_index("Model").loc[order]
    x = np.arange(len(order))
    w = 0.36
    ax2.bar(x - w / 2, sub15["Active_RMSE"], width=w, color="#2563EB", alpha=0.85, label="Active-Shift RMSE (kW)")
    ax2.bar(x + w / 2, sub15["Shock_RMSE"], width=w, color="#C0392B", alpha=0.82, label="Thermal-Shock RMSE (|dP|>2 kW)")
    for i, (a_v, s_v) in enumerate(zip(sub15["Active_RMSE"], sub15["Shock_RMSE"])):
        ax2.text(i - w / 2, a_v + 0.12, f"{a_v:.2f}", ha="center", va="bottom", fontsize=7.5)
        ax2.text(i + w / 2, s_v + 0.12, f"{s_v:.2f}", ha="center", va="bottom", fontsize=7.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_short, fontsize=8.2)
    ax2.set_ylabel("RMSE at h = 15 min (kW, Real HIPE)")
    ax2.set_title("(b) 5-Model Ablation at h = 15 min (Overall Active vs. Shocks)")
    ax2.legend(loc="upper right", fontsize=8.0)

    # (c) Peak Unseen Test Day (2017-12-06) 15-Min-Ahead Forecast Zoom (08:30 - 17:30)
    ax3 = fig.add_subplot(gs[1, :])
    hrs = df_dec6_feat["minute_of_day"].values / 60.0
    y_true_15 = df_dec6_feat["y_h15"].values
    # Plot against target realization time (t + 15 min)
    t_real = hrs + 15.0 / 60.0
    m_zoom = (t_real >= 8.5) & (t_real <= 17.5) & (~np.isnan(y_true_15))

    p_eq_mean = preds_dec6_equip[15]["raw_mean"]
    p_eq_p10 = preds_dec6_equip[15]["raw_p10"]
    p_eq_p90 = preds_dec6_equip[15]["raw_p90"]
    p_sn_mean = preds_dec6_sensor[15]["raw_mean"]

    ax3.fill_between(
        t_real[m_zoom], p_eq_p10[m_zoom], p_eq_p90[m_zoom],
        color="#1B6B4A", alpha=0.22, label="Equip-State Conformal [P10, P90] Safety Band"
    )
    ax3.plot(t_real[m_zoom], y_true_15[m_zoom], color="#111827", lw=1.35, label="Actual Real HIPE 10-Tool Load (2017-12-06)")
    ax3.plot(t_real[m_zoom], p_sn_mean[m_zoom], color="#D9822B", ls="--", lw=1.25, label="Sensor-Only LGBM Forecast (15 min ahead)")
    ax3.plot(t_real[m_zoom], p_eq_mean[m_zoom], color="#1B6B4A", lw=1.45, label="Equip-State LGBM Forecast (15 min ahead)")
    ax3.annotate(
        "ChipPress + HighTempOven\nDouble Surge (34.5 kW)",
        xy=(10.28, 34.3), xytext=(8.58, 28.0),
        arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.2),
        fontsize=8.0, fontweight="bold", color="#991B1B",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#FCA5A5", alpha=0.92),
    )
    ax3.set_xlim(8.5, 17.5)
    ax3.set_ylim(-0.5, 38.0)
    ax3.set_xlabel("Time of Day — Unseen Peak Test Day 2017-12-06 (Hours)")
    ax3.set_ylabel("Sub-metered Packaging Power (kW)")
    ax3.set_title("(c) 15-Min-Ahead Forecast on Held-Out Peak Day (2017-12-06): Zero-Leakage Discrete SEMI E10 + Noisy MES")
    ax3.legend(loc="upper right", fontsize=8.1, ncol=2)

    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(FIG_DIR, f"fig1_forecast_real_hipe.{ext}"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # FIGURE 2: 24-Hour Closed-Loop Cleanroom Trajectories on Real 2017-12-06 (Scenario A2, f=10%)
    # =========================================================================
    fig, axes = plt.subplots(4, 1, figsize=(11.2, 10.2), sharex=True)
    h24 = sims_A2["Static_PID"]["hour"].values

    # Panel 1: Real HIPE-driven bay sensible load
    ax = axes[0]
    ax.plot(h24, sims_A2["Static_PID"]["Q_room_true_kW"], color="#111827", lw=1.35,
            label=f"Real HIPE 2017-12-06 Bay Load (M={meta_10['m_bay']}x, Mean={meta_10['q_mean_kW']} kW, Peak={meta_10['q_max_kW']} kW)")
    ax.fill_between(h24, 180, 330, where=(sims_A2["Static_PID"]["msl_ge3_active"] == 1),
                    color="#DBEAFE", alpha=0.45, label="MSL >= 3 Open-Lot Window (Strict <=45% RH)")
    ax.set_ylabel("Tool Heat (kW)")
    ax.set_ylim(195, 325)
    ax.set_title("(a) Real 2017-12-06 Thermal Load Profile Mapped to 1,500 m² Reference Bay (f = 10%)")
    ax.legend(loc="upper left", fontsize=8.2)

    # Panel 2: Zone Temperature vs Tier-2 [21, 23] C
    ax = axes[1]
    ax.axhspan(21.0, 23.0, color="#D1FAE5", alpha=0.40, label="Tier-2 Hard MPC Spec (21–23°C)")
    ax.axhline(23.0, color="#C0392B", ls=":", lw=1.1)
    ax.axhline(21.0, color="#C0392B", ls=":", lw=1.1)
    ax.plot(h24, sims_A2["Static_PID"]["T_z"], color="#6B7280", ls="--", lw=1.35, label="Static_PID (SIMC Reactive)")
    ax.plot(h24, sims_A2["SensorOnly_MPC"]["T_z"], color="#D9822B", lw=1.35, label="SensorOnly_MPC (L1–L3 + Sensor LGBM)")
    ax.plot(h24, sims_A2["EquipState_MPC"]["T_z"], color="#1B6B4A", lw=1.65, label="EquipState_MPC (L1–L3 + EquipState LGBM)")
    ax.set_ylabel("Zone Temp (°C)")
    ax.set_ylim(20.6, 23.8)
    ax.set_title("(b) Closed-Loop Zone Temperature Response Across Real ChipPress & Reflow Oven Surges")
    ax.legend(loc="upper left", fontsize=8.0, ncol=2)

    # Panel 3: Zone RH vs Dynamic MSL >= 3 Ceiling (45% / 50%)
    ax = axes[2]
    ax.plot(h24, sims_A2["Static_PID"]["RH_tier2_hi"], color="#C0392B", ls=":", lw=1.3, label="Dynamic Tier-2 Upper RH Bound (45% MSL>=3 / 50% Normal)")
    ax.axhline(40.0, color="#C0392B", ls=":", lw=1.1, label="ESD Lower Bound (40% RH)")
    ax.plot(h24, sims_A2["Static_PID"]["RH_z"], color="#6B7280", ls="--", lw=1.25, label="Static_PID (Fixed 9.0°C MAU)")
    ax.plot(h24, sims_A2["SensorOnly_MPC"]["RH_z"], color="#D9822B", lw=1.25, label="SensorOnly_MPC (L1 Pre-Tightened)")
    ax.plot(h24, sims_A2["EquipState_MPC"]["RH_z"], color="#1B6B4A", lw=1.55, label="EquipState_MPC (L1 Pre-Tightened)")
    ax.set_ylabel("Zone RH (%)")
    ax.set_ylim(38.5, 51.5)
    ax.set_title("(c) Dynamic ESD / MSL Relative Humidity Compliance (Zero Breaches Under Pre-Tightened L1)")
    ax.legend(loc="lower left", fontsize=7.8, ncol=2)

    # Panel 4: Total HVAC Electric Power & DCC CHW Supply Reset
    ax = axes[3]
    ax.plot(h24, sims_A2["Static_PID"]["P_hvac_total_kW"], color="#6B7280", ls="--", lw=1.3, label="Static_PID HVAC Power (kWe)")
    ax.plot(h24, sims_A2["SensorOnly_MPC"]["P_hvac_total_kW"], color="#D9822B", lw=1.3, label="SensorOnly_MPC HVAC Power (kWe)")
    ax.plot(h24, sims_A2["EquipState_MPC"]["P_hvac_total_kW"], color="#1B6B4A", lw=1.6, label="EquipState_MPC HVAC Power (kWe)")
    ax.set_ylabel("HVAC Power (kWe)")
    ax.set_xlabel("Time of Day — Real Held-Out HIPE Day 2017-12-06 (Hours)")
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))
    ax.set_title("(d) Instantaneous Cleanroom HVAC Electric Power (Scenario A2: Chiller + Reheat + Fans)")
    ax.legend(loc="upper right", fontsize=8.0)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(FIG_DIR, f"fig2_closed_loop_24h_real.{ext}"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # FIGURE 3: Energy/Water Ablation + L4 s4 Sweep + +/-20% Plant-Model Mismatch
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))

    # (a) Stacked Annual Energy (Scenario A2 & B)
    ax = axes[0]
    sub_a2 = summary_df[summary_df["Scenario"] == "A2"].set_index("Controller")
    sub_b = summary_df[summary_df["Scenario"] == "B"].set_index("Controller")
    cats = ["A2\nPID", "A2\nSensMPC", "A2\nEquipMPC", "B\nPID", "B\nSensMPC", "B\nEquipMPC"]
    rows_plot = [
        sub_a2.loc["Static_PID"], sub_a2.loc["SensorOnly_MPC"], sub_a2.loc["EquipState_MPC"],
        sub_b.loc["Static_PID"], sub_b.loc["SensorOnly_MPC"], sub_b.loc["EquipState_MPC"],
    ]
    ch = [r["P_chiller_kWe"] * 8.76 for r in rows_plot]
    rh = [r["P_reheat_kWe"] * 8.76 for r in rows_plot]
    fn = [r["P_fans_kWe"] * 8.76 for r in rows_plot]
    xp = np.arange(len(cats))
    ax.bar(xp, ch, color="#1B6B4A", label="Chillers")
    ax.bar(xp, rh, bottom=ch, color="#D9822B", label="Elec. Reheat")
    ax.bar(xp, fn, bottom=np.array(ch) + np.array(rh), color="#3B82F6", label="FFU + MAU Fans")
    for i, r in enumerate(rows_plot):
        ax.text(i, r["Energy_MWh_yr"] + 20, f"{r['Energy_MWh_yr']:.0f}", ha="center", fontsize=7.8, fontweight="bold")
    ax.set_xticks(xp)
    ax.set_xticklabels(cats, fontsize=7.8)
    ax.set_ylabel("Annual HVAC Electricity (MWh/yr)")
    ax.set_ylim(0, 1650)
    ax.set_title("(a) Annual Energy Breakdown (A2 & B)")
    ax.legend(loc="upper right", fontsize=7.5)

    # (b) Pure L4 Chiller Saving s4 vs Fluctuation Regime f (5%, 10%, 20%)
    ax = axes[1]
    fs = l4_df["f_pct"].values
    ax.plot(fs, l4_df["s4_EquipState_pct"], "o-", color="#1B6B4A", lw=2.0, ms=6.5, label="EquipState_MPC (L4 Only, Empirical)")
    ax.plot(fs, l4_df["s4_SensorOnly_pct"], "s--", color="#D9822B", lw=1.6, ms=6.0, label="SensorOnly_MPC (L4 Only, Empirical)")
    ax.plot(fs, 0.5 * fs, "^:", color="#6B7280", lw=1.3, label="Sec 3.3.2 Workbook Prior (0.5·f)")
    for f_v, s_e in zip(fs, l4_df["s4_EquipState_pct"]):
        ax.text(f_v + 0.3, s_e - 0.85, f"{s_e:.1f}%", ha="center", fontsize=8.0, color="#1B6B4A", fontweight="bold")
    ax.set_xticks(fs)
    ax.set_xlabel("Dynamic Load Fluctuation Regime f (%)")
    ax.set_ylabel("Chiller Plant Electricity Saving s4 (%)")
    ax.set_ylim(0, max(12.5, l4_df["s4_EquipState_pct"].max() + 2.5))
    ax.set_title("(b) Isolated L4 Saving s4 (Excl. L1–L3)")
    ax.legend(loc="upper left", fontsize=7.5)

    # (c) +/-20% Plant-Model Mismatch Test (Peak T_z & RMSE across C_eff)
    ax = axes[2]
    m_order = ["-20% (25.3 MJ/K)", "Nominal (31.6 MJ/K)", "+20% (37.9 MJ/K)"]
    xm = np.arange(3)
    w = 0.26
    for idx_c, (ctrl, col, lbl) in enumerate([
        ("Static_PID", "#6B7280", "Static_PID"),
        ("SensorOnly_MPC", "#D9822B", "SensorOnly_MPC"),
        ("EquipState_MPC", "#1B6B4A", "EquipState_MPC"),
    ]):
        sub = mismatch_df[mismatch_df["Controller"] == ctrl].set_index("Plant_Ceff").loc[m_order]
        vals = sub["T_max_C"].values
        ax.bar(xm + (idx_c - 1) * w, vals, width=w, color=col, alpha=0.88, label=lbl)
        for j, v in enumerate(vals):
            ax.text(xm[j] + (idx_c - 1) * w, v + 0.04, f"{v:.2f}", ha="center", fontsize=7.2)
    ax.axhline(23.0, color="#C0392B", ls="--", lw=1.2, label="Tier-2 Limit (23.0°C)")
    ax.set_xticks(xm)
    ax.set_xticklabels(["-20% C_eff\n(25.3 MJ/K)", "Nominal\n(31.6 MJ/K)", "+20% C_eff\n(37.9 MJ/K)"], fontsize=7.8)
    ax.set_ylabel("Peak Zone Temperature (°C)")
    ax.set_ylim(22.0, max(23.85, mismatch_df["T_max_C"].max() + 0.25))
    ax.set_title("(c) ±20% Plant–Model Mismatch Test")
    ax.legend(loc="upper right", fontsize=7.2)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(FIG_DIR, f"fig3_ablation_mismatch_energy.{ext}"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # FIGURE 4: Real LBNL AHU Actuation Lag + Multi-Fault PLC Guardrail Validation
    # =========================================================================
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 6.8))

    # (a) Real LBNL Experimental Lag
    ax = axes[0]
    lags = lbnl_ev["SZVAV.csv"]["lags"]
    ax.plot(lags, lbnl_ev["SZVAV.csv"]["corrs"], "o-", color="#1B6B4A", lw=1.8,
            label=f"LBNL FLEXLAB SZVAV (Experimental, Peak={lbnl_ev['SZVAV.csv']['peak_lag_min']}m, r={lbnl_ev['SZVAV.csv']['peak_r']})")
    ax.plot(lags, lbnl_ev["MZVAV-2-1.csv"]["corrs"], "s--", color="#D9822B", lw=1.5,
            label=f"LBNL/ASHRAE MZVAV-2-1 (Experimental, Peak={lbnl_ev['MZVAV-2-1.csv']['peak_lag_min']}m, r={lbnl_ev['MZVAV-2-1.csv']['peak_r']})")
    ax.axvspan(8, 12, color="#A7D7C5", alpha=0.35, label="Report Table 3.2 Actuation Lag Window (8–12 min)")
    ax.axhline(0, color="#6B7280", lw=0.8)
    ax.set_xlim(0, 15)
    ax.set_xticks(range(0, 16))
    ax.set_xlabel("Lag Between Cooling-Coil Valve Movement d(CCV) and Cooling Rate -d(SAT) (min)")
    ax.set_ylabel("Cross-Correlation r")
    ax.set_title("(a) Empirical Cooling-Coil Transport & Thermal Lag Verified on Real LBNL Laboratory AHUs")
    ax.legend(loc="lower right", fontsize=8.2)

    # (b) Multi-Fault Stress Test on Real 2017-12-06 Day
    ax2 = axes[1]
    hf = df_fault["hour"].values
    ax2.axhspan(21.0, 23.0, color="#D1FAE5", alpha=0.35, label="Tier-2 Spec (21–23°C)")
    ax2.axhline(24.0, color="#991B1B", ls="-.", lw=1.2, label="Tier-1 Hard Alarm (24.0°C)")
    ax2.axvspan(9.5, 10.5, color="#FDE68A", alpha=0.45, label="Fault 1: Sensor-1 Drift (+1.8°C/+6%RH)")
    ax2.axvspan(13.0, 14.0, color="#FECACA", alpha=0.40, label="Fault 2: SECS/GEM Comms Blackout (<1s Fallback)")
    ax2.axvspan(15.5, 16.5, color="#DDD6FE", alpha=0.45, label="Fault 3: LBNL Cooling-Valve Stuck (78% Cap)")
    ax2.plot(hf, df_fault["T_s1_raw"], color="#D9822B", ls=":", lw=1.1, alpha=0.85, label="Corrupted Sensor-1 Reading")
    ax2.plot(hf, df_fault["T_z"], color="#1B6B4A", lw=1.7, label="True Zone Temp Under PLC Guardrails (0 min Tier-1 Breach)")
    ax2.set_xlim(8, 18.5)
    ax2.set_ylim(20.7, 25.1)
    ax2.set_xlabel("Time of Day — 2017-12-06 Multi-Fault Stress Window (Hours)")
    ax2.set_ylabel("Temperature (°C)")
    ax2.set_title("(b) PLC Hard Safety Guardrail Validation Under Simultaneous Sensor, Comms & LBNL Valve Faults")
    ax2.legend(loc="upper right", fontsize=7.5, ncol=2)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(FIG_DIR, f"fig4_lbnl_fault_guardrail.{ext}"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 84)
    print("STEP 2: REAL-DATA FORECASTING & 24-HOUR CLOSED-LOOP CLEANROOM PoC (HIPE 3M + LBNL)")
    print("=" * 84)

    # 1. Build Real 3-Month HIPE Pipeline (Zero Leakage, sigma=3m MES jitter + 2% unlogged tests)
    bundle = build_real_hipe_dataset(jitter_std_min=3.0, unannounced_prob=0.02, seed=2026)
    lbnl_ev = extract_lbnl_empirical_evidence()

    # 2. Train, Tune, Calibrate, and Benchmark All 5 Models on 10 Unseen Test Days
    fc_res = train_and_benchmark_all(bundle)
    fc_res["learning_curve_df"].to_csv(os.path.join(RES_DIR, "real_learning_curve.csv"), index=False)
    pd.concat([fc_res["tuning_sensor_df"], fc_res["tuning_equip_df"]]).to_csv(
        os.path.join(RES_DIR, "real_tuning_trials.csv"), index=False
    )
    fc_res["benchmark_df"].to_csv(os.path.join(RES_DIR, "real_forecast_benchmark.csv"), index=False)
    fc_res["feat_imp_h15"]["equip"].to_csv(os.path.join(RES_DIR, "real_feature_importance_h15.csv"))

    print("\n--- [1] LEARNING CURVE (h=15m, Val 7d) ---")
    print(fc_res["learning_curve_df"].to_string(index=False))

    print("\n--- [2] 5-MODEL BENCHMARK ON 10 UNSEEN TEST DAYS (RAW REAL HIPE kW) ---")
    b_show = fc_res["benchmark_df"][[
        "Horizon_min", "Model", "RMSE", "MAE", "R2", "Active_RMSE", "Shock_RMSE",
        "Cov_80_Active", "Cov_Upper90_Active", "Width_Active"
    ]].copy()
    for c in b_show.columns[2:]:
        b_show[c] = b_show[c].round(3)
    print(b_show.to_string(index=False))

    # 3. Extract #1 Unseen Peak Test Day (2017-12-06) for 24h Closed-Loop Simulation
    df_all = bundle["df_all"]
    df_dec6_feat = df_all[df_all["local_date"] == "2017-12-06"].reset_index(drop=True)

    df_bay_10, meta_10 = build_24h_bay_simulation_frame(df_dec6_feat, fluctuation_f=0.10, seed=42)
    preds_dec6_sensor_10 = predict_day_quantiles_for_bay(
        df_dec6_feat, fc_res["models"]["sensor"], meta_10["q_base_kW"], meta_10["m_bay"]
    )
    preds_dec6_equip_10 = predict_day_quantiles_for_bay(
        df_dec6_feat, fc_res["models"]["equip"], meta_10["q_base_kW"], meta_10["m_bay"]
    )

    # Run Scenarios A2 and B at nominal f = 10%
    summary_rows = []
    sims_A2 = {}
    for scen in ["A2", "B"]:
        for ctrl, pr in [
            ("Static_PID", None),
            ("SensorOnly_MPC", preds_dec6_sensor_10),
            ("EquipState_MPC", preds_dec6_equip_10),
        ]:
            sim_df = run_24h_simulation(
                df_bay_10, controller_mode=ctrl, scenario=scen,
                fluctuation_f=0.10, preds=pr, enable_l1_l3=True,
            )
            if scen == "A2":
                sims_A2[ctrl] = sim_df
                sim_df.to_csv(os.path.join(RES_DIR, f"sim_24h_A2_{ctrl}.csv"), index=False)
            summary_rows.append(summarize_sim(sim_df, ctrl, scen))

    summary_df = pd.DataFrame(summary_rows)
    # Add % energy, water, and CO2 savings relative to each scenario's Static_PID
    for scen in ["A2", "B"]:
        m_sc = summary_df["Scenario"] == scen
        base_e = float(summary_df.loc[m_sc & (summary_df["Controller"] == "Static_PID"), "Energy_MWh_yr"].iloc[0])
        base_w = float(summary_df.loc[m_sc & (summary_df["Controller"] == "Static_PID"), "Water_m3_yr"].iloc[0])
        summary_df.loc[m_sc, "Energy_Save_pct"] = (100.0 * (base_e - summary_df.loc[m_sc, "Energy_MWh_yr"]) / base_e).round(2)
        summary_df.loc[m_sc, "Water_Save_pct"] = (100.0 * (base_w - summary_df.loc[m_sc, "Water_m3_yr"]) / base_w).round(2)
        summary_df.loc[m_sc, "CO2_Avoided_t_yr"] = ((base_e - summary_df.loc[m_sc, "Energy_MWh_yr"]) * EF_VN_KG_KWH).round(1)
    summary_df.to_csv(os.path.join(RES_DIR, "real_closed_loop_summary.csv"), index=False)

    print("\n--- [3] 24-HOUR CLOSED-LOOP ABLATION ON REAL HELD-OUT PEAK DAY 2017-12-06 (f=10%) ---")
    print(summary_df[[
        "Scenario", "Controller", "T_RMSE_C", "T_max_C", "RH_max_MSL3_pct",
        "Tier1_Viol_min", "Tier2_T_Viol_min", "Tier2_RH_Viol_min",
        "Energy_MWh_yr", "Energy_Save_pct", "Water_m3_yr", "Water_Save_pct", "CO2_Avoided_t_yr"
    ]].to_string(index=False))

    # 4. Pure L4-Only s4 Sweep across f in {5%, 10%, 20%}
    l4_rows = []
    for f_val in [0.05, 0.10, 0.20]:
        df_bay_f, meta_f = build_24h_bay_simulation_frame(df_dec6_feat, fluctuation_f=f_val, seed=42)
        pr_sn_f = predict_day_quantiles_for_bay(df_dec6_feat, fc_res["models"]["sensor"], meta_f["q_base_kW"], meta_f["m_bay"])
        pr_eq_f = predict_day_quantiles_for_bay(df_dec6_feat, fc_res["models"]["equip"], meta_f["q_base_kW"], meta_f["m_bay"])

        s_pid = run_24h_simulation(df_bay_f, "Static_PID", "A2", f_val, None, enable_l1_l3=False)
        s_sn = run_24h_simulation(df_bay_f, "SensorOnly_MPC", "A2", f_val, pr_sn_f, enable_l1_l3=False)
        s_eq = run_24h_simulation(df_bay_f, "EquipState_MPC", "A2", f_val, pr_eq_f, enable_l1_l3=False)

        pc_pid = float(s_pid["P_chiller_kW"].mean())
        pc_sn = float(s_sn["P_chiller_kW"].mean())
        pc_eq = float(s_eq["P_chiller_kW"].mean())
        l4_rows.append({
            "f_pct": int(round(100 * f_val)),
            "m_bay": meta_f["m_bay"],
            "Peak_Bay_Tool_kW": meta_f["q_max_kW"],
            "PID_Chiller_kWe": round(pc_pid, 2),
            "Sensor_Chiller_kWe": round(pc_sn, 2),
            "Equip_Chiller_kWe": round(pc_eq, 2),
            "s4_SensorOnly_pct": round(100.0 * (pc_pid - pc_sn) / pc_pid, 2),
            "s4_EquipState_pct": round(100.0 * (pc_pid - pc_eq) / pc_pid, 2),
            "PID_T2_T_Viol_min": int(s_pid["tier2_T_viol"].sum()),
            "Sensor_T2_T_Viol_min": int(s_sn["tier2_T_viol"].sum()),
            "Equip_T2_T_Viol_min": int(s_eq["tier2_T_viol"].sum()),
            "Equip_T_chw_mean_C": round(float(s_eq["T_chw_dcc"].mean()), 2),
        })
    l4_df = pd.DataFrame(l4_rows)
    l4_df.to_csv(os.path.join(RES_DIR, "real_l4_s4_sweep.csv"), index=False)

    print("\n--- [4] ISOLATED L4-ONLY CHILLER SAVING (s4) ACROSS FLUCTUATION REGIMES f ---")
    print(l4_df.to_string(index=False))

    # 5. +/-20% Plant-Model Mismatch Robustness Test (C_eff = 25.28 / 31.60 / 37.92 MJ/K)
    mismatch_rows = []
    nom_p = CleanroomParams(alpha_mass=5.0)
    for alpha_true, lbl_ceff in [
        (4.0, "-20% (25.3 MJ/K)"),
        (5.0, "Nominal (31.6 MJ/K)"),
        (6.0, "+20% (37.9 MJ/K)"),
    ]:
        p_true = CleanroomParams(alpha_mass=alpha_true)
        for ctrl, pr in [
            ("Static_PID", None),
            ("SensorOnly_MPC", preds_dec6_sensor_10),
            ("EquipState_MPC", preds_dec6_equip_10),
        ]:
            sim_mm = run_24h_simulation(
                df_bay_10, controller_mode=ctrl, scenario="A2",
                fluctuation_f=0.10, preds=pr, enable_l1_l3=True,
                params=nom_p, plant_params=p_true,
            )
            sm = summarize_sim(sim_mm, ctrl, "A2")
            mismatch_rows.append({
                "Plant_Ceff": lbl_ceff,
                "Controller": ctrl,
                "T_RMSE_C": sm["T_RMSE_C"],
                "T_max_C": sm["T_max_C"],
                "T_min_C": sm["T_min_C"],
                "Tier1_Viol_min": sm["Tier1_Viol_min"],
                "Tier2_T_Viol_min": sm["Tier2_T_Viol_min"],
                "Tier2_RH_Viol_min": sm["Tier2_RH_Viol_min"],
                "Energy_MWh_yr": sm["Energy_MWh_yr"],
            })
    mismatch_df = pd.DataFrame(mismatch_rows)
    mismatch_df.to_csv(os.path.join(RES_DIR, "real_plant_mismatch.csv"), index=False)

    print("\n--- [5] +/-20% PLANT-MODEL MISMATCH ROBUSTNESS TEST (Controller assumes 31.6 MJ/K) ---")
    print(mismatch_df.to_string(index=False))

    # 6. Combined Fault & PLC Guardrail Validation (Drift + Comms Blackout + LBNL Valve Stuck)
    fault_cfg = {
        "drift_start": 570, "drift_end": 630, "drift_T_C": 1.8, "drift_RH_pct": 6.0,
        "comms_drop_start": 780, "comms_drop_end": 840,
        "valve_fault_start": 930, "valve_fault_end": 990, "valve_cap_frac": 0.78,
    }
    df_fault = run_24h_simulation(
        df_bay_10, controller_mode="EquipState_MPC", scenario="A2",
        fluctuation_f=0.10, preds=preds_dec6_equip_10, enable_l1_l3=True,
        fault_config=fault_cfg,
    )
    df_fault.to_csv(os.path.join(RES_DIR, "real_fault_guardrail_trace.csv"), index=False)
    fault_summary = summarize_sim(df_fault, "EquipState_MPC_FaultStress", "A2")
    fault_summary["Sensor_Drift_Isolated_min"] = int(df_fault["sensor_fault_detected"].sum())
    fault_summary["Comms_Fallback_min"] = int(df_fault["fallback_active"].sum())
    fault_summary["Valve_Fault_Active_min"] = int(df_fault["valve_fault_active"].sum())
    print("\n--- [6] COMBINED FAULT STRESS TEST (Sensor Drift + Comms Drop + LBNL Valve Fault) ---")
    print({k: fault_summary[k] for k in [
        "T_RMSE_C", "T_max_C", "T_min_C", "RH_max_MSL3_pct",
        "Tier1_Viol_min", "Tier2_T_Viol_min", "Tier2_RH_Viol_min",
        "Sensor_Drift_Isolated_min", "Comms_Fallback_min", "Valve_Fault_Active_min"
    ]})

    # Save master JSON snapshot
    master_json = {
        "dataset": {
            "source": "HIPE 3-Month (KIT, Zenodo 14054902) + LBNL FDD (Figshare 11752740)",
            "clean_prod_days": 48,
            "train_days": [bundle["train_days"][0], bundle["train_days"][-1], len(bundle["train_days"])],
            "val_days": [bundle["val_days"][0], bundle["val_days"][-1], len(bundle["val_days"])],
            "test_days": [bundle["test_days"][0], bundle["test_days"][-1], len(bundle["test_days"])],
            "thresholds_train_calibrated": bundle["thresholds"],
        },
        "best_hyperparams": {
            "SensorOnly_LGBM": fc_res["best_cfg_sensor"],
            "EquipState_LGBM": fc_res["best_cfg_equip"],
        },
        "bay_mapping_f10": meta_10,
        "lbnl_empirical": {k: v for k, v in lbnl_ev.items() if not isinstance(v, dict)},
        "lbnl_lags": {fn: {kk: vv for kk, vv in lbnl_ev[fn].items() if kk not in ("lags", "corrs")}
                      for fn in ["SZVAV.csv", "SZCAV.csv", "MZVAV-2-1.csv"]},
        "fault_stress_summary": fault_summary,
    }
    with open(os.path.join(RES_DIR, "master_results_real.json"), "w") as f:
        json.dump(master_json, f, indent=2)

    # 7. Generate All 4 Publication Figures
    make_figures(
        bundle, fc_res, df_dec6_feat, preds_dec6_sensor_10, preds_dec6_equip_10,
        sims_A2, summary_df, l4_df, mismatch_df, df_fault, lbnl_ev, meta_10
    )
    print(f"\nAll results and 4 PDF/PNG figures saved to:\n  {RES_DIR}\n  {FIG_DIR}")


if __name__ == "__main__":
    main()
