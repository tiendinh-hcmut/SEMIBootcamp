"""
Step 1: Real Dataset Preprocessing & Exploratory Data Analysis (EDA)
Processes:
  1. HIPE 1-Week Real Power-Electronics Packaging Plant Dataset (KIT, Zenodo 14054902)
     - Streams all 11 machine CSVs directly from hipe_1week.zip
     - Resamples 5-second smart-meter P_kW readings to exact 1-minute resolution
     - Saves clean 1-minute master CSV: datasets/hipe_1min_clean.csv
  2. LBNL FDD Real Experimental AHU Datasets (FLEXLAB SZVAV/SZCAV & ASHRAE RP-1312 MZVAV-2-1)
     - Empirically characterizes real valve-to-supply-air thermal lag (tau_dead + tau_coil)
     - Summarizes real fault scenarios available for guardrail testing
"""

import os
import sys
import tempfile
import zipfile
import warnings

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYDEPS_DIR = os.path.join(WORKSPACE_ROOT, ".pydeps")
if os.path.isdir(PYDEPS_DIR) and PYDEPS_DIR not in sys.path:
    sys.path.insert(0, PYDEPS_DIR)

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mpl_semi_eda"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASETS_DIR = os.path.dirname(os.path.abspath(__file__))
HIPE_ZIP = os.path.join(DATASETS_DIR, "hipe", "hipe_1week.zip")
LBNL_DIR = os.path.join(DATASETS_DIR, "lbnl_fdd")
OUT_HIPE_CSV = os.path.join(DATASETS_DIR, "hipe_1min_clean.csv")
OUT_EDA_PNG = os.path.join(DATASETS_DIR, "eda_real_datasets.png")


def process_hipe_zip():
    print("=" * 78)
    print("1. PROCESSING HIPE 1-WEEK REAL PACKAGING PLANT DATASET (KIT GERMANY)")
    print("=" * 78)

    # Full 1-minute UTC-aligned index (2017-10-23 00:00 to 2017-10-29 23:59 local CET/CEST)
    # Note: Oct 29, 2017 was EU DST switch (+02 -> +01), so use UTC or naive local timeline cleanly!
    series_dict = {}
    with zipfile.ZipFile(HIPE_ZIP) as zf:
        csv_files = sorted([f for f in zf.namelist() if f.endswith(".csv")])
        for fname in csv_files:
            machine = fname.split("_PhaseCount_")[0]
            print(f"  Streaming {machine:20s} ...", end="", flush=True)
            with zf.open(fname) as f:
                df_m = pd.read_csv(f, usecols=["SensorDateTime", "P_kW"])
            # Parse timezone-aware timestamps (some rows omit fractional seconds)
            dt = pd.to_datetime(df_m["SensorDateTime"], format="ISO8601", utc=True)
            s = pd.Series(df_m["P_kW"].astype(float).values, index=dt)
            s_1m = s.resample("1min").mean()
            series_dict[machine] = s_1m
            print(f" {len(df_m):7d} raw rows -> {len(s_1m):5d} 1-min rows | "
                  f"mean={s_1m.mean():5.2f} kW, p95={s_1m.quantile(0.95):5.2f} kW, max={s_1m.max():6.2f} kW")

    df_hipe = pd.DataFrame(series_dict)
    # Convert index to clean local timestamps (UTC+2 aligned start 2017-10-23 00:00)
    df_hipe.index = df_hipe.index + pd.Timedelta(hours=2)
    df_hipe.index = df_hipe.index.tz_localize(None)
    df_hipe.index.name = "timestamp"

    # Trim to exact 7 full days (10,080 minutes) and interpolate small telemetry gaps (<=5 min)
    t_start = pd.Timestamp("2017-10-23 00:00:00")
    full_idx = pd.date_range(t_start, periods=7 * 1440, freq="1min")
    df_hipe = df_hipe.reindex(full_idx)
    missing_before = df_hipe.isna().sum().to_dict()
    df_hipe = df_hipe.interpolate(limit=5).fillna(0.0)
    df_hipe.index.name = "timestamp"

    # Sub-metered sum of all 10 machines (excluding MainTerminal)
    machine_cols = [c for c in df_hipe.columns if c != "MainTerminal"]
    thermal_cols = ["SolderingOven", "VacuumOven", "HighTemperatureOven", "ChipPress"]
    mech_cols = ["ChipSaw", "PickAndPlaceUnit", "ScreenPrinter", "WashingMachine", "VacuumPump1", "VacuumPump2"]

    df_hipe["Submetered_10Tools_kW"] = df_hipe[machine_cols].sum(axis=1)
    df_hipe["Thermal_Ovens_Press_kW"] = df_hipe[thermal_cols].sum(axis=1)
    df_hipe["Mech_Subfab_Tools_kW"] = df_hipe[mech_cols].sum(axis=1)

    df_hipe.to_csv(OUT_HIPE_CSV)
    print(f"\nSaved clean 1-minute HIPE dataset to: {OUT_HIPE_CSV} (shape={df_hipe.shape})")
    print("Missing 1-min slots before interpolation:", {k: int(v) for k, v in missing_before.items() if v > 0})

    # Detailed Machine-Level EDA
    print("\n--- HIPE MACHINE-LEVEL SUMMARY (7 DAYS, 1-MIN ACTIVE POWER P_kW) ---")
    print(f"{'Machine':22s} | {'Active% (>0.2kW)':>16s} | {'Mean(all)':>9s} | {'Mean(act)':>9s} | {'P50(act)':>8s} | {'P95(act)':>8s} | {'Max kW':>7s}")
    print("-" * 92)
    for col in machine_cols + ["Submetered_10Tools_kW", "MainTerminal"]:
        s = df_hipe[col]
        act = s[s > 0.2]
        act_pct = 100.0 * len(act) / len(s)
        m_act = act.mean() if len(act) > 0 else 0.0
        p50_a = act.median() if len(act) > 0 else 0.0
        p95_a = act.quantile(0.95) if len(act) > 0 else 0.0
        print(f"{col:22s} | {act_pct:15.1f}% | {s.mean():9.2f} | {m_act:9.2f} | {p50_a:8.2f} | {p95_a:8.2f} | {s.max():7.2f}")

    # Day-by-Day Activity Summary (Mon Oct 23 .. Sun Oct 29)
    print("\n--- HIPE DAY-BY-DAY PRODUCTION PROFILE ---")
    df_hipe["date"] = df_hipe.index.strftime("%Y-%m-%d (%a)")
    for d_str, grp in df_hipe.groupby("date"):
        sub = grp["Submetered_10Tools_kW"]
        mt = grp["MainTerminal"]
        # Active work shift hours (06:00-22:00)
        shift = grp.between_time("06:00", "22:00")["Submetered_10Tools_kW"]
        cv_shift = (shift.std() / max(shift.mean(), 1e-6)) * 100.0
        print(f"  {d_str} | 10-Tool Mean={sub.mean():5.2f} kW, Max={sub.max():6.2f} kW | "
              f"Shift(06-22) Mean={shift.mean():5.2f} kW, Std={shift.std():5.2f} kW (CV={cv_shift:4.1f}%) | "
              f"MainTerm Mean={mt.mean():6.2f} kW")

    return df_hipe.drop(columns=["date"])


def analyze_lbnl():
    print("\n" + "=" * 78)
    print("2. ANALYZING LBNL FDD EXPERIMENTAL AHU DATASETS (FLEXLAB & ASHRAE RP-1312)")
    print("=" * 78)

    results = {}
    for fn in ["SZVAV.csv", "SZCAV.csv", "MZVAV-2-1.csv"]:
        df = pd.read_csv(os.path.join(LBNL_DIR, fn))
        clean = df[df["Fault Detection Ground Truth"] == 0].copy()
        fault = df[df["Fault Detection Ground Truth"] == 1].copy()

        sat_C = (clean["AHU: Supply Air Temperature"].astype(float) - 32.0) * 5.0 / 9.0
        rat_C = (clean["AHU: Return Air Temperature"].astype(float) - 32.0) * 5.0 / 9.0
        ccv = clean["AHU: Cooling Coil Valve Control Signal"].astype(float)
        occ = clean["Occupancy Mode Indicator"].astype(int)
        mask = (occ == 1) & (ccv > 0.02)

        dccv = ccv.diff().fillna(0).values
        dsat = -sat_C.diff().fillna(0).values
        lags = list(range(0, 16))
        corrs = []
        for lag in lags:
            if lag == 0:
                x, y = dccv[mask], dsat[mask]
            else:
                m = mask.values[lag:] & mask.values[:-lag]
                x, y = dccv[:-lag][m], dsat[lag:][m]
            corrs.append(float(np.corrcoef(x, y)[0, 1]) if len(x) > 50 and np.std(x) > 1e-6 else 0.0)

        results[fn] = {
            "clean_rows": len(clean),
            "fault_rows": len(fault),
            "lags": lags,
            "corrs": corrs,
            "peak_lag": lags[int(np.argmax(corrs))],
            "peak_corr": max(corrs),
            "df_clean": clean,
        }
        print(f"  {fn:15s}: clean={len(clean):5d} min, fault={len(fault):5d} min | "
              f"Peak CCV->SAT lag = {results[fn]['peak_lag']} min (r={results[fn]['peak_corr']:.3f})")
    return results


def plot_eda(df_hipe, lbnl_res):
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.2))

    # Panel (a): 7-Day Real HIPE Plant MainTerminal vs 10 Sub-metered Packaging Tools
    ax = axes[0, 0]
    t_days = np.arange(len(df_hipe)) / 1440.0
    ax.plot(t_days, df_hipe["MainTerminal"], color="#1F2937", lw=0.95, alpha=0.85,
            label="MainTerminal — Whole Plant (max 53.6 kW)")
    ax.fill_between(t_days, 0, df_hipe["Submetered_10Tools_kW"], color="#1B6B4A", alpha=0.55,
                    label="10 Sub-metered Tools Sum (max 12.6 kW)")
    ax.axvspan(5, 7, color="#E5E7EB", alpha=0.45, label="Weekend Shutdown (Sat–Sun)")
    ax.set_xlim(0, 7)
    ax.set_xticks(range(8))
    ax.set_xticklabels(["Mon\n23", "Tue\n24", "Wed\n25", "Thu\n26", "Fri\n27", "Sat\n28", "Sun\n29", "Mon\n30"], fontsize=8.5)
    ax.set_ylabel("Active Power (kW)")
    ax.set_title("(a) HIPE 1-Week Real Packaging Plant Telemetry (1-Min Mean)")
    ax.grid(True, alpha=0.45)
    ax.legend(loc="upper right", fontsize=8.0)

    # Panel (b): Richest Active Production Day Zoom — Thursday Oct 26 (07:00-19:30)
    ax2 = axes[0, 1]
    day4 = df_hipe.loc["2017-10-26 07:00:00":"2017-10-26 19:30:00"]
    hrs = np.arange(len(day4)) / 60.0 + 7.0
    for col, col_hex in [
        ("SolderingOven", "#C0392B"),
        ("HighTemperatureOven", "#D9822B"),
        ("VacuumPump1", "#1B6B4A"),
        ("VacuumPump2", "#0D9488"),
        ("ChipSaw", "#7C3AED"),
        ("ScreenPrinter", "#2563EB"),
        ("PickAndPlaceUnit", "#6B7280"),
    ]:
        ax2.plot(hrs, day4[col], lw=1.15, label=f"{col} ({day4[col].max():.1f} kW)", color=col_hex)
    ax2.set_xlim(7, 19.5)
    ax2.set_xlabel("Workday Time — Thursday 2017-10-26 (Hours)")
    ax2.set_ylabel("Active Power (kW)")
    ax2.set_title("(b) Real Tool-Level Waveforms on Peak Day (Thu Oct 26: 7 Active Tools)")
    ax2.grid(True, alpha=0.45)
    ax2.legend(loc="upper right", fontsize=7.5, ncol=2)

    # Panel (c): Real LBNL Experimental AHU Cooling-Coil Valve-to-SAT Cross-Correlation Lag
    ax3 = axes[1, 0]
    lags = lbnl_res["SZVAV.csv"]["lags"]
    ax3.plot(lags, lbnl_res["SZVAV.csv"]["corrs"], "o-", color="#1B6B4A", lw=1.8,
             label="LBNL FLEXLAB SZVAV (Peak lag = 6 min, r=0.47)")
    ax3.plot(lags, lbnl_res["SZCAV.csv"]["corrs"], "^:", color="#2563EB", lw=1.4,
             label="LBNL FLEXLAB SZCAV (Peak lag = 3 min, r=0.54)")
    ax3.plot(lags, lbnl_res["MZVAV-2-1.csv"]["corrs"], "s--", color="#D9822B", lw=1.5,
             label="LBNL/ASHRAE MZVAV-2-1 (Peak lag = 3 min, r=0.50)")
    ax3.axvspan(8, 12, color="#A7D7C5", alpha=0.35, label="Report Table 3.2 Lag Window (8–12 min)")
    ax3.axhline(0, color="#6B7280", lw=0.8)
    ax3.set_xlim(0, 15)
    ax3.set_xticks(range(0, 16))
    ax3.set_xlabel("Lag Between Valve Change d(CCV) and Cooling Rate -d(SAT) (min)")
    ax3.set_ylabel("Cross-Correlation r")
    ax3.set_title("(c) Empirical Cooling-Coil Transport & Thermal Lag (Real LBNL AHUs)")
    ax3.grid(True, alpha=0.45)
    ax3.legend(loc="lower right", fontsize=7.6)

    # Panel (d): Real LBNL FLEXLAB SZVAV Fault Signatures vs Unfaulted Baseline
    ax4 = axes[1, 1]
    df_sz = pd.read_csv(os.path.join(LBNL_DIR, "SZVAV.csv"))
    df_sz["dt"] = pd.to_datetime(df_sz["Datetime"])
    df_sz["date"] = df_sz["dt"].dt.strftime("%Y-%m-%d")
    df_sz["SAT_C"] = (df_sz["AHU: Supply Air Temperature"] - 32.0) * 5.0 / 9.0
    for d_str, lbl, col_hex, ls in [
        ("2017-09-20", "Unfaulted Baseline (2017-09-20)", "#1B6B4A", "-"),
        ("2017-09-22", "Fault: Cooling Valve Stuck 100% Open (Overcooling)", "#2563EB", "--"),
        ("2017-09-12", "Fault: Heating Valve Leaking 40% (Reheat Fight)", "#C0392B", "-."),
    ]:
        sub = df_sz[df_sz["date"] == d_str]
        h = (sub["dt"].dt.hour + sub["dt"].dt.minute / 60.0).values
        ax4.plot(h, sub["SAT_C"].values, color=col_hex, ls=ls, lw=1.4, label=lbl)
    ax4.axvspan(6, 18, color="#FEF3C7", alpha=0.30, label="Occupied Shift (06:00–18:00)")
    ax4.set_xlim(0, 24)
    ax4.set_xticks(range(0, 25, 4))
    ax4.set_xlabel("Time of Day (Hours)")
    ax4.set_ylabel("AHU Supply Air Temperature SAT (°C)")
    ax4.set_title("(d) Real Physical Fault Signatures in LBNL FLEXLAB SZVAV")
    ax4.grid(True, alpha=0.45)
    ax4.legend(loc="upper left", fontsize=7.5)

    plt.tight_layout()
    fig.savefig(OUT_EDA_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved EDA overview plot to: {OUT_EDA_PNG}")


if __name__ == "__main__":
    df_hipe = process_hipe_zip()
    lbnl_res = analyze_lbnl()
    plot_eda(df_hipe, lbnl_res)
