"""
Step 1 (Full 3-Month Expansion): Stream-Preprocess & EDA for HIPE 3-Month Dataset
Source: KIT Germany, Zenodo Record 14054902 (2017-10-01 to 2018-01-01)
  - Streams all 11 machine CSVs directly from hipe_3months.zip in chunks
    (zero disk unpacking of the ~30.6 GB raw archive, low RAM footprint)
  - Resamples ~5-second active power (P_kW) onto an exact 1-minute UTC grid
  - Preserves Europe/Berlin local time metadata cleanly across the Oct 29 DST switch
  - Saves clean 1-minute master dataset: datasets/hipe_3m_1min_clean.csv
  - Produces 3-month production/drift EDA & figure: datasets/eda_hipe_3months.png
"""

import os
import sys
import tempfile
import time
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
HIPE_3M_ZIP = os.path.join(DATASETS_DIR, "hipe", "hipe_3months.zip")
OUT_CSV = os.path.join(DATASETS_DIR, "hipe_3m_1min_clean.csv")
OUT_DAILY_CSV = os.path.join(DATASETS_DIR, "hipe_3m_daily_summary.csv")
OUT_PNG = os.path.join(DATASETS_DIR, "eda_hipe_3months.png")


def stream_machine_1min(zf: zipfile.ZipFile, fname: str, chunksize: int = 250_000) -> pd.Series:
    """Stream one machine CSV from zip, accumulating 1-min sum and count in UTC."""
    sum_parts = []
    cnt_parts = []
    total_rows = 0

    with zf.open(fname) as f:
        reader = pd.read_csv(f, usecols=["SensorDateTime", "P_kW"], chunksize=chunksize)
        for chunk in reader:
            total_rows += len(chunk)
            dt = pd.to_datetime(chunk["SensorDateTime"], format="ISO8601", utc=True)
            vals = pd.to_numeric(chunk["P_kW"], errors="coerce")
            valid = vals.notna()
            if not valid.any():
                continue
            dt_floor = dt[valid].dt.floor("1min")
            v = vals[valid]
            grp = v.groupby(dt_floor)
            sum_parts.append(grp.sum())
            cnt_parts.append(grp.count())

    if not sum_parts:
        return pd.Series(dtype=float), total_rows

    total_sum = pd.concat(sum_parts).groupby(level=0).sum()
    total_cnt = pd.concat(cnt_parts).groupby(level=0).sum()
    mean_1m = (total_sum / total_cnt).sort_index()
    return mean_1m, total_rows


def process_hipe_3months():
    if not os.path.isfile(HIPE_3M_ZIP):
        raise FileNotFoundError(f"Missing {HIPE_3M_ZIP}")

    t0_all = time.time()
    print("=" * 84)
    print("PROCESSING FULL 3-MONTH HIPE PACKAGING PLANT DATASET (2017-10-01 .. 2018-01-01)")
    print("=" * 84)

    series_dict = {}
    with zipfile.ZipFile(HIPE_3M_ZIP) as zf:
        csv_files = sorted([f for f in zf.namelist() if f.endswith(".csv")])
        print(f"Found {len(csv_files)} CSV files in archive.")
        for fname in csv_files:
            base = os.path.basename(fname)
            machine = base.split("_PhaseCount_")[0]
            t0 = time.time()
            s_1m, nrows = stream_machine_1min(zf, fname)
            series_dict[machine] = s_1m
            elapsed = time.time() - t0
            act = s_1m[s_1m > 0.2]
            print(
                f"  {machine:21s} | {nrows:9d} raw rows -> {len(s_1m):6d} 1-min | "
                f"active={100*len(act)/max(len(s_1m),1):5.1f}% | max={s_1m.max():6.2f} kW ({elapsed:4.1f}s)"
            )

    df_utc = pd.DataFrame(series_dict).sort_index()

    # Exact 92-day UTC range covering 2017-10-01 00:00 Europe/Berlin to 2017-12-31 23:59 Europe/Berlin
    start_utc = pd.Timestamp("2017-10-01 00:00:00", tz="Europe/Berlin").tz_convert("UTC")
    end_utc = pd.Timestamp("2017-12-31 23:59:00", tz="Europe/Berlin").tz_convert("UTC")
    full_utc = pd.date_range(start_utc, end_utc, freq="1min", tz="UTC")
    df = df_utc.reindex(full_utc)

    missing_before = {c: int(df[c].isna().sum()) for c in df.columns}
    # Interpolate short telemetry gaps (<= 10 min); record missingness before filling larger gaps
    df["any_missing_raw"] = df.isna().any(axis=1).astype(int)
    df = df.interpolate(limit=10)
    # Remaining large sensor outages (if any) marked explicitly
    df["large_gap_flag"] = df.drop(columns=["any_missing_raw"]).isna().any(axis=1).astype(int)
    df = df.fillna(0.0)
    # Clip tiny negative meter noise (-0.01 kW) cleanly to 0 for physical power
    machine_cols = [c for c in series_dict.keys() if c != "MainTerminal"]
    for c in machine_cols + ["MainTerminal"]:
        df[c] = df[c].clip(lower=0.0)

    # Local Europe/Berlin time metadata (handles Oct 29 DST cleanly!)
    local_dt = df.index.tz_convert("Europe/Berlin")
    df["local_date"] = local_dt.strftime("%Y-%m-%d")
    df["local_hour"] = local_dt.hour + local_dt.minute / 60.0
    df["day_of_week"] = local_dt.dayofweek  # 0=Mon .. 6=Sun
    df["month"] = local_dt.month

    thermal_cols = [c for c in ["SolderingOven", "VacuumOven", "HighTemperatureOven", "ChipPress"] if c in df.columns]
    mech_cols = [c for c in machine_cols if c not in thermal_cols]
    df["Submetered_10Tools_kW"] = df[machine_cols].sum(axis=1)
    df["Thermal_Tools_kW"] = df[thermal_cols].sum(axis=1)
    df["Mech_Tools_kW"] = df[mech_cols].sum(axis=1)
    df.index.name = "timestamp_utc"

    df.to_csv(OUT_CSV)
    print(f"\nSaved clean 3-month 1-min dataset: {OUT_CSV} (shape={df.shape}, total {time.time()-t0_all:.1f}s)")
    print("Missing 1-min slots before interpolation (top columns):",
          {k: v for k, v in sorted(missing_before.items(), key=lambda x: -x[1])[:6]})
    print(f"Rows with large telemetry gap (>10 min): {int(df['large_gap_flag'].sum())} / {len(df)}")

    # Machine-level summary across 3 months
    print("\n--- 3-MONTH MACHINE-LEVEL SUMMARY (1-MIN ACTIVE POWER P_kW) ---")
    print(f"{'Machine':22s} | {'Active%(>0.2kW)':>15s} | {'Active Hrs':>10s} | {'Mean(act)':>9s} | {'P95(act)':>8s} | {'Max kW':>7s}")
    print("-" * 88)
    for col in machine_cols + ["Submetered_10Tools_kW", "MainTerminal"]:
        s = df[col]
        act = s[s > 0.2]
        print(
            f"{col:22s} | {100*len(act)/len(s):14.1f}% | {len(act)/60.0:10.1f} | "
            f"{(act.mean() if len(act) else 0):9.2f} | {(act.quantile(0.95) if len(act) else 0):8.2f} | {s.max():7.2f}"
        )

    # Daily production summary (active workdays vs weekends/holidays)
    daily_rows = []
    for d_str, g in df.groupby("local_date"):
        sub = g["Submetered_10Tools_kW"]
        mt = g["MainTerminal"]
        n_active_tools = sum((g[c] > 0.2).sum() >= 15 for c in machine_cols)
        active_min = int((sub > 0.5).sum())
        gap_min = int(g["large_gap_flag"].sum())
        daily_rows.append({
            "date": d_str,
            "dow": int(g["day_of_week"].iloc[0]),
            "month": int(g["month"].iloc[0]),
            "n_rows": len(g),
            "gap_min": gap_min,
            "n_active_tools": n_active_tools,
            "sub_active_hrs": round(active_min / 60.0, 2),
            "sub_mean_kW": round(float(sub.mean()), 3),
            "sub_max_kW": round(float(sub.max()), 3),
            "sub_energy_kWh": round(float(sub.sum() / 60.0), 2),
            "mt_mean_kW": round(float(mt.mean()), 2),
            "mt_max_kW": round(float(mt.max()), 2),
            "mt_energy_kWh": round(float(mt.sum() / 60.0), 1),
        })
    df_daily = pd.DataFrame(daily_rows)
    df_daily["is_full_prod_day"] = (
        (df_daily["dow"] < 5)
        & (df_daily["gap_min"] < 30)
        & (df_daily["sub_active_hrs"] >= 2.0)
        & (df_daily["n_active_tools"] >= 3)
    ).astype(int)
    df_daily.to_csv(OUT_DAILY_CSV, index=False)

    print("\n--- MONTH-BY-MONTH PRODUCTION & DRIFT CHECK ---")
    for m, gm in df_daily.groupby("month"):
        prod = gm[gm["is_full_prod_day"] == 1]
        print(
            f"  2017-{m:02d} | Total days={len(gm):2d}, Full prod days={len(prod):2d}, "
            f"Clean gap-free days={(gm['gap_min']==0).sum():2d} | "
            f"Prod-day Sub10 mean={prod['sub_mean_kW'].mean():.2f} kW (max={prod['sub_max_kW'].max():.1f} kW), "
            f"MainTerm mean={prod['mt_mean_kW'].mean():.1f} kW (max={prod['mt_max_kW'].max():.1f} kW)"
        )

    print(f"\nTotal clean full production days (dow<5, >=3 active tools, >=2h, <30m gap): "
          f"{int(df_daily['is_full_prod_day'].sum())} / {len(df_daily)} calendar days")

    # Plot 3-Month Overview
    fig, axes = plt.subplots(2, 1, figsize=(13.0, 7.6))
    x = np.arange(len(df_daily))
    colors = ["#1B6B4A" if f == 1 else "#D1D5DB" for f in df_daily["is_full_prod_day"]]

    ax = axes[0]
    ax.bar(x, df_daily["sub_energy_kWh"], color=colors, width=0.82)
    ax.set_ylabel("10 Sub-metered Tools Daily Energy (kWh/day)")
    ax.set_title("(a) HIPE 3-Month Daily Sub-metered Packaging Energy (Green = Active Production Day, Gray = Weekend/Holiday/Low-Activity)")
    tick_idx = list(range(0, len(df_daily), 7))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([df_daily["date"].iloc[i][5:] for i in tick_idx], fontsize=8.5)
    ax.grid(True, axis="y", alpha=0.45)

    ax2 = axes[1]
    ax2.plot(x, df_daily["mt_max_kW"], color="#1F2937", lw=1.3, label="MainTerminal Daily Peak (kW)")
    ax2.plot(x, df_daily["mt_mean_kW"], color="#2563EB", lw=1.3, label="MainTerminal Daily Mean (kW)")
    ax2.bar(x, df_daily["sub_max_kW"], color="#D9822B", alpha=0.55, width=0.65, label="10 Sub-metered Tools Daily Peak (kW)")
    ax2.set_ylabel("Power (kW)")
    ax2.set_xlabel("Date (2017 MM-DD)")
    ax2.set_title("(b) Whole-Plant MainTerminal vs. Sub-metered Peak Power Across Oct–Dec 2017 (Concept-Drift & Holiday Check)")
    ax2.set_xticks(tick_idx)
    ax2.set_xticklabels([df_daily["date"].iloc[i][5:] for i in tick_idx], fontsize=8.5)
    ax2.grid(True, alpha=0.45)
    ax2.legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved 3-month EDA plot: {OUT_PNG}")


if __name__ == "__main__":
    process_hipe_3months()
