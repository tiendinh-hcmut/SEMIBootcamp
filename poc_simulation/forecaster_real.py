"""
Real-Data Multi-Horizon Thermal Load Forecaster & Fair Multi-Model Benchmark
(Section 2.3 — Grounded 100% on 3-Month HIPE Real Packaging Plant Data)

Compares 5 models on the exact same strict chronological Train(31d) / Val(7d) / Test(10d) split:
  1. Persistence          : y_hat(t+h) = y(t)
  2. Ridge (Sensor-Only)  : Regularized linear regression on causal aggregate telemetry
  3. Ridge (Equip-State)  : Regularized linear regression on SEMI E10/GEM + S23 + noisy MES
  4. LightGBM (Sensor)    : Tuned GBDT + Quantiles (P10/P50/P90) + Split Conformal Calibration
  5. LightGBM (EquipState): Tuned GBDT + Quantiles (P10/P50/P90) + Split Conformal Calibration

Includes:
  - Empirical Learning Curve (7 vs 15 vs 23 vs 31 Train days on Val RMSE)
  - Structured Validation Hyperparameter Tuning (never touching Held-Out Test)
  - Active-Window Split Conformalized Quantile Regression (CQR)
  - Granular reporting: Overall Test, Active-Shift Test, and Thermal-Shock (|dP|>2 kW) Test
"""

import os
import sys
import warnings

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYDEPS_DIR = os.path.join(WORKSPACE_ROOT, ".pydeps")
if os.path.isdir(PYDEPS_DIR) and PYDEPS_DIR not in sys.path:
    sys.path.insert(0, PYDEPS_DIR)

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from data_pipeline_real import HORIZONS, build_real_hipe_dataset


def pinball_loss(y_true, y_pred, q):
    e = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.mean(np.maximum(q * e, (q - 1.0) * e)))


def compute_metrics(y_true, y_pred, y_now=None, p10=None, p90=None):
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2)))
    mae = float(np.mean(np.abs(y_t - y_p)))
    ss_res = float(np.sum((y_t - y_p) ** 2))
    ss_tot = float(np.sum((y_t - np.mean(y_t)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-9)

    res = {"RMSE": rmse, "MAE": mae, "R2": r2}

    # Active-shift subset (when real tools are active > 0.5 kW either now or at t+h)
    act_mask = (y_t > 0.5) if y_now is None else ((y_t > 0.5) | (np.asarray(y_now) > 0.5))
    if np.sum(act_mask) > 20:
        res["Active_RMSE"] = float(np.sqrt(np.mean((y_t[act_mask] - y_p[act_mask]) ** 2)))
        res["Active_MAE"] = float(np.mean(np.abs(y_t[act_mask] - y_p[act_mask])))
    else:
        res["Active_RMSE"] = rmse
        res["Active_MAE"] = mae

    # Thermal-shock subset: large real ramps |y(t+h) - y(t)| > 2.0 kW
    if y_now is not None:
        shock_mask = np.abs(y_t - np.asarray(y_now)) > 2.0
        if np.sum(shock_mask) > 10:
            res["Shock_RMSE"] = float(np.sqrt(np.mean((y_t[shock_mask] - y_p[shock_mask]) ** 2)))
            res["Shock_MAE"] = float(np.mean(np.abs(y_t[shock_mask] - y_p[shock_mask])))
            res["Shock_n"] = int(np.sum(shock_mask))
        else:
            res["Shock_RMSE"] = np.nan
            res["Shock_MAE"] = np.nan
            res["Shock_n"] = 0

    if p10 is not None and p90 is not None:
        p10a = np.asarray(p10, dtype=float)
        p90a = np.asarray(p90, dtype=float)
        res["Pinball_P50"] = pinball_loss(y_t, y_p, 0.5)
        res["Pinball_P90"] = pinball_loss(y_t, p90a, 0.9)
        res["Cov_80_All"] = float(np.mean((y_t >= p10a) & (y_t <= p90a)))
        res["Cov_Upper90_All"] = float(np.mean(y_t <= p90a))
        res["Width_All"] = float(np.mean(p90a - p10a))
        if np.sum(act_mask) > 20:
            res["Cov_80_Active"] = float(np.mean((y_t[act_mask] >= p10a[act_mask]) & (y_t[act_mask] <= p90a[act_mask])))
            res["Cov_Upper90_Active"] = float(np.mean(y_t[act_mask] <= p90a[act_mask]))
            res["Width_Active"] = float(np.mean(p90a[act_mask] - p10a[act_mask]))
    return res


def _fit_lgbm_model(X_tr, y_tr, X_va, y_va, objective, alpha=None, cfg=None):
    cfg = cfg or {}
    params = {
        "objective": objective,
        "n_estimators": cfg.get("n_estimators", 700),
        "learning_rate": cfg.get("learning_rate", 0.04),
        "num_leaves": cfg.get("num_leaves", 31),
        "min_child_samples": cfg.get("min_child_samples", 40),
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": cfg.get("colsample_bytree", 0.8),
        "reg_alpha": cfg.get("reg_alpha", 0.5),
        "reg_lambda": cfg.get("reg_lambda", 3.0),
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    if alpha is not None:
        params["alpha"] = alpha
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )
    return model


def run_learning_curve(bundle: dict, horizon: int = 15) -> pd.DataFrame:
    """
    Answer empirically: Does training on ALL 31 active production days beat smaller
    recent windows (7, 15, 23 days) when evaluated on the exact 7-day Validation set?
    """
    df_all = bundle["df_all"]
    train_days = bundle["train_days"]
    feat_cols = bundle["equip_cols_by_h"][horizon]
    target_col = f"y_h{horizon}"

    df_va = df_all[(df_all["split"] == "val") & (df_all["valid_target"] == 1)]
    X_va = df_va[feat_cols].values
    y_va = df_va[target_col].values
    y_va_now = df_va["P_10tools_now"].values

    rows = []
    for n_days in [7, 15, 23, len(train_days)]:
        sel_days = train_days[-n_days:]
        df_tr = df_all[(df_all["local_date"].isin(sel_days)) & (df_all["valid_target"] == 1)]
        X_tr = df_tr[feat_cols].values
        y_tr = df_tr[target_col].values

        # Check how many ChipPress peak days (>15 kW) are inside sel_days
        cp_days = sum(df_tr.groupby("local_date")["s23_P_ChipPress_now"].max() > 15.0)

        m = _fit_lgbm_model(X_tr, y_tr, X_va, y_va, objective="regression")
        pred = np.clip(m.predict(X_va), 0.0, None)
        met = compute_metrics(y_va, pred, y_now=y_va_now)
        rows.append({
            "train_days": n_days,
            "date_range": f"{sel_days[0]}..{sel_days[-1]}",
            "train_rows": len(df_tr),
            "chippress_peak_days": int(cp_days),
            "val_RMSE": round(met["RMSE"], 4),
            "val_Active_RMSE": round(met["Active_RMSE"], 4),
            "val_Shock_RMSE": round(met["Shock_RMSE"], 4),
            "val_R2": round(met["R2"], 4),
        })
    return pd.DataFrame(rows)


def tune_lgbm_on_validation(bundle: dict, mode: str, horizon: int = 15) -> tuple[dict, pd.DataFrame]:
    """Structured hyperparameter grid search strictly on Validation set (zero Test leakage)."""
    df_all = bundle["df_all"]
    feat_cols = bundle["sensor_cols"] if mode == "sensor" else bundle["equip_cols_by_h"][horizon]
    target_col = f"y_h{horizon}"

    df_tr = df_all[(df_all["split"] == "train") & (df_all["valid_target"] == 1)]
    df_va = df_all[(df_all["split"] == "val") & (df_all["valid_target"] == 1)]

    X_tr, y_tr = df_tr[feat_cols].values, df_tr[target_col].values
    X_va, y_va = df_va[feat_cols].values, df_va[target_col].values
    y_va_now = df_va["P_10tools_now"].values

    grid = [
        {"num_leaves": 15, "min_child_samples": 60, "colsample_bytree": 0.7, "reg_lambda": 5.0, "learning_rate": 0.04},
        {"num_leaves": 31, "min_child_samples": 40, "colsample_bytree": 0.8, "reg_lambda": 3.0, "learning_rate": 0.04},
        {"num_leaves": 31, "min_child_samples": 80, "colsample_bytree": 0.7, "reg_lambda": 5.0, "learning_rate": 0.03},
        {"num_leaves": 63, "min_child_samples": 30, "colsample_bytree": 0.8, "reg_lambda": 2.0, "learning_rate": 0.04},
        {"num_leaves": 63, "min_child_samples": 80, "colsample_bytree": 0.7, "reg_lambda": 8.0, "learning_rate": 0.03},
    ]

    trials = []
    best_rmse = 1e9
    best_cfg = grid[0]
    for idx, cfg in enumerate(grid):
        m = _fit_lgbm_model(X_tr, y_tr, X_va, y_va, objective="regression", cfg=cfg)
        pred = np.clip(m.predict(X_va), 0.0, None)
        met = compute_metrics(y_va, pred, y_now=y_va_now)
        trials.append({
            "trial": idx,
            "mode": mode,
            **cfg,
            "best_iter": int(m.best_iteration_),
            "val_RMSE": round(met["RMSE"], 4),
            "val_Active_RMSE": round(met["Active_RMSE"], 4),
            "val_R2": round(met["R2"], 4),
        })
        if met["RMSE"] < best_rmse:
            best_rmse = met["RMSE"]
            best_cfg = cfg

    return best_cfg, pd.DataFrame(trials)


def train_and_benchmark_all(bundle: dict) -> dict:
    """
    Train all 5 benchmark models across horizons h in {5, 10, 15, 30, 60} min,
    calibrate Split Conformal P10/P90 intervals on active Validation rows,
    and evaluate strictly on the 10 unseen Held-Out Test days.
    """
    df_all = bundle["df_all"]
    df_tr = df_all[(df_all["split"] == "train") & (df_all["valid_target"] == 1)]
    df_va = df_all[(df_all["split"] == "val") & (df_all["valid_target"] == 1)]
    df_te = df_all[(df_all["split"] == "test") & (df_all["valid_target"] == 1)]

    # 1. Empirical Learning Curve on Val
    lc_df = run_learning_curve(bundle, horizon=15)

    # 2. Fair Hyperparameter Tuning on Val for BOTH Sensor-Only and Equip-State LGBM
    best_cfg_sensor, trials_sensor = tune_lgbm_on_validation(bundle, mode="sensor", horizon=15)
    best_cfg_equip, trials_equip = tune_lgbm_on_validation(bundle, mode="equip", horizon=15)

    # 3. Train & Evaluate all 5 models across all horizons
    bench_rows = []
    models_store = {"sensor": {}, "equip": {}}
    test_preds_store = {"sensor": {}, "equip": {}}
    feat_imp_h15 = {}

    for h in HORIZONS:
        tcol = f"y_h{h}"
        y_tr = df_tr[tcol].values
        y_va = df_va[tcol].values
        y_te = df_te[tcol].values
        y_te_now = df_te["P_10tools_now"].values
        y_va_now = df_va["P_10tools_now"].values

        # --- Model 1: Persistence ---
        m_pers = compute_metrics(y_te, y_te_now, y_now=y_te_now)
        bench_rows.append({"Horizon_min": h, "Model": "Persistence", **m_pers})

        # --- Model 2 & 3: Ridge Regression (Sensor-Only & Equip-State) ---
        for r_name, fcols in [("Ridge_SensorOnly", bundle["sensor_cols"]), ("Ridge_EquipState", bundle["equip_cols_by_h"][h])]:
            pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 15)))
            pipe.fit(df_tr[fcols].values, y_tr)
            pred_te = np.clip(pipe.predict(df_te[fcols].values), 0.0, None)
            met = compute_metrics(y_te, pred_te, y_now=y_te_now)
            bench_rows.append({"Horizon_min": h, "Model": r_name, **met})

        # --- Model 4 & 5: Tuned LightGBM + Quantiles + Active-Window Split Conformal ---
        for mode_key, label, fcols, cfg in [
            ("sensor", "LightGBM_SensorOnly", bundle["sensor_cols"], best_cfg_sensor),
            ("equip", "LightGBM_EquipState", bundle["equip_cols_by_h"][h], best_cfg_equip),
        ]:
            X_tr = df_tr[fcols].values
            X_va = df_va[fcols].values
            X_te = df_te[fcols].values

            m_mean = _fit_lgbm_model(X_tr, y_tr, X_va, y_va, objective="regression", cfg=cfg)
            m_p10 = _fit_lgbm_model(X_tr, y_tr, X_va, y_va, objective="quantile", alpha=0.10, cfg=cfg)
            m_p50 = _fit_lgbm_model(X_tr, y_tr, X_va, y_va, objective="quantile", alpha=0.50, cfg=cfg)
            m_p90 = _fit_lgbm_model(X_tr, y_tr, X_va, y_va, objective="quantile", alpha=0.90, cfg=cfg)

            # Raw quantile predictions on Val
            va_p10 = np.clip(m_p10.predict(X_va), 0.0, None)
            va_p50 = np.clip(m_p50.predict(X_va), 0.0, None)
            va_p90 = np.clip(m_p90.predict(X_va), 0.0, None)
            va_p10 = np.minimum(va_p10, va_p50)
            va_p90 = np.maximum(va_p90, va_p50)

            # Active-window Split Conformal Calibration (Romano et al. 2019 CQR):
            # Calibrate specifically on active/transition Val rows so midnight zeros don't shrink
            # safety margins needed during daytime production shifts!
            va_act = (y_va > 0.5) | (y_va_now > 0.5)
            # One-sided upper conformity score s_up = y - p90 (target 90% active upper coverage)
            s_up = y_va[va_act] - va_p90[va_act]
            n_c = len(s_up)
            q_level_up = min(1.0, np.ceil((n_c + 1) * 0.90) / n_c)
            delta_up = float(max(0.0, np.quantile(s_up, q_level_up)))

            # Lower conformity score s_lo = p10 - y (target 90% lower tail -> >=80% central coverage)
            s_lo = va_p10[va_act] - y_va[va_act]
            delta_lo = float(max(0.0, np.quantile(s_lo, q_level_up)))

            # Predict on Held-Out Test
            te_mean = np.clip(m_mean.predict(X_te), 0.0, None)
            te_p10_raw = np.clip(m_p10.predict(X_te), 0.0, None)
            te_p50_raw = np.clip(m_p50.predict(X_te), 0.0, None)
            te_p90_raw = np.clip(m_p90.predict(X_te), 0.0, None)

            # Apply conformal shift ONLY where model/activity is non-trivial (>0.15 kW) so idle night stays crisp
            te_active_pred = (te_p90_raw > 0.15) | (y_te_now > 0.15)
            te_p10 = np.clip(np.minimum(te_p10_raw, te_p50_raw) - np.where(te_active_pred, delta_lo, 0.0), 0.0, None)
            te_p50 = np.maximum(te_p50_raw, te_p10)
            te_p90 = np.maximum(te_p90_raw, te_p50) + np.where(te_active_pred, delta_up, 0.0)

            # Point forecast uses L2 regression mean (optimal for RMSE) while quantiles provide P10/P50/P90
            met = compute_metrics(y_te, te_mean, y_now=y_te_now, p10=te_p10, p90=te_p90)
            met["Conformal_delta_up"] = round(delta_up, 3)
            met["Conformal_delta_lo"] = round(delta_lo, 3)
            bench_rows.append({"Horizon_min": h, "Model": label, **met})

            models_store[mode_key][h] = {
                "mean": m_mean,
                "p10": m_p10,
                "p50": m_p50,
                "p90": m_p90,
                "delta_up": delta_up,
                "delta_lo": delta_lo,
                "feat_cols": fcols,
            }
            test_preds_store[mode_key][h] = {
                "mean": te_mean,
                "p10": te_p10,
                "p50": te_p50,
                "p90": te_p90,
            }

            if h == 15:
                imp = pd.Series(m_mean.feature_importances_, index=fcols).sort_values(ascending=False)
                feat_imp_h15[mode_key] = imp.head(15)

    return {
        "benchmark_df": pd.DataFrame(bench_rows),
        "learning_curve_df": lc_df,
        "tuning_sensor_df": trials_sensor,
        "tuning_equip_df": trials_equip,
        "best_cfg_sensor": best_cfg_sensor,
        "best_cfg_equip": best_cfg_equip,
        "models": models_store,
        "test_preds": test_preds_store,
        "feat_imp_h15": feat_imp_h15,
    }


def predict_day_quantiles_for_bay(
    df_day_feat: pd.DataFrame,
    models_mode: dict,
    q_base_kW: float,
    m_bay: float,
    horizons=(5, 10, 15, 30),
) -> dict:
    """
    Generate calibrated (P10, P50, P90) bay sensible heat forecasts [kW] for every minute
    of a 24h test day by mapping the trained real-HIPE models via Q_bay = q_base + m_bay * P_HIPE.
    """
    preds = {}
    p_now = df_day_feat["P_10tools_now"].values
    for h in horizons:
        entry = models_mode[h]
        X = df_day_feat[entry["feat_cols"]].values
        p10_r = np.clip(entry["p10"].predict(X), 0.0, None)
        p50_r = np.clip(entry["p50"].predict(X), 0.0, None)
        p90_r = np.clip(entry["p90"].predict(X), 0.0, None)
        mean_r = np.clip(entry["mean"].predict(X), 0.0, None)

        act_pred = (p90_r > 0.15) | (p_now > 0.15)
        p10_cal = np.clip(np.minimum(p10_r, p50_r) - np.where(act_pred, entry["delta_lo"], 0.0), 0.0, None)
        # Use blended mean/p50 clipped inside [p10, p90] so MPC sees sharp expected preheat energy
        p50_cal = np.maximum(0.5 * (p50_r + mean_r), p10_cal)
        p90_cal = np.maximum(p90_r, p50_cal) + np.where(act_pred, entry["delta_up"], 0.0)

        preds[h] = {
            "p10": q_base_kW + m_bay * p10_cal,
            "p50": q_base_kW + m_bay * p50_cal,
            "p90": q_base_kW + m_bay * p90_cal,
            "raw_mean": mean_r,
            "raw_p10": p10_cal,
            "raw_p90": p90_cal,
        }
    return preds


if __name__ == "__main__":
    print("Building real 3-month HIPE dataset (zero target leakage)...")
    bundle = build_real_hipe_dataset()
    print("Running learning curve, validation tuning, and 5-model benchmark on 10 unseen test days...")
    res = train_and_benchmark_all(bundle)

    print("\n=== 1. EMPIRICAL LEARNING CURVE (h=15 min, Evaluated on 7-Day Val Set) ===")
    print(res["learning_curve_df"].to_string(index=False))

    print("\n=== 2. BEST HYPERPARAMETERS SELECTED ON VALIDATION ===")
    print("  SensorOnly_LGBM:", res["best_cfg_sensor"])
    print("  EquipState_LGBM:", res["best_cfg_equip"])

    print("\n=== 3. 5-MODEL BENCHMARK ON 10 UNSEEN HELD-OUT TEST DAYS (RAW REAL HIPE kW) ===")
    cols_show = ["Horizon_min", "Model", "RMSE", "MAE", "R2", "Active_RMSE", "Shock_RMSE", "Cov_80_Active", "Cov_Upper90_Active", "Width_Active"]
    df_b = res["benchmark_df"][cols_show].copy()
    for c in ["RMSE", "MAE", "R2", "Active_RMSE", "Shock_RMSE", "Cov_80_Active", "Cov_Upper90_Active", "Width_Active"]:
        df_b[c] = df_b[c].round(3)
    print(df_b.to_string(index=False))

    print("\n=== 4. TOP 10 FEATURES FOR EQUIP-STATE LGBM (h=15 min) ===")
    print(res["feat_imp_h15"]["equip"].head(10).to_string())
