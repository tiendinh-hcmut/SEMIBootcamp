"""
Decompose the isolated Pure-L4 chiller saving s4 (Section 2.4, Table "L4 decomposition").

s4 combines two effects inside controllers.py:
  (a) CHW-reset COP gain  -- simulated physically (warmer chilled water -> higher COP, ~3 %/K)
  (b) standby-staging credit -- a parametric multiplier (1 + standby_margin_frac) on chiller power
This script re-runs the L4-only sweep and reports (a) and (b) separately.

Run from the repository root (retrains the forecasters, ~3 min on a laptop CPU):
    python poc_simulation/analysis_l4_decomposition.py
Output: poc_simulation/results/real_l4_staging_decomposition.csv
"""
import os
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

from data_pipeline_real import build_real_hipe_dataset, build_24h_bay_simulation_frame
from forecaster_real import train_and_benchmark_all, predict_day_quantiles_for_bay
from controllers import run_24h_simulation

RES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main():
    bundle = build_real_hipe_dataset(jitter_std_min=3.0, unannounced_prob=0.02, seed=2026)
    fc = train_and_benchmark_all(bundle)
    df_all = bundle["df_all"]
    dec6 = df_all[df_all["local_date"] == "2017-12-06"].reset_index(drop=True)

    rows = []
    for f in [0.05, 0.10, 0.20]:
        bay, meta = build_24h_bay_simulation_frame(dec6, fluctuation_f=f, seed=42)
        preds = predict_day_quantiles_for_bay(dec6, fc["models"]["equip"], meta["q_base_kW"], meta["m_bay"])
        pid = run_24h_simulation(bay, "Static_PID", "A2", f, None, enable_l1_l3=False)
        eq = run_24h_simulation(bay, "EquipState_MPC", "A2", f, preds, enable_l1_l3=False)

        p_pid = pid["P_chiller_kW"].mean()
        p_eq = eq["P_chiller_kW"].mean()
        p_eq_no_credit = (eq["P_chiller_kW"] / (1.0 + eq["standby_margin_frac"])).mean()
        rows.append({
            "f_pct": int(round(100 * f)),
            "s4_total_pct": round(100 * (p_pid - p_eq) / p_pid, 2),
            "s4_chw_reset_cop_pct": round(100 * (p_pid - p_eq_no_credit) / p_pid, 2),
            "s4_staging_credit_pct": round(100 * (p_eq_no_credit - p_eq) / p_pid, 2),
            "workbook_prior_pct": round(50 * f, 2),
            "mean_standby_margin_frac": round(float(eq["standby_margin_frac"].mean()), 4),
        })

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES_DIR, "real_l4_staging_decomposition.csv"), index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
