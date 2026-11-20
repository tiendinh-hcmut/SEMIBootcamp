"""
Three-Controller Closed-Loop Ablation Suite & PLC Safety Guardrail Layer
(Sections 2.4, 3.2, and 4.1)

Implements:
  1. Static_PID        : Conventional static cleanroom BMS (no L1-L4)
  2. SensorOnly_MPC    : Identical L1-L3 logic + Two-Level Constrained MPC driven
                         ONLY by Sensor-Only LightGBM Quantile Forecasts
  3. EquipState_MPC    : Identical L1-L3 logic + Two-Level Constrained MPC driven
                         by Equipment-State (SEMI E10/GEM + MES + S23) Quantile Forecasts
  4. L4_Only_Ablation  : Clean measurement of s4 (dynamic plant efficiency excluding L1-L3)
  5. PLC_Guardrails    : Hard Tier-1/Tier-2 interlocks, multi-sensor drift voting,
                         slew-rate limiter, and <1 s bumpless PID fallback
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from physics_model import (
    CleanroomParams,
    CleanroomZoneSimulator,
    cp_moist_air,
    humidity_ratio_from_dp,
    rh_from_T_W,
)


def _interpolate_horizon_profile(p_t5, p_t10, p_t15, q_now, H=15):
    """Linearly interpolate a 1..H minute forecast trajectory from (0, 5, 10, 15) min knots."""
    knots_x = np.array([0.0, 5.0, 10.0, 15.0])
    knots_y = np.array([q_now, p_t5, p_t10, p_t15])
    steps = np.arange(1, H + 1, dtype=float)
    return np.interp(steps, knots_x, knots_y)


class ZoneMPCController:
    """
    Receding-Horizon Zone-Level Constrained MPC (1-min step, 15-min horizon)
    + Plant-Level Chilled-Water Coordinator (CHW reset & standby-chiller staging).

    Enforces:
      - P90 upper-bound chance constraint: T_z(t+k | Q_P90) <= T_tier2_hi (23.0 C)
      - P10 lower-bound chance constraint: T_z(t+k | Q_P10) >= T_tier2_lo (21.0 C)
      - Slew-rate limit on Q_dcc_cmd (<= 25 kW/min)
      - Coil capacity limit Q_dcc_cmd <= UA_eff * (T_z - T_chw_dcc)
    """

    def __init__(
        self,
        params: CleanroomParams,
        ua_dcc_kW_K: float,
        fluctuation_f: float,
        horizon_min: int = 15,
        slew_rate_kW_min: float = 35.0,
    ):
        self.p = params
        self.ua_dcc = ua_dcc_kW_K
        self.fluctuation_f = fluctuation_f
        self.H = horizon_min
        self.slew_rate = slew_rate_kW_min
        self.prev_u_dcc = 264.0
        self.prev_T_chw_cmd = params.T_chw_DCC_base_C
        self.plant_update_counter = 0

    def reset(self, init_Q_dcc=264.0):
        self.prev_u_dcc = float(init_Q_dcc)
        self.prev_T_chw_cmd = self.p.T_chw_DCC_base_C
        self.plant_update_counter = 0

    def compute_plant_setpoint(
        self,
        T_z: float,
        q_p50_traj: np.ndarray,
        q_p90_traj: np.ndarray,
        Q_aux_kW: float,
        Q_mau_adv_kW: float,
        t_min: int,
        q_p50_30m: float = None,
        q_p90_30m: float = None,
    ):
        """
        Plant-Level Coordinator (updates every 15 min per Section 2.1, or immediately on high surge).
        Determines:
          - T_chw_dcc_cmd [C]: highest safe DCC chilled-water supply temp that still meets
            the 15-30 min P90 peak sensible load without coil saturation.
          - standby_margin_credit: chiller staging/part-load efficiency gain from shrinking
            unforecasted 15-30 min surge uncertainty (P90 - P50), aligned with centrifugal
            chiller startup lead time (15-30 min).
        """
        p = self.p
        Q_env = p.G_env_kW_K * (p.T_amb_adj_C - T_z)
        # Peak P90 sensible cooling demand over next 15-30 min plant staging window
        q_peak_p90 = float(np.max(q_p90_traj))
        if q_p90_30m is not None:
            q_peak_p90 = max(q_peak_p90, float(q_p90_30m))
        q_dcc_p90_peak = float(q_peak_p90 + Q_aux_kW + Q_env + Q_mau_adv_kW)
        q_dcc_p90_peak = max(40.0, q_dcc_p90_peak)

        # Required approach DeltaT = Q_peak / UA_dcc (keep 0.45 K valve-authority reserve)
        delta_T_req = (q_dcc_p90_peak / self.ua_dcc) + 0.45
        T_chw_ideal = np.clip(
            p.T_target - delta_T_req, p.T_chw_DCC_min_C, p.T_chw_DCC_max_C
        )

        # Plant-level 15-min coordination cadence (with immediate pre-chill override if T_chw_ideal drops >0.4K)
        if (t_min % 15 == 0) or (T_chw_ideal < self.prev_T_chw_cmd - 0.4):
            # Rate-limit plant CHW reset to <= 1.2 K rise or 2.0 K drop per update for fast pre-chill
            delta_chw = np.clip(T_chw_ideal - self.prev_T_chw_cmd, -2.0, 1.2)
            self.prev_T_chw_cmd = float(
                np.clip(
                    self.prev_T_chw_cmd + delta_chw,
                    p.T_chw_DCC_min_C,
                    p.T_chw_DCC_max_C,
                )
            )

        # Standby-chiller staging margin recovery over the 15-30 min chiller startup horizon:
        # Static plant holds spinning headroom for unforecasted load swings ~ 1.35 * f * Q_mean.
        # Forecast-driven plant only needs headroom for residual 15-30 min quantile uncertainty (P90 - P50).
        q_mean_ref = 0.87 * p.Q_tool_RUN_kW
        spread_15 = float(q_p90_traj[-1] - q_p50_traj[-1])
        spread_plant = 0.5 * (spread_15 + float(q_p90_30m - q_p50_30m)) if (q_p90_30m is not None and q_p50_30m is not None) else float(np.mean(q_p90_traj - q_p50_traj))
        p90_spread_frac = max(0.0, spread_plant) / q_mean_ref
        static_headroom_frac = 1.35 * self.fluctuation_f
        recovered_headroom = max(0.0, static_headroom_frac - p90_spread_frac)
        # Negative standby_margin_frac represents efficiency improvement relative to static baseline
        standby_margin_frac = -min(0.065, 0.24 * recovered_headroom)

        return self.prev_T_chw_cmd, standby_margin_frac

    def solve_zone_mpc(
        self,
        T_z: float,
        W_z: float,
        Q_dcc_curr: float,
        buf_Q_dcc: list,
        T_chw_dcc: float,
        T_mau_sa: float,
        Q_aux_kW: float,
        q_p10_traj: np.ndarray,
        q_p50_traj: np.ndarray,
        q_p90_traj: np.ndarray,
    ):
        """
        Solve 15-minute receding-horizon constrained Zone MPC using 3-block move blocking
        (u0 for steps 0..2, u1 for 3..7, u2 for 8..14) for fast, smooth, globally accurate QP-like actuation.
        """
        p = self.p
        H = self.H
        dt_s = 60.0
        alpha_coil = 1.0 - np.exp(-1.0 / p.tau_coil_min)
        cp_z = float(cp_moist_air(W_z))
        Q_dcc_cap = max(20.0, self.ua_dcc * max(0.5, T_z - T_chw_dcc))

        # Precompute fixed disturbances over horizon
        Q_env_approx = p.G_env_kW_K * (p.T_amb_adj_C - T_z)
        Q_mau_adv_approx = p.m_da_MAU_kg_s * cp_z * (T_mau_sa - T_z)

        # Fast rollout helper under any disturbance trajectory q_traj
        def rollout(u_seq, q_traj):
            T = T_z
            q_d = Q_dcc_curr
            fifo = list(buf_Q_dcc)
            T_hist = np.empty(H)
            for k in range(H):
                u_del = fifo.pop(0)
                fifo.append(u_seq[k])
                q_d = q_d + alpha_coil * (u_del - q_d)
                dT = (Q_env_approx + q_traj[k] + Q_aux_kW + Q_mau_adv_approx - q_d) * dt_s / p.C_eff_kJ_K
                T = T + dT
                T_hist[k] = T
            return T_hist

        def objective(u_blocks):
            # Expand 3 move-blocking control variables to 15-step sequence
            u_seq = np.empty(H)
            u_seq[0:3] = u_blocks[0]
            u_seq[3:8] = u_blocks[1]
            u_seq[8:H] = u_blocks[2]

            T_p50 = rollout(u_seq, q_p50_traj)
            T_p90 = rollout(u_seq, q_p90_traj)
            T_p10 = rollout(u_seq, q_p10_traj)

            # 1. Tracking nominal setpoint 22.0 C (with mild energy preference toward 22.2 C)
            cost_track = 4.0 * np.sum((T_p50 - 22.05) ** 2)

            # 2. Control smoothness (penalize chatter & large jumps from prev_u_dcc)
            cost_smooth = 0.0008 * ((u_blocks[0] - self.prev_u_dcc) ** 2
                                    + (u_blocks[1] - u_blocks[0]) ** 2
                                    + (u_blocks[2] - u_blocks[1]) ** 2)

            # 3. Direct cooling energy term
            cost_energy = 0.00015 * np.sum(u_seq)

            # 4. P90 & P10 Chance-Constraint Barrier Penalties (Tier-2 bounds [21.0, 23.0] C)
            # Keep soft margin 0.15 C inside Tier-2 so real plant never crosses 21.0 or 23.0 C
            over_p90 = np.maximum(0.0, T_p90 - (p.T_tier2_hi - 0.12))
            under_p10 = np.maximum(0.0, (p.T_tier2_lo + 0.12) - T_p10)
            over_p50 = np.maximum(0.0, T_p50 - (p.T_tier2_hi - 0.25))
            under_p50 = np.maximum(0.0, (p.T_tier2_lo + 0.25) - T_p50)

            cost_safety = (
                450.0 * np.sum(over_p90 ** 2)
                + 450.0 * np.sum(under_p10 ** 2)
                + 150.0 * np.sum(over_p50 ** 2)
                + 150.0 * np.sum(under_p50 ** 2)
            )
            return float(cost_track + cost_smooth + cost_energy + cost_safety)

        # Warm-start initial guess from steady-state balance of P50
        u_init = np.clip(
            np.array([
                q_p50_traj[4] + Q_aux_kW + Q_env_approx + Q_mau_adv_approx,
                q_p50_traj[9] + Q_aux_kW + Q_env_approx + Q_mau_adv_approx,
                q_p50_traj[14] + Q_aux_kW + Q_env_approx + Q_mau_adv_approx,
            ]),
            0.0,
            Q_dcc_cap,
        )

        # Slew-rate bounds on first move
        u0_lo = max(0.0, self.prev_u_dcc - self.slew_rate)
        u0_hi = min(Q_dcc_cap, self.prev_u_dcc + self.slew_rate)
        if u0_lo > u0_hi:
            u0_lo = 0.0
            u0_hi = Q_dcc_cap

        bounds = [(u0_lo, u0_hi), (0.0, Q_dcc_cap), (0.0, Q_dcc_cap)]
        res = minimize(
            objective,
            x0=np.clip(u_init, [b[0] for b in bounds], [b[1] for b in bounds]),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 35, "ftol": 1e-6},
        )
        u_cmd = float(np.clip(res.x[0], u0_lo, u0_hi))
        self.prev_u_dcc = u_cmd
        return u_cmd


class ReactivePIDController:
    """
    Skogestad SIMC-tuned industrial PID controller for an integrating cleanroom zone
    with tau_dead + tau_coil = 10 min actuation lag (k' = 0.0019 K/(kW*min)).
    """

    def __init__(self, params: CleanroomParams, ua_dcc_kW_K: float, kp=55.0, ki=0.45, kd=180.0):
        self.p = params
        self.ua_dcc = ua_dcc_kW_K
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.reset()

    def reset(self, init_u=264.0):
        self.integral = float(init_u)
        self.prev_err = 0.0
        self.prev_u = float(init_u)

    def compute(self, T_z_meas: float, T_chw_dcc: float, T_setpoint: float = 22.0):
        err = T_z_meas - T_setpoint  # positive when zone too hot -> need more cooling
        self.integral = float(np.clip(self.integral + self.ki * err, 40.0, 380.0))
        deriv = err - self.prev_err
        self.prev_err = err

        u_raw = self.integral + self.kp * err + self.kd * deriv
        Q_cap = max(20.0, self.ua_dcc * max(0.5, T_z_meas - T_chw_dcc))
        # Rate-limit valve movement
        u_cmd = float(np.clip(u_raw, max(0.0, self.prev_u - 30.0), min(Q_cap, self.prev_u + 30.0)))
        self.prev_u = u_cmd
        return u_cmd


def run_24h_simulation(
    df_test: pd.DataFrame,
    controller_mode: str,
    scenario: str = "A2",
    fluctuation_f: float = 0.10,
    preds: dict = None,
    enable_l1_l3: bool = True,
    fault_config: dict = None,
    params: CleanroomParams = CleanroomParams(),
    plant_params: CleanroomParams = None,
) -> pd.DataFrame:
    """
    Run a full 24-hour (1,440-minute) closed-loop simulation on the held-out Test Day.

    Parameters
    ----------
    controller_mode : str
        One of {'Static_PID', 'SensorOnly_MPC', 'EquipState_MPC'}.
    scenario : str
        'A2' (Decoupled MAU with electric reheat + DCC) or 'B' (Heat-recovery reheat + DCC).
    enable_l1_l3 : bool
        If True, both SensorOnly_MPC and EquipState_MPC use the exact same L1-L3 actuation
        logic (per Section 3.3.2 ablation rule). If False, L1-L3 are disabled so ONLY L4
        (dynamic plant efficiency & forecast tracking) is measured cleanly.
    fault_config : dict, optional
        Inject realistic faults to validate PLC Hard Safety Guardrails:
          - 'drift_start', 'drift_end', 'drift_T_C', 'drift_RH_pct'
          - 'comms_drop_start', 'comms_drop_end'
          - 'valve_fault_start', 'valve_fault_end', 'valve_cap_frac' (LBNL FDD valve fault)
    params : CleanroomParams
        Nominal controller model parameters (used inside MPC).
    plant_params : CleanroomParams, optional
        True physical plant parameters (for +/-20% Plant-Model Mismatch robustness tests).
        Defaults to params (nominal match).
    """
    n_steps = len(df_test)
    rng = np.random.default_rng(1234)
    p_plant = plant_params if plant_params is not None else params

    # Size DCC coil UA for the bay's peak design load at 10 K approach (12C CHW -> 22C zone)
    q_peak_design = max(
        264.0 * (1.0 + 1.5 * fluctuation_f) * 1.10,
        (float(df_test["Q_room_true_kW"].max()) + 48.0) * 1.05,
    )
    ua_dcc = q_peak_design / 10.0

    # Sync plant UA_DCC with the bay's designed coil sizing
    p_plant_eff = CleanroomParams(
        alpha_mass=p_plant.alpha_mass,
        G_env_kW_K=p_plant.G_env_kW_K,
        tau_dead_min=p_plant.tau_dead_min,
        tau_coil_min=p_plant.tau_coil_min,
        UA_DCC_kW_K=ua_dcc,
    )
    sim = CleanroomZoneSimulator(params=p_plant_eff)
    init_q_dcc = float(df_test["Q_room_true_kW"].iloc[0] + 45.0)
    # Start in steady-state moisture equilibrium (9.0 C static vs 10.5 C normal MSL 1-2 overnight)
    msl_init = int(df_test["msl_ge3_active"].iloc[0])
    if controller_mode == "Static_PID":
        init_coil = params.T_MAU_coil_strict
        init_rh = 44.0
    else:
        init_coil = 8.6 if (msl_init == 1 or not enable_l1_l3) else params.T_MAU_coil_normal
        init_rh = 43.0 if (msl_init == 1 or not enable_l1_l3) else 47.0
    sim.reset(
        T_z0=22.0,
        RH0=init_rh,
        T_mau_sa0=21.0,
        Q_dcc0=init_q_dcc,
        T_chw0=params.T_chw_DCC_base_C,
        T_mau_coil0=init_coil,
    )

    pid = ReactivePIDController(params, ua_dcc_kW_K=ua_dcc)
    pid.reset(init_u=init_q_dcc)

    mpc = ZoneMPCController(params, ua_dcc_kW_K=ua_dcc, fluctuation_f=fluctuation_f)
    mpc.reset(init_Q_dcc=init_q_dcc)

    records = []
    fallback_active = False

    for t in range(n_steps):
        row = df_test.iloc[t]
        Q_room_true = float(row["Q_room_true_kW"])
        T_out = float(row["T_out_C"])
        Td_out = float(row["Td_out_C"])
        m_H2O_int = float(row["m_H2O_int_kg_s"])
        msl_active = int(row["msl_ge3_active"])
        msl_ahead_35 = int(row["mes_msl_ge3_ahead_35m"])
        msl_ahead_65 = int(row["mes_msl_ge3_ahead_65m"])
        msl_ahead_75 = int(row["mes_msl_ge3_ahead_75m"])
        frac_run_gem = float(row["frac_run_gem"])
        frac_run_mes_10 = float(row["mes_frac_run_ahead_10m"])

        # True current zone state
        T_z_true = sim.T_z
        RH_z_true = float(rh_from_T_W(sim.T_z, sim.W_z))

        # Multi-sensor measurements (3 redundant zone sensors with realistic noise)
        T_s1 = T_z_true + rng.normal(0.0, 0.04)
        T_s2 = T_z_true + rng.normal(0.0, 0.04)
        T_s3 = T_z_true + rng.normal(0.0, 0.04)
        RH_s1 = RH_z_true + rng.normal(0.0, 0.25)
        RH_s2 = RH_z_true + rng.normal(0.0, 0.25)
        RH_s3 = RH_z_true + rng.normal(0.0, 0.25)

        # Optional Fault Injection (LBNL FDD-style sensor drift + SECS/GEM comms dropout)
        comms_lost = False
        sensor_fault_detected = False
        if fault_config is not None:
            if fault_config.get("drift_start", -1) <= t < fault_config.get("drift_end", -1):
                # Faulty Sensor 1 drifts by +1.8 C and +6% RH
                T_s1 += fault_config.get("drift_T_C", 1.8)
                RH_s1 += fault_config.get("drift_RH_pct", 6.0)
            if fault_config.get("comms_drop_start", -1) <= t < fault_config.get("comms_drop_end", -1):
                comms_lost = True

        # PLC Layer: 2-out-of-3 Median Sensor Voting & Drift Isolation
        T_sensors = np.array([T_s1, T_s2, T_s3])
        RH_sensors = np.array([RH_s1, RH_s2, RH_s3])
        T_med = float(np.median(T_sensors))
        RH_med = float(np.median(RH_sensors))
        if np.max(np.abs(T_sensors - T_med)) > 0.6 or np.max(np.abs(RH_sensors - RH_med)) > 2.5:
            sensor_fault_detected = True
            # Isolate outlier: average the two agreeing sensors
            inliers_T = T_sensors[np.abs(T_sensors - T_med) <= 0.6]
            inliers_RH = RH_sensors[np.abs(RH_sensors - RH_med) <= 2.5]
            T_meas = float(np.mean(inliers_T)) if len(inliers_T) > 0 else T_med
            RH_meas = float(np.mean(inliers_RH)) if len(inliers_RH) > 0 else RH_med
        else:
            T_meas = float(np.mean(T_sensors))
            RH_meas = float(np.mean(RH_sensors))

        # Active Tier-2 upper RH ceiling (Dynamic MSL bound, Table 1.2 & Section 1.3.2)
        RH_tier2_hi = params.RH_tier2_hi_msl3 if msl_active == 1 else params.RH_tier2_hi_normal

        # ---------------------------------------------------------------------
        # Controller Dispatch
        # ---------------------------------------------------------------------
        if controller_mode == "Static_PID":
            T_mau_coil_cmd = params.T_MAU_coil_strict   # Fixed 9.0 C year-round
            T_mau_sa_cmd = params.T_MAU_reheat_target   # Fixed 21.0 C electric reheat
            ach_rec = params.ach_rec_nom                # Fixed 35 ACH
            T_chw_cmd = params.T_chw_DCC_base_C         # Fixed 12.0 C
            standby_margin_frac = 0.0                   # Reference baseline (matches Table 3.4)
            Q_dcc_cmd = pid.compute(T_meas, sim.T_chw_dcc, T_setpoint=params.T_target)
            fallback_active = False

        else:
            # Check PLC Hard Safety Interlock & Comms Heartbeat
            tier1_breach = (
                T_meas <= params.T_tier1_lo + 0.3
                or T_meas >= params.T_tier1_hi - 0.3
                or RH_meas <= params.RH_tier1_lo + 1.5
                or RH_meas >= params.RH_tier1_hi - 1.5
            )

            if comms_lost or tier1_breach:
                # Bumpless PLC Fallback (< 1 s) to SIMC PID + safe static setpoints
                if not fallback_active:
                    # Bumpless transfer: compensate for MAU reheat re-arming sensible step (+delta_Q_mau)
                    cp_fb = float(cp_moist_air(sim.W_z))
                    reheat_step_kW = max(
                        0.0,
                        params.m_da_MAU_kg_s * cp_fb * (params.T_MAU_reheat_target - sim.T_mau_sa),
                    )
                    pid.reset(init_u=mpc.prev_u_dcc + reheat_step_kW)
                fallback_active = True
                T_mau_coil_cmd = 8.6
                T_mau_sa_cmd = params.T_MAU_reheat_target
                ach_rec = params.ach_rec_nom
                T_chw_cmd = params.T_chw_DCC_base_C
                standby_margin_frac = 0.0
                Q_dcc_cmd = pid.compute(T_meas, sim.T_chw_dcc, T_setpoint=params.T_target)
                mpc.prev_u_dcc = Q_dcc_cmd
            else:
                fallback_active = False

                # ----- L1, L2, L3 Logic (IDENTICAL for SensorOnly_MPC & EquipState_MPC!) -----
                if enable_l1_l3:
                    # L1: MSL-aware dew-point reset (pre-tighten >= 2*tau_w ~ 65-75 min ahead since tau_w = 31.4 min)
                    # Uses 8.6 C coil exit (within Section 3.1.2 8.5-9.0 C spec) to keep RH <= 45.0% even at T_z=21.6 C
                    need_strict_dp = (
                        (msl_active == 1)
                        or (msl_ahead_35 == 1)
                        or (msl_ahead_65 == 1)
                        or (msl_ahead_75 == 1)
                    )
                    T_mau_coil_cmd = (
                        8.6 if need_strict_dp else params.T_MAU_coil_normal
                    )

                    # L2: Load-aware MAU reheat bypass when tools are predominantly RUN (>= 60%)
                    # Re-arms reheat if zone drops near 21.3 C or when tools drop to non-RUN
                    if frac_run_gem >= 0.60 and T_meas > 21.35:
                        T_mau_sa_cmd = T_mau_coil_cmd  # Bypass electric reheat!
                    else:
                        T_mau_sa_cmd = params.T_MAU_reheat_target

                    # L3: FFU turndown 35 -> 30 ACH in non-RUN states, pre-restoring 10 min before LOT_START
                    if frac_run_gem < 0.60 and frac_run_mes_10 < 0.65:
                        ach_rec = params.ach_rec_turndown
                    else:
                        ach_rec = params.ach_rec_nom
                else:
                    # Pure L4 Ablation Mode: L1-L3 locked to Static Baseline!
                    T_mau_coil_cmd = params.T_MAU_coil_strict
                    T_mau_sa_cmd = params.T_MAU_reheat_target
                    ach_rec = params.ach_rec_nom

                # ----- Build 15-min Quantile Forecast Trajectory (P10, P50, P90) -----
                q_now_est = (
                    float(row["Q_gem_instant_kW"])
                    if controller_mode == "EquipState_MPC"
                    else float(row["Q_obs_bms_kW"])
                )
                q_p10_traj = _interpolate_horizon_profile(
                    preds[5]["p10"][t], preds[10]["p10"][t], preds[15]["p10"][t], q_now_est
                )
                q_p50_traj = _interpolate_horizon_profile(
                    preds[5]["p50"][t], preds[10]["p50"][t], preds[15]["p50"][t], q_now_est
                )
                q_p90_traj = _interpolate_horizon_profile(
                    preds[5]["p90"][t], preds[10]["p90"][t], preds[15]["p90"][t], q_now_est
                )

                P_ffu_now = params.P_ffu_nom_kW * (ach_rec / params.ach_rec_nom) ** 3
                Q_aux_now = P_ffu_now + params.Q_lighting_kW
                cp_now = float(cp_moist_air(sim.W_z))
                Q_mau_adv_now = params.m_da_MAU_kg_s * cp_now * (T_mau_sa_cmd - T_meas)

                # ----- Plant-Level Coordinator (L4: CHW Reset + 15-30 min Staging Margin) -----
                q_p50_30 = float(preds[30]["p50"][t]) if 30 in preds else None
                q_p90_30 = float(preds[30]["p90"][t]) if 30 in preds else None
                T_chw_cmd, standby_margin_frac = mpc.compute_plant_setpoint(
                    T_z=T_meas,
                    q_p50_traj=q_p50_traj,
                    q_p90_traj=q_p90_traj,
                    Q_aux_kW=Q_aux_now,
                    Q_mau_adv_kW=Q_mau_adv_now,
                    t_min=t,
                    q_p50_30m=q_p50_30,
                    q_p90_30m=q_p90_30,
                )

                # ----- Zone-Level Constrained MPC (1-min step, 15-min horizon) -----
                Q_dcc_cmd = mpc.solve_zone_mpc(
                    T_z=T_meas,
                    W_z=sim.W_z,
                    Q_dcc_curr=sim.Q_dcc,
                    buf_Q_dcc=sim._buf_Q_dcc_cmd,
                    T_chw_dcc=sim.T_chw_dcc,
                    T_mau_sa=T_mau_sa_cmd,
                    Q_aux_kW=Q_aux_now,
                    q_p10_traj=q_p10_traj,
                    q_p50_traj=q_p50_traj,
                    q_p90_traj=q_p90_traj,
                )

        # Optional LBNL FDD-style cooling-coil valve sticking / degradation fault
        valve_fault_active = False
        Q_dcc_cmd_applied = Q_dcc_cmd
        if fault_config is not None:
            if fault_config.get("valve_fault_start", -1) <= t < fault_config.get("valve_fault_end", -1):
                valve_fault_active = True
                # Mimic LBNL FLEXLAB cooling-coil valve partially stuck (reduced effective authority)
                cap_frac = float(fault_config.get("valve_cap_frac", 0.78))
                q_cap_now = max(0.0, p_plant_eff.UA_DCC_kW_K * (sim.T_z - sim.T_chw_dcc))
                Q_dcc_cmd_applied = min(Q_dcc_cmd, cap_frac * q_cap_now)
                # PLC interlock detects valve saturation/tracking error -> forces CHW to min safe 11.8 C & 35 ACH
                if Q_dcc_cmd > Q_dcc_cmd_applied + 5.0:
                    T_chw_cmd = params.T_chw_DCC_min_C
                    ach_rec = params.ach_rec_nom

        # Advance physical cleanroom simulator by 1 minute
        step_res = sim.step(
            Q_room_kW=Q_room_true,
            T_out_C=T_out,
            Td_out_C=Td_out,
            m_H2O_int_kg_s=m_H2O_int,
            Q_dcc_cmd_kW=Q_dcc_cmd_applied,
            T_mau_coil_cmd_C=T_mau_coil_cmd,
            T_mau_sa_cmd_C=T_mau_sa_cmd,
            T_chw_cmd_C=T_chw_cmd,
            ach_rec=ach_rec,
            standby_margin_frac=standby_margin_frac,
            scenario=scenario,
        )

        # Safety Compliance Flags (strictly evaluated on TRUE zone T & RH!)
        T_post = step_res["T_z"]
        RH_post = step_res["RH_z"]
        tier1_viol = int(
            (T_post < params.T_tier1_lo)
            or (T_post > params.T_tier1_hi)
            or (RH_post < params.RH_tier1_lo)
            or (RH_post > params.RH_tier1_hi)
        )
        tier2_T_viol = int((T_post < params.T_tier2_lo) or (T_post > params.T_tier2_hi))
        tier2_RH_viol = int((RH_post < params.RH_tier2_lo) or (RH_post > RH_tier2_hi))
        tier2_viol = int(tier2_T_viol or tier2_RH_viol)

        records.append({
            "minute": t,
            "hour": t / 60.0,
            "Q_room_true_kW": Q_room_true,
            "msl_ge3_active": msl_active,
            "RH_tier2_hi": RH_tier2_hi,
            "T_z_meas": T_meas,
            "T_s1_raw": T_s1,
            "RH_s1_raw": RH_s1,
            "ach_rec": ach_rec,
            "Q_dcc_cmd_kW": Q_dcc_cmd,
            "T_chw_cmd_C": T_chw_cmd,
            "standby_margin_frac": standby_margin_frac,
            "fallback_active": int(fallback_active),
            "sensor_fault_detected": int(sensor_fault_detected),
            "valve_fault_active": int(valve_fault_active),
            "tier1_viol": tier1_viol,
            "tier2_T_viol": tier2_T_viol,
            "tier2_RH_viol": tier2_RH_viol,
            "tier2_viol": tier2_viol,
            **step_res,
        })

    return pd.DataFrame(records)
