"""
Cleanroom Thermal-Moisture Physics & HVAC Plant Model
Aligned strictly with Section 3.1 (Eqs. 3.1-3.18) and Section 3.3 (Eqs. 3.19-3.21)
of the Group 3 SEMI TECH Zoomers Challenge 2026 Report.
"""

from dataclasses import dataclass
import numpy as np


# =============================================================================
# 1. Psychrometric Functions (Eqs. 3.1 - 3.5, Arden Buck Correlation)
# =============================================================================
P_ATM_KPA = 101.20  # Standard barometric pressure at Tan Son Nhat (WMO 489000)


def saturation_vapor_pressure_kpa(T_c):
    """Arden Buck correlation for saturation vapor pressure over liquid water (Eq. 3.1)."""
    T = np.asarray(T_c, dtype=float)
    return 0.61121 * np.exp((18.678 - T / 234.5) * (T / (257.14 + T)))


def humidity_ratio_from_dp(Td_c, p_atm=P_ATM_KPA):
    """Humidity ratio W [kg_w / kg_da] from dew point Td [C] (Eq. 3.2)."""
    p_w = saturation_vapor_pressure_kpa(Td_c)
    return 0.621945 * p_w / (p_atm - p_w)


def humidity_ratio_from_T_RH(T_c, RH_pct, p_atm=P_ATM_KPA):
    """Humidity ratio W [kg_w / kg_da] from dry-bulb T [C] and RH [%]."""
    p_ws = saturation_vapor_pressure_kpa(T_c)
    p_w = (np.asarray(RH_pct, dtype=float) / 100.0) * p_ws
    return 0.621945 * p_w / (p_atm - p_w)


def rh_from_T_W(T_c, W, p_atm=P_ATM_KPA):
    """Relative humidity RH [%] from dry-bulb T [C] and humidity ratio W [kg_w/kg_da]."""
    W_arr = np.asarray(W, dtype=float)
    p_w = (W_arr * p_atm) / (0.621945 + W_arr)
    p_ws = saturation_vapor_pressure_kpa(T_c)
    return np.clip(100.0 * p_w / p_ws, 0.0, 100.0)


def dew_point_from_W(W, p_atm=P_ATM_KPA):
    """Dew point temperature Td [C] from humidity ratio W using Magnus-Tetens inverse."""
    W_arr = np.maximum(np.asarray(W, dtype=float), 1e-6)
    p_w = (W_arr * p_atm) / (0.621945 + W_arr)
    # Magnus-Tetens inverse matching Table 1.2 (a=17.625, b=243.04 C)
    gamma = np.log(p_w / 0.61094)
    return (243.04 * gamma) / (17.625 - gamma)


def dry_air_density(T_c, W, p_atm=P_ATM_KPA):
    """Dry-air partial density rho_da [kg_da / m^3] (Eq. 3.3)."""
    return p_atm / (0.287042 * (T_c + 273.15) * (1.0 + 1.607858 * W))


def cp_moist_air(W):
    """Moist-air specific heat per unit dry air [kJ/(kg_da*K)] (Section 3.1.2)."""
    return 1.006 + 1.86 * np.asarray(W, dtype=float)


def specific_enthalpy(T_c, W):
    """Moist-air specific enthalpy h [kJ/kg_da] (Eq. 3.5)."""
    return 1.006 * T_c + W * (2501.0 + 1.86 * T_c)


# =============================================================================
# 2. Master Physical Parameters (Table 3.2 & Section 3.3.1)
# =============================================================================
@dataclass(frozen=True)
class CleanroomParams:
    # Geometry & Air Mass (Table 3.2)
    area_m2: float = 1500.0
    height_m: float = 3.5
    volume_m3: float = 5250.0
    M_da_kg: float = 6197.5              # Dry-air mass at 22C, 45% RH
    C_air_kJ_K: float = 6320.0           # Air-only thermal capacitance [kJ/K]
    alpha_mass: float = 5.0              # Midpoint of [4, 6] thermal mass factor
    G_env_kW_K: float = 0.50             # Envelope conductance 1/R_env [kW/K], midpoint of [0.3, 0.8]
    T_amb_adj_C: float = 28.0            # Adjacent corridor / plenum temperature [C]

    # Actuation Lags (Eq. 3.14 & Table 3.2: tau_dead + tau_coil = 8-12 min)
    tau_dead_min: int = 4                # Pure hydronic transport / valve dead time [min]
    tau_coil_min: float = 6.0            # First-order coil thermal lag [min]
    tau_plant_chw_min: float = 10.0      # Central CHW loop reset thermal lag [min]

    # Airflows (Table 3.2 & Section 3.3.2)
    m_da_MAU_kg_s: float = 3.287         # 2 ACH at HCMC base outdoor density
    ach_rec_nom: float = 35.0            # Nominal FFU recirculation ACH
    ach_rec_turndown: float = 30.0       # Non-RUN FFU turndown ACH (L3)
    m_da_rec_nom_kg_s: float = 60.25     # 35 ACH dry-air flow
    m_H2O_int_base_kg_s: float = 3.125e-4  # 15 personnel * 75 g/h

    # Auxiliary Gains & Fan Power (Section 3.3.1 & Table 3.4)
    P_ffu_nom_kW: float = 30.0           # FFU bank electric power at 35 ACH
    P_mau_fan_kW: float = 5.0            # MAU fan electric power (total fans = 35.0 kW)
    Q_lighting_kW: float = 15.0          # Lighting sensible gain inside bay
    Q_tool_RUN_kW: float = 252.0         # Nominal peak RUN tool sensible heat into room

    # Chiller Plant & Cooling Tower (Section 3.3.1 & Eq. 3.21)
    COP_MAU: float = 4.4                 # Low-temp loop (5-7C CHW, 0.80 kW/RT)
    COP_DCC_base: float = 5.28           # 1.2 * COP_MAU at nominal 12.0C CHW
    T_chw_DCC_base_C: float = 12.0       # Static DCC chilled-water supply temp [C]
    T_chw_DCC_min_C: float = 11.8        # Minimum safe dry-coil CHW temp (> room DP)
    T_chw_DCC_max_C: float = 16.5        # Maximum CHW reset ceiling (valve authority)
    cop_gain_per_K: float = 0.03         # ~3%/K COP improvement on CHW reset
    UA_DCC_kW_K: float = 33.0            # Effective coil UA (330 kW peak at 10 K approach)
    evap_L_per_kWh: float = 1.2          # Cooling-tower evaporation [L / kWh_th]
    CoC: float = 4.0                     # Cycles of concentration

    # Environmental Bounds (Table 1.2)
    T_tier1_lo: float = 20.0
    T_tier1_hi: float = 24.0
    RH_tier1_lo: float = 35.0
    RH_tier1_hi: float = 55.0

    T_tier2_lo: float = 21.0
    T_tier2_hi: float = 23.0
    RH_tier2_lo: float = 40.0
    RH_tier2_hi_normal: float = 50.0
    RH_tier2_hi_msl3: float = 45.0
    T_target: float = 22.0

    # MAU Coil Setpoints (Table 3.1 & Section 3.3.2)
    T_MAU_coil_strict: float = 9.0       # Satisfies MSL >= 3 (<= 45% RH w/ 15 staff)
    T_MAU_coil_normal: float = 10.5      # Satisfies MSL 1-2 (<= 50% RH)
    T_MAU_reheat_target: float = 21.0    # Nominal MAU reheat discharge temp

    @property
    def C_eff_kJ_K(self) -> float:
        return self.alpha_mass * self.C_air_kJ_K  # 31,600 kJ/K = 31.60 MJ/K

    @property
    def tau_w_min(self) -> float:
        return (self.M_da_kg / self.m_da_MAU_kg_s) / 60.0  # 31.42 min


# =============================================================================
# 3. MAU Coil Balance & Chiller/Tower Plant Equations
# =============================================================================
def compute_mau_loads(T_out_C, Td_out_C, T_coil_C, T_mau_sa_C, params: CleanroomParams):
    """
    Compute MAU total cooling, sensible/latent split, condensate rate, and reheat power
    per Eqs. (3.6)-(3.10).
    """
    W_out = humidity_ratio_from_dp(Td_out_C)
    W_coil = humidity_ratio_from_dp(T_coil_C)
    m_da = params.m_da_MAU_kg_s

    h_out = specific_enthalpy(T_out_C, W_out)
    h_coil = specific_enthalpy(T_coil_C, W_coil)
    Q_mau_tot = max(0.0, m_da * (h_out - h_coil))
    Q_mau_sens = m_da * cp_moist_air(W_out) * (T_out_C - T_coil_C)
    Q_mau_lat = m_da * max(0.0, W_out - W_coil) * (2501.0 + 1.86 * T_coil_C)
    m_cond_kg_s = m_da * max(0.0, W_out - W_coil)
    Q_mau_reheat = max(0.0, m_da * cp_moist_air(W_coil) * (T_mau_sa_C - T_coil_C))

    return {
        "W_out": W_out,
        "W_coil": W_coil,
        "Q_mau_tot_kW": Q_mau_tot,
        "Q_mau_sens_kW": Q_mau_sens,
        "Q_mau_lat_kW": Q_mau_lat,
        "m_cond_kg_s": m_cond_kg_s,
        "Q_mau_reheat_kW": Q_mau_reheat,
    }


def compute_plant_power_and_water(
    Q_mau_tot_kW,
    Q_dcc_kW,
    Q_reheat_kW,
    ach_rec,
    T_chw_dcc_C,
    standby_margin_frac,
    scenario: str,
    params: CleanroomParams,
):
    """
    Compute instantaneous HVAC electric power [kWe] and cooling-tower make-up water [L/min].

    Parameters
    ----------
    standby_margin_frac : float
        Fractional chiller plant part-load / standby-staging efficiency penalty
        required when load surges are unforecasted (0.0 = perfectly staged; >0 = spinning
        headroom / part-load penalty).
    scenario : str
        'A2' (Decoupled MAU with electric reheat + DCC) or
        'B'  (Decoupled MAU with passive heat-recovery reheat + DCC).
    """
    # Chiller COP responds to DCC chilled-water reset (~3%/K, Section 3.3.1)
    cop_dcc = params.COP_DCC_base * (
        1.0 + params.cop_gain_per_K * (T_chw_dcc_C - params.T_chw_DCC_base_C)
    )
    cop_dcc = max(3.5, cop_dcc)

    # Plant electricity: MAU chiller loop + DCC chiller loop with staging margin penalty
    P_chiller_mau = Q_mau_tot_kW / params.COP_MAU
    P_chiller_dcc_ideal = max(0.0, Q_dcc_kW) / cop_dcc
    # Staging / part-load headroom penalty applies to the central chiller plant serving the zone
    P_chiller_total = (P_chiller_mau + P_chiller_dcc_ideal) * (1.0 + standby_margin_frac)

    # Reheat electric power (COP = 1.0 in A2; ~0 in B)
    P_reheat_elec = Q_reheat_kW if scenario == "A2" else 0.0

    # Fan power: cubic VFD law on FFU recirculation + fixed MAU fan
    P_ffu = params.P_ffu_nom_kW * (ach_rec / params.ach_rec_nom) ** 3
    P_fans = P_ffu + params.P_mau_fan_kW

    P_hvac_total = P_chiller_total + P_reheat_elec + P_fans

    # Cooling-tower heat rejection & make-up water (Eq. 3.21)
    Q_plant_cool = Q_mau_tot_kW + max(0.0, Q_dcc_kW)
    Q_rej_kW = Q_plant_cool + P_chiller_total
    # Make-up water [L/min] = evap_L_per_kWh * Q_rej_kW / 60 * CoC / (CoC - 1)
    makeup_L_min = (
        params.evap_L_per_kWh
        * (Q_rej_kW / 60.0)
        * (params.CoC / (params.CoC - 1.0))
    )

    return {
        "P_chiller_kW": P_chiller_total,
        "P_reheat_kW": P_reheat_elec,
        "P_fans_kW": P_fans,
        "P_ffu_kW": P_ffu,
        "P_hvac_total_kW": P_hvac_total,
        "COP_dcc": cop_dcc,
        "Q_rej_kW": Q_rej_kW,
        "makeup_L_min": makeup_L_min,
    }


# =============================================================================
# 4. Dynamic Zone Simulator (1-Minute Integration of Eqs. 3.12 - 3.14)
# =============================================================================
class CleanroomZoneSimulator:
    """
    Grey-box 1-minute discrete simulator of the ISO 7 packaging cleanroom
    (Eqs. 3.12 - 3.14, Scenarios A2 and B: Decoupled MAU + DCC).
    """

    def __init__(self, params: CleanroomParams = CleanroomParams(), substeps: int = 6):
        self.p = params
        self.substeps = substeps  # 10-second internal RK/Euler substeps for numerical stability
        self.reset()

    def reset(
        self,
        T_z0: float = 22.0,
        RH0: float = 44.5,
        T_mau_sa0: float = 21.0,
        Q_dcc0: float = 264.0,
        T_chw0: float = 12.0,
        T_mau_coil0: float = 9.0,
    ):
        self.T_z = float(T_z0)
        self.W_z = float(humidity_ratio_from_T_RH(T_z0, RH0))
        self.T_mau_sa = float(T_mau_sa0)
        self.Q_dcc = float(Q_dcc0)
        self.T_chw_dcc = float(T_chw0)
        self.T_mau_coil = float(T_mau_coil0)

        # Pure transport dead-time FIFOs (tau_dead_min steps)
        d = self.p.tau_dead_min
        w0 = float(humidity_ratio_from_dp(T_mau_coil0))
        self._buf_T_mau_sa_cmd = [float(T_mau_sa0)] * d
        self._buf_Q_dcc_cmd = [float(Q_dcc0)] * d
        self._buf_W_mau_cmd = [w0] * d

    def step(
        self,
        Q_room_kW: float,
        T_out_C: float,
        Td_out_C: float,
        m_H2O_int_kg_s: float,
        Q_dcc_cmd_kW: float,
        T_mau_coil_cmd_C: float,
        T_mau_sa_cmd_C: float,
        T_chw_cmd_C: float,
        ach_rec: float,
        standby_margin_frac: float,
        scenario: str = "A2",
    ):
        p = self.p
        dt_min = 1.0
        dt_sub_s = (dt_min * 60.0) / self.substeps

        # 1. Update central CHW temperature with plant loop first-order lag
        alpha_chw = 1.0 - np.exp(-dt_min / p.tau_plant_chw_min)
        self.T_chw_dcc += alpha_chw * (
            np.clip(T_chw_cmd_C, p.T_chw_DCC_min_C, p.T_chw_DCC_max_C) - self.T_chw_dcc
        )

        # 2. Update MAU coil exit temperature (small 2-min coil lag)
        alpha_mau_coil = 1.0 - np.exp(-dt_min / 2.0)
        self.T_mau_coil += alpha_mau_coil * (T_mau_coil_cmd_C - self.T_mau_coil)
        W_mau_exit_now = float(humidity_ratio_from_dp(self.T_mau_coil))

        # 3. Physical DCC capacity limit given current T_chw_dcc and T_z
        Q_dcc_cap = max(0.0, p.UA_DCC_kW_K * (self.T_z - self.T_chw_dcc))
        Q_dcc_cmd_eff = float(np.clip(Q_dcc_cmd_kW, 0.0, Q_dcc_cap))
        T_mau_sa_cmd_eff = float(np.clip(T_mau_sa_cmd_C, self.T_mau_coil, 23.0))

        # 4. Transport dead time FIFO (tau_dead_min)
        Q_dcc_delayed = self._buf_Q_dcc_cmd.pop(0)
        self._buf_Q_dcc_cmd.append(Q_dcc_cmd_eff)

        T_mau_sa_delayed = self._buf_T_mau_sa_cmd.pop(0)
        self._buf_T_mau_sa_cmd.append(T_mau_sa_cmd_eff)

        W_mau_delayed = self._buf_W_mau_cmd.pop(0)
        self._buf_W_mau_cmd.append(W_mau_exit_now)

        # 5. FFU + Lighting sensible heat gain inside zone
        P_ffu_kW = p.P_ffu_nom_kW * (ach_rec / p.ach_rec_nom) ** 3
        Q_aux_kW = P_ffu_kW + p.Q_lighting_kW

        # 6. Substep integration of Eqs. (3.12), (3.13), and (3.14)
        tau_coil_s = p.tau_coil_min * 60.0
        for _ in range(self.substeps):
            # Eq. (3.14): first-order coil lag on Q_dcc and T_mau_sa
            dQ_dcc = (Q_dcc_delayed - self.Q_dcc) / tau_coil_s
            dT_sa = (T_mau_sa_delayed - self.T_mau_sa) / tau_coil_s
            self.Q_dcc += dQ_dcc * dt_sub_s
            self.T_mau_sa += dT_sa * dt_sub_s

            # Eq. (3.12): Zone energy balance
            cp_z = float(cp_moist_air(self.W_z))
            Q_env = p.G_env_kW_K * (p.T_amb_adj_C - self.T_z)
            Q_mau_adv = p.m_da_MAU_kg_s * cp_z * (self.T_mau_sa - self.T_z)
            dTz_dt = (Q_env + Q_room_kW + Q_aux_kW + Q_mau_adv - self.Q_dcc) / p.C_eff_kJ_K
            self.T_z += dTz_dt * dt_sub_s

            # Eq. (3.13): Zone water-vapor balance (decoupled 2 ACH MAU stream)
            dWz_dt = (
                p.m_da_MAU_kg_s * (W_mau_delayed - self.W_z) + m_H2O_int_kg_s
            ) / p.M_da_kg
            self.W_z = max(1e-4, self.W_z + dWz_dt * dt_sub_s)

        # 7. Compute resulting zone psychrometrics & plant energy/water
        RH_z = float(rh_from_T_W(self.T_z, self.W_z))
        DP_z = float(dew_point_from_W(self.W_z))

        mau_res = compute_mau_loads(
            T_out_C=T_out_C,
            Td_out_C=Td_out_C,
            T_coil_C=self.T_mau_coil,
            T_mau_sa_C=self.T_mau_sa,
            params=p,
        )
        plant_res = compute_plant_power_and_water(
            Q_mau_tot_kW=mau_res["Q_mau_tot_kW"],
            Q_dcc_kW=self.Q_dcc,
            Q_reheat_kW=mau_res["Q_mau_reheat_kW"],
            ach_rec=ach_rec,
            T_chw_dcc_C=self.T_chw_dcc,
            standby_margin_frac=standby_margin_frac,
            scenario=scenario,
            params=p,
        )

        return {
            "T_z": self.T_z,
            "W_z": self.W_z,
            "RH_z": RH_z,
            "DP_z": DP_z,
            "Q_dcc_kW": self.Q_dcc,
            "Q_dcc_cap_kW": Q_dcc_cap,
            "T_mau_coil": self.T_mau_coil,
            "T_mau_sa": self.T_mau_sa,
            "T_chw_dcc": self.T_chw_dcc,
            "Q_mau_tot_kW": mau_res["Q_mau_tot_kW"],
            "m_cond_L_min": mau_res["m_cond_kg_s"] * 60.0,
            **plant_res,
        }
