"""
Case study definitions for STHE DRO optimization.

Case 1: Methanol (shell) / Seawater (tube)
Case 2: Kerosene (shell) / Crude oil (tube)

Operating specifications and thermophysical properties are taken from the
benchmark datasets of Caputo, Pelagagge & Salini (2008), "Heat exchanger
design based on economic optimisation", Applied Thermal Engineering 28,
1151-1159, as adopted in Lara-Montano et al. (2021, 2025). These are the
exact values tabulated in Table 1 (tab:case_specifications) of the manuscript
-- the code and the paper now share a single source of truth (fixes N1).

Thermophysical properties are evaluated once at the nominal mean stream
temperature and treated as deterministic constants, following Caputo (2008).
Over the +/-5% temperature uncertainty band the induced property variation is
small; this is stated and justified in the manuscript (addresses N3 / R2-M5).

Design/rating convention (fixes N2):
    - The design vector x fully fixes the geometry, INCLUDING tube length
      (12th variable). Hence the heat-transfer area A = A(x) depends only on
      the design, never on the operating scenario.
    - Under each scenario xi the achieved duty is obtained by epsilon-NTU
      RATING of the fixed unit (see sthe_model.py); the geometry is never
      re-sized per scenario.
    - The process specification is a target hot-outlet temperature
      T_h_out_target (fixed). The required duty therefore SCALES with the
      scenario: Q_req(xi) = m_h * cp_s * (T_h_in - T_h_out_target).
"""

import numpy as np

# ==============================================================================
# Case Study 1: Methanol (shell side, HOT) / Seawater (tube side, COLD)
# Caputo et al. (2008), Q_req = 4.34 MW
# ==============================================================================

CASE1 = {
    "name": "Case 1: Methanol / Seawater",

    # --- Nominal uncertain parameters xi_0 ---
    # xi = [T_h_in, T_c_in, m_h, m_c, R_f_t, R_f_s]
    # Temperatures in K, mass flows in kg/s, fouling in m^2*K/W
    "T_h_in_0": 95.0 + 273.15,    # Hot inlet (shell, methanol) [K]
    "T_c_in_0": 25.0 + 273.15,    # Cold inlet (tube, seawater) [K]
    "T_h_out_0": 40.0 + 273.15,   # Hot outlet TARGET (process spec) [K]
    "T_c_out_0": 40.0 + 273.15,   # Cold outlet at nominal [K] (reference only)
    "m_h_0": 27.80,               # Shell-side (methanol) mass flow [kg/s]
    "m_c_0": 68.90,               # Tube-side (seawater) mass flow [kg/s]
    "R_f_t_0": 2.00e-4,           # Tube-side fouling (seawater) [m^2*K/W]
    "R_f_s_0": 3.30e-4,           # Shell-side fouling (methanol) [m^2*K/W]

    # --- Fluid properties (shell side - methanol) at mean T ---
    "Den_s": 750.0,               # Density [kg/m^3]
    "Cp_s": 2840.0,               # Specific heat [J/(kg*K)]
    "Vis_s": 3.40e-4,             # Viscosity [Pa*s]
    "k_s": 0.19,                  # Thermal conductivity [W/(m*K)]

    # --- Fluid properties (tube side - seawater) at mean T ---
    "Den_t": 995.0,               # Density [kg/m^3]
    "Cp_t": 4200.0,               # Specific heat [J/(kg*K)]
    "Vis_t": 8.00e-4,             # Viscosity [Pa*s]
    "k_t": 0.59,                  # Thermal conductivity [W/(m*K)]

    # --- Tube wall ---
    "k_wall": 16.0,               # Tube wall conductivity [W/(m*K)] (stainless steel tubes)

    # --- Viscosity correlation params (wall correction, tube side) ---
    "A_vis": -11.6225,
    "B_vis": 1.949e3,
    "C_vis": 2.1641e-2,
    "D_vis": -1.599e-5,

    # --- Pressure drop limits [Pa] ---
    "dP_t_max": 70000.0,
    "dP_s_max": 70000.0,

    # --- Velocity limits [m/s] ---
    "v_t_min": 0.5,
    "v_t_max": 3.0,

    # --- L/D limit ---
    "L_D_max": 15.0,

    # --- Economic parameters ---
    "eff_pump": 0.85,
    "hours_year": 8000,
    "interest": 0.05,
    "n_years": 20.0,
    "C_p": 1.0,          # pressure-rating cost factor
    "C_t": 1.0,          # tube-type cost factor  (Case 1: C_T=1.0)
    "C_m": 1.7,          # material cost factor   (Case 1: C_M=1.7, CS shell / SS tubes)
    "cepci_act": 750,
    "cepci_ref": 394,
    "elec_price": 0.1,   # USD/kWh
}

# Compute derived quantities for Case 1
CASE1["Pr_s"] = CASE1["Cp_s"] * CASE1["Vis_s"] / CASE1["k_s"]
CASE1["Pr_t"] = CASE1["Cp_t"] * CASE1["Vis_t"] / CASE1["k_t"]
# Nominal required duty (fixed reference used for penalty normalization).
# The scenario-dependent requirement Q_req(xi) is computed inside sthe_model.
CASE1["Q_req"] = CASE1["m_h_0"] * CASE1["Cp_s"] * (CASE1["T_h_in_0"] - CASE1["T_h_out_0"])
CASE1["T_h_out_target"] = CASE1["T_h_out_0"]


# ==============================================================================
# Case Study 2: Kerosene (shell side, HOT) / Crude oil (tube side, COLD)
# Caputo et al. (2008), Q_req = 1.44 MW
# ==============================================================================

CASE2 = {
    "name": "Case 2: Kerosene / Crude oil",

    # --- Nominal uncertain parameters xi_0 ---
    "T_h_in_0": 199.0 + 273.15,   # Hot inlet (shell, kerosene) [K]
    "T_c_in_0": 37.8 + 273.15,    # Cold inlet (tube, crude oil) [K]
    "T_h_out_0": 93.3 + 273.15,   # Hot outlet TARGET (process spec) [K]
    "T_c_out_0": 76.7 + 273.15,   # Cold outlet at nominal [K] (reference only)
    "m_h_0": 5.52,                # Shell-side (kerosene) mass flow [kg/s]
    "m_c_0": 18.80,               # Tube-side (crude oil) mass flow [kg/s]
    "R_f_t_0": 6.10e-4,           # Tube-side fouling (crude) [m^2*K/W]
    "R_f_s_0": 6.10e-4,           # Shell-side fouling (kerosene) [m^2*K/W]

    # --- Fluid properties (shell side - kerosene) at mean T ---
    "Den_s": 850.0,
    "Cp_s": 2470.0,
    "Vis_s": 4.00e-4,
    "k_s": 0.13,

    # --- Fluid properties (tube side - crude oil) at mean T ---
    "Den_t": 995.0,
    "Cp_t": 2050.0,
    "Vis_t": 3.58e-3,
    "k_t": 0.13,

    # --- Tube wall ---
    "k_wall": 45.0,               # carbon steel (both sides)

    # --- Viscosity correlation params ---
    "A_vis": -11.6225,
    "B_vis": 1.949e3,
    "C_vis": 2.1641e-2,
    "D_vis": -1.599e-5,

    # --- Constraints ---
    "dP_t_max": 70000.0,
    "dP_s_max": 70000.0,
    "v_t_min": 0.5,
    "v_t_max": 3.0,
    "L_D_max": 15.0,

    # --- Economic parameters ---
    "eff_pump": 0.85,
    "hours_year": 8000,
    "interest": 0.05,
    "n_years": 20.0,
    "C_p": 1.0,
    "C_t": 1.6,          # tube-type cost factor  (Case 2: C_T=1.6)
    "C_m": 1.0,          # material cost factor   (Case 2: C_M=1.0, CS both)
    "cepci_act": 750,
    "cepci_ref": 394,
    "elec_price": 0.1,
}

CASE2["Pr_s"] = CASE2["Cp_s"] * CASE2["Vis_s"] / CASE2["k_s"]
CASE2["Pr_t"] = CASE2["Cp_t"] * CASE2["Vis_t"] / CASE2["k_t"]
CASE2["Q_req"] = CASE2["m_h_0"] * CASE2["Cp_s"] * (CASE2["T_h_in_0"] - CASE2["T_h_out_0"])
CASE2["T_h_out_target"] = CASE2["T_h_out_0"]


# ==============================================================================
# Case Study 3: Distilled water (shell, HOT) / Raw water (tube, COLD)
# Caputo et al. (2008) Table 1, Q_req ~ 0.42 MW (Caputo report 0.46 MW).
# Low driving force (dT_h = 4.5 C) -> large-area, water/water service.
# ==============================================================================

CASE3 = {
    "name": "Case 3: Distilled water / Raw water",

    "T_h_in_0": 33.9 + 273.15,    # Hot inlet (shell, distilled water) [K]
    "T_c_in_0": 23.9 + 273.15,    # Cold inlet (tube, raw water) [K]
    "T_h_out_0": 29.4 + 273.15,   # Hot outlet TARGET [K]
    "T_c_out_0": 26.7 + 273.15,   # Cold outlet at nominal [K] (reference)
    "m_h_0": 22.07,               # Shell-side (distilled) mass flow [kg/s]
    "m_c_0": 35.31,               # Tube-side (raw) mass flow [kg/s]
    "R_f_t_0": 1.70e-4,           # Tube-side fouling (raw water) [m^2*K/W]
    "R_f_s_0": 1.70e-4,           # Shell-side fouling (distilled) [m^2*K/W]

    # --- Fluid properties (shell side - distilled water) ---
    "Den_s": 995.0,
    "Cp_s": 4180.0,
    "Vis_s": 8.00e-4,
    "k_s": 0.62,

    # --- Fluid properties (tube side - raw water) ---
    "Den_t": 999.0,
    "Cp_t": 4180.0,
    "Vis_t": 9.20e-4,
    "k_t": 0.62,

    # --- Tube wall ---
    "k_wall": 45.0,               # carbon steel

    # --- Viscosity correlation params (wall correction) ---
    "A_vis": -11.6225,
    "B_vis": 1.949e3,
    "C_vis": 2.1641e-2,
    "D_vis": -1.599e-5,

    # --- Constraints ---
    "dP_t_max": 70000.0,
    "dP_s_max": 70000.0,
    "v_t_min": 0.5,
    "v_t_max": 3.0,
    "L_D_max": 15.0,

    # --- Economic parameters (Lara-Montano 2021 basis, as in C1/C2) ---
    "eff_pump": 0.85,
    "hours_year": 8000,
    "interest": 0.05,
    "n_years": 20.0,
    "C_p": 1.0,
    "C_t": 1.0,          # TODO: confirm material/type factors for C3 with coauthor
    "C_m": 1.0,          # carbon steel default (water/water service)
    "cepci_act": 750,
    "cepci_ref": 394,
    "elec_price": 0.1,
}

CASE3["Pr_s"] = CASE3["Cp_s"] * CASE3["Vis_s"] / CASE3["k_s"]
CASE3["Pr_t"] = CASE3["Cp_t"] * CASE3["Vis_t"] / CASE3["k_t"]
CASE3["Q_req"] = CASE3["m_h_0"] * CASE3["Cp_s"] * (CASE3["T_h_in_0"] - CASE3["T_h_out_0"])
CASE3["T_h_out_target"] = CASE3["T_h_out_0"]


# ==============================================================================
# Case Study 4 (NEW): Glycerol (shell, HOT) / Cooling water (tube, COLD)
# Biorefinery-relevant, viscous aqueous-organic service (glycerol = biodiesel
# co-product). Pure-glycerol properties at mean stream temperature (75 C) from
# standard tables: viscosity Segur & Oberstar (1951) ~40 cP; density/cp/k from
# glycerol property compilations (Glycerine Producers' Assoc.; Cheng 2008).
# High shell-side viscosity -> shell pressure-drop tail under flow drift, i.e.
# the regime where distributional robustness (WDRO) is expected to pay off.
#
# ENERGY BALANCE (nominal):
#   Q_h = m_h*cp_h*(T_h_in - T_h_out) = 10.0*2560*(100-50)   = 1.280 MW
#   Q_c = m_c*cp_c*(T_c_out - T_c_in) = 20.0*4180*(40.3-25.0) = 1.279 MW  (matches)
# T_c_out is the balance-consistent nominal cold outlet (an OUTPUT of the rating).
# NOTE: glycerol viscosity is strongly temperature-dependent (14.8 cP at 100 C
# to 142 cP at 50 C); the constant-mean-property assumption (following Caputo)
# is therefore weakest for this case and is flagged as a limitation.
# ==============================================================================

CASE4 = {
    "name": "Case 4: Glycerol / Cooling water",

    "T_h_in_0": 100.0 + 273.15,   # Hot inlet (shell, glycerol) [K]
    "T_c_in_0": 25.0 + 273.15,    # Cold inlet (tube, water) [K]
    "T_h_out_0": 50.0 + 273.15,   # Hot outlet TARGET [K]
    "T_c_out_0": 40.3 + 273.15,   # Cold outlet at nominal [K] (balance-consistent)
    "m_h_0": 10.00,               # Shell-side (glycerol) mass flow [kg/s]
    "m_c_0": 20.00,               # Tube-side (water) mass flow [kg/s]
    "R_f_t_0": 2.00e-4,           # Tube-side fouling (water) [m^2*K/W]
    "R_f_s_0": 3.50e-4,           # Shell-side fouling (glycerol) [m^2*K/W]

    # --- Fluid properties (shell side - glycerol) at mean T = 75 C ---
    "Den_s": 1228.0,
    "Cp_s": 2560.0,
    "Vis_s": 4.00e-2,             # 40 cP at 75 C (Segur & Oberstar 1951)
    "k_s": 0.286,

    # --- Fluid properties (tube side - cooling water) at mean T ~ 33 C ---
    "Den_t": 995.0,
    "Cp_t": 4180.0,
    "Vis_t": 7.70e-4,
    "k_t": 0.62,

    # --- Tube wall ---
    "k_wall": 45.0,

    # --- Viscosity correlation params (wall correction) ---
    "A_vis": -11.6225,
    "B_vis": 1.949e3,
    "C_vis": 2.1641e-2,
    "D_vis": -1.599e-5,

    # --- Constraints ---
    "dP_t_max": 70000.0,
    "dP_s_max": 70000.0,
    "v_t_min": 0.5,
    "v_t_max": 3.0,
    "L_D_max": 15.0,

    # --- Economic parameters ---
    "eff_pump": 0.85,
    "hours_year": 8000,
    "interest": 0.05,
    "n_years": 20.0,
    "C_p": 1.0,
    "C_t": 1.0,
    "C_m": 1.0,          # carbon steel
    "cepci_act": 750,
    "cepci_ref": 394,
    "elec_price": 0.1,
}

CASE4["Pr_s"] = CASE4["Cp_s"] * CASE4["Vis_s"] / CASE4["k_s"]
CASE4["Pr_t"] = CASE4["Cp_t"] * CASE4["Vis_t"] / CASE4["k_t"]
CASE4["Q_req"] = CASE4["m_h_0"] * CASE4["Cp_s"] * (CASE4["T_h_in_0"] - CASE4["T_h_out_0"])
CASE4["T_h_out_target"] = CASE4["T_h_out_0"]


# ==============================================================================
# R2-4b: temperature-dependent glycerol properties (Case 4 sensitivity)
# ==============================================================================

def glycerol_shell_props(T_K):
    """
    Temperature-dependent pure-glycerol properties for the R2-4b sensitivity
    variants of Case 4.

    Viscosity: Cheng (2008), Ind. Eng. Chem. Res. 47(9), 3285-3288, pure-
    glycerol limit with T in Celsius (output mPa*s -> Pa*s). Reproduces the
    Segur & Oberstar (1951) values used to define the base case within ~3%
    over 50-100 C: 39.1 cP at 75 C (base 40), 14.9 at 100 C (14.8), 146 at
    50 C (142).
    Density and heat capacity: linear in T anchored at the base-case 75 C
    values (1228 kg/m3, 2560 J/kg/K), slopes from the Glycerine Producers'
    Association tables (-0.66 kg/m3 per K, +4.3 J/kg/K per K).
    Conductivity: nearly constant for glycerol; 0.286 W/m/K with a +1e-4 per K
    slope.

    Returns (rho, cp, mu, k) in SI units.
    """
    T_C = float(T_K) - 273.15
    T_C = min(max(T_C, 20.0), 110.0)   # correlation validity clamp
    mu = 12100.0 * np.exp((-1233.0 + T_C) * T_C / (9900.0 + 70.0 * T_C)) * 1e-3
    rho = 1228.0 - 0.66 * (T_C - 75.0)
    cp = 2560.0 + 4.3 * (T_C - 75.0)
    k = 0.286 + 1.0e-4 * (T_C - 75.0)
    return rho, cp, mu, k


def make_case4_props_variant(mode):
    """
    Case-4 variants for the R2-4b property-sensitivity study.

    mode = 'scenarioT' : shell properties evaluated per scenario at the mean
                         shell temperature (model flag props_of_T);
    mode = 'const65' / 'const85' : constant shell properties re-evaluated at
                         an alternative reference temperature, bracketing the
                         lumped-property choice (65 C: ~63 cP; 85 C: ~26 cP)
                         against the base 75 C (~40 cP).
    """
    c = dict(CASE4)
    if mode == 'scenarioT':
        c['props_of_T'] = 'glycerol'
        c['name'] = CASE4['name'] + ' (scenario-T props)'
        return c
    T_ref = {'const65': 65.0, 'const85': 85.0}[mode]
    rho, cp, mu, k = glycerol_shell_props(T_ref + 273.15)
    c.update(Den_s=rho, Cp_s=cp, Vis_s=mu, k_s=k)
    c['Pr_s'] = cp * mu / k
    c['Q_req'] = c['m_h_0'] * cp * (c['T_h_in_0'] - c['T_h_out_0'])
    c['name'] = CASE4['name'] + f' (const props at {T_ref:.0f} C)'
    return c


# ==============================================================================
# Design variable bounds (shared across cases, TEMA-consistent)
# x = [D_s, selTub, F_pt, F_layout, F_Np, F_baffle_cut,
#      F_Lbc, F_Lbi, F_dsb, F_dtb, F_Dotl, L_tubes]
# NOTE (fixes N2): L_tubes (tube length, m) is now an explicit DESIGN variable,
# so the geometry -- and therefore the area A(x) -- is fully fixed by x and
# never re-sized per scenario. Dimension is 12.
# ==============================================================================

DESIGN_BOUNDS = {
    "lb": np.array([0.3, 1, 0, 0, 0, 0, 0.20, 1.0, 0.01, 0.01, 0.80, 1.0]),
    "ub": np.array([1.0, 21, 1, 3, 3, 4, 0.55, 1.6, 0.10, 0.10, 0.95, 12.0]),
    "names": [
        "D_s (shell diameter)",
        "selTub (tube OD index)",
        "F_pt (pitch ratio selector)",
        "F_layout (tube layout selector)",
        "F_Np (tube passes selector)",
        "F_baffle_cut (baffle cut selector)",
        "F_Lbc (central baffle spacing ratio)",
        "F_Lbi (inlet baffle spacing ratio)",
        "F_dsb (shell-baffle clearance ratio)",
        "F_dtb (tube-baffle clearance ratio)",
        "F_Dotl (tube bundle OD ratio)",
        "L_tubes (tube length, m)",
    ],
    "dim": 12,
}


# ==============================================================================
# Uncertainty bands (Eq. 42-44 in paper)
# ==============================================================================

# Uncertainty band widths.
# FIX (audit M6): inlet-temperature uncertainty is now an ABSOLUTE band in Kelvin
# (measurement/operational-level uncertainty), NOT a percentage of the absolute
# temperature. Applying +/-5% to an absolute Kelvin value is dimensionally
# arbitrary (it scales with the arbitrary Celsius/Kelvin offset) and produced
# +/-15-24 C swings that injected thermodynamically impossible (T_c_in > target)
# scenarios in the tight-approach Case 3. Flows and fouling keep multiplicative
# bands (physically meaningful for a rate and a growing resistance).
DT_BAND = 3.0        # absolute inlet-temperature half-width [K]  (confirm magnitude)
MFLOW_FRAC = 0.10    # +/-10% mass-flow band
RFOUL_FRAC = 0.50    # 0 .. +50% fouling band


def get_xi_bounds(case):
    """
    Return (xi_0, xi_lb, xi_ub) for the 6D uncertain vector.
    xi = [T_h_in, T_c_in, m_h, m_c, R_f_t, R_f_s]
    """
    xi_0 = np.array([
        case["T_h_in_0"],
        case["T_c_in_0"],
        case["m_h_0"],
        case["m_c_0"],
        case["R_f_t_0"],
        case["R_f_s_0"],
    ])

    # Per-case band overrides (used by the uncertainty-magnitude sweep, Phase 2).
    dt = float(case.get("dt_band", DT_BAND))
    mf = float(case.get("mflow_frac", MFLOW_FRAC))
    rf = float(case.get("rfoul_frac", RFOUL_FRAC))

    xi_lb = np.array([
        case["T_h_in_0"] - dt,
        case["T_c_in_0"] - dt,
        (1 - mf) * case["m_h_0"],
        (1 - mf) * case["m_c_0"],
        0.0,
        0.0,
    ])

    xi_ub = np.array([
        case["T_h_in_0"] + dt,
        case["T_c_in_0"] + dt,
        (1 + mf) * case["m_h_0"],
        (1 + mf) * case["m_c_0"],
        (1 + rf) * case["R_f_t_0"],
        (1 + rf) * case["R_f_s_0"],
    ])

    return xi_0, xi_lb, xi_ub


# ==============================================================================
# Experimental parameters
# ==============================================================================

EXPERIMENT_CONFIG = {
    "N_tr": 500,       # Training samples
    "N_te": 2000,      # Test samples (i.i.d.)
    "N_dep": 5000,     # Deployment samples per kappa
    "alpha": 0.95,     # Risk level for quantile/CVaR
    "kappa_ladder": [0.0, 0.25, 0.50, 0.75, 1.00],
    "N_pop": 50,       # Population size per solver
    "I_max": 50,       # Max iterations per solver
    "R_runs": 10,      # Independent runs per method-solver combo
    "rho_grid_size": 15,
    "rho_min": 0.01,
    "rho_max": 1.0,
    "lambda_penalty": 1e6,  # Penalty weight (raised for epsilon-NTU rating model:
                            # must exceed feasible TAC so violations never pay off)
    "inner_max_iter": 60,   # Inner Nelder-Mead iterations for WDRO
    "inner_tol": 1e-4,
}
