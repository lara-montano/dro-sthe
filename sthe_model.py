"""
Parametric Shell-and-Tube Heat Exchanger (STHE) model.

Bell-Delaware method for shell-side rating and pressure drop, combined with an
epsilon-NTU RATING of a geometrically fixed unit.

Design/rating convention (fixes audit item N2):
    - The design vector x fully fixes the geometry, INCLUDING tube length
      (x[11], m). The heat-transfer area therefore satisfies A = A(x): it is a
      function of the design ONLY and is never re-sized per scenario.
    - Given the operating scenario xi, the achieved duty Q(x,xi) is obtained by
      epsilon-NTU rating of the fixed unit. No per-scenario geometry solve.
    - The process specification is a target hot-outlet temperature
      T_h_out_target (from the case). The required duty scales with the
      scenario:  Q_req(xi) = m_h * cp_s * (T_h_in - T_h_out_target).
      The duty constraint is  g_Q = Q_req(xi) - Q(x,xi) <= 0, i.e. the fixed
      unit must remove at least enough heat to meet the cooling spec under xi.

Thermophysical properties are evaluated once at the nominal mean stream
temperature and treated as constants (following Caputo 2008); see config.

References:
    - Serna & Jimenez (2005) compact Bell-Delaware formulation
    - Lara-Montano et al. (2021) metaheuristic benchmarking
    - TEMA Standards (2019)
"""

import math
import numpy as np
from tube_diameters import select_tube
from config_case_studies import glycerol_shell_props


# =============================================================================
# Geometry helper functions (Bell-Delaware)
# =============================================================================

def cal_pitch(d_o, F_p_t):
    """Tube pitch from OD and pitch ratio selector."""
    if F_p_t == 1:
        return 1.25 * d_o
    elif F_p_t == 2:
        return 1.5 * d_o
    else:
        return 1.25 * d_o


def decode_discrete(F_p_t, F_layout, F_Np, F_baffle_cut):
    """Decode continuous decision variables into discrete STHE parameters."""
    pitch_sel = 1 if F_p_t <= 0.5 else 2

    if F_layout <= 1:
        layout = 30
    elif F_layout <= 2:
        layout = 45
    else:
        layout = 90

    if F_Np <= 1:
        Np = 1
    elif F_Np <= 2:
        Np = 2
    else:
        Np = 4

    if F_baffle_cut <= 1:
        bc = 25
    elif F_baffle_cut <= 2:
        bc = 30
    elif F_baffle_cut <= 3:
        bc = 40
    else:
        bc = 45

    return pitch_sel, layout, Np, bc


def cal_A_o_cr(layout, D_s, D_otl, D_ctl, d_o, L_b_c, p_t):
    """Cross-flow area at shell centerline."""
    if layout == 30 or layout == 90:
        X_t = p_t
        A_o_cr = (D_s - D_otl + (D_ctl / X_t) * (X_t - d_o)) * L_b_c
        Coef_t = 0.866 if layout == 30 else 1.0
    elif layout == 45:
        X_t = math.sqrt(2.0) * p_t
        Coef_t = 1.0
        if (p_t / d_o) >= 1.707:
            A_o_cr = (D_s - D_otl + (D_ctl / X_t) * (X_t - d_o)) * L_b_c
        else:
            A_o_cr = (D_s - D_otl + (2.0 * D_ctl / X_t) * (p_t - d_o)) * L_b_c
    elif layout == 60:
        Coef_t = 0.866
        X_t = math.sqrt(3.0) * p_t
        if (p_t / d_o) >= 3.732:
            A_o_cr = (D_s - D_otl + (D_ctl / X_t) * (X_t - d_o)) * L_b_c
        else:
            A_o_cr = (D_s - D_otl + (2.0 * D_ctl / X_t) * (p_t - d_o)) * L_b_c
    else:
        X_t = p_t
        A_o_cr = (D_s - D_otl + (D_ctl / X_t) * (X_t - d_o)) * L_b_c
        Coef_t = 1.0
    return max(A_o_cr, 1e-8), Coef_t


def longitudinal_pitch(p_t, layout):
    """Longitudinal pitch X_l."""
    if layout == 30:
        return (math.sqrt(3.0) / 2.0) * p_t
    elif layout == 60:
        return p_t / 2.0
    elif layout == 90:
        return p_t
    elif layout == 45:
        return p_t / math.sqrt(2.0)
    return p_t


# Bell-Delaware banded curve-fit coefficients (Taborek tables, as compiled by
# Shah & Sekulic and the Serna-Jimenez compact formulation). Each row is
# (upper Re limit, a1, a2, a3, a4, b1, b2, b3, b4); bands are selected by
# Re_s <= limit, i.e. rows cover Re <10, 10-10^2, 10^2-10^3, 10^3-10^4, >10^4.
_JF_RANGES = {
    30: [
        (10, 1.4, -0.667, 1.45, 0.519, 48.0, -1.0, 7.0, 0.5),
        (100, 1.36, -0.657, 1.45, 0.519, 45.1, -0.973, 7.0, 0.5),
        (1000, 0.593, -0.477, 1.45, 0.519, 4.57, -0.476, 7.0, 0.5),
        (10000, 0.321, -0.388, 1.45, 0.519, 0.486, -0.152, 7.0, 0.5),
        (1e30, 0.321, -0.388, 1.45, 0.519, 0.372, -0.123, 7.0, 0.5),
    ],
    45: [
        (10, 1.55, -0.667, 1.93, 0.5, 32.0, -1.0, 6.59, 0.52),
        # AUDIT R2-4a: a1 = 1.498 per the published table (Taborek; Shah &
        # Sekulic Table 8.6). The R1 code had 0.498 (dropped leading digit),
        # a -67%/+200% j discontinuity at Re_s = 10/100 that under-rated
        # 45-deg layouts by ~3x over 10 < Re_s < 100 (the Case-4 shell-Re
        # range). With 1.498 both band edges are continuous within 1%.
        (100, 1.498, -0.656, 1.93, 0.5, 26.2, -0.913, 6.59, 0.52),
        (1000, 0.73, -0.50, 1.93, 0.5, 3.5, -0.476, 6.59, 0.52),
        (10000, 0.37, -0.396, 1.93, 0.5, 0.333, -0.136, 6.59, 0.52),
        (1e30, 0.37, -0.396, 1.93, 0.5, 0.303, -0.126, 6.59, 0.52),
    ],
    90: [
        (10, 0.97, -0.667, 1.187, 0.37, 35.0, -1.0, 6.3, 0.378),
        (100, 0.9, -0.631, 1.187, 0.37, 32.1, -0.963, 6.3, 0.378),
        (1000, 0.408, -0.46, 1.187, 0.37, 6.09, -0.602, 6.3, 0.378),
        (10000, 0.107, -0.266, 1.187, 0.37, 0.0815, 0.022, 6.3, 0.378),
        # AUDIT R2-4a: b2 = -0.148 per the published table (Taborek; Shah &
        # Sekulic Table 8.6). The R1 code had -0.418 (transposed digits),
        # which made the 90-deg friction factor jump ~12x when Re_s crossed
        # 1e4 downward -- the spurious "regime boundary" that dominated the
        # Case-3 nominal response. With -0.148 the band edges are continuous
        # (f = 0.100 on both sides of Re_s = 1e4), as in the other layouts.
        (1e30, 0.37, -0.395, 1.187, 0.37, 0.391, -0.148, 6.3, 0.378),
    ],
}

# R1 coefficient sets for A/B attribution (legacy_r1_model): the transposed
# 90-deg top-band friction exponent and the dropped leading digit in the
# 45-deg 10<Re<100 j coefficient.
_JF_RANGES_LEGACY = {
    45: [_JF_RANGES[45][0],
         (100, 0.498, -0.656, 1.93, 0.5, 26.2, -0.913, 6.59, 0.52)] +
        list(_JF_RANGES[45][2:]),
    90: list(_JF_RANGES[90][:-1]) + [
        (1e30, 0.37, -0.395, 1.187, 0.37, 0.391, -0.418, 6.3, 0.378)],
}


def j_colburn_params(layout, Re_s):
    """Bell-Delaware j-factor and friction factor parameters."""
    Re_s = max(Re_s, 1.0)
    ranges = _JF_RANGES.get(layout, _JF_RANGES[30])
    a1, a2, a3, a4 = 0.321, -0.388, 1.45, 0.519
    b1, b2, b3, b4 = 0.372, -0.123, 7.0, 0.5

    for (lim, _a1, _a2, _a3, _a4, _b1, _b2, _b3, _b4) in ranges:
        if Re_s <= lim:
            a1, a2, a3, a4 = _a1, _a2, _a3, _a4
            b1, b2, b3, b4 = _b1, _b2, _b3, _b4
            break

    a = a3 / (1.0 + 0.14 * Re_s ** a4)
    return a, a1, a2, a3, a4, b1, b2, b3, b4


# ---------------------------------------------------------------------------
# R2-4a: smoothed-regime variant of the banded correlations
# ---------------------------------------------------------------------------

SMOOTH_W = 1.2   # blending window: Re in [lim/W, lim*W] around each boundary
                 # (windows of adjacent boundaries do not overlap for W=1.2)


def _smoothstep(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _jf_eval_row(row, Re_s, ptr):
    """Colburn j and ideal-bank friction factor from one coefficient row."""
    _, a1, a2, a3, a4, b1, b2, b3, b4 = row
    a = a3 / (1.0 + 0.14 * Re_s ** a4)
    b = b3 / (1.0 + 0.14 * Re_s ** b4)
    j = a1 * ((1.33 / max(ptr, 1.01)) ** a) * (Re_s ** a2)
    f = b1 * ((1.33 / max(ptr, 1.01)) ** b) * (Re_s ** b2)
    return j, f


def _blend_log(v_lo, v_hi, Re, lim, W=SMOOTH_W):
    """Log-space smoothstep blend of two positive values across Re = lim."""
    t = (math.log(Re) - math.log(lim / W)) / (2.0 * math.log(W))
    s = _smoothstep(t)
    return math.exp((1.0 - s) * math.log(max(v_lo, 1e-30)) +
                    s * math.log(max(v_hi, 1e-30)))


def bell_delaware_jf(layout, Re_s, ptr, smooth=False, legacy90=False):
    """
    Colburn j and ideal-bank friction factor of the Bell-Delaware method.

    smooth=False reproduces the banded coefficient tables exactly (base
    model). smooth=True (R2-4a sensitivity variant) replaces each hard band
    switch by a smoothstep blend of log j / log f between the adjacent bands
    within Re in [lim/SMOOTH_W, lim*SMOOTH_W] of the boundary.
    legacy90=True restores the R1 coefficient typos (90-deg friction
    exponent, 45-deg low-Re j coefficient) for A/B attribution.
    """
    Re_s = max(Re_s, 1.0)
    if legacy90 and layout in _JF_RANGES_LEGACY:
        ranges = _JF_RANGES_LEGACY[layout]
    else:
        ranges = _JF_RANGES.get(layout, _JF_RANGES[30])
    if smooth:
        for b_idx in range(len(ranges) - 1):
            lim = ranges[b_idx][0]
            if lim / SMOOTH_W <= Re_s <= lim * SMOOTH_W:
                j_lo, f_lo = _jf_eval_row(ranges[b_idx], Re_s, ptr)
                j_hi, f_hi = _jf_eval_row(ranges[b_idx + 1], Re_s, ptr)
                return (_blend_log(j_lo, j_hi, Re_s, lim),
                        _blend_log(f_lo, f_hi, Re_s, lim))
    for row in ranges:
        if Re_s <= row[0]:
            return _jf_eval_row(row, Re_s, ptr)
    return _jf_eval_row(ranges[-1], Re_s, ptr)


def vis_wall(Tw, A=-11.6225, B=1.949e3, C=2.1641e-2, D=-1.599e-5):
    """Wall viscosity correlation."""
    Tw = max(Tw, 200.0)
    return (10.0 ** (A + B / Tw + C * Tw + D * Tw ** 2.0)) * 0.001


# =============================================================================
# Tube-side heat transfer coefficients
# =============================================================================

def h_tube_laminar(L, Vis_t, d_i, Den_t, k_t, Pr_t, v):
    Re = max(Den_t * v * d_i / Vis_t, 1.0)
    return 1.86 * (k_t / d_i) * (Re * Pr_t * (d_i / max(L, 0.01))) ** (1.0 / 3.0)


def h_tube_transition(L, Vis_t, d_i, Den_t, k_t, Pr_t, v):
    Re = max(Den_t * v * d_i / Vis_t, 1.0)
    h1 = Re ** (2.0 / 3.0)
    h2 = 0.116 * (k_t / d_i) * (h1 - 125.0)
    h3 = (1 + (d_i / max(L, 0.01)) ** (2.0 / 3.0)) * Pr_t ** (1.0 / 3.0)
    return max(h2 * h3, 1.0)


def h_tube_turbulent(Vis_w, Vis_t, k_t, d_i, Re_t, Pr_t):
    Re_t = max(Re_t, 1.0)
    return 0.023 * (k_t / d_i) * (Re_t ** 0.8) * (Pr_t ** 0.4) * (Vis_t / max(Vis_w, 1e-10)) ** 0.14


# =============================================================================
# epsilon-NTU effectiveness
# =============================================================================

def effectiveness_ntu(NTU, Cr, multi_tube_pass):
    """
    Effectiveness of a single-shell-pass STHE.

    multi_tube_pass=False -> pure counterflow (1 tube pass, Np=1).
    multi_tube_pass=True  -> 1 shell pass, 2n tube passes (TEMA E, 1-2 relation;
                             standard approximation for Np in {2,4}).

    Cr = C_min / C_max in [0, 1]; NTU = UA / C_min.
    """
    NTU = max(NTU, 0.0)
    Cr = min(max(Cr, 0.0), 1.0)

    if not multi_tube_pass:
        # Counterflow
        if abs(Cr - 1.0) < 1e-6:
            return NTU / (1.0 + NTU)
        e = math.exp(-NTU * (1.0 - Cr))
        return (1.0 - e) / max(1.0 - Cr * e, 1e-12)

    # One shell pass, 2n tube passes
    root = math.sqrt(1.0 + Cr * Cr)
    ex = math.exp(-NTU * root)  # in (0, 1]
    ratio = (1.0 + ex) / max(1.0 - ex, 1e-12)
    eps1 = 2.0 / (1.0 + Cr + root * ratio)
    return min(max(eps1, 0.0), 1.0)


# =============================================================================
# Main parametric STHE model (epsilon-NTU rating of fixed geometry)
# =============================================================================

def sthe_model(x, xi, case):
    """
    Rate a geometrically fixed STHE (design x) at operating point xi.

    Returns dict with keys including:
      - 'feasible_core': based on (g_Q,g_t,g_s) only (paper feasibility)
      - 'feasible_full': includes extra checks (v_t, L/D, etc.)
      - 'feasible'     : alias of feasible_full (compat)
    """
    # Unpack uncertain parameters
    T_h_in = float(xi[0])
    T_c_in = float(xi[1])
    m_h = float(xi[2])
    m_c = float(xi[3])
    R_f_t = float(xi[4])
    R_f_s = float(xi[5])

    # --- Robust admissibility check (prevents weird drift edge cases) ---
    if (T_h_in <= T_c_in) or (m_h <= 0.0) or (m_c <= 0.0) or (R_f_t < 0.0) or (R_f_s < 0.0):
        big = 1e8
        return {
            'TAC': big, 'Q': 0.0, 'Q_req': float(case["Q_req"]),
            'Q_req_xi': float(case["Q_req"]), 'dP_t': big, 'dP_s': big,
            'Area': 0.0, 'L_tubes': 0.0, 'U': 1.0, 'v_t': 0.0, 'L_D': big,
            'N_tubes': 0, 'T_h_out': T_h_in, 'T_c_out': T_c_in,
            'g_Q': float(case["Q_req"]), 'g_t': big, 'g_s': big,
            'feasible_core': False, 'feasible_full': False, 'feasible': False,
            'v_ok': False, 'ld_ok': False,
        }

    # Unpack case properties (constant, at nominal mean stream temperature)
    Cp_s = case["Cp_s"]
    Cp_t = case["Cp_t"]
    Den_s = case["Den_s"]
    Den_t = case["Den_t"]
    Vis_s = case["Vis_s"]
    Vis_t = case["Vis_t"]
    k_s = case["k_s"]
    k_t = case["k_t"]
    k_wall = case["k_wall"]
    Pr_s = case["Pr_s"]
    Pr_t = case["Pr_t"]
    A_vis = case["A_vis"]
    B_vis = case["B_vis"]
    C_vis = case["C_vis"]
    D_vis = case["D_vis"]
    Q_req_nom = case["Q_req"]
    T_h_out_target = case.get("T_h_out_target", case["T_h_out_0"])
    dP_t_max = case["dP_t_max"]
    dP_s_max = case["dP_s_max"]

    # R2-4a sensitivity variant: smoothstep blending across the Reynolds-band
    # switches of the banded correlations (default off = base model).
    smooth = bool(case.get("smooth_regimes", False))

    # R2-4b sensitivity variant: scenario-dependent shell-side properties for
    # the glycerol service, evaluated at the scenario mean shell temperature
    # (same mean-temperature convention as the base model, which uses the
    # NOMINAL mean; only the scenario dependence is added). The penalty
    # normalization Q_req_nom stays fixed at the nominal value.
    if case.get("props_of_T") == 'glycerol':
        T_ref_s = 0.5 * (T_h_in + T_h_out_target)
        Den_s, Cp_s, Vis_s, k_s = glycerol_shell_props(T_ref_s)
        Pr_s = Cp_s * Vis_s / k_s

    # ----------------------------------------------------------------------
    # Unpack design variables with clipping. x[11] = tube length (FIXED geom).
    # ----------------------------------------------------------------------
    D_s = np.clip(x[0], 0.3, 1.0)
    selTub = int(np.clip(x[1], 1, 21))
    F_p_t = np.clip(x[2], 0, 1)
    F_layout = np.clip(x[3], 0, 3)
    F_Np = np.clip(x[4], 0, 3)
    F_baffle_cut = np.clip(x[5], 0, 4)
    F_L_b_c = np.clip(x[6], 0.20, 0.55)
    F_L_b_i = np.clip(x[7], 1.0, 1.6)
    F_delta_sb = np.clip(x[8], 0.01, 0.10)
    F_delta_tb = np.clip(x[9], 0.01, 0.10)
    F_D_otl = np.clip(x[10], 0.80, 0.95)
    L_tubes = float(np.clip(x[11] if len(x) > 11 else 4.0, 1.0, 12.0))

    pitch_sel, layout, Np, baffle_cut = decode_discrete(F_p_t, F_layout, F_Np, F_baffle_cut)

    diam = select_tube(selTub)
    d_i = diam[1] * 0.0254
    d_o = diam[0] * 0.0254

    L_b_c = F_L_b_c * D_s
    L_b_i = F_L_b_i * F_L_b_c * D_s
    L_b_o = L_b_i
    delta_sb = F_delta_sb * D_s
    delta_tb = F_delta_tb * d_o
    D_otl = F_D_otl * (D_s - delta_sb)

    p_t = cal_pitch(d_o, pitch_sel)
    D_ctl = D_otl - d_o
    l_c = (baffle_cut / 100.0) * D_s

    A_o_cr, Coef_t = cal_A_o_cr(layout, D_s, D_otl, D_ctl, d_o, L_b_c, p_t)

    # Number of tubes (geometry only) and FIXED area A = A(x)
    N_t = max(int(round((math.pi / 4.0) * D_ctl ** 2 / (Coef_t * p_t ** 2))), 1)
    Area = math.pi * d_o * N_t * L_tubes

    # ----------------------------------------------------------------------
    # Shell-side ideal coefficient (Bell-Delaware)
    # ----------------------------------------------------------------------
    G_s = m_h / A_o_cr
    Re_s = max((G_s * d_o) / Vis_s, 1.0)

    legacy90 = bool(case.get("bd90_f_legacy", False) or
                    case.get("legacy_r1_model", False))
    j_col, f_id = bell_delaware_jf(layout, Re_s, p_t / d_o, smooth=smooth,
                                   legacy90=legacy90)
    h_s_i = j_col * (m_h * Cp_s * Pr_s ** (-2.0 / 3.0)) / A_o_cr

    # --- Bell-Delaware correction factors (geometry + Re_s only) ---

    # J_c baffle cut correction
    F_w = 0.0
    theta_ctl = 0.0
    if abs(D_ctl) < 1e-10:
        J_c = 0.55
    else:
        cos_arg = np.clip((D_s - 2 * l_c) / max(D_ctl, 1e-6), -1, 1)
        theta_ctl = 2.0 * math.acos(cos_arg)
        F_w = theta_ctl / (2 * math.pi) - math.sin(theta_ctl) / (2 * math.pi)
        F_c = 1.0 - 2.0 * F_w
        J_c = 0.55 + 0.72 * F_c

    # J_l leakage correction
    theta_b_arg = np.clip(1.0 - (2.0 * l_c) / max(D_s, 1e-6), -1, 1)
    theta_b = 2.0 * math.acos(theta_b_arg)
    A_o_sb = math.pi * D_s * (delta_sb / 2.0) * (1 - theta_b / (2 * math.pi))
    A_o_tb = (math.pi / 4.0) * ((d_o + delta_tb) ** 2 - d_o ** 2) * N_t * (1 - F_w)
    A_o_tb = max(A_o_tb, 1e-10)
    A_o_sb = max(A_o_sb, 1e-10)
    r_s = A_o_sb / (A_o_sb + A_o_tb)
    r_lm = (A_o_sb + A_o_tb) / A_o_cr
    J_l = 0.44 * (1 - r_s) + (1 - 0.44 * (1 - r_s)) * math.exp(-2.2 * r_lm)

    # J_b bypass correction
    w_p = 0.05 * D_s
    A_o_bp = (D_s - D_otl + 0.5 * Np * w_p) * L_b_c
    r_b = A_o_bp / A_o_cr
    N_ss = 2.0
    X_l = longitudinal_pitch(p_t, layout)
    N_r_ss = max((D_s - 2.0 * l_c) / max(X_l, 1e-6), 1e-6)
    N_ss_plus = N_ss / N_r_ss
    if smooth and (100.0 / SMOOTH_W) <= Re_s <= (100.0 * SMOOTH_W):
        t_fac = (math.log(Re_s) - math.log(100.0 / SMOOTH_W)) / (2.0 * math.log(SMOOTH_W))
        FacC = 1.35 + _smoothstep(t_fac) * (1.25 - 1.35)
    else:
        FacC = 1.35 if Re_s <= 100 else 1.25

    if N_ss_plus >= 0.5:
        J_b = 1.0
    else:
        J_b = math.exp(-FacC * r_b * (1 - (2.0 * N_ss_plus) ** (1.0 / 3.0)))

    # Baffle counts and row counts (geometry, with FIXED tube length)
    N_b = max(((L_tubes - L_b_i - L_b_o) / max(L_b_c, 1e-6)) + 1, 1)
    N_r_cc = max((D_s - 2 * l_c) / max(X_l, 1e-6), 1)
    N_r_cw = max((0.8 / max(X_l, 1e-6)) * (l_c - 0.5 * (D_s - D_ctl)), 0.1)
    N_r_c = N_r_cc + N_r_cw

    # J_r laminar correction.
    # AUDIT R2-4a: the Bell-Delaware method specifies J_r = (10/N_r_c)^0.18
    # for Re_s <= 20, J_r = 1 for Re_s >= 100, and LINEAR interpolation in
    # Re_s between 20 and 100 (Taborek; Shah & Sekulic). The R1 code set
    # J_r = 1 over the whole 20-100 band (a step at Re_s = 20); the corrected
    # default follows the literature. jr_step_legacy=True reproduces the R1
    # behavior for A/B attribution.
    if Re_s >= 100:
        J_r = 1.0
    elif case.get("jr_step_legacy", False) or case.get("legacy_r1_model", False):
        J_r = (10.0 / max(N_r_c, 1)) ** 0.18 if Re_s <= 20 else 1.0
    else:
        J_r20 = (10.0 / max(N_r_c, 1)) ** 0.18
        if Re_s <= 20:
            J_r = J_r20
        else:
            J_r = J_r20 + (Re_s - 20.0) / 80.0 * (1.0 - J_r20)

    # J_s unequal baffle spacing correction
    L_i_plus = 1.0
    L_o_plus = L_b_o / max(L_b_c, 1e-6)
    if smooth and (5000.0 / SMOOTH_W) <= Re_s <= (5000.0 * SMOOTH_W):
        t_ns = (math.log(Re_s) - math.log(5000.0 / SMOOTH_W)) / (2.0 * math.log(SMOOTH_W))
        n_exp = (1.0 / 3.0) + _smoothstep(t_ns) * (0.6 - 1.0 / 3.0)
    else:
        n_exp = 1.0 / 3.0 if Re_s < 5000 else 0.6
    J_s_denom = max(N_b - 1 + L_i_plus + L_o_plus, 1e-6)
    J_s = (N_b - 1 + L_i_plus ** (1 - n_exp) + L_o_plus ** (1 - n_exp)) / J_s_denom

    h_s = max(h_s_i * J_b * J_c * J_l * J_r * J_s, 1.0)

    # ----------------------------------------------------------------------
    # Tube-side coefficient (light fixed-point on wall temperature for the
    # turbulent viscosity correction; geometry is fixed so this is cheap).
    # ----------------------------------------------------------------------
    A_per_tube = (math.pi * d_i ** 2) / 4.0
    tubes_per_pass = max(N_t / Np, 1)
    A_flow_t = A_per_tube * tubes_per_pass
    v_t = m_c / max(A_flow_t * Den_t, 1e-10)
    Re_t = max(Den_t * v_t * d_i / Vis_t, 1.0)

    # Mean stream temperatures (target-based; properties are constant anyway)
    T_med_s = 0.5 * (T_h_in + T_h_out_target)
    T_med_t = 0.5 * (T_c_in + case.get("T_c_out_0", T_c_in + 10.0))
    Tw = 0.5 * (T_med_s + T_med_t)

    h_t = 1.0
    for _ in range(6):
        Vis_w_val = vis_wall(Tw, A_vis, B_vis, C_vis, D_vis)
        if smooth and (2100.0 / SMOOTH_W) <= Re_t <= (2100.0 * SMOOTH_W):
            h_lam = h_tube_laminar(L_tubes, Vis_t, d_i, Den_t, k_t, Pr_t, v_t)
            h_tra = h_tube_transition(L_tubes, Vis_t, d_i, Den_t, k_t, Pr_t, v_t)
            h_t = _blend_log(max(h_lam, 1.0), max(h_tra, 1.0), Re_t, 2100.0)
        elif smooth and (1e5 / SMOOTH_W) <= Re_t <= (1e5 * SMOOTH_W):
            h_tra = h_tube_transition(L_tubes, Vis_t, d_i, Den_t, k_t, Pr_t, v_t)
            h_tur = h_tube_turbulent(Vis_w_val, Vis_t, k_t, d_i, Re_t, Pr_t)
            h_t = _blend_log(max(h_tra, 1.0), max(h_tur, 1.0), Re_t, 1e5)
        elif Re_t <= 2100:
            h_t = h_tube_laminar(L_tubes, Vis_t, d_i, Den_t, k_t, Pr_t, v_t)
        elif Re_t < 1e5:
            h_t = h_tube_transition(L_tubes, Vis_t, d_i, Den_t, k_t, Pr_t, v_t)
        else:
            h_t = h_tube_turbulent(Vis_w_val, Vis_t, k_t, d_i, Re_t, Pr_t)
        h_t = max(h_t, 1.0)

        Tw_new = (h_t * T_med_t + h_s * (d_o / d_i) * T_med_s) / max(h_t + h_s * (d_o / d_i), 1e-6)
        if abs(Tw_new - Tw) / max(abs(Tw_new), 1e-6) < 1e-3:
            Tw = Tw_new
            break
        Tw = Tw_new

    # ----------------------------------------------------------------------
    # Overall coefficient (referenced to outside area) and UA
    # ----------------------------------------------------------------------
    R_wall = (d_o * math.log(max(d_o / d_i, 1.001))) / (2.0 * k_wall)
    U = 1.0 / (1.0 / h_s + R_f_s + R_wall + R_f_t * (d_o / d_i) + d_o / (h_t * d_i))
    U = max(U, 1.0)
    UA = U * Area

    # ----------------------------------------------------------------------
    # epsilon-NTU RATING (fixed geometry) -> achieved duty
    # ----------------------------------------------------------------------
    C_h = m_h * Cp_s
    C_c = m_c * Cp_t
    C_min = min(C_h, C_c)
    C_max = max(C_h, C_c)
    Cr = C_min / max(C_max, 1e-12)
    NTU = UA / max(C_min, 1e-12)

    eps = effectiveness_ntu(NTU, Cr, multi_tube_pass=(Np >= 2))
    Q_max = C_min * (T_h_in - T_c_in)
    Q_achieved = eps * Q_max

    T_h_out = T_h_in - Q_achieved / max(C_h, 1e-12)
    T_c_out = T_c_in + Q_achieved / max(C_c, 1e-12)

    # Scenario-dependent required duty (target hot-outlet spec)
    Q_req_xi = m_h * Cp_s * (T_h_in - T_h_out_target)

    # ----------------------------------------------------------------------
    # Pressure drops
    # ----------------------------------------------------------------------
    # Tube-side (regime-consistent friction factor, audit m4: laminar 16/Re
    # below Re=2100, Blasius turbulent above; matches the h_t regime switch)
    if smooth and (2100.0 / SMOOTH_W) <= Re_t <= (2100.0 * SMOOTH_W):
        f_t = _blend_log(16.0 / max(Re_t, 1.0), 0.046 * Re_t ** (-0.2),
                         Re_t, 2100.0)
    elif Re_t <= 2100.0:
        f_t = 16.0 / max(Re_t, 1.0)
    else:
        f_t = 0.046 * Re_t ** (-0.2)
    dP_t = Np * ((4.0 * f_t * L_tubes / d_i) + 2.5) * (Den_t * v_t ** 2 / 2.0)

    # Shell-side (Bell-Delaware); f_id computed with j_col above (same band,
    # smoothed across band boundaries when smooth_regimes is set)
    dPb_id = (4 * f_id * G_s ** 2 * N_r_cc) / (2 * Den_s)

    Afr_w = (math.pi * D_s ** 2 / 4.0) * (theta_b / (2 * math.pi) - math.sin(theta_b) / (2 * math.pi))
    Afr_t = (math.pi * d_o ** 2 * F_w * N_t) / 4.0
    A_o_w = max(Afr_w - Afr_t, 1e-8)

    dPw_id = (2.0 + 0.6 * N_r_cw) * (m_h / (A_o_cr * Den_s)) * (m_h / (A_o_w * Den_s))

    if N_ss_plus < 0.5:
        Sig_b = math.exp(-3.7 * r_b * (1 - (2.0 * N_ss) ** (1.0 / 3.0)))
    else:
        Sig_b = 1.0

    p_pres = -0.15 * (1 + r_s) + 0.8
    r_lm_safe = max(r_lm, 1e-10)
    Sig_l = math.exp(-1.33 * (1 + r_s) * r_lm_safe ** p_pres)
    Sig_s = (L_b_c / max(L_b_o, 1e-6)) ** 1.8 + (L_b_c / max(L_b_i, 1e-6)) ** 1.8

    N_b_int = max(int(round(N_b)), 1)
    dP_s = ((N_b_int - 1) * dPb_id * Sig_b + N_b_int * dPw_id) * Sig_l + \
           2 * dPb_id * (1 + N_r_cw / max(N_r_cc, 1e-6)) * Sig_b * Sig_s

    L_D = L_tubes / max(D_s, 1e-6)

    # ----------------------------------------------------------------------
    # Cost
    # ----------------------------------------------------------------------
    TAC = compute_TAC(Area, dP_t, dP_s, v_t, m_h, m_c, Den_s, Den_t, L_D, case)

    # ----------------------------------------------------------------------
    # Constraints (positive = violation)
    # ----------------------------------------------------------------------
    g_Q = Q_req_xi - Q_achieved     # duty shortfall vs. target cooling spec
    g_t = dP_t - dP_t_max
    g_s = dP_s - dP_s_max

    feasible_core = (g_Q <= 0) and (g_t <= 0) and (g_s <= 0)

    v_ok = (v_t >= case.get("v_t_min", 0.5)) and (v_t <= case.get("v_t_max", 3.0))
    ld_ok = (L_D <= case.get("L_D_max", 15.0))

    feasible_full = feasible_core and v_ok and ld_ok

    return {
        'TAC': float(TAC),
        'Q': float(Q_achieved),
        'Q_req': float(Q_req_nom),        # nominal (penalty normalization)
        'Q_req_xi': float(Q_req_xi),      # scenario-dependent requirement
        'dP_t': float(dP_t),
        'dP_s': float(dP_s),
        'Area': float(Area),
        'L_tubes': float(L_tubes),
        'U': float(U),
        'v_t': float(v_t),
        'L_D': float(L_D),
        'N_tubes': int(N_t),
        'T_h_out': float(T_h_out),
        'T_c_out': float(T_c_out),
        'g_Q': float(g_Q),
        'g_t': float(g_t),
        'g_s': float(g_s),
        'feasible_core': bool(feasible_core),
        'feasible_full': bool(feasible_full),
        'feasible': bool(feasible_full),  # compat alias
        'v_ok': bool(v_ok),
        'ld_ok': bool(ld_ok),
    }


def compute_TAC(Area, dP_t, dP_s, v_t, m_h, m_c, Den_s, Den_t, L_D, case):
    """
    Compute Total Annual Cost (Eqs. 8-13 in paper).
    """
    eff = case["eff_pump"]
    hrs = case["hours_year"]
    r = case["interest"]
    n = case["n_years"]
    Cm = case["C_m"]
    Cp = case["C_p"]
    Ct = case["C_t"]
    cepci_act = case["cepci_act"]
    cepci_ref = case["cepci_ref"]
    elec = case["elec_price"]

    C_E = 3.28e4 * (max(Area, 0.1) / 80.0) ** 0.68
    C_i = (cepci_act / cepci_ref) * Cm * Cp * Ct * C_E

    CRF = (r * (1 + r) ** n) / ((1 + r) ** n - 1)
    C_fix = C_i * CRF

    W_s = m_h * abs(dP_s) / (Den_s * eff)
    W_t = m_c * abs(dP_t) / (Den_t * eff)
    C_op = (W_s + W_t) * elec * hrs / 1000.0

    return float(C_fix + C_op)
