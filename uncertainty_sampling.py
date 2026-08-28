"""
Uncertainty sampling and drift construction.

Implements the train-test-shift protocol described in Section 3:
- Training/test data: i.i.d. Gaussian perturbations (Eq. 45)
- Drift: mean shift (Eq. 46) and variance inflation (Eq. 47)
- Drift ladder: kappa in {0, 0.25, 0.50, 0.75, 1.00}

NOTES (paper-ready):
- Drift is constructed to be *adversarial* for thermal feasibility:
    Th_in decreases, Tc_in increases, fouling increases
  and optionally *harder hydraulically*:
    flows increase (ΔP tends to increase).
- All samples are projected to the physical support set Xi via clipping
  and the temperature ordering constraint Th_in > Tc_in is enforced.
- This version adds a "drift profile" mechanism so you can run:
    'paper' (recommended), 'strong', 'temps', 'fouling', 'hydraulics', 'custom'
  to avoid pathological collapse of feasibility at large kappa while keeping drift meaningful.
"""

import numpy as np
from config_case_studies import get_xi_bounds


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _enforce_admissibility(samples, xi_lb, xi_ub):
    """
    Project to Xi and enforce basic admissibility:
      - clipping to support
      - Th_in > Tc_in (temperature ordering)
      - positive flows
      - nonnegative fouling
    Returns valid subset (may be smaller than input).
    """
    s = np.clip(samples, xi_lb, xi_ub)
    mask = (
        (s[:, 0] > s[:, 1]) &
        (s[:, 2] > 0) & (s[:, 3] > 0) &
        (s[:, 4] >= 0) & (s[:, 5] >= 0)
    )
    return s[mask]


def _get_nominals(case, xi_0):
    """Robust access to nominal values: prefer case keys if present; otherwise xi_0."""
    Th0 = float(case.get("T_h_in_0", xi_0[0]))
    Tc0 = float(case.get("T_c_in_0", xi_0[1]))
    mh0 = float(case.get("m_h_0",   xi_0[2]))
    mc0 = float(case.get("m_c_0",   xi_0[3]))
    Rft0 = float(case.get("R_f_t_0", xi_0[4]))
    Rfs0 = float(case.get("R_f_s_0", xi_0[5]))
    return Th0, Tc0, mh0, mc0, Rft0, Rfs0


# -------------------------------------------------------------------------
# IID sampling
# -------------------------------------------------------------------------

def generate_iid_samples(case, N, seed=None):
    """
    Generate N i.i.d. operating-point samples around nominal xi_0.

    Uses:
      xi = clip(xi_0 + sigma * z, lb, ub)
      sigma_j = (ub_j - lb_j) / 4  (two-sigma rule)

    Notes
    -----
    We enforce Th_in > Tc_in using rejection sampling to keep distributions plausible.
    If rejection cannot fill N, we fallback to small-noise draws around nominal.

    Returns
    -------
    samples : ndarray, shape (N, 6)
    """
    rng = np.random.RandomState(seed)
    xi_0, xi_lb, xi_ub = get_xi_bounds(case)

    sigma = (xi_ub - xi_lb) / 4.0

    samples = np.zeros((N, 6), dtype=float)
    count = 0

    max_attempts = max(N * 25, 2000)
    for _ in range(max_attempts):
        if count >= N:
            break
        z = rng.randn(6)
        xi = xi_0 + sigma * z
        xi = np.clip(xi, xi_lb, xi_ub)
        if (xi[0] > xi[1]) and (xi[2] > 0) and (xi[3] > 0) and (xi[4] >= 0) and (xi[5] >= 0):
            samples[count] = xi
            count += 1

    # fallback if needed
    if count < N:
        for i in range(count, N):
            xi = xi_0 + 0.1 * sigma * rng.randn(6)
            xi = np.clip(xi, xi_lb, xi_ub)

            # minimal correction to satisfy Th_in > Tc_in
            if xi[0] <= xi[1]:
                mid = 0.5 * (xi[0] + xi[1])
                xi[0] = min(xi_ub[0], mid + 1e-3)
                xi[1] = max(xi_lb[1], mid - 1e-3)
                if xi[0] <= xi[1]:
                    xi[0], xi[1] = xi_0[0], xi_0[1]

            samples[i] = xi

    return samples


def generate_train_test(case, N_tr, N_te, seed_tr=42, seed_te=123):
    """Generate training and test datasets (i.i.d.)."""
    D_tr = generate_iid_samples(case, N_tr, seed=seed_tr)
    D_te = generate_iid_samples(case, N_te, seed=seed_te)
    return D_tr, D_te


# -------------------------------------------------------------------------
# Drift profiles (paper-friendly defaults)
# -------------------------------------------------------------------------

# FIX (audit M6): temperature drift shifts are ABSOLUTE in Kelvin (Th_K, Tc_K),
# not fractions of the absolute inlet temperature (which was dimensionally
# arbitrary and produced ~11 K shifts from a nominal "3%"). Flows and fouling
# stay fractional. At kappa=1 the 'paper' profile shifts the hot inlet down and
# the cold inlet up by Th_K/Tc_K Kelvin (adversarial: less driving force).
DRIFT_PROFILES = {
    # Recommended for paper: adversarial, but not "kill feasibility instantly".
    'paper': {
        'Th_K': 3.0,       # Th_in down 3 K at kappa=1
        'Tc_K': 3.0,       # Tc_in up 3 K at kappa=1
        'mh_frac': 0.05,   # flows up 5%
        'mc_frac': 0.05,
        'Rf_frac': 0.20,   # fouling mean up 20%
    },

    # Strong
    'strong': {
        'Th_K': 5.0,
        'Tc_K': 5.0,
        'mh_frac': 0.10,
        'mc_frac': 0.10,
        'Rf_frac': 0.50,
    },

    # Ablations
    'temps': {
        'Th_K': 5.0,
        'Tc_K': 5.0,
        'mh_frac': 0.00,
        'mc_frac': 0.00,
        'Rf_frac': 0.00,
    },
    'fouling': {
        'Th_K': 0.0,
        'Tc_K': 0.0,
        'mh_frac': 0.00,
        'mc_frac': 0.00,
        'Rf_frac': 0.30,
    },
    'hydraulics': {
        'Th_K': 0.0,
        'Tc_K': 0.0,
        'mh_frac': 0.10,
        'mc_frac': 0.10,
        'Rf_frac': 0.00,
    },
}


# -------------------------------------------------------------------------
# Drift mechanisms
# -------------------------------------------------------------------------

def apply_mean_shift(samples, case, kappa,
                     include_flows=True, include_fouling=True,
                     profile='paper', custom=None):
    """
    Apply adversarial mean shift to samples.

    Parameters
    ----------
    samples : ndarray (N, 6)
    case : dict
    kappa : float in [0, 1]
    include_flows : bool
    include_fouling : bool
    profile : str
        One of DRIFT_PROFILES keys, e.g. 'paper' (default) or 'strong'.
    custom : dict or None
        If provided, overrides profile with:
          {'Th_frac','Tc_frac','mh_frac','mc_frac','Rf_frac'}
        Fractions are applied to nominal values.

    Returns
    -------
    shifted : ndarray (N, 6)
    """
    xi_0, xi_lb, xi_ub = get_xi_bounds(case)
    Th0, Tc0, mh0, mc0, Rft0, Rfs0 = _get_nominals(case, xi_0)

    if custom is None:
        cfg = DRIFT_PROFILES.get(profile, DRIFT_PROFILES['paper'])
    else:
        cfg = custom

    # Absolute-Kelvin temperature shifts (audit M6); fractional flow/fouling.
    # Back-compat: fall back to legacy *_frac*nominal only if *_K keys absent.
    Th_K = float(cfg.get('Th_K', cfg.get('Th_frac', 0.0) * Th0))
    Tc_K = float(cfg.get('Tc_K', cfg.get('Tc_frac', 0.0) * Tc0))
    mh_frac = float(cfg.get('mh_frac', 0.0))
    mc_frac = float(cfg.get('mc_frac', 0.0))
    Rf_frac = float(cfg.get('Rf_frac', 0.0))

    d = np.zeros(6, dtype=float)
    d[0] = -Th_K
    d[1] = +Tc_K

    if include_flows:
        d[2] = +mh_frac * mh0
        d[3] = +mc_frac * mc0

    if include_fouling:
        d[4] = +Rf_frac * Rft0
        d[5] = +Rf_frac * Rfs0

    shifted = samples + float(kappa) * d
    shifted = np.clip(shifted, xi_lb, xi_ub)
    return shifted


def apply_variance_inflation(samples, case, kappa, inflate_fouling_only=True, factor=None):
    """
    Apply variance inflation around nominal xi_0.

    Default:
      xi_j^(kappa) = xi0_j + (1 + kappa) * (xi_j - xi0_j)   for j in {4,5}
    Optionally, set `factor` to override (e.g., 1 + 0.5*kappa).

    Parameters
    ----------
    samples : ndarray (N, 6)
    case : dict
    kappa : float in [0, 1]
    inflate_fouling_only : bool
    factor : float or None
        Multiplicative factor for deviations. If None -> (1 + kappa).

    Returns
    -------
    inflated : ndarray (N, 6)
    """
    xi_0, xi_lb, xi_ub = get_xi_bounds(case)
    inflated = samples.copy()

    if factor is None:
        fac = 1.0 + float(kappa)
    else:
        fac = float(factor)

    idxs = [4, 5] if inflate_fouling_only else [0, 1, 2, 3, 4, 5]

    for j in idxs:
        inflated[:, j] = xi_0[j] + fac * (samples[:, j] - xi_0[j])

    inflated = np.clip(inflated, xi_lb, xi_ub)
    return inflated


# -------------------------------------------------------------------------
# Deployment sampling
# -------------------------------------------------------------------------

def generate_deployment_samples(case, kappa, N_dep, seed=None,
                                apply_mean=True, apply_var=True,
                                include_flows_in_mean=True,
                                include_fouling_in_mean=True,
                                inflate_fouling_only=True,
                                mean_profile='paper',
                                mean_custom=None,
                                var_factor=None):
    """
    Generate deployment samples at drift severity kappa.

    Steps:
      1) Generate i.i.d. base samples
      2) Apply mean shift (optional)
      3) Apply variance inflation (optional)
      4) Project to Xi and enforce Th_in > Tc_in by rejection
      5) If rejection reduces sample count, top up with additional draws

    Parameters
    ----------
    mean_profile : str
        'paper' recommended; 'strong' replicates harsher drift.
    mean_custom : dict or None
        Overrides mean_profile if provided.
    var_factor : float or None
        Overrides deviation multiplier for variance inflation. If None -> (1 + kappa).

    Returns
    -------
    samples : ndarray, shape (N_dep, 6)
    """
    xi_0, xi_lb, xi_ub = get_xi_bounds(case)

    base = generate_iid_samples(case, N_dep, seed=seed)

    if float(kappa) == 0.0:
        return base

    s = base.copy()
    if apply_mean:
        s = apply_mean_shift(
            s, case, kappa,
            include_flows=include_flows_in_mean,
            include_fouling=include_fouling_in_mean,
            profile=mean_profile,
            custom=mean_custom
        )
    if apply_var:
        s = apply_variance_inflation(
            s, case, kappa,
            inflate_fouling_only=inflate_fouling_only,
            factor=var_factor
        )

    valid = _enforce_admissibility(s, xi_lb, xi_ub)

    # Top-up if needed
    attempts = 0
    while len(valid) < N_dep and attempts < 25:
        attempts += 1
        extra_seed = None if seed is None else int(seed + 1000 + attempts * 17)
        extra = generate_iid_samples(case, N_dep, seed=extra_seed)

        if apply_mean:
            extra = apply_mean_shift(
                extra, case, kappa,
                include_flows=include_flows_in_mean,
                include_fouling=include_fouling_in_mean,
                profile=mean_profile,
                custom=mean_custom
            )
        if apply_var:
            extra = apply_variance_inflation(
                extra, case, kappa,
                inflate_fouling_only=inflate_fouling_only,
                factor=var_factor
            )

        extra_valid = _enforce_admissibility(extra, xi_lb, xi_ub)
        if len(extra_valid) > 0:
            valid = np.vstack([valid, extra_valid])

    # Hard fallback: pad with nominal (rare)
    if len(valid) < N_dep:
        pad = np.tile(xi_0, (N_dep - len(valid), 1))
        pad = np.clip(pad, xi_lb, xi_ub)
        valid = np.vstack([valid, pad])

    return valid[:N_dep]


def generate_drift_ladder(case, N_dep, kappa_ladder=None, seed_base=500,
                          apply_mean=True, apply_var=True,
                          include_flows_in_mean=True,
                          include_fouling_in_mean=True,
                          inflate_fouling_only=True,
                          mean_profile='paper',
                          mean_custom=None,
                          var_factor=None):
    """
    Generate deployment datasets for all kappa in the drift ladder.

    Parameters
    ----------
    mean_profile : str
        Default 'paper' (recommended). Use 'strong' to replicate harsher drift.
    mean_custom : dict or None
        Overrides mean_profile if provided.
    var_factor : float or None
        Overrides (1 + kappa) in variance inflation.

    Returns
    -------
    drift_datasets : dict
        {kappa: ndarray of shape (N_dep, 6)}
    """
    if kappa_ladder is None:
        kappa_ladder = [0.0, 0.25, 0.50, 0.75, 1.00]

    drift_datasets = {}
    for i, kappa in enumerate(kappa_ladder):
        drift_datasets[float(kappa)] = generate_deployment_samples(
            case, float(kappa), N_dep,
            seed=int(seed_base + i * 100),
            apply_mean=apply_mean,
            apply_var=apply_var,
            include_flows_in_mean=include_flows_in_mean,
            include_fouling_in_mean=include_fouling_in_mean,
            inflate_fouling_only=inflate_fouling_only,
            mean_profile=mean_profile,
            mean_custom=mean_custom,
            var_factor=var_factor
        )
    return drift_datasets


# -------------------------------------------------------------------------
# WDRO ground-cost scaling helper
# -------------------------------------------------------------------------

def compute_empirical_std(D_tr):
    """Compute coordinate-wise empirical standard deviation of training data."""
    return np.std(D_tr, axis=0) + 1e-10