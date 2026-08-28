"""
Evaluation metrics for STHE designs under uncertainty and drift.

Implements:
- Mean TAC (Eq. 53)
- Quantile TAC (Eq. 54)
- CVaR TAC (Eq. 55)
- Joint feasibility probability (Eq. 56)
- Marginal violation probabilities (Eq. 27)
- Mean violation magnitude (Eq. 58)
- Parallel evaluation via multiprocessing
- Sanity-check diagnostics (failure counts, dominant violations)

Notes (important for your paper consistency)
-------------------------------------------
- The paper feasibility definition (Eq. 56) is based on (g_Q, g_t, g_s) only.
  That is reported as 'joint_feasibility'.
- If your sthe_model also enforces extra checks (e.g., velocity bounds, L/D),
  expose them as res['feasible_full'] (or res['feasible']) and we also report
  'joint_feasibility_full' for engineering diagnostics.
"""

import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
from sthe_model import sthe_model


# =========================================================================
# Single-scenario evaluation (picklable for multiprocessing)
# =========================================================================

def _eval_single(xi, x, case):
    """Evaluate one scenario. Returns a dict of scalars (always)."""
    try:
        res = sthe_model(x, xi, case)

        g_Q = float(res['g_Q'])
        g_t = float(res['g_t'])
        g_s = float(res['g_s'])

        feasible_core = bool(res.get('feasible_core', (g_Q <= 0.0 and g_t <= 0.0 and g_s <= 0.0)))
        feasible_full = bool(res.get('feasible_full', res.get('feasible', feasible_core)))

        return {
            'TAC': float(res['TAC']),
            'g_Q': g_Q,
            'g_t': g_t,
            'g_s': g_s,
            'feasible_core': feasible_core,
            'feasible_full': feasible_full,
            'failed': False,
        }
    except Exception:
        # Hard-fail fallback: very large penalty values
        return {
            'TAC': 1e8,
            'g_Q': 1e6,
            'g_t': 1e6,
            'g_s': 1e6,
            'feasible_core': False,
            'feasible_full': False,
            'failed': True,
        }


# =========================================================================
# Dataset evaluation (serial or parallel)
# =========================================================================

def evaluate_design_on_dataset(x, dataset, case, n_workers=1):
    """
    Evaluate design x on a dataset of operating points.

    Parameters
    ----------
    x : array-like, shape (11,)
    dataset : ndarray, shape (N, 6)
    case : dict
    n_workers : int
        Number of parallel workers. 1 = serial (safest).
        Set to -1 for cpu_count()-1.

    Returns
    -------
    results : dict
        Keys:
          - 'TAC', 'g_Q', 'g_t', 'g_s'
          - 'feasible_core' : feasibility based on (g_Q,g_t,g_s) only (paper definition)
          - 'feasible_full' : feasibility including extra checks if provided by sthe_model
          - 'n_failed', 'failure_rate'
    """
    dataset = np.asarray(dataset)
    N = int(len(dataset))

    if n_workers == -1:
        n_workers = max(1, cpu_count() - 1)
    n_workers = int(max(1, n_workers))

    if N == 0:
        # empty dataset safety
        return {
            'TAC': np.array([], dtype=float),
            'g_Q': np.array([], dtype=float),
            'g_t': np.array([], dtype=float),
            'g_s': np.array([], dtype=float),
            'feasible_core': np.array([], dtype=bool),
            'feasible_full': np.array([], dtype=bool),
            'n_failed': 0,
            'failure_rate': 0.0,
        }

    if n_workers > 1:
        worker = partial(_eval_single, x=x, case=case)
        with Pool(n_workers) as pool:
            raw = pool.map(worker, [dataset[i] for i in range(N)])
    else:
        raw = [_eval_single(dataset[i], x, case) for i in range(N)]

    TACs = np.array([r['TAC'] for r in raw], dtype=float)
    g_Qs = np.array([r['g_Q'] for r in raw], dtype=float)
    g_ts = np.array([r['g_t'] for r in raw], dtype=float)
    g_ss = np.array([r['g_s'] for r in raw], dtype=float)

    feasible_core = np.array([r['feasible_core'] for r in raw], dtype=bool)
    feasible_full = np.array([r['feasible_full'] for r in raw], dtype=bool)

    failed = np.array([r['failed'] for r in raw], dtype=bool)

    return {
        'TAC': TACs,
        'g_Q': g_Qs,
        'g_t': g_ts,
        'g_s': g_ss,
        'feasible_core': feasible_core,
        'feasible_full': feasible_full,
        'n_failed': int(np.sum(failed)),
        'failure_rate': float(np.mean(failed)),
    }


# =========================================================================
# Metrics computation
# =========================================================================

def compute_metrics(results, alpha=0.95):
    """
    Compute all performance and feasibility metrics.

    Returns
    -------
    metrics : dict
    """
    TACs = np.asarray(results['TAC'], dtype=float)
    g_Qs = np.asarray(results['g_Q'], dtype=float)
    g_ts = np.asarray(results['g_t'], dtype=float)
    g_ss = np.asarray(results['g_s'], dtype=float)

    N = len(TACs)
    if N == 0:
        return {
            'mu_TAC': np.nan,
            'std_TAC': np.nan,
            'q_alpha': np.nan,
            'CVaR': np.nan,
            'joint_feasibility': np.nan,
            'joint_feasibility_full': np.nan,
            'p_viol_Q': np.nan,
            'p_viol_t': np.nan,
            'p_viol_s': np.nan,
            'v_Q_mean': np.nan,
            'v_t_mean': np.nan,
            'v_s_mean': np.nan,
        }

    # Mean TAC (Eq. 53)
    mu_TAC = float(np.mean(TACs))
    std_TAC = float(np.std(TACs))

    # Quantile TAC (Eq. 54)
    q_alpha = float(np.percentile(TACs, alpha * 100.0))

    # CVaR TAC (Eq. 55)
    tail_mask = TACs >= q_alpha
    cvar = float(np.mean(TACs[tail_mask])) if np.any(tail_mask) else q_alpha

    # Joint feasibility probability (Eq. 56) -- paper definition (core constraints only)
    joint_feas = float(np.mean((g_Qs <= 0.0) & (g_ts <= 0.0) & (g_ss <= 0.0)))

    # Optional: full feasibility if evaluation provided it (extra engineering checks)
    feas_full_arr = results.get('feasible_full', None)
    if feas_full_arr is not None:
        feas_full_arr = np.asarray(feas_full_arr, dtype=bool)
        joint_feas_full = float(np.mean(feas_full_arr)) if len(feas_full_arr) == N else joint_feas
    else:
        joint_feas_full = joint_feas

    # Marginal violation probabilities (Eq. 27)
    p_viol_Q = float(np.mean(g_Qs > 0.0))
    p_viol_t = float(np.mean(g_ts > 0.0))
    p_viol_s = float(np.mean(g_ss > 0.0))

    # Mean positive-part violation magnitude (Eq. 58)
    v_Q = float(np.mean(np.maximum(g_Qs, 0.0)))
    v_t = float(np.mean(np.maximum(g_ts, 0.0)))
    v_s = float(np.mean(np.maximum(g_ss, 0.0)))

    return {
        'mu_TAC': mu_TAC,
        'std_TAC': std_TAC,
        'q_alpha': q_alpha,
        'CVaR': cvar,

        # Core feasibility (matches paper Eq. 56)
        'joint_feasibility': joint_feas,

        # Full feasibility (optional; useful for diagnostics/appendix)
        'joint_feasibility_full': joint_feas_full,

        # Constraint violation stats
        'p_viol_Q': p_viol_Q,
        'p_viol_t': p_viol_t,
        'p_viol_s': p_viol_s,
        'v_Q_mean': v_Q,
        'v_t_mean': v_t,
        'v_s_mean': v_s,
    }


# =========================================================================
# Sanity-check diagnostics
# =========================================================================

def compute_diagnostics(results):
    """
    Compute sanity-check diagnostics for a single design evaluation.

    Uses FULL feasibility flag if available; otherwise falls back to core feasibility.

    Returns
    -------
    diag : dict
        'n_failed'            : int
        'failure_rate'        : float
        'n_infeasible'        : int
        'infeasibility_rate'  : float
        'dominant_violation'  : str
        'constraint_profile'  : dict
    """
    g_Qs = np.asarray(results['g_Q'], dtype=float)
    g_ts = np.asarray(results['g_t'], dtype=float)
    g_ss = np.asarray(results['g_s'], dtype=float)
    N = int(len(g_Qs))

    n_failed = int(results.get('n_failed', 0))
    failure_rate = float(results.get('failure_rate', 0.0))

    # choose which feasibility flag to diagnose
    feas = results.get('feasible_full', None)
    if feas is None:
        feas = results.get('feasible_core', None)
    if feas is None:
        feas = (g_Qs <= 0.0) & (g_ts <= 0.0) & (g_ss <= 0.0)
    feas = np.asarray(feas, dtype=bool)

    n_infeasible = int(np.sum(~feas))
    infeasibility_rate = float(n_infeasible / max(N, 1))

    profile = {}
    for name, g in [('g_Q', g_Qs), ('g_t', g_ts), ('g_s', g_ss)]:
        violations = np.maximum(g, 0.0)
        p_viol = float(np.mean(g > 0.0)) if N > 0 else 0.0
        v_mean = float(np.mean(violations)) if N > 0 else 0.0
        v_max = float(np.max(violations)) if N > 0 else 0.0
        v_p95 = float(np.percentile(violations, 95)) if N > 0 else 0.0
        profile[name] = {'p_viol': p_viol, 'v_mean': v_mean, 'v_max': v_max, 'v_p95': v_p95}

    probs = {k: v['p_viol'] for k, v in profile.items()}
    dominant = max(probs, key=probs.get) if any(v > 0.0 for v in probs.values()) else 'none'

    return {
        'n_failed': n_failed,
        'failure_rate': failure_rate,
        'n_infeasible': n_infeasible,
        'infeasibility_rate': infeasibility_rate,
        'dominant_violation': dominant,
        'constraint_profile': profile,
    }


# =========================================================================
# Drift evaluation
# =========================================================================

def evaluate_across_drift(x, case, drift_datasets, alpha=0.95, n_workers=1):
    """
    Evaluate design across the full drift ladder.

    Returns
    -------
    drift_metrics : dict {kappa: metrics_dict}
    drift_diagnostics : dict {kappa: diagnostics_dict}
    """
    drift_metrics = {}
    drift_diagnostics = {}

    for kappa, samples in sorted(drift_datasets.items()):
        res = evaluate_design_on_dataset(x, samples, case, n_workers=n_workers)
        drift_metrics[kappa] = compute_metrics(res, alpha)
        drift_diagnostics[kappa] = compute_diagnostics(res)

    return drift_metrics, drift_diagnostics


# =========================================================================
# Pretty-print
# =========================================================================

def print_metrics_table(drift_metrics, design_name="Design"):
    """Pretty-print metrics across drift levels (core feasibility column)."""
    print(f"\n{'='*95}")
    print(f"  {design_name}")
    print(f"{'='*95}")
    header = (f"{'kappa':>7} | {'mu_TAC':>12} | {'q_95':>12} | {'CVaR_95':>12} | "
              f"{'Feas%':>7} | {'pV_Q':>6} | {'pV_t':>6} | {'pV_s':>6}")
    print(header)
    print("-" * 95)
    for kappa in sorted(drift_metrics.keys()):
        m = drift_metrics[kappa]
        print(f"{kappa:7.2f} | {m['mu_TAC']:12.2f} | {m['q_alpha']:12.2f} | "
              f"{m['CVaR']:12.2f} | {m['joint_feasibility']*100:6.1f}% | "
              f"{m['p_viol_Q']:6.3f} | {m['p_viol_t']:6.3f} | {m['p_viol_s']:6.3f}")
    print(f"{'='*95}")


def print_diagnostics(diagnostics, design_name="Design"):
    """Pretty-print sanity-check diagnostics."""
    print(f"\n  Diagnostics for {design_name}:")
    for kappa, d in sorted(diagnostics.items()):
        dom = d['dominant_violation']
        print(f"    κ={kappa:.2f}: failed={d['n_failed']}, "
              f"infeasible={d['n_infeasible']} ({d['infeasibility_rate']*100:.1f}%), "
              f"dominant={dom}", end="")
        if dom != 'none':
            p = d['constraint_profile'][dom]
            print(f" [p={p['p_viol']:.3f}, v_mean={p['v_mean']:.1f}, v_max={p['v_max']:.1f}]")
        else:
            print()