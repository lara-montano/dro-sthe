"""
Wasserstein Distributionally Robust Optimization (WDRO) core.

Implements:
- Wasserstein ball definition (Eq. 29)
- Ground cost with unit-consistent scaling (Eq. 30/61)
- Penalized loss (Eq. 33/62)
- Dual representation (Eq. 35)
- Inner maximization via bounded Nelder-Mead simplex (Eq. 63, §5.7)

All stochastic components receive an explicit numpy.random.RandomState
to guarantee reproducibility across evaluations and runs.
"""

import numpy as np
from scipy.optimize import minimize as scipy_minimize
from sthe_model import sthe_model


# =========================================================================
# Ground cost
# =========================================================================

def ground_cost_l1(xi_a, xi_b, W_diag):
    """
    Weighted L1 ground cost (Eq. 61).
    c(xi_a, xi_b) = ||W(xi_a - xi_b)||_1
    W = diag(1/sigma_1, ..., 1/sigma_6)
    """
    return np.sum(np.abs(W_diag * (xi_a - xi_b)))


# =========================================================================
# Penalized loss
# =========================================================================

def penalized_loss(x, xi, case, lambda_penalty=1e6):
    """
    Penalized loss L(x, xi) = TAC(x,xi) + penalty terms (Eq. 33/62).

    Normalized violations (Eq. 59) ensure unit-independent penalties.

    Constraints penalized (all declared in the case specification):
        - duty shortfall vs. the target-outlet requirement Q_req(xi)   (g_Q)
        - tube- and shell-side pressure drop limits                    (g_t, g_s)
        - tube-side velocity window [v_t_min, v_t_max]
        - slenderness ratio L/D <= L_D_max
    The last two are needed once tube length is an explicit design variable,
    otherwise the optimizer exploits unphysical long/thin bundles to cut cost.
    """
    try:
        res = sthe_model(x, xi, case)
        TAC = res['TAC']
        g_Q = res['g_Q']
        g_t = res['g_t']
        g_s = res['g_s']

        Q_req = case['Q_req']
        dP_t_max = case['dP_t_max']
        dP_s_max = case['dP_s_max']

        v_Q = max(0.0, g_Q) / max(Q_req, 1.0)
        v_t = max(0.0, g_t) / max(dP_t_max, 1.0)
        v_s = max(0.0, g_s) / max(dP_s_max, 1.0)

        # Geometric/operability constraints (normalized)
        vel = res['v_t']
        v_min = case.get('v_t_min', 0.5)
        v_max = case.get('v_t_max', 3.0)
        v_vel = (max(0.0, v_min - vel) + max(0.0, vel - v_max)) / max(v_max, 1e-6)

        LD = res['L_D']
        LD_max = case.get('L_D_max', 15.0)
        v_LD = max(0.0, LD - LD_max) / max(LD_max, 1e-6)

        return TAC + lambda_penalty * (v_Q + v_t + v_s + v_vel + v_LD)
    except Exception:
        return 1e8


# =========================================================================
# Inner maximization — bounded Nelder-Mead (consistent with paper §5.7)
# =========================================================================

def _clip_and_validate(xi, xi_lb, xi_ub):
    """Coordinate-wise clip + enforce T_h_in > T_c_in (Eq. 41)."""
    xi = np.clip(xi, xi_lb, xi_ub)
    # If temperature ordering violated, push T_h up slightly
    if xi[0] <= xi[1]:
        xi[0] = xi[1] + 0.5  # small margin in K
        xi[0] = min(xi[0], xi_ub[0])
    return xi


def _nelder_mead_inner_max(x, xi_i, lam, W_diag, case, xi_lb, xi_ub,
                            lambda_penalty, max_iter, tol, rng):
    """
    Bounded Nelder-Mead simplex for inner maximization (Eq. 63).

    Solves:  max_{xi in Xi}  L(x, xi) - lambda * c(xi, xi_i)

    The simplex is initialized at xi_i with vertices perturbed along
    each coordinate.  After each reflection/expansion/contraction step
    every candidate is projected back to Xi via coordinate clipping and
    temperature-ordering enforcement, as described in §5.7 of the paper.

    Parameters
    ----------
    rng : numpy.random.RandomState
        Controlled random state for reproducibility.
    """
    dim = len(xi_i)

    # --- inner objective (to MAXIMIZE) → we MINIMIZE the negative ----------
    def neg_obj(xi_raw):
        xi = _clip_and_validate(xi_raw.copy(), xi_lb, xi_ub)
        loss = penalized_loss(x, xi, case, lambda_penalty)
        transport = lam * ground_cost_l1(xi, xi_i, W_diag)
        return -(loss - transport)

    # --- build initial simplex centred at xi_i -------------------------
    # perturbation scale ~ 1/(lambda * W_j), capped at 15% of domain width
    safe_lam_W = np.maximum(lam * W_diag, 1e-6)
    scale = np.minimum(1.0 / safe_lam_W, (xi_ub - xi_lb) * 0.15)

    simplex = np.empty((dim + 1, dim))
    simplex[0] = xi_i.copy()
    for j in range(dim):
        vertex = xi_i.copy()
        vertex[j] += scale[j] * (1.0 if rng.rand() > 0.5 else -1.0)
        simplex[j + 1] = _clip_and_validate(vertex, xi_lb, xi_ub)

    # --- scipy Nelder-Mead with bounded callback -----------------------
    try:
        result = scipy_minimize(
            neg_obj,
            xi_i,
            method='Nelder-Mead',
            options={
                'maxiter': max_iter,
                'xatol': tol,
                'fatol': tol,
                'initial_simplex': simplex,
                'adaptive': True,
            },
        )
        xi_star = _clip_and_validate(result.x.copy(), xi_lb, xi_ub)
        return -neg_obj(xi_star)
    except Exception:
        # Fallback: evaluate at the scenario itself
        return penalized_loss(x, xi_i, case, lambda_penalty)


def wdro_inner_max(x, xi_i, lam, W_diag, case, xi_lb, xi_ub,
                   lambda_penalty=1e4, max_iter=60, tol=1e-4, rng=None):
    """
    Public interface for inner maximization (Eq. 63).

    Uses bounded Nelder-Mead simplex as stated in the paper (§5.7).

    Parameters
    ----------
    rng : numpy.random.RandomState or None
        If None a deterministic RNG (seed=0) is created.
    """
    if rng is None:
        rng = np.random.RandomState(0)

    return _nelder_mead_inner_max(
        x, xi_i, lam, W_diag, case, xi_lb, xi_ub,
        lambda_penalty, max_iter, tol, rng
    )


# =========================================================================
# Full WDRO dual — "exact" version (used for final post-opt evaluation)
# =========================================================================

def wdro_dual_objective(x, D_tr, rho, W_diag, case, xi_lb, xi_ub,
                        lambda_penalty=1e4, inner_max_iter=60, inner_tol=1e-4,
                        rng=None):
    """
    WDRO dual objective (Eq. 35):

        inf_{λ≥0}  { λρ  +  (1/N) Σ_i  sup_{ξ∈Ξ}
                      [ L(x,ξ) − λ c(ξ,ξ_i) ] }

    Uses Nelder-Mead inner max (I_in=60) and grid search over λ.
    """
    if rng is None:
        rng = np.random.RandomState(0)
    N = len(D_tr)

    lambda_candidates = np.logspace(-2, 4, 15)
    best_dual = np.inf

    for lam in lambda_candidates:
        inner_vals = np.zeros(N)
        for i in range(N):
            inner_vals[i] = _nelder_mead_inner_max(
                x, D_tr[i], lam, W_diag, case, xi_lb, xi_ub,
                lambda_penalty, inner_max_iter, inner_tol, rng
            )
        dual_val = lam * rho + np.mean(inner_vals)
        if dual_val < best_dual:
            best_dual = dual_val

    return best_dual


# =========================================================================
# Fast WDRO dual — used INSIDE metaheuristic loops
# =========================================================================

def wdro_dual_objective_fast(x, D_tr, rho, W_diag, case, xi_lb, xi_ub,
                              lambda_penalty=1e4, n_lambda=8,
                              inner_budget=15, rng=None):
    """
    Fast approximate WDRO dual for use inside metaheuristic search.

    Identical formulation to ``wdro_dual_objective`` but with:
    * fewer λ-grid points (``n_lambda``, default 8),
    * reduced Nelder-Mead budget per inner problem (``inner_budget``).

    The reduced budget yields an *under-estimate* of the true worst-case,
    which is conservative in the right direction: the metaheuristic sees
    a softer objective during search; the final selected design is then
    re-evaluated with the full budget in the evaluation phase.

    Parameters
    ----------
    rng : numpy.random.RandomState
        **Required** for reproducibility.  If None a deterministic
        fallback (seed=0) is used, but callers should always provide one.
    """
    if rng is None:
        rng = np.random.RandomState(0)

    N = len(D_tr)
    lambda_candidates = np.logspace(-1, 3, n_lambda)
    best_dual = np.inf

    for lam in lambda_candidates:
        inner_sum = 0.0
        for i in range(N):
            inner_sum += _nelder_mead_inner_max(
                x, D_tr[i], lam, W_diag, case, xi_lb, xi_ub,
                lambda_penalty,
                max_iter=inner_budget,
                tol=1e-3,
                rng=rng,
            )
        dual_val = lam * rho + inner_sum / N
        if dual_val < best_dual:
            best_dual = dual_val

    return best_dual
