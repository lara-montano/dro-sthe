"""
Discretized-support WDRO inner maximization (rebuild of the WDRO numerics).

Motivation
----------
The submitted version solved the sample-wise inner maximization
    sup_{xi in Xi} [ L(x,xi) - lambda * c(xi, xi_i) ]
with a LOCAL Nelder-Mead started at xi_i. A local-vs-global diagnostic showed
this recovers < 10% of the true worst case (global search found dual values
~9x larger), so the WDRO dual collapsed numerically to the SAA mean and every
WDRO design was identical to SAA. This directly reproduces reviewer concerns
R1-min5 and R2-M4a.

Key idea for a GLOBAL yet CHEAP inner max
-----------------------------------------
In the dual, the term L(x,xi) does NOT depend on the data point xi_i -- only the
transport cost c(xi, xi_i) does. Therefore we may:
  1. Build a fixed candidate support S of the uncertainty set Xi (box vertices,
     admissible w.r.t. T_h_in > T_c_in, plus a Latin-hypercube fill and the data
     points themselves).
  2. Evaluate L(x, .) ONCE on S per design x  (|S| model calls).
  3. Precompute the transport-cost matrix C[j,i] = c(S_j, xi_i)  ONCE (it does
     not depend on x), so the inner sup for every (i, lambda) is a vectorized
     finite maximum:  sup_i(lambda) = max_j ( L_S[j] - lambda * C[j,i] ).
The dual is then  inf_{lambda>=0} [ lambda*rho + mean_i sup_i(lambda) ],
a 1-D convex piecewise-linear program.

R2 revision (reviewer comment 1a): the outer minimization over lambda is now
solved EXACTLY (to numerical tolerance) by derivative-sign bisection on the
convex piecewise-linear dual, replacing the previous lambda grid. With the
outer problem solved exactly, the computed value equals the optimal value of
the support-restricted WDRO problem (finite LP strong duality), which is a
valid lower bound on the continuous worst case by feasible-set inclusion --
no regularity conditions on the loss are needed for that chain. The legacy
grid solver is retained for A/B diagnostics (solver='grid').

This is a discretized-support inner maximization for Wasserstein-DRO: the
worst-case measure is restricted to a finite candidate set S subset Xi. Because
the max over a finite subset is <= the continuous sup, the resulting dual is a
LOWER BOUND on the true continuous WDRO dual (audit M5 -- it is NOT "exact" and
the worst case is generally NOT attained at a box vertex; for tight-approach
services the bound keeps rising as |S| grows and the operating |S| under-provisions
the worst case, so the gap must be reported). It is used as a fast, consistent
surrogate for the metaheuristic search; convergence is quantified separately by a
densification study (see densification_gap). Cost per objective call ~ |S| model
evaluations, comparable to SAA.
"""

import numpy as np
from wdro_core import penalized_loss, ground_cost_l1
from config_case_studies import get_xi_bounds


def build_candidate_support(case, D_tr, n_lhs=160, seed=0, include_data=True):
    """
    Build a candidate support S for the inner maximization.

    S = admissible box vertices  +  Latin-hypercube fill  +  (optionally) data.
    Admissibility: clip to [xi_lb, xi_ub] and require T_h_in > T_c_in.
    """
    xi_0, xi_lb, xi_ub = get_xi_bounds(case)
    rng = np.random.RandomState(seed)
    dim = 6

    # --- box vertices (2^6 = 64) ---
    verts = []
    for mask in range(2 ** dim):
        v = np.array([xi_ub[d] if (mask >> d) & 1 else xi_lb[d] for d in range(dim)])
        verts.append(v)
    verts = np.array(verts)

    # --- Latin-hypercube fill ---
    lhs = np.zeros((n_lhs, dim))
    for d in range(dim):
        cuts = (np.arange(n_lhs) + rng.rand(n_lhs)) / n_lhs
        lhs[:, d] = xi_lb[d] + cuts[rng.permutation(n_lhs)] * (xi_ub[d] - xi_lb[d])

    S = np.vstack([verts, lhs])
    if include_data:
        S = np.vstack([S, np.asarray(D_tr)])

    # enforce admissibility (T_h_in > T_c_in); nudge if needed
    S = np.clip(S, xi_lb, xi_ub)
    bad = S[:, 0] <= S[:, 1]
    S[bad, 0] = np.minimum(S[bad, 1] + 0.5, xi_ub[0])
    # De-duplicate on rounded coordinates but KEEP the original (unrounded)
    # rows, so the training scenarios are contained in S exactly (as stated
    # in the paper) and their transport cost to themselves is exactly zero --
    # required for the rho=0 <-> SAA identity of the exact dual (R2-3a).
    _, keep = np.unique(np.round(S, 8), axis=0, return_index=True)
    S = S[np.sort(keep)]
    return S


def precompute_cost_matrix(S, D_tr, W_diag):
    """C[j,i] = || W (S_j - xi_i) ||_1 ; shape (|S|, N)."""
    S = np.asarray(S)[:, None, :]          # (M,1,6)
    D = np.asarray(D_tr)[None, :, :]        # (1,N,6)
    return np.sum(np.abs(W_diag * (S - D)), axis=2)   # (M,N)


def wdro_dual_discrete(L_S, C, rho, lambdas):
    """
    Global-over-support WDRO dual.

        inf_{lambda in lambdas} [ lambda*rho + (1/N) sum_i max_j ( L_S[j] - lambda*C[j,i] ) ]

    L_S : (M,) losses on the candidate support
    C   : (M,N) transport-cost matrix
    """
    L_S = np.asarray(L_S)[:, None]          # (M,1)
    best = np.inf
    for lam in lambdas:
        inner = L_S - lam * C               # (M,N)
        sup_i = inner.max(axis=0)           # (N,)
        val = lam * rho + sup_i.mean()
        if val < best:
            best = float(val)
    return best


DEFAULT_LAMBDAS = np.concatenate([[0.0], np.logspace(-2, 7, 48)])


# ---------------------------------------------------------------------------
# Exact 1-D convex solve of the outer minimization (R2 reviewer comment 1a)
# ---------------------------------------------------------------------------

def _g_and_right_derivative(L_S_col, C, rho, lam):
    """
    Evaluate g(lam) = lam*rho + mean_i sup_i(lam) and its RIGHT derivative.

    sup_i(lam) = max_j ( L_S[j] - lam*C[j,i] ) is an upper envelope of affine
    functions of lam, so g is convex piecewise-linear. Its right derivative is
    rho - mean_i C[j_i^+, i], where j_i^+ attains the max with the SMALLEST
    transport cost among (numerical) ties -- the line that stays active as
    lam increases.
    """
    inner = L_S_col - lam * C                 # (M, N)
    sup_i = inner.max(axis=0)                 # (N,)
    # ties within a small relative tolerance of the max
    tol = 1e-9 * np.maximum(1.0, np.abs(sup_i))
    is_tie = inner >= (sup_i - tol)           # (M, N) boolean
    # smallest cost among tied argmaxes -> right derivative
    C_masked = np.where(is_tie, C, np.inf)
    c_right = C_masked.min(axis=0)            # (N,)
    g = lam * rho + sup_i.mean()
    g_prime_right = rho - c_right.mean()
    return float(g), float(g_prime_right)


def wdro_dual_discrete_exact(L_S, C, rho, tol_rel=1e-12, return_info=False):
    """
    Exact (to numerical tolerance) solution of

        inf_{lambda >= 0} [ lambda*rho + (1/N) sum_i max_j ( L_S[j] - lambda*C[j,i] ) ]

    by bisection on the sign of the right derivative of the convex
    piecewise-linear dual. Replaces the grid search of wdro_dual_discrete
    (R2 reviewer comment 1a): a grid minimum is an UPPER bound on the true
    infimum, which broke the lower-bound chain to the continuous worst case;
    the bisection solution restores it.

    Special case rho = 0: g is non-increasing and its infimum (attained as
    lambda -> inf) equals mean_i max{ L_S[j] : C[j,i] = 0 }, i.e. the SAA
    objective when the training points are contained in the support. This is
    the exact value of the radius-0 restricted problem (the Wasserstein ball
    collapses to the empirical distribution), so WDRO with rho = 0 recovers
    SAA by construction.

    Returns the optimal value; with return_info=True also a dict with
    lam_star, the final bracket, and the certified optimality gap
    (g(best) - max of the two supporting-line lower bounds at the bracket).
    """
    L_S_col = np.asarray(L_S, dtype=float)[:, None]     # (M,1)
    C = np.asarray(C, dtype=float)

    # --- rho = 0: exact closed form (SAA on the zero-cost support points) ---
    if rho <= 0.0:
        # Data-in-support gives an exactly-zero column minimum; the max()
        # guard degrades gracefully to the nearest support point if a caller
        # ever passes a support that does not contain the data.
        col_min = C.min(axis=0, keepdims=True)
        zero_cost = C <= np.maximum(1e-12, col_min)
        masked = np.where(zero_cost, L_S_col, -np.inf)
        val = float(masked.max(axis=0).mean())
        if return_info:
            return val, {'lam_star': np.inf, 'bracket': (np.inf, np.inf),
                         'cert_gap': 0.0, 'n_iter': 0}
        return val

    # --- lambda = 0: if right derivative already >= 0 the minimum is at 0 ---
    g0, d0 = _g_and_right_derivative(L_S_col, C, rho, 0.0)
    if d0 >= 0.0:
        if return_info:
            return g0, {'lam_star': 0.0, 'bracket': (0.0, 0.0),
                        'cert_gap': 0.0, 'n_iter': 0}
        return g0

    # --- bracket [lam_lo, lam_hi] with d(lam_lo) < 0 <= d(lam_hi) ----------
    # As lam -> inf the active lines are the zero-cost (data) points, so the
    # derivative tends to rho > 0: a finite upper end always exists.
    lam_lo, g_lo = 0.0, g0
    lam_hi = 1.0
    g_hi, d_hi = _g_and_right_derivative(L_S_col, C, rho, lam_hi)
    n_iter = 1
    while d_hi < 0.0 and n_iter < 80:
        lam_lo, g_lo = lam_hi, g_hi
        lam_hi *= 4.0
        g_hi, d_hi = _g_and_right_derivative(L_S_col, C, rho, lam_hi)
        n_iter += 1

    # --- bisection on the derivative sign ---------------------------------
    for _ in range(200):
        if (lam_hi - lam_lo) <= tol_rel * max(1.0, lam_hi):
            break
        lam_mid = 0.5 * (lam_lo + lam_hi)
        g_mid, d_mid = _g_and_right_derivative(L_S_col, C, rho, lam_mid)
        n_iter += 1
        if d_mid < 0.0:
            lam_lo, g_lo = lam_mid, g_mid
        else:
            lam_hi, g_hi = lam_mid, g_mid

    best = min(g_lo, g_hi)
    if return_info:
        # convexity certificate: supporting lines at the bracket ends bound
        # the minimum from below
        _, d_lo_end = _g_and_right_derivative(L_S_col, C, rho, lam_lo)
        lower1 = g_lo + d_lo_end * (lam_hi - lam_lo)   # extend left support line
        lower = max(min(lower1, g_hi), min(g_lo, g_hi) - abs(d_lo_end) * (lam_hi - lam_lo))
        info = {'lam_star': 0.5 * (lam_lo + lam_hi),
                'bracket': (lam_lo, lam_hi),
                'cert_gap': float(best - lower),
                'n_iter': n_iter}
        return float(best), info
    return float(best)


def make_wdro_objective_discrete(case, D_tr, rho, W_diag,
                                 lambda_penalty=1e6, n_lhs=160,
                                 support_seed=0, lambdas=None, solver='exact'):
    """
    Build a fast WDRO objective using the discretized-support global inner max.

    The candidate support S and the cost matrix C are precomputed ONCE; each
    objective evaluation costs |S| penalized-loss (model) calls plus vectorized
    algebra.

    solver='exact' (default, R2): bisection solve of the 1-D convex dual.
    solver='grid': legacy lambda-grid search (R1 numbers), kept for A/B checks.
    """
    S = build_candidate_support(case, D_tr, n_lhs=n_lhs, seed=support_seed)
    C = precompute_cost_matrix(S, D_tr, W_diag)
    lam_grid = DEFAULT_LAMBDAS if lambdas is None else np.asarray(lambdas)

    if solver == 'exact':
        def objective(x):
            L_S = np.array([penalized_loss(x, S[j], case, lambda_penalty)
                            for j in range(len(S))])
            return wdro_dual_discrete_exact(L_S, C, rho)
    else:
        def objective(x):
            L_S = np.array([penalized_loss(x, S[j], case, lambda_penalty)
                            for j in range(len(S))])
            return wdro_dual_discrete(L_S, C, rho, lam_grid)

    objective.support_size = len(S)
    return objective


def wdro_dual_discrete_at(x, case, D_tr, rho, W_diag, lambda_penalty=1e6,
                          n_lhs=160, support_seed=0, lambdas=None,
                          solver='exact', return_info=False):
    """Convenience: evaluate the discretized-support dual at a single design x."""
    S = build_candidate_support(case, D_tr, n_lhs=n_lhs, seed=support_seed)
    C = precompute_cost_matrix(S, D_tr, W_diag)
    L_S = np.array([penalized_loss(x, S[j], case, lambda_penalty) for j in range(len(S))])
    if solver == 'exact':
        if return_info:
            val, info = wdro_dual_discrete_exact(L_S, C, rho, return_info=True)
            return val, len(S), info
        return wdro_dual_discrete_exact(L_S, C, rho), len(S)
    lam_grid = DEFAULT_LAMBDAS if lambdas is None else np.asarray(lambdas)
    return wdro_dual_discrete(L_S, C, rho, lam_grid), len(S)


def densification_gap(x, case, D_tr, rho, W_diag, lambda_penalty=1e6,
                      n_small=160, n_large=1800, support_seed=0):
    """
    Report how much the discretized-support dual UNDER-estimates the worst case.

    Returns (dual_small, dual_large, rel_gap) where rel_gap = (large-small)/small.
    A large positive gap (e.g. tight-approach Case 3) means the operating support
    under-provisions the worst case and the reported dual is a loose lower bound.
    Intended for final selected designs only (not inside the search loop).
    """
    d_small, _ = wdro_dual_discrete_at(x, case, D_tr, rho, W_diag, lambda_penalty,
                                       n_lhs=n_small, support_seed=support_seed)
    d_large, _ = wdro_dual_discrete_at(x, case, D_tr, rho, W_diag, lambda_penalty,
                                       n_lhs=n_large, support_seed=support_seed)
    return d_small, d_large, (d_large - d_small) / max(abs(d_small), 1e-9)
