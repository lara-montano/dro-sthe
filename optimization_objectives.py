"""
Optimization objective functions for each approach:
1. Deterministic baseline (Eq. 21)
2. SAA (Eq. 22)
3. Percentile-based baseline (Eq. 25)
4. WDRO (Eq. 34)

All include penalized feasibility handling (Eq. 60).
Every closure that involves randomness accepts an explicit
numpy.random.RandomState for reproducibility.
"""

import numpy as np
from sthe_model import sthe_model
from wdro_core import (penalized_loss, wdro_dual_objective_fast,
                       ground_cost_l1)
from config_case_studies import get_xi_bounds


# =========================================================================
# 1. Deterministic baseline
# =========================================================================

def make_deterministic_objective(case, lambda_penalty=1e4):
    """
    Deterministic baseline: minimize TAC at nominal xi_0 (Eq. 21).
    Fully deterministic — no RNG needed.
    """
    xi_0, _, _ = get_xi_bounds(case)

    def objective(x):
        return penalized_loss(x, xi_0, case, lambda_penalty)

    return objective


# =========================================================================
# 2. Sample Average Approximation (SAA)
# =========================================================================

def make_saa_objective(case, D_tr, lambda_penalty=1e4):
    """
    Full SAA: mean TAC over *all* training scenarios (Eq. 22).
    Deterministic given D_tr.
    """
    N = len(D_tr)

    def objective(x):
        total = 0.0
        for i in range(N):
            total += penalized_loss(x, D_tr[i], case, lambda_penalty)
        return total / N

    return objective


def make_saa_objective_subsample(case, D_tr, subsample_size=50,
                                  lambda_penalty=1e4, rng=None):
    """
    SAA with fixed-seed subsampling for computational efficiency.

    The RNG is reset to a *deterministic child seed* derived from a
    hash of x, so the same x always produces the same objective value.
    This avoids artificial noise during metaheuristic search while still
    using a subset of training data.
    """
    N = len(D_tr)
    if rng is None:
        rng = np.random.RandomState(42)

    # Pre-generate a fixed set of index permutations to cycle through.
    # This makes the objective fully deterministic for a given x within
    # the same run.
    _n_perms = 20
    _perms = [rng.choice(N, min(subsample_size, N), replace=False)
              for _ in range(_n_perms)]
    _call_counter = [0]      # mutable counter inside closure

    def objective(x):
        idx = _perms[_call_counter[0] % _n_perms]
        _call_counter[0] += 1
        total = 0.0
        for i in idx:
            total += penalized_loss(x, D_tr[i], case, lambda_penalty)
        return total / len(idx)

    return objective


# =========================================================================
# 3. Percentile-based baseline
# =========================================================================

def make_percentile_objective(case, D_tr, alpha=0.95, lambda_penalty=1e4):
    """
    Percentile-based: minimize empirical alpha-quantile of TAC (Eq. 25).
    Deterministic given D_tr.
    """
    N = len(D_tr)

    def objective(x):
        losses = np.array([
            penalized_loss(x, D_tr[i], case, lambda_penalty)
            for i in range(N)
        ])
        return np.percentile(losses, alpha * 100)

    return objective


# =========================================================================
# 3b. Risk-averse empirical baselines (R2 reviewer comment 5)
# =========================================================================

def make_saa_cvar_objective(case, D_tr, alpha=0.95, lambda_penalty=1e6):
    """
    Empirical CVaR baseline: minimize CVaR_alpha of the penalized loss over
    the training scenarios (R2-C5). Uses the same tail convention as the
    evaluation metrics (mean of losses >= the empirical alpha-percentile,
    numpy linear-interpolated percentile), so objective and reported metric
    are consistent. Deterministic given D_tr.
    """
    N = len(D_tr)

    def objective(x):
        losses = np.array([
            penalized_loss(x, D_tr[i], case, lambda_penalty)
            for i in range(N)
        ])
        q = np.percentile(losses, alpha * 100)
        tail = losses[losses >= q]
        return float(tail.mean()) if len(tail) else float(q)

    return objective


def make_worstcase_objective(case, D_tr, lambda_penalty=1e6):
    """
    Worst-case-over-samples baseline: minimize the maximum penalized loss over
    the training scenarios (R2-C5). This is the sample-robust counterpart of
    SAA: it hedges against the worst OBSERVED scenario but, unlike WDRO,
    cannot provision for scenarios outside the empirical support.
    Deterministic given D_tr.
    """
    N = len(D_tr)

    def objective(x):
        worst = -np.inf
        for i in range(N):
            val = penalized_loss(x, D_tr[i], case, lambda_penalty)
            if val > worst:
                worst = val
        return float(worst)

    return objective


# =========================================================================
# 4. WDRO
# =========================================================================

def make_wdro_objective(case, D_tr, rho, W_diag, lambda_penalty=1e4,
                        n_lambda=8, inner_budget=15, rng=None):
    """
    WDRO: worst-case expected penalized loss over Wasserstein ball (Eq. 34).
    Uses fast approximate dual with bounded Nelder-Mead inner max.

    Parameters
    ----------
    rng : numpy.random.RandomState
        Passed through to the inner Nelder-Mead; guarantees identical
        objective values for identical x across evaluations.
    """
    xi_0, xi_lb, xi_ub = get_xi_bounds(case)
    if rng is None:
        rng = np.random.RandomState(42)

    def objective(x):
        return wdro_dual_objective_fast(
            x, D_tr, rho, W_diag, case, xi_lb, xi_ub,
            lambda_penalty=lambda_penalty,
            n_lambda=n_lambda,
            inner_budget=inner_budget,
            rng=rng,
        )

    return objective


def make_wdro_objective_subsample(case, D_tr, rho, W_diag,
                                   subsample_size=50, lambda_penalty=1e4,
                                   n_lambda=6, inner_budget=15, rng=None):
    """
    WDRO with fixed-seed subsampling for computational efficiency.

    Same deterministic-closure strategy as ``make_saa_objective_subsample``:
    pre-generated index permutations ensure that the same x always
    evaluates against the same scenario subset within a run.
    """
    xi_0, xi_lb, xi_ub = get_xi_bounds(case)
    N = len(D_tr)
    if rng is None:
        rng = np.random.RandomState(42)

    _n_perms = 20
    _perm_rng = np.random.RandomState(rng.randint(0, 2**31))
    _perms = [_perm_rng.choice(N, min(subsample_size, N), replace=False)
              for _ in range(_n_perms)]
    _call_counter = [0]

    # Separate RNG for inner NM to keep it isolated
    _inner_rng = np.random.RandomState(rng.randint(0, 2**31))

    def objective(x):
        idx = _perms[_call_counter[0] % _n_perms]
        _call_counter[0] += 1
        D_sub = D_tr[idx]
        return wdro_dual_objective_fast(
            x, D_sub, rho, W_diag, case, xi_lb, xi_ub,
            lambda_penalty=lambda_penalty,
            n_lambda=n_lambda,
            inner_budget=inner_budget,
            rng=_inner_rng,
        )

    return objective
