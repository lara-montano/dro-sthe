"""
Metaheuristic solvers for STHE design optimization.

Implements:
- Differential Evolution (DE)
- Grey Wolf Optimizer (GWO)
- Success-Based Optimization Algorithm (SBOA)

All share the same interface: minimize f(x) subject to lb <= x <= ub.
"""

import numpy as np
import time


def differential_evolution(fobj, lb, ub, dim, pop_size=50, max_iter=50,
                           F=0.7, CR=0.9, seed=None):
    """
    Differential Evolution (DE/rand/1/bin).

    Parameters
    ----------
    fobj : callable
        Objective function f(x) -> float.
    lb, ub : ndarray
        Lower and upper bounds.
    dim : int
        Dimension.
    pop_size : int
        Population size.
    max_iter : int
        Maximum iterations.
    F : float
        Mutation factor.
    CR : float
        Crossover probability.
    seed : int
        Random seed.

    Returns
    -------
    best_val : float
    best_x : ndarray
    convergence : ndarray of shape (max_iter,)
    elapsed : float
    """
    rng = np.random.RandomState(seed)
    t0 = time.time()

    # Initialize population
    pop = lb + rng.rand(pop_size, dim) * (ub - lb)
    fitness = np.array([fobj(pop[i]) for i in range(pop_size)])

    best_idx = np.argmin(fitness)
    best_val = fitness[best_idx]
    best_x = pop[best_idx].copy()
    convergence = np.zeros(max_iter)

    for it in range(max_iter):
        for i in range(pop_size):
            # Select three distinct random indices
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1, r2, r3 = rng.choice(idxs, 3, replace=False)

            # Mutation
            mutant = pop[r1] + F * (pop[r2] - pop[r3])
            mutant = np.clip(mutant, lb, ub)

            # Crossover
            trial = pop[i].copy()
            j_rand = rng.randint(dim)
            for j in range(dim):
                if rng.rand() < CR or j == j_rand:
                    trial[j] = mutant[j]

            trial = np.clip(trial, lb, ub)

            # Selection
            f_trial = fobj(trial)
            if f_trial <= fitness[i]:
                pop[i] = trial
                fitness[i] = f_trial

                if f_trial < best_val:
                    best_val = f_trial
                    best_x = trial.copy()

        convergence[it] = best_val

    elapsed = time.time() - t0
    return best_val, best_x, convergence, elapsed


def grey_wolf_optimizer(fobj, lb, ub, dim, pop_size=50, max_iter=50, seed=None):
    """
    Grey Wolf Optimizer (GWO).

    Parameters
    ----------
    fobj : callable
        Objective function f(x) -> float.
    lb, ub : ndarray
    dim, pop_size, max_iter : int
    seed : int

    Returns
    -------
    best_val, best_x, convergence, elapsed
    """
    rng = np.random.RandomState(seed)
    t0 = time.time()

    # Initialize
    pop = lb + rng.rand(pop_size, dim) * (ub - lb)
    fitness = np.array([fobj(pop[i]) for i in range(pop_size)])

    # Sort and identify alpha, beta, delta
    sorted_idx = np.argsort(fitness)
    alpha = pop[sorted_idx[0]].copy()
    beta = pop[sorted_idx[1]].copy()
    delta = pop[sorted_idx[2]].copy()
    alpha_score = fitness[sorted_idx[0]]

    convergence = np.zeros(max_iter)

    for it in range(max_iter):
        a = 2.0 - 2.0 * it / max_iter  # Linearly decreasing

        for i in range(pop_size):
            for j in range(dim):
                # Alpha
                r1, r2 = rng.rand(), rng.rand()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha[j] - pop[i, j])
                X1 = alpha[j] - A1 * D_alpha

                # Beta
                r1, r2 = rng.rand(), rng.rand()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta[j] - pop[i, j])
                X2 = beta[j] - A2 * D_beta

                # Delta
                r1, r2 = rng.rand(), rng.rand()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta[j] - pop[i, j])
                X3 = delta[j] - A3 * D_delta

                pop[i, j] = (X1 + X2 + X3) / 3.0

            pop[i] = np.clip(pop[i], lb, ub)
            fitness[i] = fobj(pop[i])

        # Update leaders
        sorted_idx = np.argsort(fitness)
        if fitness[sorted_idx[0]] < alpha_score:
            alpha = pop[sorted_idx[0]].copy()
            alpha_score = fitness[sorted_idx[0]]
        beta = pop[sorted_idx[1]].copy() if pop_size > 1 else alpha.copy()
        delta = pop[sorted_idx[2]].copy() if pop_size > 2 else alpha.copy()

        convergence[it] = alpha_score

    elapsed = time.time() - t0
    return alpha_score, alpha, convergence, elapsed


def sboa(fobj, lb, ub, dim, pop_size=50, max_iter=50, seed=None):
    """
    Success-Based Optimization Algorithm (SBOA).

    Reference: Lara-Montaño & Gómez-Castro (2025)

    The algorithm partitions the population into successful and unsuccessful
    individuals based on median fitness, then applies different perturbation
    strategies to each group.

    Parameters
    ----------
    fobj : callable
    lb, ub : ndarray
    dim, pop_size, max_iter : int
    seed : int

    Returns
    -------
    best_val, best_x, convergence, elapsed
    """
    rng = np.random.RandomState(seed)
    t0 = time.time()

    # Initialize
    pop = lb + rng.rand(pop_size, dim) * (ub - lb)
    fitness = np.array([fobj(pop[i]) for i in range(pop_size)])

    best_idx = np.argmin(fitness)
    best_val = fitness[best_idx]
    best_x = pop[best_idx].copy()
    convergence = np.zeros(max_iter)

    for it in range(max_iter):
        # Partition into successful / unsuccessful based on median
        median_fit = np.median(fitness)

        # Adaptive parameter
        t_ratio = it / max_iter  # 0 -> 1
        alpha = 2.0 * (1.0 - t_ratio)  # Exploration decreases

        for i in range(pop_size):
            new_x = pop[i].copy()

            if fitness[i] <= median_fit:
                # Successful: exploitation around best
                # Move toward global best with Gaussian perturbation
                r1 = rng.rand()
                r2 = rng.randn(dim)
                sigma = alpha * (ub - lb) * 0.1

                # Attraction to best + local search
                new_x = pop[i] + r1 * (best_x - pop[i]) + sigma * r2
            else:
                # Unsuccessful: exploration
                # Random walk with Lévy-like jumps
                r1 = rng.rand()
                r2 = rng.rand()

                # Pick a random successful individual
                successful_mask = fitness <= median_fit
                if np.any(successful_mask):
                    successful_pop = pop[successful_mask]
                    r_idx = rng.randint(len(successful_pop))
                    ref = successful_pop[r_idx]
                else:
                    ref = best_x

                # Combination of random partner and best
                j = rng.randint(pop_size)
                while j == i:
                    j = rng.randint(pop_size)

                new_x = pop[i] + alpha * r1 * (ref - pop[i]) + r2 * (pop[j] - pop[i])

            new_x = np.clip(new_x, lb, ub)
            f_new = fobj(new_x)

            if f_new <= fitness[i]:
                pop[i] = new_x
                fitness[i] = f_new

                if f_new < best_val:
                    best_val = f_new
                    best_x = new_x.copy()

        convergence[it] = best_val

    elapsed = time.time() - t0
    return best_val, best_x, convergence, elapsed


# Dictionary for easy access
SOLVERS = {
    'DE': differential_evolution,
    'GWO': grey_wolf_optimizer,
    'SBOA': sboa,
}
