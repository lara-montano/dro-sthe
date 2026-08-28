"""
R2-3a (supplement): stability of the radius calibration vs. training-set size.

The reviewer notes that 60 observations with 3-fold CV (20 scenarios per
validation fold) may make the radius selection noisy. N_tr = 60 is the
deliberate small-data regime of the study, so the main protocol keeps it;
this check quantifies the selection noise and shows how it shrinks when more
data are available: the per-seed CV selection is repeated at N_tr = 60 and
N_tr = 120 (folds of 20 vs 40 scenarios) on 8 seeds for Cases 1 and 4, and
the dispersion of rho* across seeds is compared.

Run:  python ntr_stability_check.py
"""
import json
import os
import time
import numpy as np

from config_case_studies import CASE1, CASE4, DESIGN_BOUNDS
from uncertainty_sampling import generate_iid_samples, compute_empirical_std
from optimization_objectives import make_saa_objective
from wdro_discrete import make_wdro_objective_discrete
from sthe_model import sthe_model
from metaheuristics import SOLVERS

lb, ub, dim = DESIGN_BOUNDS['lb'], DESIGN_BOUNDS['ub'], DESIGN_BOUNDS['dim']
LAM = 1e6
RHO_GRID = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40]
SEEDS = [11, 23, 42, 57, 71, 88, 103, 119]
CV_FOLDS = 3
CV_POP, CV_IT = 18, 18


def opt_one(obj, seed=42):
    v, x, _, _ = SOLVERS['DE'](obj, lb, ub, dim, pop_size=CV_POP, max_iter=CV_IT, seed=seed)
    return x


def val_feas(x, case, D_val):
    return float(np.mean([sthe_model(x, xi, case)['feasible_core'] for xi in D_val]))


def select_rho(case, seed, n_tr):
    D = generate_iid_samples(case, n_tr, seed=seed)
    rng = np.random.RandomState(seed)
    folds = np.array_split(rng.permutation(len(D)), CV_FOLDS)
    scores = {}
    for rho in RHO_GRID:
        vals = []
        for f in range(CV_FOLDS):
            tr_idx = np.concatenate([folds[g] for g in range(CV_FOLDS) if g != f])
            D_tr, D_val = D[tr_idx], D[folds[f]]
            W = 1.0 / compute_empirical_std(D_tr)
            if rho <= 0.0:
                obj = make_saa_objective(case, D_tr, LAM)
            else:
                obj = make_wdro_objective_discrete(case, D_tr, rho, W, lambda_penalty=LAM)
            vals.append(val_feas(opt_one(obj), case, D_val))
        scores[rho] = float(np.mean(vals))
    best = max(scores.values())
    return min([r for r in RHO_GRID if scores[r] >= best - 1e-9]), scores


def main():
    t0 = time.time()
    out = {}
    for case, cname in [(CASE1, 'Case1'), (CASE4, 'Case4')]:
        out[cname] = {}
        for n_tr in (60, 120):
            hist = []
            for s in SEEDS:
                rho_star, _ = select_rho(case, s, n_tr)
                hist.append(rho_star)
                print(f"{cname} N_tr={n_tr} seed {s}: rho*={rho_star} ({time.time()-t0:.0f}s)",
                      flush=True)
            vals = np.array(hist)
            out[cname][str(n_tr)] = dict(
                rho_star_hist=hist,
                mode=float(max(set(hist), key=hist.count)),
                n_distinct=int(len(set(hist))),
                iqr=[float(np.percentile(vals, 25)), float(np.percentile(vals, 75))])
            print(f"  -> {cname} N_tr={n_tr}: hist={hist} distinct={len(set(hist))}",
                  flush=True)

    path = os.path.join(os.path.dirname(__file__), 'results', 'ntr_stability.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Saved -> {path}  ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
