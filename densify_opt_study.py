"""
R2-1b: optimize WDRO ON the dense support (not just re-evaluate).

The R1 densification study evaluated the FINAL designs on a denser candidate
support (|S| 284 -> ~1924) and reported the dual-value gap. The reviewer
correctly notes this validates the evaluation, not the optimization: a search
run on the denser support could select a different design. This study re-runs
the WDRO optimization itself with the dense support (n_lhs = 1800) under the
identical protocol (same per-seed calibrated rho from the main run, same DE
restarts, same polish) on 4 seeds x 4 cases, and compares:
  - the selected designs (coarse-search vs dense-search),
  - their drift-regime feasibility and TAC-CVaR on the shared ladder,
  - their dual values under both supports.

Requires the main-run checkpoints results/consolidation_R2_polished_Case*.json
(for the per-seed rho* and the coarse-search designs).

Run:  python densify_opt_study.py
"""
import json
import os
import time
import numpy as np
from scipy.optimize import minimize as scipy_minimize

from config_case_studies import CASE1, CASE2, CASE3, CASE4, DESIGN_BOUNDS
from uncertainty_sampling import (generate_iid_samples, compute_empirical_std,
                                  generate_drift_ladder)
from wdro_discrete import make_wdro_objective_discrete, wdro_dual_discrete_at
from sthe_model import sthe_model
from metaheuristics import SOLVERS

lb, ub, dim = DESIGN_BOUNDS['lb'], DESIGN_BOUNDS['ub'], DESIGN_BOUNDS['dim']
LAM = 1e6
N_TR, N_DEP = 60, 3000
KAPPAS = [0.0, 0.5, 0.75, 1.0]
STUDY_SEEDS = [11, 23, 42, 57]
DATA_SEEDS = [11, 23, 42, 57, 71, 88, 103, 119, 134, 150,
              167, 181, 202, 219, 233, 251]
OPT_SEEDS = [42, 7, 123]
POP, IT = 24, 24
N_DENSE = 1800
CONT_IDX = [0, 6, 7, 8, 9, 10, 11]
ALPHA = 0.95
HERE = os.path.dirname(__file__)


def opt_best_polish(obj):
    bx, bv = None, np.inf
    for s in OPT_SEEDS:
        v, x, _, _ = SOLVERS['DE'](obj, lb, ub, dim, pop_size=POP, max_iter=IT, seed=s)
        if v < bv:
            bv, bx = v, x
    rng = np.random.RandomState(0)
    bounds = [(lb[i], ub[i]) for i in CONT_IDX]

    def wrapped(z):
        x = bx.copy()
        x[CONT_IDX] = z
        return obj(x)

    best_z, best_v = bx[CONT_IDX].copy(), bv
    starts = [bx[CONT_IDX].copy()] + [
        np.array([rng.uniform(lo, hi) for lo, hi in bounds]) for _ in range(7)]
    for z0 in starts:
        try:
            r = scipy_minimize(wrapped, z0, method='Nelder-Mead', bounds=bounds,
                               options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-3})
            if r.fun < best_v:
                best_v, best_z = float(r.fun), r.x.copy()
        except Exception:
            continue
    x = bx.copy()
    x[CONT_IDX] = np.clip(best_z, lb[CONT_IDX], ub[CONT_IDX])
    return x


def eval_regime(x, case, drift):
    feas, cvs = [], []
    for k in KAPPAS:
        if k < 0.5:
            continue
        tac, f = [], 0.0
        for xi in drift[k]:
            r = sthe_model(x, xi, case)
            tac.append(r['TAC'])
            f += r['feasible_core']
        tac = np.array(tac)
        q = np.percentile(tac, ALPHA * 100)
        feas.append(f / len(drift[k]))
        cvs.append(float(tac[tac >= q].mean()))
    return float(np.mean(feas)), float(np.mean(cvs))


def main():
    t0 = time.time()
    out = {}
    for case, cname in [(CASE1, 'Case1'), (CASE2, 'Case2'),
                        (CASE3, 'Case3'), (CASE4, 'Case4')]:
        main_path = os.path.join(HERE, 'results', f'consolidation_R2_polished_{cname}.json')
        with open(main_path) as f:
            main_res = json.load(f)
        rho_hist = main_res['rho_star_hist']
        coarse_designs = main_res['designs']['WDRO']
        drift = generate_drift_ladder(case, N_DEP, KAPPAS, seed_base=500)
        rows = []
        # Study the first (up to) 4 seeds whose calibrated radius is NONZERO:
        # at rho* = 0 the WDRO design is the SAA design and densification is
        # moot, so fixed seed lists would leave cases with mostly-zero
        # calibrations (e.g., Case 2) without coverage.
        study = [s for si, s in enumerate(DATA_SEEDS)
                 if rho_hist[si] > 0][:len(STUDY_SEEDS)]
        if not study:
            print(f"{cname}: all rho*=0 -- densification moot (WDRO==SAA)", flush=True)
            out[cname] = [dict(note='all rho*=0 (WDRO==SAA)')]
            continue
        for s in study:
            si = DATA_SEEDS.index(s)
            rho = float(rho_hist[si])
            D = generate_iid_samples(case, N_TR, seed=s)
            W = 1.0 / compute_empirical_std(D)
            x_coarse = np.array(coarse_designs[si])
            obj_dense = make_wdro_objective_discrete(case, D, rho, W,
                                                     lambda_penalty=LAM,
                                                     n_lhs=N_DENSE)
            x_dense = opt_best_polish(obj_dense)

            f_c, cv_c = eval_regime(x_coarse, case, drift)
            f_d, cv_d = eval_regime(x_dense, case, drift)
            d_c_small, _ = wdro_dual_discrete_at(x_coarse, case, D, rho, W, LAM, n_lhs=160)
            d_c_large, _ = wdro_dual_discrete_at(x_coarse, case, D, rho, W, LAM, n_lhs=N_DENSE)
            d_d_small, _ = wdro_dual_discrete_at(x_dense, case, D, rho, W, LAM, n_lhs=160)
            d_d_large, nS = wdro_dual_discrete_at(x_dense, case, D, rho, W, LAM, n_lhs=N_DENSE)
            rows.append(dict(
                seed=s, rho=rho, support_dense=int(nS),
                coarse=dict(feas=f_c, cvarTAC=cv_c,
                            dual160=d_c_small, dual1800=d_c_large),
                dense=dict(feas=f_d, cvarTAC=cv_d,
                           dual160=d_d_small, dual1800=d_d_large),
                dfeas_pp=(f_d - f_c) * 100,
                dcvar_rel=(cv_d - cv_c) / max(cv_c, 1e-9) * 100,
                x_dense=[float(v) for v in x_dense]))
            print(f"{cname} seed {s} rho={rho}: coarse feas={f_c*100:.1f}% "
                  f"dense feas={f_d*100:.1f}% dFeas={(f_d-f_c)*100:+.2f}pp "
                  f"dCVaR={(cv_d-cv_c)/max(cv_c,1e-9)*100:+.2f}% ({time.time()-t0:.0f}s)",
                  flush=True)
        out[cname] = rows

    path = os.path.join(HERE, 'results', 'densify_opt_study.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Saved -> {path}  ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
