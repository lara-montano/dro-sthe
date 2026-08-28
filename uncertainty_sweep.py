"""
STAR FIGURE (Phase 2): the WDRO-over-SAA joint-feasibility advantage under drift
as a function of the MAGNITUDE of uncertainty (a single controlled axis u that
scales both the i.i.d. training bands and the deployment drift). Two base services
(C1 methanol/seawater, C4 glycerol/water). Fixed Wasserstein radius (stated), so
the message is clean: at a fixed radius, robustness pays more as uncertainty grows,
until uncertainty is so large that even robustness cannot keep the design feasible.
"""
import copy
import json
import os
import time
import numpy as np
from config_case_studies import CASE1, CASE4, DESIGN_BOUNDS
from sthe_model import sthe_model
from uncertainty_sampling import (generate_iid_samples, compute_empirical_std,
                                  generate_drift_ladder)
from optimization_objectives import make_saa_objective
from wdro_discrete import make_wdro_objective_discrete
from metaheuristics import SOLVERS

lb, ub, dim = DESIGN_BOUNDS['lb'], DESIGN_BOUNDS['ub'], DESIGN_BOUNDS['dim']
LAM = 1e6
KAPPAS = [0.0, 0.5, 1.0]
U_GRID = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
SEEDS = [11, 23, 42, 57, 71, 88, 103, 119]     # 8 seeds
RHO = 0.2
N_TR, N_DEP = 55, 1500
OPT_SEEDS = [42, 7]
POP, IT = 22, 22


def unc_case(base, u):
    c = copy.deepcopy(base)
    c['dt_band'] = 3.0 * u
    c['mflow_frac'] = 0.10 * u
    c['rfoul_frac'] = 0.50 * u
    return c


def drift_custom(u):
    return dict(Th_K=3.0 * u, Tc_K=3.0 * u, mh_frac=0.05 * u, mc_frac=0.05 * u, Rf_frac=0.20 * u)


def opt_best(obj, seeds):
    bx, bv = None, np.inf
    for sd in seeds:
        v, x, _, _ = SOLVERS['DE'](obj, lb, ub, dim, pop_size=POP, max_iter=IT, seed=sd)
        if v < bv:
            bv, bx = v, x
    return bx


def drift_feas(x, case, drift):
    return float(np.mean([np.mean([sthe_model(x, xi, case)['feasible_core'] for xi in drift[k]])
                          for k in KAPPAS if k >= 0.5]))


def bootstrap_ci(vals, n_boot=4000, seed=0):
    vals = np.asarray(vals, float); rng = np.random.RandomState(seed)
    m = [vals[rng.randint(0, len(vals), len(vals))].mean() for _ in range(n_boot)]
    return float(np.mean(vals)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    t0 = time.time()
    out = {'u_grid': U_GRID, 'rho': RHO, 'n_seeds': len(SEEDS), 'bases': {}}
    for base, bname in [(CASE1, 'C1'), (CASE4, 'C4')]:
        rows = []
        for u in U_GRID:
            case = unc_case(base, u)
            drift = generate_drift_ladder(case, N_DEP, KAPPAS, seed_base=500, mean_custom=drift_custom(u))
            saa_f, wdro_f, dfs = [], [], []
            for sd in SEEDS:
                D = generate_iid_samples(case, N_TR, seed=sd)
                W = 1.0 / compute_empirical_std(D)
                x_saa = opt_best(make_saa_objective(case, D, LAM), OPT_SEEDS)
                x_wdro = opt_best(make_wdro_objective_discrete(case, D, RHO, W, lambda_penalty=LAM), OPT_SEEDS)
                sf, wf = drift_feas(x_saa, case, drift), drift_feas(x_wdro, case, drift)
                saa_f.append(sf); wdro_f.append(wf); dfs.append(wf - sf)
            md, lo, hi = bootstrap_ci(dfs)
            rows.append(dict(u=u, saa=float(np.mean(saa_f)), wdro=float(np.mean(wdro_f)),
                             dfeas=md, dfeas_ci=[lo, hi]))
            print(f"  {bname} u={u:.2f}: SAA={np.mean(saa_f)*100:.1f} WDRO={np.mean(wdro_f)*100:.1f} "
                  f"dFeas={md*100:+.1f}pp CI[{lo*100:+.1f},{hi*100:+.1f}] ({time.time()-t0:.0f}s)", flush=True)
        out['bases'][bname] = rows
        outdir = os.path.join(os.path.dirname(__file__), 'results')
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, 'uncertainty_sweep.json'), 'w') as f:
            json.dump(out, f, indent=2)
        print(f"  [checkpoint] {bname} saved", flush=True)
    print(f"Total {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
