"""
PCTL diagnostic re-run for the R2 revision (Section 6 'Baseline and solver
diagnostics'): the percentile baseline vs a recomputed SAA under the CORRECTED
model, identical 16-seed protocol at stage-1 budget (DE only, 3 restarts, no
polish), paired on the shared deployment ladder. Also records per-constraint
violation channels (duty vs tube/shell pressure drop vs velocity window) at
kappa = 0 and in the drift regime.

Run:  python pctl_diag_run.py
"""
import json
import os
import time
import numpy as np

from config_case_studies import CASE1, CASE2, CASE3, CASE4, DESIGN_BOUNDS
from uncertainty_sampling import (generate_iid_samples, compute_empirical_std,
                                  generate_drift_ladder)
from optimization_objectives import make_saa_objective, make_percentile_objective
from sthe_model import sthe_model
from metaheuristics import SOLVERS

lb, ub, dim = DESIGN_BOUNDS['lb'], DESIGN_BOUNDS['ub'], DESIGN_BOUNDS['dim']
LAM = 1e6
N_TR, N_DEP = 60, 3000
KAPPAS = [0.0, 0.5, 0.75, 1.0]
DATA_SEEDS = [11, 23, 42, 57, 71, 88, 103, 119, 134, 150,
              167, 181, 202, 219, 233, 251]
OPT_SEEDS = [42, 7, 123]
POP, IT = 24, 24
ALPHA = 0.95


def opt_best(obj):
    bx, bv = None, np.inf
    for s in OPT_SEEDS:
        v, x, _, _ = SOLVERS['DE'](obj, lb, ub, dim, pop_size=POP, max_iter=IT, seed=s)
        if v < bv:
            bv, bx = v, x
    return bx


def eval_full(x, case, D):
    """Feasibility + per-constraint violation rates + mean area proxy."""
    n = len(D)
    feas = 0.0
    viol = {'duty': 0.0, 'dPt': 0.0, 'dPs': 0.0, 'vel': 0.0}
    area = 0.0
    for xi in D:
        r = sthe_model(x, xi, case)
        feas += r['feasible_core']
        viol['duty'] += (r['g_Q'] > 0)
        viol['dPt'] += (r['g_t'] > 0)
        viol['dPs'] += (r['g_s'] > 0)
        viol['vel'] += (not r['v_ok'])
        area = r['Area']
    return feas / n, {k: v / n for k, v in viol.items()}, area


def main():
    t0 = time.time()
    out = {}
    for case, cname in [(CASE1, 'Case1'), (CASE2, 'Case2'),
                        (CASE3, 'Case3'), (CASE4, 'Case4')]:
        drift = generate_drift_ladder(case, N_DEP, KAPPAS, seed_base=500)
        di = [k for k in KAPPAS if k >= 0.5]
        rows = {'SAA': [], 'PCTL': []}
        areas = {'SAA': [], 'PCTL': []}
        viol0 = {'SAA': [], 'PCTL': []}
        for s in DATA_SEEDS:
            D = generate_iid_samples(case, N_TR, seed=s)
            x_saa = opt_best(make_saa_objective(case, D, LAM))
            x_pctl = opt_best(make_percentile_objective(case, D, ALPHA, LAM))
            for m, x in (('SAA', x_saa), ('PCTL', x_pctl)):
                fs = []
                for k in di:
                    f, _, _ = eval_full(x, case, drift[k])
                    fs.append(f)
                rows[m].append(float(np.mean(fs)))
                f0, v0, a = eval_full(x, case, drift[0.0])
                viol0[m].append(v0)
                areas[m].append(a)
            print(f"{cname} seed {s}: SAA {rows['SAA'][-1]*100:.1f}% "
                  f"PCTL {rows['PCTL'][-1]*100:.1f}% ({time.time()-t0:.0f}s)", flush=True)
        out[cname] = {}
        for m in ('SAA', 'PCTL'):
            v_mean = {k: float(np.mean([v[k] for v in viol0[m]])) for k in ('duty', 'dPt', 'dPs', 'vel')}
            out[cname][m] = dict(regime_feas_mean=float(np.mean(rows[m])),
                                 regime_feas_per_seed=rows[m],
                                 area_mean=float(np.mean(areas[m])),
                                 viol_channels_k0=v_mean)
        d = np.array(rows['PCTL']) - np.array(rows['SAA'])
        out[cname]['delta_pctl_saa_pp'] = float(d.mean() * 100)
        print(f"== {cname}: PCTL-SAA regime dFeas {d.mean()*100:+.1f}pp; "
              f"areas SAA {np.mean(areas['SAA']):.0f} vs PCTL {np.mean(areas['PCTL']):.0f} m2",
              flush=True)

    path = os.path.join(os.path.dirname(__file__), 'results', 'pctl_diag_R2.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Saved -> {path}  ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
