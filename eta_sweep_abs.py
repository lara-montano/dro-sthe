"""
R2-2b: penalty-weight (eta) sensitivity with ABSOLUTE metrics per method.

The R1 manuscript reported only the WDRO-SAA feasibility GAP at
eta in {1e5, 1e6, 1e7}; the reviewer asks for the absolute feasibility and
cost of SAA and WDRO separately at every eta, so the reader can see how each
formulation responds to the penalty weight (not just their difference).

Protocol (mirrors the R1 eta study): Case 1, fixed rho = 0.2, stage-1 budget
(DE only, 3 restarts, no polish), 6 data seeds, drift-regime metrics
(kappa >= 0.5) on the shared deployment ladder. Case 4 is included as a
second service (viscous) since the corrected model changed its regime.

Run:  python eta_sweep_abs.py
"""
import json
import os
import time
import numpy as np

from config_case_studies import CASE1, CASE4, DESIGN_BOUNDS
from uncertainty_sampling import (generate_iid_samples, compute_empirical_std,
                                  generate_drift_ladder)
from optimization_objectives import make_saa_objective
from wdro_discrete import make_wdro_objective_discrete
from sthe_model import sthe_model
from metaheuristics import SOLVERS

lb, ub, dim = DESIGN_BOUNDS['lb'], DESIGN_BOUNDS['ub'], DESIGN_BOUNDS['dim']
ETAS = [1e5, 1e6, 1e7]
RHO_FIX = 0.2
N_TR, N_DEP = 60, 3000
SEEDS = [11, 23, 42, 57, 71, 88]          # first 6 of the paper's seed list
OPT_SEEDS = [42, 7, 123]                  # stage-1: DE only, 3 restarts
POP, IT = 24, 24
KAPPAS = [0.0, 0.5, 0.75, 1.0]
ALPHA = 0.95


def opt_best(obj):
    bx, bv = None, np.inf
    for s in OPT_SEEDS:
        v, x, _, _ = SOLVERS['DE'](obj, lb, ub, dim, pop_size=POP, max_iter=IT, seed=s)
        if v < bv:
            bv, bx = v, x
    return bx


def eval_regime(x, case, drift):
    """Drift-regime (kappa >= 0.5) feasibility, mean TAC, TAC-CVaR."""
    feas, mus, cvs = [], [], []
    for k in KAPPAS:
        if k < 0.5:
            continue
        tac = []
        f = 0.0
        for xi in drift[k]:
            r = sthe_model(x, xi, case)
            tac.append(r['TAC'])
            f += r['feasible_core']
        tac = np.array(tac)
        q = np.percentile(tac, ALPHA * 100)
        feas.append(f / len(drift[k]))
        mus.append(float(tac.mean()))
        cvs.append(float(tac[tac >= q].mean()))
    return float(np.mean(feas)), float(np.mean(mus)), float(np.mean(cvs))


def main():
    t0 = time.time()
    out = {}
    for case, cname in [(CASE1, 'Case1'), (CASE4, 'Case4')]:
        print(f"===== {cname} =====", flush=True)
        drift = generate_drift_ladder(case, N_DEP, KAPPAS, seed_base=500)
        rows = []
        for eta in ETAS:
            per = {'SAA': {'feas': [], 'muTAC': [], 'cvarTAC': []},
                   'WDRO': {'feas': [], 'muTAC': [], 'cvarTAC': []}}
            for s in SEEDS:
                D = generate_iid_samples(case, N_TR, seed=s)
                W = 1.0 / compute_empirical_std(D)
                x_saa = opt_best(make_saa_objective(case, D, eta))
                x_wdro = opt_best(make_wdro_objective_discrete(
                    case, D, RHO_FIX, W, lambda_penalty=eta))
                for m, x in (('SAA', x_saa), ('WDRO', x_wdro)):
                    f, mu, cv = eval_regime(x, case, drift)
                    per[m]['feas'].append(f)
                    per[m]['muTAC'].append(mu)
                    per[m]['cvarTAC'].append(cv)
            row = {'eta': eta}
            for m in ('SAA', 'WDRO'):
                row[m] = {k: [float(np.mean(v)), float(np.std(v))]
                          for k, v in per[m].items()}
            gap = np.array(per['WDRO']['feas']) - np.array(per['SAA']['feas'])
            row['gap_feas_pp'] = [float(gap.mean() * 100), float(gap.std() * 100)]
            rows.append(row)
            print(f"  eta={eta:.0e}: SAA feas={row['SAA']['feas'][0]*100:5.1f}% "
                  f"cvarTAC={row['SAA']['cvarTAC'][0]:8.0f} | "
                  f"WDRO feas={row['WDRO']['feas'][0]*100:5.1f}% "
                  f"cvarTAC={row['WDRO']['cvarTAC'][0]:8.0f} | "
                  f"gap={row['gap_feas_pp'][0]:+.1f}pp  ({time.time()-t0:.0f}s)",
                  flush=True)
        out[cname] = rows

    path = os.path.join(os.path.dirname(__file__), 'results', 'eta_sweep_abs.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Saved -> {path}  ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
