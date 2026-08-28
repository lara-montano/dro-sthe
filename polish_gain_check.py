"""
R2 diagnostics: measure the deterministic-polish objective gains under the
CORRECTED model (the R1-reported gains were measured pre-audit and cannot be
cited). For 2 seeds x 4 cases, SAA and WDRO (rho = 0.1): DE incumbent value
vs. polished value, and the per-evaluation cost of one WDRO objective call vs
one SAA call on this platform.

Run:  python polish_gain_check.py
"""
import json
import os
import time
import numpy as np

from config_case_studies import CASE1, CASE2, CASE3, CASE4, DESIGN_BOUNDS
from uncertainty_sampling import generate_iid_samples, compute_empirical_std
from optimization_objectives import make_saa_objective
from wdro_discrete import make_wdro_objective_discrete
from consolidation_run import opt_best, _polish, OPT_SEEDS
from metaheuristics import SOLVERS

lb, ub, dim = DESIGN_BOUNDS['lb'], DESIGN_BOUNDS['ub'], DESIGN_BOUNDS['dim']
LAM = 1e6
SEEDS = [11, 42]


def de_then_polish(obj):
    bx, bv = None, np.inf
    for s in OPT_SEEDS:
        v, x, _, _ = SOLVERS['DE'](obj, lb, ub, dim, pop_size=24, max_iter=24, seed=s)
        if v < bv:
            bv, bx = v, x
    px, pv = _polish(obj, bx, bv)
    return float(bv), float(pv)


def main():
    t0 = time.time()
    out = {'gains': [], 'timing': {}}
    for case, cname in [(CASE1, 'Case1'), (CASE2, 'Case2'),
                        (CASE3, 'Case3'), (CASE4, 'Case4')]:
        for s in SEEDS:
            D = generate_iid_samples(case, 60, seed=s)
            W = 1.0 / compute_empirical_std(D)
            for m, obj in (('SAA', make_saa_objective(case, D, LAM)),
                           ('WDRO', make_wdro_objective_discrete(case, D, 0.1, W,
                                                                 lambda_penalty=LAM))):
                v_de, v_pol = de_then_polish(obj)
                gain = (v_de - v_pol) / max(abs(v_de), 1e-9) * 100
                out['gains'].append(dict(case=cname, seed=s, method=m,
                                         de=v_de, polished=v_pol, gain_pct=gain))
                print(f"{cname} seed {s} {m:5}: DE {v_de:.4g} -> polished {v_pol:.4g} "
                      f"({gain:+.2f}%)  ({time.time()-t0:.0f}s)", flush=True)

    # per-evaluation timing on this platform
    case = CASE1
    D = generate_iid_samples(case, 60, seed=11)
    W = 1.0 / compute_empirical_std(D)
    x0 = (lb + ub) / 2.0
    f_saa = make_saa_objective(case, D, LAM)
    f_wdro = make_wdro_objective_discrete(case, D, 0.1, W, lambda_penalty=LAM)
    for name, f in (('SAA', f_saa), ('WDRO', f_wdro)):
        f(x0)
        t1 = time.time()
        n = 50
        for _ in range(n):
            f(x0)
        out['timing'][name + '_ms'] = (time.time() - t1) / n * 1e3
    out['timing']['support_size'] = int(getattr(f_wdro, 'support_size', -1))
    print("timing:", out['timing'])

    gains = [g['gain_pct'] for g in out['gains']]
    by = {}
    for g in out['gains']:
        by.setdefault(g['method'], []).append(g['gain_pct'])
    for m, v in by.items():
        print(f"{m}: polish gain {min(v):.2f}..{max(v):.2f}% (median {np.median(v):.2f}%)")

    path = os.path.join(os.path.dirname(__file__), 'results', 'polish_gain_R2.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Saved -> {path}  ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
