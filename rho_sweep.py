"""
Radius sensitivity sweep (R1-M6) — CORRECTED FOUNDATIONS.

Re-run of the rho sweep after the audit fixes (absolute-Kelvin uncertainty bands
M6, regime-consistent friction m4). The pre-fix sweep was archived under
results/archive_stale_prekelvinfix/ and must NOT be cited.

For each case: sweep rho, measure drift-regime (kappa>=0.5) joint feasibility and
PURE TAC-CVaR (audit M2: no penCVaR headline), with SAA as the rho->0 reference,
over several data seeds. Output: results/rho_sweep_corrected.json
"""
import json
import os
import time
import numpy as np
from config_case_studies import CASE1, CASE2, CASE3, CASE4, DESIGN_BOUNDS
from uncertainty_sampling import (generate_iid_samples, compute_empirical_std,
                                  generate_drift_ladder)
from optimization_objectives import make_saa_objective
from wdro_discrete import make_wdro_objective_discrete
from sthe_model import sthe_model
from metaheuristics import SOLVERS

LAM = 1e6
# R2: near-zero radii added so the dial connects continuously to the SAA
# reference (the honest calibration grid frequently selects rho* = 0).
RHO_GRID = [0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60]
KAPPAS = [0.0, 0.25, 0.50, 0.75, 1.00]
N_TR, N_DEP, POP, IT = 60, 3000, 24, 24
SEEDS = [11, 23, 42, 57, 71, 88]
OPT_SEEDS = [42, 7, 123]
lb, ub, dim = DESIGN_BOUNDS['lb'], DESIGN_BOUNDS['ub'], DESIGN_BOUNDS['dim']
DI = [k for k in KAPPAS if k >= 0.5]


def opt_best(obj, seeds):
    bx, bv = None, np.inf
    for s in seeds:
        v, x, _, _ = SOLVERS['DE'](obj, lb, ub, dim, pop_size=POP, max_iter=IT, seed=s)
        if v < bv:
            bv, bx = v, x
    return bx


def cvar(a, alpha=0.95):
    a = np.asarray(a); q = np.percentile(a, alpha * 100); t = a[a >= q]
    return float(t.mean()) if len(t) else float(q)


def drift_metrics(x, case, drift):
    fs, cs = [], []
    for k in DI:
        D = drift[k]
        feas = np.empty(len(D), dtype=bool); tac = np.empty(len(D))
        for i, xi in enumerate(D):
            r = sthe_model(x, xi, case)
            feas[i] = r['feasible_core']; tac[i] = r['TAC']
        fs.append(feas.mean()); cs.append(cvar(tac))
    return float(np.mean(fs)), float(np.mean(cs))


def run_case(case, cname):
    t0 = time.time()
    drift = generate_drift_ladder(case, N_DEP, KAPPAS, seed_base=500)
    rows = {'SAA': {'feas': [], 'cvarTAC': []}}
    rows.update({str(r): {'feas': [], 'cvarTAC': []} for r in RHO_GRID})
    for s in SEEDS:
        D_tr = generate_iid_samples(case, N_TR, seed=s)
        W = 1.0 / compute_empirical_std(D_tr)
        x_saa = opt_best(make_saa_objective(case, D_tr, LAM), OPT_SEEDS)
        f, c = drift_metrics(x_saa, case, drift)
        rows['SAA']['feas'].append(f); rows['SAA']['cvarTAC'].append(c)
        for r in RHO_GRID:
            x = opt_best(make_wdro_objective_discrete(case, D_tr, r, W, lambda_penalty=LAM), OPT_SEEDS)
            f, c = drift_metrics(x, case, drift)
            rows[str(r)]['feas'].append(f); rows[str(r)]['cvarTAC'].append(c)
        print(f"  {cname} seed {s} done ({time.time()-t0:.0f}s)", flush=True)
    agg = {}
    for key, d in rows.items():
        agg[key] = dict(feas_mean=float(np.mean(d['feas'])), feas_sd=float(np.std(d['feas'])),
                        cvarTAC_mean=float(np.mean(d['cvarTAC'])), cvarTAC_sd=float(np.std(d['cvarTAC'])))
    print(f"\n{cname} corrected radius sweep (drift regime, {len(SEEDS)} seeds):")
    print(f"  {'radius':>8} {'feas%':>8} {'cvarTAC':>10}")
    print(f"  {'SAA':>8} {agg['SAA']['feas_mean']*100:8.1f} {agg['SAA']['cvarTAC_mean']:10.0f}")
    for r in RHO_GRID:
        print(f"  {r:>8} {agg[str(r)]['feas_mean']*100:8.1f} {agg[str(r)]['cvarTAC_mean']:10.0f}")
    return {'rho_grid': RHO_GRID, 'agg': agg, 'n_seeds': len(SEEDS)}


def main():
    outdir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, 'rho_sweep_corrected.json')
    out = {}
    for case, cname in [(CASE1, 'Case1'), (CASE2, 'Case2'), (CASE3, 'Case3'), (CASE4, 'Case4')]:
        print(f"\n########## {cname} ##########", flush=True)
        out[cname] = run_case(case, cname)
        with open(path, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"  [checkpoint] {cname} saved", flush=True)


if __name__ == '__main__':
    main()
