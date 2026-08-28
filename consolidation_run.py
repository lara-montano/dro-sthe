"""
CONSOLIDATION RUN (audit-corrected): final numbers for the revised paper.

Corrected model (fixed geometry + epsilon-NTU, absolute-K uncertainty bands) and
the discretized-support WDRO surrogate. Audit-driven changes vs the first run:

  M4  rho is selected by NESTED k-fold CV PER DATA SEED (not one hard-coded seed),
      and the CV criterion is VALIDATION JOINT FEASIBILITY (lambda-invariant),
      NOT penCVaR. The distribution of rho* across seeds is reported.
  M2  Primary metrics are joint feasibility (%) and PURE TAC-CVaR (cost). The
      cost-of-robustness Delta(cvarTAC) WDRO-vs-SAA is reported explicitly.
      penCVaR is kept only as a secondary/appendix diagnostic, never cross-case.
  M1  More data seeds; Holm-Bonferroni correction across all case x metric tests.
  M7  Final designs use 3 restarts; CV uses a lighter budget only to rank rho.

  M7b Optional '--polish': after DE, every design (NOMINAL/SAA/WDRO alike) is
      polished by deterministic bounded Nelder-Mead multistart on the continuous
      variables with discretes fixed (R1-min3). Batch A showed DE-only SAA
      designs were under-converged (polish gains 4-13%), inflating the WDRO gap;
      polished numbers are the defensible headline.

  R2 revision (second major revision, reviewer #1):
  1a  The WDRO dual is solved EXACTLY in lambda (bisection on the convex
      piecewise-linear dual; wdro_discrete.solver='exact' is now the default).
  2a  Joint feasibility is reported BOTH as the core metric (duty + two
      pressure drops, the paper Eq. 19 definition) and as the full metric
      including the tube-side velocity window and L/D; the velocity-window
      violation rate is also reported separately.
  3a  RHO_GRID now includes 0 (WDRO can degenerate to SAA) and near-zero
      radii; rho=0 uses the SAA objective directly (mathematically identical:
      the radius-0 Wasserstein ball is the empirical distribution).
  3b  The CV records per-fold validation cost AND feasibility for every rho,
      enabling the post-hoc combined-criterion analysis in the supplement.
  5   Two risk-averse empirical baselines added under the identical protocol:
      CVAR (CVaR_0.95 of the empirical penalized loss) and WCMAX (worst case
      over the training scenarios).
      Confirmatory tests remain the WDRO-SAA family (Holm across 8 tests);
      the WDRO-CVAR family is corrected separately (Holm across 8 tests) and
      CVAR-SAA is reported descriptively.
      Designs and per-seed metrics are stored for re-analysis.

Run:
    python consolidation_run.py --polish             # full run (all 4 cases)
    python consolidation_run.py --polish --cases 2   # one case (parallel launch)
    python consolidation_run.py --smoke              # tiny smoke test
"""
import sys
import json
import time
import os
import numpy as np
from scipy.stats import wilcoxon
from scipy.optimize import minimize as scipy_minimize

from config_case_studies import (CASE1, CASE2, CASE3, CASE4, DESIGN_BOUNDS,
                                 get_xi_bounds)
from uncertainty_sampling import (generate_iid_samples, compute_empirical_std,
                                  generate_drift_ladder)
from optimization_objectives import (make_deterministic_objective,
                                     make_saa_objective,
                                     make_saa_cvar_objective,
                                     make_worstcase_objective)
from wdro_discrete import make_wdro_objective_discrete
from sthe_model import sthe_model
from wdro_core import penalized_loss
from metaheuristics import SOLVERS

LAM = 1e6
RHO_GRID = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40]   # R2-3a: includes 0 and near-0
KAPPAS = [0.0, 0.25, 0.50, 0.75, 1.00]
ALPHA = 0.95
METHODS = ('NOMINAL', 'SAA', 'CVAR', 'WCMAX', 'WDRO')
lb, ub, dim = DESIGN_BOUNDS['lb'], DESIGN_BOUNDS['ub'], DESIGN_BOUNDS['dim']

SMOKE = '--smoke' in sys.argv
POLISH = '--polish' in sys.argv
CONT_IDX = [0, 6, 7, 8, 9, 10, 11]    # continuous design variables (discretes fixed in polish)
if SMOKE:
    N_TR, N_DEP = 40, 400
    POP, IT = 14, 14
    CV_POP, CV_IT = 12, 12
    DATA_SEEDS = [11, 23]
    CV_FOLDS = 2
    OPT_SEEDS = [42, 7]
else:
    N_TR, N_DEP = 60, 3000
    POP, IT = 24, 24
    CV_POP, CV_IT = 18, 18
    DATA_SEEDS = [11, 23, 42, 57, 71, 88, 103, 119, 134, 150,
                  167, 181, 202, 219, 233, 251]     # 16 seeds (M1)
    CV_FOLDS = 3
    OPT_SEEDS = [42, 7, 123]                          # 3 restarts (M7)


def opt_best(make_obj, seeds, pop=POP, it=IT, polish=False):
    bx, bv = None, np.inf
    for s in seeds:
        v, x, _, _ = SOLVERS['DE'](make_obj, lb, ub, dim, pop_size=pop, max_iter=it, seed=s)
        if v < bv:
            bv, bx = v, x
    if polish:
        bx, bv = _polish(make_obj, bx, bv)
    return bx


def _polish(obj, x0, v0, n_starts=8, seed=0, maxiter=300):
    """Deterministic bounded Nelder-Mead multistart on the continuous variables,
    discrete variables fixed at the DE solution (R1-min3). Applied identically
    to every method, so the comparison stays paired and fair."""
    rng = np.random.RandomState(seed)
    bounds = [(lb[i], ub[i]) for i in CONT_IDX]

    def wrapped(z):
        x = x0.copy()
        x[CONT_IDX] = z
        return obj(x)

    best_z, best_v = x0[CONT_IDX].copy(), v0
    starts = [x0[CONT_IDX].copy()] + [
        np.array([rng.uniform(lo, hi) for lo, hi in bounds]) for _ in range(n_starts - 1)]
    for z0 in starts:
        try:
            r = scipy_minimize(wrapped, z0, method='Nelder-Mead', bounds=bounds,
                               options={'maxiter': maxiter, 'xatol': 1e-5, 'fatol': 1e-3})
            if r.fun < best_v:
                best_v, best_z = float(r.fun), r.x.copy()
        except Exception:
            continue
    x = x0.copy()
    x[CONT_IDX] = np.clip(best_z, lb[CONT_IDX], ub[CONT_IDX])
    return x, best_v


def cvar(a, alpha=ALPHA):
    a = np.asarray(a)
    q = np.percentile(a, alpha * 100)
    tail = a[a >= q]
    return float(tail.mean()) if len(tail) else float(q)


def eval_design(x, case, D):
    """Feasibility (core = paper Eq. 19, and full incl. velocity window and
    L/D), velocity-window violation rate, pure TAC-CVaR (cost), penCVaR."""
    tac = np.empty(len(D)); pen = np.empty(len(D))
    feas = np.empty(len(D), dtype=bool); feasF = np.empty(len(D), dtype=bool)
    vok = np.empty(len(D), dtype=bool)
    for i, xi in enumerate(D):
        r = sthe_model(x, xi, case)
        tac[i] = r['TAC']
        pen[i] = penalized_loss(x, xi, case, LAM)
        feas[i] = r['feasible_core']
        feasF[i] = r['feasible_full']
        vok[i] = r['v_ok']
    return dict(feas=float(feas.mean()), feasFull=float(feasF.mean()),
                vViol=float(1.0 - vok.mean()),
                cvarTAC=cvar(tac), muTAC=float(tac.mean()), penCVaR=cvar(pen))


def val_metrics(x, case, D_val):
    """Validation joint feasibility (CV criterion) and mean TAC (recorded for
    the R2-3b combined-criterion analysis; not used for selection)."""
    feas = 0.0; mu = 0.0
    for xi in D_val:
        r = sthe_model(x, xi, case)
        feas += float(r['feasible_core'])
        mu += r['TAC']
    n = max(len(D_val), 1)
    return feas / n, mu / n


def _wdro_or_saa_objective(case, D_tr, rho, W):
    """WDRO objective; at rho = 0 the radius-0 Wasserstein ball collapses to
    the empirical distribution, so the SAA objective is used directly
    (identical value, |S|-times cheaper). Equivalence asserted in the pilot."""
    if rho <= 0.0:
        return make_saa_objective(case, D_tr, LAM)
    return make_wdro_objective_discrete(case, D_tr, rho, W, lambda_penalty=LAM)


def select_rho_cv(case, seed):
    """
    Nested k-fold CV on the TRAINING draw of one data seed (M4). Criterion:
    maximize validation JOINT FEASIBILITY (lambda-invariant). Ties -> smaller
    rho (with 0 in the grid, a tie at the feasibility ceiling degenerates
    WDRO to SAA -- the R2-3a requirement). Per-fold validation cost and
    feasibility are recorded for every rho (R2-3b).
    """
    D = generate_iid_samples(case, N_TR, seed=seed)
    rng = np.random.RandomState(seed)
    folds = np.array_split(rng.permutation(len(D)), CV_FOLDS)
    scores = {}
    detail = []
    for rho in RHO_GRID:
        vals = []
        for f in range(CV_FOLDS):
            val_idx = folds[f]
            tr_idx = np.concatenate([folds[g] for g in range(CV_FOLDS) if g != f])
            D_tr, D_val = D[tr_idx], D[val_idx]
            W = 1.0 / compute_empirical_std(D_tr)
            obj = _wdro_or_saa_objective(case, D_tr, rho, W)
            x = opt_best(obj, OPT_SEEDS[:1], pop=CV_POP, it=CV_IT)
            v_feas, v_mu = val_metrics(x, case, D_val)
            vals.append(v_feas)
            detail.append({'rho': rho, 'fold': f, 'val_feas': v_feas,
                           'val_muTAC': v_mu})
        scores[rho] = float(np.mean(vals))
    best = max(scores.values())
    rho_star = min([r for r in RHO_GRID if scores[r] >= best - 1e-9])  # tie -> smaller
    return rho_star, scores, detail


def bootstrap_ci(vals, n_boot=5000, seed=0):
    vals = np.asarray(vals, dtype=float)
    rng = np.random.RandomState(seed)
    means = [vals[rng.randint(0, len(vals), len(vals))].mean() for _ in range(n_boot)]
    return float(np.mean(vals)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))


def safe_wilcoxon(d):
    d = np.asarray(d)
    if np.allclose(d, 0):
        return 1.0
    try:
        return float(wilcoxon(d).pvalue)
    except Exception:
        return 1.0


METRIC_KEYS = ('feas', 'feasFull', 'vViol', 'cvarTAC', 'muTAC', 'penCVaR')


def _paired_contrast(per, a, b, di, n_seeds, metric):
    """Per-seed drift-regime means of `metric` for methods a and b."""
    va = np.array([[per[a][k][metric][i] for k in di] for i in range(n_seeds)]).mean(axis=1)
    vb = np.array([[per[b][k][metric][i] for k in di] for i in range(n_seeds)]).mean(axis=1)
    return va, vb


def _gate_block(per, a, b, di, n_seeds):
    """Drift-regime paired contrast a-b: feasibility and TAC-CVaR."""
    af, bf = _paired_contrast(per, a, b, di, n_seeds, 'feas')
    ac, bc = _paired_contrast(per, a, b, di, n_seeds, 'cvarTAC')
    dfeas, dcost = af - bf, ac - bc
    mdf, lodf, hidf = bootstrap_ci(dfeas)
    mdc, lodc, hidc = bootstrap_ci(dcost)
    return dict(
        dfeas_mean=mdf, dfeas_ci=[lodf, hidf],
        wilcoxon_feas_p=safe_wilcoxon(dfeas), cliffs_delta_feas=cliffs_delta(af, bf),
        dcvarTAC_mean=mdc, dcvarTAC_ci=[lodc, hidc],
        dcvarTAC_rel=float(mdc / max(np.mean(bc), 1e-9) * 100),
        wilcoxon_cost_p=safe_wilcoxon(dcost),
    )


def run_case(case, cname):
    t0 = time.time()
    print(f"\n########## {cname} ##########", flush=True)

    x_nom = opt_best(make_deterministic_objective(case, LAM), OPT_SEEDS, polish=POLISH)
    drift = generate_drift_ladder(case, N_DEP, KAPPAS, seed_base=500)   # shared deployment

    per = {m: {k: {mk: [] for mk in METRIC_KEYS} for k in KAPPAS} for m in METHODS}
    rho_hist, cv_details = [], []
    designs = {m: [] for m in METHODS}
    for si, s in enumerate(DATA_SEEDS):
        D_tr = generate_iid_samples(case, N_TR, seed=s)
        W = 1.0 / compute_empirical_std(D_tr)
        rho_star, _, cv_det = select_rho_cv(case, s)   # per-seed rho (M4)
        rho_hist.append(rho_star)
        cv_details.append(cv_det)
        x_saa = opt_best(make_saa_objective(case, D_tr, LAM), OPT_SEEDS, polish=POLISH)
        x_cvar = opt_best(make_saa_cvar_objective(case, D_tr, ALPHA, LAM), OPT_SEEDS, polish=POLISH)
        x_wc = opt_best(make_worstcase_objective(case, D_tr, LAM), OPT_SEEDS, polish=POLISH)
        x_wdro = opt_best(_wdro_or_saa_objective(case, D_tr, rho_star, W),
                          OPT_SEEDS, polish=POLISH)
        for m, x in (('NOMINAL', x_nom), ('SAA', x_saa), ('CVAR', x_cvar),
                     ('WCMAX', x_wc), ('WDRO', x_wdro)):
            designs[m].append([float(v) for v in x])
            for k in KAPPAS:
                e = eval_design(x, case, drift[k])
                for mk in METRIC_KEYS:
                    per[m][k][mk].append(e[mk])
        print(f"  seed {s} ({si+1}/{len(DATA_SEEDS)}) rho*={rho_star} ({time.time()-t0:.0f}s)", flush=True)

    out = {'case': cname, 'n_seeds': len(DATA_SEEDS), 'kappas': KAPPAS,
           'rho_grid': RHO_GRID, 'rho_star_hist': rho_hist,
           'rho_star_mode': float(max(set(rho_hist), key=rho_hist.count)),
           'cv_details': cv_details, 'designs': designs,
           'per_seed': {m: {str(k): {mk: per[m][k][mk] for mk in METRIC_KEYS}
                            for k in KAPPAS} for m in METHODS},
           'metrics': {}, 'gate': {}, 'gate_cvar': {}, 'desc_cvar_saa': {}}
    for m in METHODS:
        out['metrics'][m] = {}
        for k in KAPPAS:
            mf, lof, hif = bootstrap_ci(per[m][k]['feas'])
            mff, loff, hiff = bootstrap_ci(per[m][k]['feasFull'])
            mc, loc, hic = bootstrap_ci(per[m][k]['cvarTAC'])
            out['metrics'][m][str(k)] = dict(
                feas=mf, feas_ci=[lof, hif],
                feasFull=mff, feasFull_ci=[loff, hiff],
                vViol=float(np.mean(per[m][k]['vViol'])),
                cvarTAC=mc, cvarTAC_ci=[loc, hic],
                penCVaR=float(np.mean(per[m][k]['penCVaR'])))

    # ---- Gates (drift regime kappa>=0.5): confirmatory WDRO-SAA family, ----
    # ---- secondary WDRO-CVAR family, descriptive CVAR-SAA               ----
    di = [k for k in KAPPAS if k >= 0.5]
    n_seeds = len(DATA_SEEDS)
    out['gate'] = _gate_block(per, 'WDRO', 'SAA', di, n_seeds)
    out['gate_cvar'] = _gate_block(per, 'WDRO', 'CVAR', di, n_seeds)
    cf, sf = _paired_contrast(per, 'CVAR', 'SAA', di, n_seeds, 'feas')
    cc, sc = _paired_contrast(per, 'CVAR', 'SAA', di, n_seeds, 'cvarTAC')
    mdf, lodf, hidf = bootstrap_ci(cf - sf)
    mdc, lodc, hidc = bootstrap_ci(cc - sc)
    out['desc_cvar_saa'] = dict(dfeas_mean=mdf, dfeas_ci=[lodf, hidf],
                                dcvarTAC_mean=mdc, dcvarTAC_ci=[lodc, hidc],
                                dcvarTAC_rel=float(mdc / max(np.mean(sc), 1e-9) * 100))

    g = out['gate']; g2 = out['gate_cvar']
    print(f"\n{cname} VERDICT (drift regime, {n_seeds} seeds, rho* mode={out['rho_star_mode']}):")
    print(f"  dFeas WDRO-SAA  = {g['dfeas_mean']*100:+.1f}pp CI[{g['dfeas_ci'][0]*100:+.1f},{g['dfeas_ci'][1]*100:+.1f}] "
          f"p={g['wilcoxon_feas_p']:.4f} delta={g['cliffs_delta_feas']:+.2f}")
    print(f"  cost-of-robustness vs SAA dTAC-CVaR = {g['dcvarTAC_rel']:+.1f}% p={g['wilcoxon_cost_p']:.4f}")
    print(f"  dFeas WDRO-CVAR = {g2['dfeas_mean']*100:+.1f}pp CI[{g2['dfeas_ci'][0]*100:+.1f},{g2['dfeas_ci'][1]*100:+.1f}] "
          f"p={g2['wilcoxon_feas_p']:.4f}  dCost {g2['dcvarTAC_rel']:+.1f}%")
    print(f"  dFeas CVAR-SAA  = {out['desc_cvar_saa']['dfeas_mean']*100:+.1f}pp (descriptive)")
    print(f"  ({time.time()-t0:.0f}s)")
    return out


def holm_bonferroni(pvals_named):
    """Return {name: (p_raw, p_holm, reject@0.05)}."""
    items = sorted(pvals_named.items(), key=lambda kv: kv[1])
    m = len(items)
    out = {}
    max_adj = 0.0
    for i, (name, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        max_adj = max(max_adj, adj)     # enforce monotonicity
        out[name] = (p, max_adj, max_adj < 0.05)
    return out


def apply_variant(case, variant):
    """Return a case-dict copy with the requested R2 sensitivity variant."""
    c = dict(case)
    if variant == 'smooth':
        c['smooth_regimes'] = True          # R2-4a smoothed regime boundaries
    elif variant == 'propsT':
        c['props_of_T'] = 'glycerol'        # R2-4b scenario-T shell properties
    elif variant in ('props65', 'props85'):
        from config_case_studies import make_case4_props_variant
        c = make_case4_props_variant({'props65': 'const65', 'props85': 'const85'}[variant])
    elif variant == 'r1model':
        c['legacy_r1_model'] = True         # R1 J_r step + transposed 90-deg
                                            # friction exponent (A/B attribution)
    elif variant:
        raise SystemExit(f"unknown --variant {variant}")
    return c


def _cli_value(flag, default=None):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


def main():
    t0 = time.time()
    outdir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(outdir, exist_ok=True)
    tag = 'smoke' if SMOKE else ('polished' if POLISH else 'full')
    variant = _cli_value('--variant', '')
    vtag = f'_{variant}' if variant else ''
    cases_arg = _cli_value('--cases', '1,2,3,4')
    wanted = [int(t) for t in cases_arg.replace(',', ' ').split()]

    all_cases = {1: (CASE1, 'Case1'), 2: (CASE2, 'Case2'),
                 3: (CASE3, 'Case3'), 4: (CASE4, 'Case4')}

    # One checkpoint file PER CASE so parallel per-case launches never race.
    for n in wanted:
        case, cname = all_cases[n]
        cpath = os.path.join(outdir, f'consolidation_R2{vtag}_{tag}_{cname}.json')
        if os.path.exists(cpath):
            print(f"[resume] {cname} checkpoint exists -- skipping", flush=True)
            continue
        res = run_case(apply_variant(case, variant), cname)
        with open(cpath, 'w') as f:
            json.dump(res, f, indent=2)
        print(f"  [checkpoint] saved {cname} -> {cpath}", flush=True)

    # Combine whatever case files exist; apply the two Holm families when all
    # four cases are present (confirmatory WDRO-SAA; secondary WDRO-CVAR).
    results = {}
    for n in (1, 2, 3, 4):
        _, cname = all_cases[n]
        cpath = os.path.join(outdir, f'consolidation_R2{vtag}_{tag}_{cname}.json')
        if os.path.exists(cpath):
            with open(cpath) as f:
                results[cname] = json.load(f)
    if len(results) == 4:
        pv1 = {}
        pv2 = {}
        for cname, r in results.items():
            pv1[f"{cname}:feas"] = r['gate']['wilcoxon_feas_p']
            pv1[f"{cname}:cost"] = r['gate']['wilcoxon_cost_p']
            pv2[f"{cname}:feas"] = r['gate_cvar']['wilcoxon_feas_p']
            pv2[f"{cname}:cost"] = r['gate_cvar']['wilcoxon_cost_p']
        holm1 = holm_bonferroni(pv1)
        holm2 = holm_bonferroni(pv2)
        results['_holm'] = {k: {'p_raw': v[0], 'p_holm': v[1], 'reject': v[2]}
                            for k, v in holm1.items()}
        results['_holm_cvar'] = {k: {'p_raw': v[0], 'p_holm': v[1], 'reject': v[2]}
                                 for k, v in holm2.items()}
        combined = os.path.join(outdir, f'consolidation_R2{vtag}_{tag}.json')
        with open(combined, 'w') as f:
            json.dump(results, f, indent=2)
        print("\n===== Holm (WDRO-SAA confirmatory family) =====")
        for k, v in sorted(holm1.items(), key=lambda kv: kv[1][0]):
            print(f"  {k:16} p_raw={v[0]:.4f}  p_holm={v[1]:.4f}  {'REJECT (sig)' if v[2] else 'ns'}")
        print("===== Holm (WDRO-CVAR secondary family) =====")
        for k, v in sorted(holm2.items(), key=lambda kv: kv[1][0]):
            print(f"  {k:16} p_raw={v[0]:.4f}  p_holm={v[1]:.4f}  {'REJECT (sig)' if v[2] else 'ns'}")
        print(f"\nSaved -> {combined}")
    else:
        print(f"\n[{len(results)}/4 case files present; combined file deferred]")
    print(f"Total {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
