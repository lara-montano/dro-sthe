"""
PILOT GATE R2: verify every R2 code change and read the direction of the
result deltas BEFORE committing to the full re-runs (PLAN_REVISION_R2 sec. 5).

Checks
------
A  Exact-lambda dual vs. legacy grid dual at fixed designs (rel. gap, lam*),
   plus the rho=0 == SAA identity (R2-1a, R2-3a).
B  feasible_core vs feasible_full and velocity-violation rate on the drift
   ladder (R2-2a): does the velocity window change the feasibility numbers?
C  J_r audit correction impact (R2-4a): rate the SAME designs with the
   literature J_r interpolation vs. the R1 step (jr_step_legacy).
D  Case-4 property variants (R2-4b): scenario-T props, const-65C, const-85C.
E  Case-3 smoothed-regime rating (R2-4a): base vs smooth_regimes.
F  Quick CVaR / worst-case baselines (R2-5): where do they land vs SAA/WDRO?

Budget: quick DE (pop 14, it 14, 2 restarts, no polish), ladder N=800,
kappa in {0, 0.5, 1}. Runtime target: minutes.
"""
import json
import os
import time
import numpy as np

from config_case_studies import (CASE1, CASE2, CASE3, CASE4, DESIGN_BOUNDS,
                                 make_case4_props_variant)
from uncertainty_sampling import (generate_iid_samples, compute_empirical_std,
                                  generate_drift_ladder)
from optimization_objectives import (make_deterministic_objective,
                                     make_saa_objective,
                                     make_saa_cvar_objective,
                                     make_worstcase_objective)
from wdro_discrete import (make_wdro_objective_discrete, build_candidate_support,
                           precompute_cost_matrix, wdro_dual_discrete,
                           wdro_dual_discrete_exact, DEFAULT_LAMBDAS)
from wdro_core import penalized_loss
from sthe_model import sthe_model
from metaheuristics import SOLVERS

lb, ub, dim = DESIGN_BOUNDS['lb'], DESIGN_BOUNDS['ub'], DESIGN_BOUNDS['dim']
LAM = 1e6
N_TR, N_LAD = 60, 800
KAP = [0.0, 0.5, 1.0]
SEED = 11
PILOT_RHO = 0.10


def qopt(obj, seeds=(42, 7)):
    bx, bv = None, np.inf
    for s in seeds:
        v, x, _, _ = SOLVERS['DE'](obj, lb, ub, dim, pop_size=14, max_iter=14, seed=s)
        if v < bv:
            bv, bx = v, x
    return bx


def rate(x, case, D):
    out = {'feas': 0.0, 'feasFull': 0.0, 'vviol': 0.0, 'tac': []}
    for xi in D:
        r = sthe_model(x, xi, case)
        out['feas'] += r['feasible_core']
        out['feasFull'] += r['feasible_full']
        out['vviol'] += (not r['v_ok'])
        out['tac'].append(r['TAC'])
    n = len(D)
    tac = np.array(out['tac'])
    q = np.percentile(tac, 95)
    return dict(feas=out['feas'] / n, feasFull=out['feasFull'] / n,
                vviol=out['vviol'] / n, cvarTAC=float(tac[tac >= q].mean()))


def regime_feas(x, case, drift, key='feas'):
    vals = [rate(x, case, drift[k])[key] for k in KAP if k >= 0.5]
    return float(np.mean(vals))


def main():
    t0 = time.time()
    report = {}
    for case, cname in [(CASE1, 'Case1'), (CASE2, 'Case2'),
                        (CASE3, 'Case3'), (CASE4, 'Case4')]:
        print(f"\n===== {cname} =====", flush=True)
        rep = {}
        D = generate_iid_samples(case, N_TR, seed=SEED)
        W = 1.0 / compute_empirical_std(D)
        drift = generate_drift_ladder(case, N_LAD, KAP, seed_base=500)

        # --- quick designs -------------------------------------------------
        x_nom = qopt(make_deterministic_objective(case, LAM))
        x_saa = qopt(make_saa_objective(case, D, LAM))
        x_cvar = qopt(make_saa_cvar_objective(case, D, 0.95, LAM))
        x_wc = qopt(make_worstcase_objective(case, D, LAM))
        x_wdro = qopt(make_wdro_objective_discrete(case, D, PILOT_RHO, W,
                                                   lambda_penalty=LAM))
        designs = {'NOM': x_nom, 'SAA': x_saa, 'CVAR': x_cvar,
                   'WCMAX': x_wc, 'WDRO': x_wdro}

        # --- A: exact vs grid dual + rho=0 identity ------------------------
        S = build_candidate_support(case, D, n_lhs=160, seed=0)
        C = precompute_cost_matrix(S, D, W)
        A = {}
        for dname in ('SAA', 'WDRO'):
            x = designs[dname]
            L_S = np.array([penalized_loss(x, S[j], case, LAM) for j in range(len(S))])
            for rho in (0.05, 0.40):
                g_grid = wdro_dual_discrete(L_S, C, rho, DEFAULT_LAMBDAS)
                g_ex, info = wdro_dual_discrete_exact(L_S, C, rho, return_info=True)
                A[f"{dname}_rho{rho}"] = dict(
                    grid=g_grid, exact=g_ex,
                    rel_gap=(g_grid - g_ex) / max(abs(g_ex), 1e-9),
                    lam_star=info['lam_star'], cert_gap=info['cert_gap'])
            # rho = 0 identity vs SAA objective
            saa_val = float(np.mean([penalized_loss(x, D[i], case, LAM)
                                     for i in range(len(D))]))
            wdro0 = wdro_dual_discrete_exact(L_S, C, 0.0)
            A[f"{dname}_rho0_identity"] = dict(
                wdro0=wdro0, saa=saa_val,
                rel=(wdro0 - saa_val) / max(abs(saa_val), 1e-9))
        rep['A_exact_vs_grid'] = A
        worst_gap = max(abs(v['rel_gap']) for k, v in A.items() if 'rel_gap' in v)
        id_err = max(abs(v['rel']) for k, v in A.items() if 'rel' in v)
        print(f"  A: max |grid-exact|/exact = {worst_gap:.3e}; rho0-SAA identity err = {id_err:.3e}")

        # --- B: core vs full feasibility on the ladder ---------------------
        B = {}
        for dname, x in designs.items():
            r0 = rate(x, case, drift[0.0]); r1 = rate(x, case, drift[1.0])
            B[dname] = dict(k0=r0, k1=r1)
            print(f"  B: {dname:5} k=0 feas={r0['feas']*100:5.1f}% full={r0['feasFull']*100:5.1f}% vviol={r0['vviol']*100:4.1f}% | "
                  f"k=1 feas={r1['feas']*100:5.1f}% full={r1['feasFull']*100:5.1f}% vviol={r1['vviol']*100:4.1f}%")
        rep['B_core_vs_full'] = {k: {kk: vv for kk, vv in v.items()} for k, v in B.items()}

        # --- C: audit-correction impact (same designs, legacy R1 model vs ----
        # --- corrected: J_r interpolation + 90-deg friction exponent fix) ----
        case_legacy = dict(case); case_legacy['legacy_r1_model'] = True
        Cc = {}
        for dname in ('NOM', 'SAA', 'WDRO'):
            x = designs[dname]
            f_new = regime_feas(x, case, drift)
            f_old = regime_feas(x, case_legacy, drift)
            k0_new = rate(x, case, drift[0.0])['feas']
            k0_old = rate(x, case_legacy, drift[0.0])['feas']
            Cc[dname] = dict(corrected=f_new, legacy=f_old,
                             delta_pp=(f_new - f_old) * 100,
                             k0_corrected=k0_new, k0_legacy=k0_old,
                             k0_delta_pp=(k0_new - k0_old) * 100)
        rep['C_audit_model_impact'] = Cc
        print(f"  C: audit corrections dFeas (drift regime / k=0): "
              + ", ".join(f"{k} {v['delta_pp']:+.1f}/{v['k0_delta_pp']:+.1f}pp"
                          for k, v in Cc.items()))

        # --- E: Case-3 smoothed rating -------------------------------------
        if cname == 'Case3':
            case_sm = dict(case); case_sm['smooth_regimes'] = True
            E = {}
            for dname in ('NOM', 'SAA', 'WDRO'):
                x = designs[dname]
                E[dname] = dict(
                    base_k0=rate(x, case, drift[0.0])['feas'],
                    base_k1=rate(x, case, drift[1.0])['feas'],
                    sm_k0=rate(x, case_sm, drift[0.0])['feas'],
                    sm_k1=rate(x, case_sm, drift[1.0])['feas'])
                print(f"  E: {dname:5} feas k0 {E[dname]['base_k0']*100:.1f}->{E[dname]['sm_k0']*100:.1f}% "
                      f"k1 {E[dname]['base_k1']*100:.1f}->{E[dname]['sm_k1']*100:.1f}% (base->smooth)")
            rep['E_smooth_c3'] = E

        # --- D: Case-4 property variants -----------------------------------
        if cname == 'Case4':
            Dv = {}
            variants = {
                'scenarioT': make_case4_props_variant('scenarioT'),
                'const65': make_case4_props_variant('const65'),
                'const85': make_case4_props_variant('const85'),
            }
            for dname in ('SAA', 'WDRO'):
                x = designs[dname]
                row = {'base': regime_feas(x, case, drift)}
                for vn, vc in variants.items():
                    row[vn] = regime_feas(x, vc, drift)
                Dv[dname] = row
                print(f"  D: {dname:5} regime feas base={row['base']*100:.1f}% "
                      f"scenT={row['scenarioT']*100:.1f}% c65={row['const65']*100:.1f}% c85={row['const85']*100:.1f}%")
            gap_base = (Dv['WDRO']['base'] - Dv['SAA']['base']) * 100
            gaps = {vn: (Dv['WDRO'][vn] - Dv['SAA'][vn]) * 100
                    for vn in ('scenarioT', 'const65', 'const85')}
            print(f"  D: WDRO-SAA gap: base {gap_base:+.1f}pp | " +
                  " ".join(f"{k} {v:+.1f}pp" for k, v in gaps.items()))
            rep['D_props_c4'] = Dv

        # --- F: baseline landscape -----------------------------------------
        F = {d: regime_feas(x, case, drift) for d, x in designs.items()}
        rep['F_regime_feas'] = F
        print("  F: drift-regime feas: " +
              " ".join(f"{k}={v*100:.1f}%" for k, v in F.items()))

        report[cname] = rep

    outp = os.path.join(os.path.dirname(__file__), 'results', 'pilot_r2.json')
    os.makedirs(os.path.dirname(outp), exist_ok=True)

    def scrub(o):
        if isinstance(o, dict):
            return {k: scrub(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [scrub(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        return o

    with open(outp, 'w') as f:
        json.dump(scrub(report), f, indent=2)
    print(f"\nSaved -> {outp}  ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
