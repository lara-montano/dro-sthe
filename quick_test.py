"""
Quick test/demo of the DRO STHE optimization framework.

Runs a minimal experiment to verify all components work together.
Use --full for publication-quality runs.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from config_case_studies import CASE1, CASE2, DESIGN_BOUNDS, EXPERIMENT_CONFIG, get_xi_bounds
from sthe_model import sthe_model
from uncertainty_sampling import (
    generate_train_test, generate_drift_ladder, compute_empirical_std
)
from optimization_objectives import (
    make_deterministic_objective, make_saa_objective_subsample,
    make_wdro_objective_subsample
)
from metaheuristics import SOLVERS
from evaluation_metrics import (
    evaluate_design_on_dataset, compute_metrics,
    evaluate_across_drift, print_metrics_table
)
from visualization import generate_all_plots


def test_sthe_model():
    """Test the STHE model with nominal conditions."""
    print("=" * 60)
    print("  Test 1: STHE model evaluation at nominal conditions")
    print("=" * 60)

    case = CASE1
    xi_0, _, _ = get_xi_bounds(case)

    # A reasonable design vector (12 vars; last = tube length L in m)
    x_test = np.array([0.6, 11, 0.3, 1.5, 1.5, 1.5, 0.35, 1.2, 0.05, 0.05, 0.88, 5.0])

    res = sthe_model(x_test, xi_0, case)

    print(f"  TAC         = {res['TAC']:.2f} USD/yr")
    print(f"  Q_achieved  = {res['Q']:.0f} W")
    print(f"  Q_required  = {res['Q_req']:.0f} W")
    print(f"  dP_tube     = {res['dP_t']:.0f} Pa (max {case['dP_t_max']:.0f})")
    print(f"  dP_shell    = {res['dP_s']:.0f} Pa (max {case['dP_s_max']:.0f})")
    print(f"  Area        = {res['Area']:.2f} m²")
    print(f"  L_tubes     = {res['L_tubes']:.3f} m")
    print(f"  U           = {res['U']:.1f} W/(m²·K)")
    print(f"  v_tube      = {res['v_t']:.3f} m/s")
    print(f"  N_tubes     = {res['N_tubes']}")
    print(f"  Feasible    = {res['feasible']}")
    print(f"  g_Q = {res['g_Q']:.0f}, g_t = {res['g_t']:.0f}, g_s = {res['g_s']:.0f}")
    print()

    return res


def test_uncertainty_sampling():
    """Test data generation."""
    print("=" * 60)
    print("  Test 2: Uncertainty sampling")
    print("=" * 60)

    case = CASE1
    D_tr, D_te = generate_train_test(case, N_tr=100, N_te=100, seed_tr=42, seed_te=123)
    print(f"  Training set: {D_tr.shape}")
    print(f"  Test set:     {D_te.shape}")
    print(f"  Train mean:   {D_tr.mean(axis=0)}")
    print(f"  Train std:    {D_tr.std(axis=0)}")

    drift = generate_drift_ladder(case, N_dep=100, kappa_ladder=[0, 0.5, 1.0])
    for k, v in drift.items():
        print(f"  Drift kappa={k:.2f}: shape={v.shape}, mean={v.mean(axis=0)[:4].round(2)}")
    print()
    return D_tr, D_te


def test_deterministic_optimization():
    """Test deterministic optimization with DE."""
    print("=" * 60)
    print("  Test 3: Deterministic optimization (DE, small budget)")
    print("=" * 60)

    case = CASE1
    lb = DESIGN_BOUNDS['lb']
    ub = DESIGN_BOUNDS['ub']
    dim = DESIGN_BOUNDS['dim']
    lam_pen = 1e4

    fobj = make_deterministic_objective(case, lam_pen)

    t0 = time.time()
    best_val, best_x, conv, elapsed = SOLVERS['DE'](
        fobj, lb, ub, dim, pop_size=30, max_iter=30, seed=42
    )
    print(f"  Best objective: {best_val:.2f}")
    print(f"  Best design:    {np.round(best_x, 4)}")
    print(f"  Time:           {elapsed:.1f}s")

    # Evaluate at nominal
    xi_0, _, _ = get_xi_bounds(case)
    res = sthe_model(best_x, xi_0, case)
    print(f"  TAC={res['TAC']:.2f}, dP_t={res['dP_t']:.0f}, dP_s={res['dP_s']:.0f}, feas={res['feasible']}")
    print()
    return best_x, best_val


def run_quick_experiment():
    """Run a minimal version of the full experiment."""
    print("=" * 60)
    print("  Test 4: Quick full pipeline (Case 1)")
    print("=" * 60)

    case = CASE1
    case_name = "Case1_Quick"
    lb = DESIGN_BOUNDS['lb']
    ub = DESIGN_BOUNDS['ub']
    dim = DESIGN_BOUNDS['dim']
    lam_pen = 1e4

    # Small settings
    N_tr, N_te, N_dep = 80, 100, 200
    pop_size, max_iter, n_runs = 20, 20, 2
    alpha = 0.95
    kappa_ladder = [0.0, 0.5, 1.0]

    # Generate data
    print("  Generating data...")
    D_tr, D_te = generate_train_test(case, N_tr, N_te)
    drift_datasets = generate_drift_ladder(case, N_dep, kappa_ladder)
    sigma_tr = compute_empirical_std(D_tr)
    W_diag = 1.0 / sigma_tr

    # Build objectives
    obj_det = make_deterministic_objective(case, lam_pen)
    obj_saa = make_saa_objective_subsample(
        case, D_tr, subsample_size=30, lambda_penalty=lam_pen,
        rng=np.random.RandomState(42)
    )
    obj_wdro = make_wdro_objective_subsample(
        case, D_tr, rho=0.1, W_diag=W_diag,
        subsample_size=30, lambda_penalty=lam_pen, n_lambda=5,
        rng=np.random.RandomState(42)
    )

    methods = {
        'DET': obj_det,
        'SAA': obj_saa,
        'WDRO': obj_wdro,
    }

    # Optimize each method with DE
    designs = {}
    for name, fobj in methods.items():
        print(f"  Optimizing {name}... ", end="", flush=True)
        t0 = time.time()
        best_val, best_x, conv, elapsed = SOLVERS['DE'](
            fobj, lb, ub, dim, pop_size=pop_size, max_iter=max_iter, seed=42
        )
        # Second run
        val2, x2, _, _ = SOLVERS['DE'](
            fobj, lb, ub, dim, pop_size=pop_size, max_iter=max_iter, seed=99
        )
        if val2 < best_val:
            best_val, best_x = val2, x2
        designs[name] = best_x
        print(f"obj={best_val:.2f}, t={time.time()-t0:.1f}s")

    # Evaluate
    print("\n  Evaluating across drift ladder...")
    evaluation_results = {}
    for name, x in designs.items():
        drift_metrics = evaluate_across_drift(x, case, drift_datasets, alpha)
        res_te = evaluate_design_on_dataset(x, D_te, case)
        metrics_te = compute_metrics(res_te, alpha)
        evaluation_results[name] = {
            'iid_test': metrics_te,
            'drift': drift_metrics,
        }
        print_metrics_table(drift_metrics, name)

    # Generate plots
    output_dir = os.path.join(os.path.dirname(__file__), 'results_quick')
    os.makedirs(output_dir, exist_ok=True)
    generate_all_plots(evaluation_results, case_name, output_dir)
    print(f"\n  Plots saved to {output_dir}/")

    return designs, evaluation_results


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  DRO STHE Optimization Framework - Quick Tests")
    print("=" * 60 + "\n")

    # Test 1: Model
    test_sthe_model()

    # Test 2: Sampling
    test_uncertainty_sampling()

    # Test 3: Deterministic opt
    test_deterministic_optimization()

    # Test 4: Full pipeline
    designs, eval_results = run_quick_experiment()

    print("\n" + "=" * 60)
    print("  All tests completed successfully!")
    print("=" * 60)
