"""
Structured results I/O for reproducible experiments.

Saves to:  results/<case>/<timestamp>/
    config.json              — full experiment configuration + seeds
    best_designs.csv         — best design vector per method
    metrics_iid_test.csv     — i.i.d. test metrics per method
    metrics_by_kappa.csv     — all metrics per method × kappa
    convergence_curves.csv   — convergence per method × solver × run
    solver_stats.csv         — per-run objective values and times
    diagnostics.json         — sanity-check counters (failures, dominant violations)
"""

import os
import json
import csv
import datetime
import numpy as np


def _make_output_dir(base_dir, case_name):
    """Create timestamped output directory."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base_dir, case_name, ts)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def save_config(out_dir, case_name, config, extra=None):
    """
    Save full experiment configuration to config.json.

    Includes all seeds, sample sizes, bounds, rho, mode, etc.
    """
    payload = {
        'case': case_name,
        'timestamp': datetime.datetime.now().isoformat(),
        'config': {},
    }
    for k, v in config.items():
        if isinstance(v, (int, float, str, bool, list)):
            payload['config'][k] = v
        elif isinstance(v, np.ndarray):
            payload['config'][k] = v.tolist()
        elif isinstance(v, np.integer):
            payload['config'][k] = int(v)
        elif isinstance(v, np.floating):
            payload['config'][k] = float(v)
    if extra:
        payload.update(extra)

    path = os.path.join(out_dir, 'config.json')
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2, default=str)
    return path


def save_best_designs(out_dir, best_designs, design_names=None):
    """
    Save best_designs to CSV.

    best_designs: {method: {'x': ndarray, 'val': float, 'solver': str}}
    """
    if design_names is None:
        design_names = [f'x{i}' for i in range(11)]

    path = os.path.join(out_dir, 'best_designs.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['method', 'solver', 'obj_value'] + design_names
        writer.writerow(header)
        for method, info in best_designs.items():
            row = [method, info['solver'], f"{info['val']:.6f}"]
            row += [f"{v:.8f}" for v in info['x']]
            writer.writerow(row)
    return path


def save_metrics_iid(out_dir, evaluation_results):
    """Save i.i.d. test metrics to CSV."""
    path = os.path.join(out_dir, 'metrics_iid_test.csv')
    methods = list(evaluation_results.keys())
    if not methods:
        return path
    metric_keys = list(evaluation_results[methods[0]]['iid_test'].keys())

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method'] + metric_keys)
        for method in methods:
            m = evaluation_results[method]['iid_test']
            row = [method] + [f"{m[k]:.8f}" for k in metric_keys]
            writer.writerow(row)
    return path


def save_metrics_by_kappa(out_dir, evaluation_results):
    """Save drift metrics to CSV: one row per (method, kappa)."""
    path = os.path.join(out_dir, 'metrics_by_kappa.csv')
    methods = list(evaluation_results.keys())
    if not methods:
        return path

    # Get metric keys from first available entry
    first_drift = evaluation_results[methods[0]]['drift']
    first_kappa = sorted(first_drift.keys())[0]
    metric_keys = list(first_drift[first_kappa].keys())

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'kappa'] + metric_keys)
        for method in methods:
            drift = evaluation_results[method]['drift']
            for kappa in sorted(drift.keys()):
                m = drift[kappa]
                row = [method, f"{kappa:.4f}"]
                row += [f"{m[k]:.8f}" for k in metric_keys]
                writer.writerow(row)
    return path


def save_solver_stats(out_dir, all_results):
    """
    Save per-run solver statistics.

    all_results: {method: {solver: {'all_vals': [...], 'mean_time': float, ...}}}
    """
    path = os.path.join(out_dir, 'solver_stats.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'solver', 'run', 'obj_value', 'time_s'])
        for method, solvers in all_results.items():
            for solver, info in solvers.items():
                for run_i, (val, t) in enumerate(
                        zip(info['all_vals'], info.get('all_times', [0]*len(info['all_vals'])))):
                    writer.writerow([method, solver, run_i,
                                     f"{val:.6f}", f"{t:.3f}"])
    return path


def save_convergence_curves(out_dir, all_results):
    """
    Save convergence curves to CSV.

    One row per (method, solver, run, iteration).
    """
    path = os.path.join(out_dir, 'convergence_curves.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'solver', 'run', 'iteration', 'best_value'])
        for method, solvers in all_results.items():
            for solver, info in solvers.items():
                for run_i, conv in enumerate(info.get('all_convergence', [])):
                    if conv is not None:
                        for it, val in enumerate(conv):
                            writer.writerow([method, solver, run_i, it,
                                             f"{val:.6f}"])
    return path


def save_diagnostics(out_dir, diagnostics):
    """
    Save sanity-check diagnostics to JSON.

    diagnostics: {method: {
        'n_eval_failures': int,
        'failure_rate': float,
        'dominant_violation': str,
        'constraint_profiles': {...},
        ...
    }}
    """
    path = os.path.join(out_dir, 'diagnostics.json')
    # Convert numpy types
    clean = {}
    for method, diag in diagnostics.items():
        clean[method] = {}
        for k, v in diag.items():
            if isinstance(v, (np.integer, np.int64)):
                clean[method][k] = int(v)
            elif isinstance(v, (np.floating, np.float64)):
                clean[method][k] = float(v)
            elif isinstance(v, np.ndarray):
                clean[method][k] = v.tolist()
            elif isinstance(v, dict):
                clean[method][k] = {
                    kk: float(vv) if isinstance(vv, (np.floating, np.float64))
                    else int(vv) if isinstance(vv, (np.integer, np.int64))
                    else vv
                    for kk, vv in v.items()
                }
            else:
                clean[method][k] = v

    with open(path, 'w') as f:
        json.dump(clean, f, indent=2, default=str)
    return path


def save_all_results(out_dir, case_name, config, best_designs,
                      evaluation_results, all_results, diagnostics,
                      extra_config=None):
    """
    One-call function to save everything.

    Returns
    -------
    out_dir : str
        Path to the output directory with all saved files.
    """
    save_config(out_dir, case_name, config, extra=extra_config)
    save_best_designs(out_dir, best_designs)
    save_metrics_iid(out_dir, evaluation_results)
    save_metrics_by_kappa(out_dir, evaluation_results)
    save_solver_stats(out_dir, all_results)
    save_convergence_curves(out_dir, all_results)
    save_diagnostics(out_dir, diagnostics)

    print(f"\n  All results saved to: {out_dir}/")
    for f in sorted(os.listdir(out_dir)):
        size = os.path.getsize(os.path.join(out_dir, f))
        print(f"    {f:40s}  {size:>8,d} bytes")

    return out_dir
