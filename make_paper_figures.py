"""
Centerpiece figures from consolidation_full.json (4 case studies).

New narrative: with a properly FIXED design the cost is near-deterministic, so
the decision-relevant risk is FEASIBILITY under distribution shift. Across four
case studies WDRO (at a cross-validated radius) improves reliability over SAA
by an amount that DEPENDS ON THE CASE -- large where the design is tightly
constrained (Case 1), negligible where SAA is already near-ceiling or where
i.i.d. cross-validation under-selects the radius.

fig_feasibility_vs_drift : 2x2 joint-feasibility vs kappa, 3 methods, bootstrap CI
fig_pencvar_vs_drift     : 2x2 penalized-loss CVaR vs kappa (per-case y-scale)
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
FIGDIR = os.path.join(HERE, 'results', 'figures')
os.makedirs(FIGDIR, exist_ok=True)

COLORS = {'NOMINAL': '#B0413E', 'SAA': '#2E6E9E', 'CVAR': '#8B6BB7',
          'WCMAX': '#B08C2E', 'WDRO': '#2A9D63'}
MARK = {'NOMINAL': 'o', 'SAA': 's', 'CVAR': 'D', 'WCMAX': 'v', 'WDRO': '^'}
LABEL = {'NOMINAL': 'Nominal', 'SAA': 'SAA', 'CVAR': 'CVaR$_{0.95}$',
         'WCMAX': 'Worst-case', 'WDRO': 'WDRO ($\\rho^*$)'}
METHODS = ('NOMINAL', 'SAA', 'CVAR', 'WCMAX', 'WDRO')
TITLES = {'Case1': 'Case 1: methanol/seawater (4.34 MW)',
          'Case2': 'Case 2: kerosene/crude oil (1.44 MW)',
          'Case3': 'Case 3: distilled/raw water (0.42 MW)',
          'Case4': 'Case 4: glycerol/water (1.28 MW)'}
CASES = ['Case1', 'Case2', 'Case3', 'Case4']


def save_fig(fig, stem):
    fig.savefig(os.path.join(FIGDIR, f'{stem}.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIGDIR, f'{stem}.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)


def load():
    # CANONICAL (R2): the polished consolidation with exact-lambda WDRO,
    # extended rho grid, and the CVaR / worst-case baselines.
    with open(os.path.join(HERE, 'results', 'consolidation_R2_polished.json')) as f:
        return json.load(f)


def panel(ax, res, metric, ci_key, scale=1.0):
    kappas = res['kappas']
    for m in METHODS:
        md = res['metrics'][m]
        y = np.array([md[str(k)][metric] for k in kappas]) * scale
        lo = np.array([md[str(k)][ci_key][0] for k in kappas]) * scale
        hi = np.array([md[str(k)][ci_key][1] for k in kappas]) * scale
        ax.plot(kappas, y, marker=MARK[m], color=COLORS[m], label=LABEL[m], lw=2, ms=6)
        ax.fill_between(kappas, lo, hi, color=COLORS[m], alpha=0.18, lw=0)
    ax.set_xlabel('Drift severity $\\kappa$')
    ax.grid(alpha=0.3)


def grid_fig(metric, ci_key, ylabel, stem, scale=1.0, ylim=None, legend_loc='best'):
    data = load()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, ck in zip(axes.ravel(), CASES):
        panel(ax, data[ck], metric, ci_key, scale=scale)
        ax.set_ylabel(ylabel)
        rho = data[ck].get('rho_star_mode', data[ck].get('rho_star', '?'))
        ax.set_title(f"{TITLES[ck]}  ($\\rho^*$ mode $= {rho}$)", fontsize=10)
        if ylim:
            ax.set_ylim(*ylim)
    axes.ravel()[0].legend(loc=legend_loc, fontsize=9, framealpha=0.9)
    fig.tight_layout()
    save_fig(fig, stem)


def main():
    grid_fig('feas', 'feas_ci', 'Joint feasibility (%)', 'fig_feasibility_vs_drift',
             scale=100.0, ylim=(0, 103), legend_loc='lower left')
    grid_fig('cvarTAC', 'cvarTAC_ci', 'TAC CVaR$_{95}$ (kUSD/yr)', 'fig_cost_vs_drift',
             scale=1e-3, legend_loc='upper left')

    data = load()
    for gate_key, contrast in (('gate', 'WDRO-SAA'), ('gate_cvar', 'WDRO-CVAR')):
        print(f"\n===== {contrast} (drift regime) =====")
        print(f"{'Case':>6} {'rho*mode':>8} | {'dFeas(pp)':>10} {'CI':>16} {'p_feas':>8} | {'dCost%':>8} {'p_cost':>8}")
        for ck in CASES:
            g = data[ck][gate_key]
            ci = f"[{g['dfeas_ci'][0]*100:+.1f},{g['dfeas_ci'][1]*100:+.1f}]"
            print(f"{ck:>6} {data[ck]['rho_star_mode']:>8} | {g['dfeas_mean']*100:>+9.1f} {ci:>16} "
                  f"{g['wilcoxon_feas_p']:>8.4f} | {g['dcvarTAC_rel']:>+7.1f} {g['wilcoxon_cost_p']:>8.4f}")
    print("\n===== CVAR-SAA (descriptive) =====")
    for ck in CASES:
        d = data[ck]['desc_cvar_saa']
        print(f"{ck:>6}  dFeas={d['dfeas_mean']*100:+.1f}pp  dCost={d['dcvarTAC_rel']:+.1f}%")
    print("\n===== rho* histograms =====")
    for ck in CASES:
        print(f"{ck:>6}  {data[ck]['rho_star_hist']}")


if __name__ == '__main__':
    main()
