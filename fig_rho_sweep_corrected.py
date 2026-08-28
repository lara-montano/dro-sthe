"""Radius-sweep figure (corrected foundations, 2x2): per case, drift-regime joint
feasibility (left axis) and pure TAC-CVaR (right axis) vs the Wasserstein radius,
with SAA as the rho->0 reference. Message: the radius is a reliability-vs-cost
dial — feasibility rises monotonically to 100% in all four services, at a
case-specific capital premium."""
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
FIGDIR = os.path.join(HERE, 'results', 'figures')
with open(os.path.join(HERE, 'results', 'rho_sweep_corrected.json')) as f:
    data = json.load(f)

TITLES = {'Case1': 'Case 1: methanol/seawater',
          'Case2': 'Case 2: kerosene/crude oil',
          'Case3': 'Case 3: distilled/raw water',
          'Case4': 'Case 4: glycerol/water'}
CASES = ['Case1', 'Case2', 'Case3', 'Case4']

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, ck in zip(axes.ravel(), CASES):
    d = data[ck]
    rhos = d['rho_grid']
    feas = [d['agg'][str(r)]['feas_mean'] * 100 for r in rhos]
    cost = [d['agg'][str(r)]['cvarTAC_mean'] / 1e3 for r in rhos]
    saa_f = d['agg']['SAA']['feas_mean'] * 100
    saa_c = d['agg']['SAA']['cvarTAC_mean'] / 1e3

    ax.axhline(saa_f, ls='--', color='#2E6E9E', lw=1.6)
    ax.plot(rhos, feas, marker='^', color='#2A9D63', lw=2, ms=7, label='WDRO feasibility')
    ax.set_xlabel('Wasserstein radius $\\rho$')
    ax.set_ylabel('Joint feasibility under drift (%)', color='#2A9D63')
    ax.tick_params(axis='y', labelcolor='#2A9D63')
    ax.set_ylim(60, 103)
    ax.set_title(TITLES[ck], fontsize=10)
    ax.grid(alpha=0.3)
    ax.text(rhos[-1], saa_f - 2.5, 'SAA feas.', color='#2E6E9E', fontsize=8, ha='right')

    ax2 = ax.twinx()
    ax2.axhline(saa_c, ls=':', color='#C77B30', lw=1.4)
    ax2.plot(rhos, cost, marker='o', color='#C77B30', lw=1.8, ms=5, alpha=0.85,
             label='WDRO cost')
    ax2.set_ylabel('TAC CVaR$_{95}$ (kUSD/yr)', color='#C77B30')
    ax2.tick_params(axis='y', labelcolor='#C77B30')
    ax2.text(rhos[0], saa_c, 'SAA cost', color='#C77B30', fontsize=8, va='bottom')

fig.suptitle('The Wasserstein radius as a reliability-vs-cost dial (drift regime, 6 seeds)',
             fontsize=11, y=1.0)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'fig_rho_sweep.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(FIGDIR, 'fig_rho_sweep.png'), bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved fig_rho_sweep (corrected, 2x2, dual-axis)')
