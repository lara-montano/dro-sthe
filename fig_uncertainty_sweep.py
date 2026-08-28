"""Star figure (2x2): top row = joint feasibility under drift for SAA and WDRO as
the uncertainty magnitude u grows (the 'scissors'); bottom row = the WDRO-SAA
feasibility gap with its bootstrap 95% CI. Robustness is worthless at low
uncertainty (gap ~ 0) and increasingly valuable as u grows."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
FIGDIR = os.path.join(HERE, 'results', 'figures')
with open(os.path.join(HERE, 'results', 'uncertainty_sweep.json')) as f:
    d = json.load(f)

TITLES = {'C1': 'Case 1 base: methanol/seawater', 'C4': 'Case 4 base: glycerol/water'}
fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), sharex='col',
                         gridspec_kw={'height_ratios': [2, 1]})
for j, bk in enumerate(['C1', 'C4']):
    rows = d['bases'][bk]
    u = [r['u'] for r in rows]
    saa = [r['saa'] * 100 for r in rows]
    wdro = [r['wdro'] * 100 for r in rows]
    gap = [r['dfeas'] * 100 for r in rows]
    lo = [r['dfeas_ci'][0] * 100 for r in rows]
    hi = [r['dfeas_ci'][1] * 100 for r in rows]

    ax = axes[0, j]
    ax.plot(u, saa, marker='s', color='#2E6E9E', lw=2, ms=6, label='SAA')
    ax.plot(u, wdro, marker='^', color='#2A9D63', lw=2, ms=6,
            label='WDRO ($\\rho=%.1f$)' % d['rho'])
    ax.fill_between(u, saa, wdro, color='#2A9D63', alpha=0.12)
    ax.set_ylabel('Joint feasibility under drift (%)')
    ax.set_title(TITLES[bk], fontsize=10)
    ax.grid(alpha=0.3); ax.set_ylim(40, 103)
    ax.legend(fontsize=9, loc='lower left')

    ax = axes[1, j]
    ax.axhline(0, color='gray', lw=1)
    ax.plot(u, gap, marker='o', color='#2A9D63', lw=2, ms=5)
    ax.fill_between(u, lo, hi, color='#2A9D63', alpha=0.25,
                    label=f"bootstrap 95% CI ({d['n_seeds']} seeds)")
    ax.set_xlabel('Uncertainty magnitude $u$ (× nominal bands)')
    ax.set_ylabel('$\\Delta$feasibility\nWDRO$-$SAA (pp)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='upper left')

fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'fig_uncertainty_sweep.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(FIGDIR, 'fig_uncertainty_sweep.png'), bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved fig_uncertainty_sweep (2x2 with CI panels)')
