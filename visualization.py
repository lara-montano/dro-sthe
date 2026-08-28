"""
Visualization and reporting utilities for DRO STHE results.

Generates:
- Drift degradation curves (TAC vs kappa)
- Feasibility comparison plots
- Cost distribution plots
- Summary tables
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Style
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

METHOD_STYLES = {
    'DET':  {'color': '#2196F3', 'marker': 's', 'ls': '--', 'label': 'Deterministic'},
    'SAA':  {'color': '#4CAF50', 'marker': '^', 'ls': '-.',  'label': 'SAA'},
    'PCTL': {'color': '#FF9800', 'marker': 'D', 'ls': ':',   'label': 'Percentile'},
    'WDRO': {'color': '#E91E63', 'marker': 'o', 'ls': '-',   'label': 'WDRO'},
}


def plot_drift_degradation(evaluation_results, metric_key='mu_TAC',
                           ylabel='Mean TAC (USD/yr)', title='Cost Degradation Under Drift',
                           output_path=None):
    """
    Plot metric vs. drift severity kappa for all methods.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for method, data in evaluation_results.items():
        drift = data['drift']
        kappas = sorted(drift.keys())
        vals = [drift[k][metric_key] for k in kappas]

        style = METHOD_STYLES.get(method, {'color': 'gray', 'marker': 'x', 'ls': '-', 'label': method})
        ax.plot(kappas, vals, color=style['color'], marker=style['marker'],
                ls=style['ls'], label=style['label'], markersize=6, linewidth=1.5)

    ax.set_xlabel(r'Drift severity $\kappa$')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    return fig


def plot_feasibility_degradation(evaluation_results, output_path=None):
    """Plot joint feasibility probability vs. kappa."""
    return plot_drift_degradation(
        evaluation_results, metric_key='joint_feasibility',
        ylabel='Joint Feasibility Probability',
        title='Feasibility Degradation Under Drift',
        output_path=output_path
    )


def plot_cvar_degradation(evaluation_results, output_path=None):
    """Plot CVaR_95 vs. kappa."""
    return plot_drift_degradation(
        evaluation_results, metric_key='CVaR',
        ylabel=r'CVaR$_{0.95}$ TAC (USD/yr)',
        title='Tail-Cost Degradation Under Drift',
        output_path=output_path
    )


def plot_combined_panels(evaluation_results, case_name='', output_path=None):
    """
    Combined 2x2 panel: mu_TAC, CVaR, feasibility, violation probability.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f'Robustness Under Distribution Shift — {case_name}', fontsize=14, y=1.01)

    panels = [
        (axes[0, 0], 'mu_TAC', r'Mean TAC (USD/yr)'),
        (axes[0, 1], 'CVaR', r'CVaR$_{0.95}$ TAC (USD/yr)'),
        (axes[1, 0], 'joint_feasibility', 'Joint Feasibility Probability'),
        (axes[1, 1], 'p_viol_Q', 'Duty Violation Probability'),
    ]

    for ax, metric_key, ylabel in panels:
        for method, data in evaluation_results.items():
            drift = data['drift']
            kappas = sorted(drift.keys())
            vals = [drift[k][metric_key] for k in kappas]

            style = METHOD_STYLES.get(method, {'color': 'gray', 'marker': 'x', 'ls': '-', 'label': method})
            ax.plot(kappas, vals, color=style['color'], marker=style['marker'],
                    ls=style['ls'], label=style['label'], markersize=5, linewidth=1.5)

        ax.set_xlabel(r'Drift severity $\kappa$')
        ax.set_ylabel(ylabel)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    return fig


def generate_latex_table(evaluation_results, alpha=0.95, output_path=None):
    """
    Generate LaTeX table of metrics across drift levels.
    """
    methods = list(evaluation_results.keys())
    kappas = sorted(list(evaluation_results[methods[0]]['drift'].keys()))

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Performance metrics under distribution shift.}")
    lines.append(r"\label{tab:drift_metrics}")
    lines.append(r"\small")

    # Subtable for TAC
    lines.append(r"\begin{tabular}{l" + "r" * len(kappas) + "}")
    lines.append(r"\toprule")
    header = r"Method & " + " & ".join([rf"$\kappa={k:.2f}$" for k in kappas]) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Mean TAC
    lines.append(r"\multicolumn{" + str(len(kappas) + 1) + r"}{l}{\textbf{Mean TAC (USD/yr)}} \\")
    for method in methods:
        label = METHOD_STYLES.get(method, {}).get('label', method)
        vals = [evaluation_results[method]['drift'][k]['mu_TAC'] for k in kappas]
        row = f"{label} & " + " & ".join([f"{v:,.0f}" for v in vals]) + r" \\"
        lines.append(row)

    lines.append(r"\midrule")

    # Joint Feasibility
    lines.append(r"\multicolumn{" + str(len(kappas) + 1) + r"}{l}{\textbf{Joint Feasibility (\%)}} \\")
    for method in methods:
        label = METHOD_STYLES.get(method, {}).get('label', method)
        vals = [evaluation_results[method]['drift'][k]['joint_feasibility'] * 100 for k in kappas]
        row = f"{label} & " + " & ".join([f"{v:.1f}" for v in vals]) + r" \\"
        lines.append(row)

    lines.append(r"\midrule")

    # CVaR
    lines.append(r"\multicolumn{" + str(len(kappas) + 1) + r"}{l}{\textbf{CVaR$_{0.95}$ TAC (USD/yr)}} \\")
    for method in methods:
        label = METHOD_STYLES.get(method, {}).get('label', method)
        vals = [evaluation_results[method]['drift'][k]['CVaR'] for k in kappas]
        row = f"{label} & " + " & ".join([f"{v:,.0f}" for v in vals]) + r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    table_str = "\n".join(lines)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(table_str)

    return table_str


def generate_all_plots(evaluation_results, case_name, output_dir):
    """Generate all plots for a case study."""
    os.makedirs(output_dir, exist_ok=True)

    plot_drift_degradation(
        evaluation_results, metric_key='mu_TAC',
        ylabel='Mean TAC (USD/yr)',
        title=f'{case_name} — Mean Cost Under Drift',
        output_path=os.path.join(output_dir, 'drift_mean_tac.pdf')
    )

    plot_feasibility_degradation(
        evaluation_results,
        output_path=os.path.join(output_dir, 'drift_feasibility.pdf')
    )

    plot_cvar_degradation(
        evaluation_results,
        output_path=os.path.join(output_dir, 'drift_cvar.pdf')
    )

    plot_combined_panels(
        evaluation_results, case_name=case_name,
        output_path=os.path.join(output_dir, 'combined_panels.pdf')
    )

    generate_latex_table(
        evaluation_results,
        output_path=os.path.join(output_dir, 'metrics_table.tex')
    )

    print(f"  All plots saved to {output_dir}/")
