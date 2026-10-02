# dro-sthe — Wasserstein DRO for Shell-and-Tube Heat Exchanger Design

Complete implementation for the paper:

> *Wasserstein Distributionally Robust Optimization for Shell-and-Tube Heat
> Exchanger Design Under Operational Uncertainty and Distribution Shift*,
> O. D. Lara-Montaño, S. I. Martínez-Guido, F. I. Gómez-Castro,
> Computers & Chemical Engineering (manuscript CACE-D-26-00376).

A geometrically fixed shell-and-tube heat exchanger (12 design variables) is
rated at each operating scenario with an effectiveness–NTU / Bell–Delaware
model; designs are optimized under a nominal, SAA, percentile, empirical-CVaR,
worst-case-over-samples, or Wasserstein-DRO objective and stress-tested on a
train–test–shift protocol (deployment drift ladder). The WDRO dual is solved
over a discretized candidate support with the outer dual variable minimized
**exactly** (bisection on the convex piecewise-linear dual), so every reported
WDRO value is the exact optimum of the support-restricted problem — a valid
lower bound on the continuous worst case with no regularity assumptions.

## Module map

| Module | Paper section | Contents |
|---|---|---|
| `config_case_studies.py` | Sec. 5.1–5.2 | Case specifications (Caputo et al. 2008 benchmarks + glycerol service), design bounds, uncertainty bands, glycerol property correlations (Cheng 2008) |
| `tube_diameters.py` | Sec. 2 | TEMA tube OD/ID lookup |
| `sthe_model.py` | Sec. 2 + Supp. S1–S3 | Bell–Delaware + ε-NTU rating of a fixed design; audited coefficient tables; `smooth_regimes` and `props_of_T` sensitivity variants; `legacy_r1_model` A/B flag |
| `uncertainty_sampling.py` | Sec. 3 | i.i.d. training draws, deployment drift ladder |
| `wdro_core.py` | Sec. 4.1, 4.6 | Ground cost, penalized loss |
| `wdro_discrete.py` | Sec. 4.6.4 | Candidate support, exact 1-D convex dual solve (`wdro_dual_discrete_exact`), densification diagnostics |
| `optimization_objectives.py` | Sec. 4.2–4.6 | Nominal, SAA, percentile, empirical-CVaR, worst-case objectives |
| `metaheuristics.py` | Sec. 4.8 | DE (and GWO/SBOA) with fixed-seed interface |
| `evaluation_metrics.py` | Sec. 3.3 | Joint feasibility (core and full-operability), velocity-window rate, TAC mean/CVaR |

## Reproducing the paper

Each table/figure has one runner. All seeds are fixed in the scripts; runs are
deterministic end to end.

| Result | Runner | Output |
|---|---|---|
| Reliability and cost tables, feasibility/cost vs drift figures (headline, 16 seeds, polished) | `python consolidation_run.py --polish` (or `--cases N` per case, in parallel) | `results/consolidation_R2_polished*.json` |
| Headline figures + gate summaries | `python make_paper_figures.py` | `results/figures/` |
| Radius sweep figure | `python rho_sweep.py` then `python fig_rho_sweep_corrected.py` | `results/rho_sweep_corrected.json` |
| Uncertainty-magnitude sweep figure | `python uncertainty_sweep.py` then `python fig_uncertainty_sweep.py` | `results/uncertainty_sweep.json` |
| Penalty-weight (η) sensitivity, absolute metrics | `python eta_sweep_abs.py` | `results/eta_sweep_abs.json` |
| Dense-support optimization study | `python densify_opt_study.py` (after the consolidation run) | `results/densify_opt_study.json` |
| Calibration stability at N_tr = 120 | `python ntr_stability_check.py` | `results/ntr_stability.json` |
| Case-3 smoothed-regime sensitivity | `python consolidation_run.py --polish --variant smooth --cases 3` | `results/consolidation_R2_smooth_polished_Case3.json` |
| Case-4 property sensitivity | `python consolidation_run.py --polish --variant propsT --cases 4` (also `props65`, `props85`) | `results/consolidation_R2_props*_polished_Case4.json` |
| PCTL diagnostic | `python pctl_diag_run.py` | `results/pctl_diag_R2.json` |
| R1-model A/B attribution (legacy coefficients) | `python consolidation_run.py --polish --variant r1model` | `results/consolidation_R2_r1model_polished*.json` |
| Sanity gate for all R2 changes | `python pilot_gate_r2.py` | `results/pilot_r2.json` |
| Band-edge continuity audit of the correlation set (Supp. S2) | `python check_band_edges.py` | console table |

Aggregated result JSONs behind the published tables/figures are included under
`results/` so numbers can be checked without re-running.

## Environment

Python 3.13; `pip install -r requirements.txt` (numpy 2.3.5, scipy 1.16.3,
matplotlib 3.10.6). Single-threaded; the four case pipelines can be run as
concurrent processes. Reference wall-clock times are reported in Section 6.5
of the paper.

## Implementation audit (second revision)

The Bell–Delaware coefficient tables were verified coefficient-by-coefficient
against the published set (Taborek, as compiled by Shah & Sekulić and
Serna & Jiménez 2005), including a numerical continuity check at every
Reynolds-band edge. Two transcription errors and one missing interpolation
band present in the first-revision code were found and corrected (see
Supplementary Material S2 of the paper); `legacy_r1_model=True` in a case
dict reproduces the pre-audit behavior for attribution.

## Consistency checks

Every number in the paper is produced by running the scripts in this
repository. The consistency checks used during development are included:
`pilot_gate_r2.py` (among other checks, the identity between the WDRO
objective at rho = 0 and the SAA objective) and `check_band_edges.py` (the
band-edge continuity audit of the correlation set).

## License

MIT — see `LICENSE`.
