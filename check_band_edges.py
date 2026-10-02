"""Band-edge continuity audit of the Bell-Delaware ideal-bank correlations.

The banded curve fits (Taborek; Kakac & Liu, Table 8.6; Serth & Lestina,
Table 6.1) approximate single continuous curves, so the Colburn factor j and
the friction factor f must agree across every Reynolds-band edge. A large jump
at an edge indicates a defective coefficient.

The script prints the relative jump of j and f at each edge, for the audited
coefficient set and for the pre-audit set (flag legacy90), and exits with a
nonzero status if the audited set exceeds the tolerance. See Supplementary
Material S2 of the paper.

Usage:  python check_band_edges.py
"""
import sys
import sthe_model as sm

PTR = 1.25                    # pitch ratio used for the check
EDGES = (10, 100, 1000, 10000)
TOL = 0.06                    # residual mismatch of the published fits (< 6 %)


def jumps(legacy):
    rows = []
    for layout in (30, 45, 90):
        for lim in EDGES:
            j_lo, f_lo = sm.bell_delaware_jf(layout, lim * 0.9999, PTR, legacy90=legacy)
            j_hi, f_hi = sm.bell_delaware_jf(layout, lim * 1.0001, PTR, legacy90=legacy)
            rows.append((layout, lim, (j_hi - j_lo) / j_lo, (f_hi - f_lo) / f_lo))
    return rows


def report(title, rows):
    print(f"\n{title}")
    print(f"  {'layout':>6} {'Re_s edge':>10} {'jump in j':>10} {'jump in f':>10}")
    for layout, lim, dj, df in rows:
        flag = "  <-- discontinuity" if max(abs(dj), abs(df)) > TOL else ""
        print(f"  {layout:>5}d {lim:>10} {dj:>+10.1%} {df:>+10.1%}{flag}")
    worst = max(max(abs(dj), abs(df)) for _, _, dj, df in rows)
    print(f"  largest |jump| = {worst:.1%}")
    return worst


if __name__ == '__main__':
    worst_audited = report("Audited coefficient set (used for all reported results)", jumps(False))
    report("Pre-audit coefficient set (legacy90=True; for attribution only)", jumps(True))
    if worst_audited > TOL:
        sys.exit(f"audited set exceeds the {TOL:.0%} tolerance")
    print(f"\nOK: audited set is continuous within {TOL:.0%} at every band edge.")
