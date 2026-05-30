"""
layer2_speed_validation.py
============================
Track 1 Markov Chain — Layer 2 Physical Plausibility Filter

Layer 2 contains a single substantive check:
    Per-leg speed plausibility — is the transit time physically achievable
    given the sea-route distance between consecutive ports?

Distance strategy:
    Primary:  sea_distance_matrix.pkl  (port_name → port_name, nautical miles)
    Fallback: great-circle × GC_CORRECTION_FACTOR for port pairs not in matrix
              (factor 1.3 reflects average sea-route / great-circle ratio)

Design rationale (see thesis Section X):
    - Port reachability is not separately checked: all ports are globally
      connected by sea, and any reachability concern is already captured by
      the speed check below.
    - Dwell time is not separately checked: already validated at the population
      level by Layer 1, and individual extreme values are negligible given
      LogNormal sampling from observed data.
    - Total span and no-repeat constraints are enforced during generation and
      not re-checked here.

Speed thresholds are derived empirically from the observed AIS data:
    min_speed (p5 per vessel type)  → upper bound on transit time
    max_speed (p95 per vessel type) → lower bound on transit time

Filter formula (per leg):
    min_transit_h = dist_nm / (max_speed_kt * SPEED_UPPER_BUFFER)
    max_transit_h = dist_nm / (min_speed_kt * SPEED_LOWER_BUFFER)
    PASS if  min_transit_h <= actual_transit_h <= max_transit_h

A voyage fails Layer 2 if ANY of its legs fails the speed check.

Outputs:
    data/markov/validation.result/mc.validation/layer2_speed_filter/speed_thresholds.csv
    data/markov/validation.result/mc.validation/layer2_speed_filter/speed_distribution.png
    data/markov/validation.result/mc.validation/layer2_speed_filter/mc.syn.all.layer2.csv
    data/markov/validation.result/mc.validation/layer2_speed_filter/layer2_report.txt
"""

import ast
import re
import warnings
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════
# 0. Configuration
# ══════════════════════════════════════════════════════════

SCRIPT_DIR      = Path(__file__).resolve().parent
BASE_DIR        = SCRIPT_DIR.parent.parent.parent   # → track1_markovchain/
OBS_PATH        = BASE_DIR / 'data/clustering/cluster.m1.labels.k5.csv'
SYN_PATH        = BASE_DIR / 'data/markov/output/mc.syn.all.csv'
SEA_DIST_PATH   = SCRIPT_DIR.parent / 'mc_generator/sea_distance_matrix.pkl'
OUT_DIR         = BASE_DIR / 'data/markov/validation.result/mc.validation/layer2_speed_filter'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Fallback correction: great-circle × this factor when port pair not in matrix.
# 1.3 is a conservative average sea-route / great-circle ratio.
GC_CORRECTION_FACTOR = 1.3

# Speed filter buffers
# Upper: v95 × 1.05
# Lower: v5  × 1.00  (p5 is the hard lower bound, no further relaxation)
SPEED_UPPER_BUFFER = 1.05   # applied to p95 speed → upper plausibility ceiling
SPEED_LOWER_BUFFER = 1.00   # applied to p5  speed → lower plausibility floor

# Sanity bounds for speed derivation from observed data.
# Legs outside these bounds are data artefacts, excluded from threshold fitting.
OBS_SPEED_MIN_KT = 0.5    # slower than this → vessel at anchor or data error
OBS_SPEED_MAX_KT = 30.0   # faster than this → AIS interpolation artefact

# Minimum distance to apply speed check.
# Legs shorter than this are often within a port complex;
# skipped to avoid noisy speed estimates.
MIN_DIST_NM = 50.0

# Fallback speed thresholds used when a vessel type has fewer than MIN_OBS_LEGS
# observations — set conservatively wide to avoid spurious rejections.
MIN_OBS_LEGS    = 20
FALLBACK_MIN_KT = 2.0
FALLBACK_MAX_KT = 22.0

COLOR_OBS = '#2E86AB'
COLOR_SYN = '#E84855'

# ══════════════════════════════════════════════════════════
# 1. Utilities
# ══════════════════════════════════════════════════════════

def haversine_nm(lat1, lon1, lat2, lon2):
    """Great-circle distance in nautical miles (used as fallback only)."""
    R_NM = 3440.065
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return R_NM * 2 * atan2(sqrt(a), sqrt(1 - a))


def load_sea_matrix(path: Path) -> dict:
    """
    Load sea distance matrix pickle.
    Expected structure: {'km': {...}, 'nm': {(port_a, port_b): float}, 'source': {...}}
    Returns the 'nm' sub-dict directly. Falls back to empty dict on failure.
    """
    import pickle
    try:
        with open(path, 'rb') as f:
            mat = pickle.load(f)
        nm_dict = mat['nm']
        print(f"[SeaDist] Loaded {len(nm_dict):,} port pairs (nm) from {path.name}")
        return nm_dict
    except Exception as e:
        print(f"[SeaDist] WARNING: could not load matrix ({e}). "
              f"All legs will use great-circle x {GC_CORRECTION_FACTOR}.")
        return {}


def get_dist_nm(port_a: str, port_b: str,
                lat_a: float, lon_a: float,
                lat_b: float, lon_b: float,
                sea_matrix: dict) -> tuple:
    """
    Return (distance_nm, source) where source is 'sea' or 'gc_fallback'.
    Lookup order:
        1. (port_a, port_b) in matrix
        2. (port_b, port_a) in matrix  (symmetric fallback)
        3. great-circle × GC_CORRECTION_FACTOR
    """
    for key in [(port_a, port_b), (port_b, port_a)]:
        if key in sea_matrix:
            return float(sea_matrix[key]), 'sea'
    gc = haversine_nm(lat_a, lon_a, lat_b, lon_b)
    return gc * GC_CORRECTION_FACTOR, 'gc_fallback'


def safe_eval(val):
    """Parse list fields from CSV (handles np.float64 etc.)."""
    if pd.isna(val):
        return []
    s = re.sub(r'np\.\w+\(([^)]+)\)', r'\1', str(val))
    try:
        return ast.literal_eval(s)
    except Exception:
        return []


def load_df(path, list_cols):
    df = pd.read_csv(path)
    for col in list_cols:
        if col in df.columns:
            df[col] = df[col].apply(safe_eval)
    return df


# ══════════════════════════════════════════════════════════
# 2. Derive speed thresholds from observed data
# ══════════════════════════════════════════════════════════

def derive_speed_thresholds(obs: pd.DataFrame, sea_matrix: dict):
    """
    For every leg in every observed voyage, compute actual speed (knots).

    Returns two threshold tables:
      global_thresh   : DataFrame indexed by vessel_type (fallback)
      cluster_thresh  : dict {(cluster, vtype): {'p5_kt', 'max_kt', 'fallback'}}

    Lookup priority in check_voyage:
      1. (cluster, vtype)  — cluster-specific, matches generator's speed model
      2. vtype             — global fallback when cluster has too few legs
    """
    print("[Thresholds] Computing per-leg speeds from observed data...")

    records = []
    n_sea = 0
    n_gc  = 0
    for row in obs.itertuples(index=False):
        vtype    = row.nbic_type_group
        cluster  = int(row.cluster)
        ports    = row.ports
        lats     = row.latitudes
        lons     = row.longitudes
        transits = row.transits_hours
        n        = int(row.n_ports)

        for i in range(1, n):
            t_h = transits[i] if i < len(transits) else None
            if t_h is None or t_h <= 0:
                continue
            try:
                dist_nm, src = get_dist_nm(
                    ports[i-1], ports[i],
                    lats[i-1], lons[i-1],
                    lats[i],   lons[i],
                    sea_matrix
                )
                if src == 'sea':
                    n_sea += 1
                else:
                    n_gc += 1
            except Exception:
                continue
            if dist_nm < MIN_DIST_NM:
                continue
            speed_kt = dist_nm / t_h
            if OBS_SPEED_MIN_KT <= speed_kt <= OBS_SPEED_MAX_KT:
                records.append({'vessel_type': vtype,
                                 'cluster':     cluster,
                                 'dist_nm':     dist_nm,
                                 'transit_h':   t_h,
                                 'speed_kt':    speed_kt,
                                 'dist_source': src})

    legs_df = pd.DataFrame(records)
    print(f"         {len(legs_df)} valid legs across "
          f"{legs_df['vessel_type'].nunique()} vessel types, "
          f"{legs_df['cluster'].nunique()} clusters")

    # ── 1. Global thresholds (per vessel type, used as fallback) ──────────────
    global_rows = []
    for vtype, grp in legs_df.groupby('vessel_type'):
        n = len(grp)
        if n < MIN_OBS_LEGS:
            print(f"  [WARN] {vtype}: only {n} legs → using fallback thresholds")
            global_rows.append({'vessel_type': vtype,
                                 'n_legs': n,
                                 'p5_kt':  FALLBACK_MIN_KT,
                                 'p50_kt': (FALLBACK_MIN_KT + FALLBACK_MAX_KT) / 2,
                                 'max_kt': FALLBACK_MAX_KT,
                                 'fallback': True})
        else:
            global_rows.append({'vessel_type': vtype,
                                 'n_legs': n,
                                 'p5_kt':  float(np.percentile(grp['speed_kt'], 5)),
                                 'p50_kt': float(np.percentile(grp['speed_kt'], 50)),
                                 'max_kt': float(np.percentile(grp['speed_kt'], 95)),
                                 'fallback': False})

    global_thresh = pd.DataFrame(global_rows).set_index('vessel_type')

    print("\n  Global speed thresholds (knots) — used as fallback:")
    print(f"  {'Vessel type':20} {'N_legs':>8} {'p5':>7} {'p50':>7} "
          f"{'max':>7} {'max×1.1':>8} {'fallback':>9}")
    print("  " + "-" * 70)
    for vt, r in global_thresh.iterrows():
        fb = "YES" if r['fallback'] else ""
        print(f"  {vt:20} {r['n_legs']:>8.0f} {r['p5_kt']:>7.2f} "
              f"{r['p50_kt']:>7.2f} {r['max_kt']:>7.2f} "
              f"{r['max_kt']*SPEED_UPPER_BUFFER:>8.2f} {fb:>9}")

    # ── 2. Cluster-level thresholds (per (cluster, vessel_type)) ─────────────
    cluster_thresh = {}
    print("\n  Cluster-level speed thresholds (knots):")
    print(f"  {'Cluster':>8} {'Vessel type':20} {'N_legs':>8} "
          f"{'p5':>7} {'max':>7} {'source':>10}")
    print("  " + "-" * 68)

    for (cl, vt), grp in legs_df.groupby(['cluster', 'vessel_type']):
        n = len(grp)
        if n < MIN_OBS_LEGS:
            # Fall back to global vessel-type threshold
            if vt in global_thresh.index:
                entry = {'p5_kt':   global_thresh.loc[vt, 'p5_kt'],
                         'max_kt':  global_thresh.loc[vt, 'max_kt'],
                         'n_legs':  n,
                         'fallback': True}
                source = 'global_fallback'
            else:
                entry = {'p5_kt':   FALLBACK_MIN_KT,
                         'max_kt':  FALLBACK_MAX_KT,
                         'n_legs':  n,
                         'fallback': True}
                source = 'hardcoded_fallback'
        else:
            entry = {'p5_kt':   float(np.percentile(grp['speed_kt'], 5)),
                     'max_kt':  float(np.percentile(grp['speed_kt'], 95)),
                     'n_legs':  n,
                     'fallback': False}
            source = 'cluster_observed'
        cluster_thresh[(cl, vt)] = entry
        print(f"  {cl:>8} {vt:20} {n:>8} "
              f"{entry['p5_kt']:>7.2f} {entry['max_kt']:>7.2f} {source:>18}")

    return global_thresh, cluster_thresh, legs_df


# ══════════════════════════════════════════════════════════
# 3. Apply Layer 2 filter to synthetic voyages
# ══════════════════════════════════════════════════════════

def check_voyage(row, global_thresh: pd.DataFrame,
                 cluster_thresh: dict, sea_matrix: dict):
    """
    Check every leg of a single synthetic voyage.

    Threshold lookup priority:
      1. cluster_thresh[(cluster, vtype)]  — cluster-specific
      2. global_thresh[vtype]              — global fallback
      3. hardcoded FALLBACK constants      — last resort

    Returns (passed: bool, n_legs_checked: int, n_legs_failed: int,
             fail_details: list of dicts)
    """
    vtype    = row.vessel_type
    cluster  = int(row.cluster)
    ports    = row.ports
    lats     = row.latitudes
    lons     = row.longitudes
    transits = row.transits_hours
    n        = int(row.n_ports)

    # Resolve speed thresholds by priority
    key = (cluster, vtype)
    if key in cluster_thresh:
        min_speed = cluster_thresh[key]['p5_kt']
        max_speed = cluster_thresh[key]['max_kt']
    elif vtype in global_thresh.index:
        min_speed = global_thresh.loc[vtype, 'p5_kt']
        max_speed = global_thresh.loc[vtype, 'max_kt']
    else:
        min_speed = FALLBACK_MIN_KT
        max_speed = FALLBACK_MAX_KT

    min_speed_eff = min_speed * SPEED_LOWER_BUFFER
    max_speed_eff = max_speed * SPEED_UPPER_BUFFER

    fail_details = []
    n_checked = 0

    for i in range(1, n):
        t_h = transits[i] if i < len(transits) else None
        if t_h is None or t_h <= 0:
            continue
        try:
            dist_nm, _ = get_dist_nm(
                ports[i-1], ports[i],
                lats[i-1], lons[i-1],
                lats[i],   lons[i],
                sea_matrix
            )
        except Exception:
            continue
        if dist_nm < MIN_DIST_NM:
            continue

        n_checked += 1
        min_t = dist_nm / max_speed_eff
        max_t = dist_nm / min_speed_eff
        actual_speed = dist_nm / t_h

        if not (min_t <= t_h <= max_t):
            fail_details.append({
                'leg': i,
                'dist_nm': round(dist_nm, 1),
                'transit_h': round(t_h, 1),
                'actual_speed_kt': round(actual_speed, 2),
                'min_transit_h': round(min_t, 1),
                'max_transit_h': round(max_t, 1),
            })

    passed = len(fail_details) == 0
    return passed, n_checked, len(fail_details), fail_details


def apply_filter(syn: pd.DataFrame, global_thresh: pd.DataFrame,
                 cluster_thresh: dict, sea_matrix: dict):
    """Apply Layer 2 speed check to all synthetic voyages."""
    print("\n[Filter] Applying Layer 2 speed check to synthetic voyages...")

    results = []
    for row in syn.itertuples(index=False):
        passed, n_checked, n_failed, details = check_voyage(
            row, global_thresh, cluster_thresh, sea_matrix)
        results.append({
            'passed':    passed,
            'n_checked': n_checked,
            'n_failed':  n_failed,
        })

    result_df = pd.DataFrame(results, index=syn.index)
    syn_filtered = syn[result_df['passed']].copy()
    syn_rejected = syn[~result_df['passed']].copy()

    pass_rate = len(syn_filtered) / len(syn) * 100
    print(f"  Total:    {len(syn):>6}")
    print(f"  Passed:   {len(syn_filtered):>6}  ({pass_rate:.1f}%)")
    print(f"  Rejected: {len(syn_rejected):>6}  ({100-pass_rate:.1f}%)")

    return syn_filtered, syn_rejected, result_df


# ══════════════════════════════════════════════════════════
# 4. Diagnostic plot
# ══════════════════════════════════════════════════════════

def make_diagnostic_plot(legs_df: pd.DataFrame,
                         thresholds: pd.DataFrame,
                         path: Path):
    """Speed distribution per vessel type with threshold lines."""
    vtypes = sorted(thresholds.index.tolist())
    n = len(vtypes)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    fig.patch.set_facecolor('#F8F9FA')
    axes = axes.flatten() if n > 1 else [axes]

    for ax, vt in zip(axes, vtypes):
        grp = legs_df[legs_df['vessel_type'] == vt]['speed_kt']
        ax.hist(grp, bins=40, color=COLOR_OBS, alpha=0.7, density=True)
        p5  = thresholds.loc[vt, 'p5_kt']
        mx  = thresholds.loc[vt, 'max_kt']
        ax.axvline(p5,           color='#E84855', lw=1.5, linestyle='--',
                   label=f'p5={p5:.1f} kt')
        ax.axvline(mx,           color='#333333', lw=1.5, linestyle='--',
                   label=f'p95={mx:.1f} kt')
        ax.axvline(mx * SPEED_UPPER_BUFFER, color='#888888', lw=1.0,
                   linestyle=':', label=f'p95×1.05={mx*SPEED_UPPER_BUFFER:.1f} kt')
        ax.set_title(f"{vt}\n(n={len(grp):,} legs)", fontsize=9, fontweight='bold')
        ax.set_xlabel("Speed (knots)", fontsize=8)
        ax.legend(fontsize=7, framealpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('#FFFFFF')

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Observed leg speed distributions by vessel type\n"
                 "Dashed lines = p5 (min_speed) and p95 (max_speed) thresholds",
                 fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Plot saved: {path.name}")


# ══════════════════════════════════════════════════════════
# 5. Text report
# ══════════════════════════════════════════════════════════

def write_report(syn: pd.DataFrame,
                 syn_filtered: pd.DataFrame,
                 syn_rejected: pd.DataFrame,
                 result_df: pd.DataFrame,
                 global_thresh: pd.DataFrame,
                 path: Path):
    lines = []
    lines.append("=" * 65)
    lines.append("Track 1 Markov Chain — Layer 2 Physical Plausibility Report")
    lines.append("=" * 65)
    lines.append("")
    lines.append(f"Total synthetic voyages:  {len(syn):>6}")
    lines.append(f"Passed:                   {len(syn_filtered):>6}  "
                 f"({len(syn_filtered)/len(syn)*100:.1f}%)")
    lines.append(f"Rejected:                 {len(syn_rejected):>6}  "
                 f"({len(syn_rejected)/len(syn)*100:.1f}%)")
    lines.append("")
    lines.append("Note: thresholds applied per (cluster, vessel_type); "
                 "global vtype thresholds used as fallback.")
    lines.append("")

    lines.append("Global speed thresholds used as fallback (knots):")
    lines.append(f"  {'Vessel type':20} {'min_speed(p5)':>14} "
                 f"{'max_speed(p95)':>15} {'ceil(×1.05)':>11} {'fallback':>9}")
    lines.append("  " + "-" * 72)
    for vt, r in global_thresh.iterrows():
        fb = "YES" if r['fallback'] else ""
        lines.append(f"  {vt:20} {r['p5_kt']:>14.2f} {r['max_kt']:>15.2f} "
                     f"{r['max_kt']*SPEED_UPPER_BUFFER:>11.2f} {fb:>9}")
    lines.append("")

    lines.append("Rejection breakdown by vessel type:")
    lines.append(f"  {'Vessel type':20} {'Total':>7} {'Rejected':>9} {'Rate':>7}")
    lines.append("  " + "-" * 46)
    for vt, grp in syn.groupby('vessel_type'):
        rej = syn_rejected[syn_rejected['vessel_type'] == vt]
        rate = len(rej) / len(grp) * 100 if len(grp) > 0 else 0
        lines.append(f"  {vt:20} {len(grp):>7} {len(rej):>9} {rate:>6.1f}%")
    lines.append("")

    lines.append("Rejection breakdown by cluster:")
    lines.append(f"  {'Cluster':>8} {'Total':>7} {'Rejected':>9} {'Rate':>7}")
    lines.append("  " + "-" * 36)
    for cl, grp in syn.groupby('cluster'):
        rej = syn_rejected[syn_rejected['cluster'] == cl]
        rate = len(rej) / len(grp) * 100 if len(grp) > 0 else 0
        lines.append(f"  {cl:>8} {len(grp):>7} {len(rej):>9} {rate:>6.1f}%")

    text = "\n".join(lines)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(text)


# ══════════════════════════════════════════════════════════
# 6. Main
# ══════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("Track 1 Markov Chain — Layer 2 Physical Plausibility Filter")
    print("=" * 65)

    LIST_COLS = ['ports', 'latitudes', 'longitudes',
                 'transits_hours', 'dwells_hours', 'ecoprovinces']

    print("\n[Load] Observed data...")
    obs = load_df(OBS_PATH, LIST_COLS)
    print(f"       {len(obs)} observed voyages")

    print("\n[Load] Synthetic data...")
    syn = load_df(SYN_PATH, LIST_COLS)
    print(f"       {len(syn)} synthetic voyages")

    # Load sea distance matrix
    sea_matrix = load_sea_matrix(SEA_DIST_PATH)

    # Step 1: derive thresholds
    global_thresh, cluster_thresh, legs_df = derive_speed_thresholds(obs, sea_matrix)
    global_thresh.to_csv(OUT_DIR / 'speed_thresholds.csv')
    print(f"\n  Thresholds saved: speed_thresholds.csv")

    # Step 2: diagnostic plot
    make_diagnostic_plot(legs_df, global_thresh,
                         OUT_DIR / 'speed_distribution.png')

    # Step 3: apply filter
    syn_filtered, syn_rejected, result_df = apply_filter(
        syn, global_thresh, cluster_thresh, sea_matrix)
    syn_filtered.to_csv(OUT_DIR / 'mc.syn.all.layer2.csv', index=False)
    print(f"\n  Filtered CSV saved: mc.syn.all.layer2.csv")

    # Step 4: report
    print("\n[Report]")
    write_report(syn, syn_filtered, syn_rejected, result_df, global_thresh,
                 OUT_DIR / 'layer2_report.txt')
    print(f"\n  Report saved: layer2_report.txt")
    print("\nDone.")


if __name__ == '__main__':
    main()