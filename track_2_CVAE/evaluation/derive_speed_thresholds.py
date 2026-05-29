"""
derive_speed_thresholds.py
==========================
Track 2 CVAE — Data-driven vessel speed threshold derivation

Computes per-vessel-type speed ranges from the observed AIS voyage data:
    v_min  = p5  of observed leg speeds  → upper bound on transit time
    v_max  = p95 of observed leg speeds × 1.05  → lower bound on transit time

These values are intended to replace the hardcoded VESSEL_SPEED dict in
config.py, making the Layer 2 clamping constraint data-driven.

Derivation method (per leg):
    1. Retrieve sea distance (nm) from the pre-computed matrix; skip leg
       if the pair is absent (rather than using a fallback, to avoid
       introducing distance estimation error into speed calibration).
    2. Skip legs with dist < MIN_DIST_NM (intra-port-complex noise).
    3. Compute implied speed = dist_nm / transit_hours.
    4. Discard speeds outside [OBS_SPEED_MIN_KT, OBS_SPEED_MAX_KT]
       (AIS artefacts and anchor/congestion events).
    5. Aggregate per vessel type: p5 → v_min, max × 1.05 → v_max.

Outputs:
    evaluation/outputs/speed_thresholds.csv      per-vessel-type thresholds
    evaluation/outputs/speed_distribution.png    diagnostic plot
    (printed to stdout)                          suggested VESSEL_SPEED dict
"""

import ast, sys, pickle, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parents[1]
OBS_PATH     = BASE_DIR / 'data' / 'cluster.m1.labels.k5.csv'
SEA_DIST_PKL = BASE_DIR / 'data' / 'sea_distance_matrix.pkl'
OUT_DIR      = Path(__file__).resolve().parent / 'outputs'
OUT_DIR.mkdir(exist_ok=True)

# ── Constants (mirroring Track 1 script design rationale) ────────────────────
OBS_SPEED_MIN_KT = 0.5    # below this → anchor / AIS error
OBS_SPEED_MAX_KT = 30.0   # above this → AIS interpolation artefact
MIN_DIST_NM      = 50.0   # legs shorter than this skipped (port complex noise)
MIN_OBS_LEGS     = 10     # fallback if vessel type has too few observations

FALLBACK_VMIN_KT = 2.0
FALLBACK_VMAX_KT = 22.0

VMAX_BUFFER = 1.05        # p95 × 1.05 → v_max

# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_list(val):
    if isinstance(val, list):
        return val
    try:
        return ast.literal_eval(str(val))
    except Exception:
        return []


def load_sea_matrix(path: Path) -> dict:
    with open(path, 'rb') as f:
        data = pickle.load(f)
    nm = data['nm']
    print(f"[SeaDist] Loaded {len(nm):,} port pairs (nm)")
    return nm


# ── Speed derivation ──────────────────────────────────────────────────────────

def derive_speed_thresholds(obs: pd.DataFrame, sea_nm: dict):
    """
    Returns:
        legs_df    : DataFrame of all valid legs with implied speed
        thresholds : DataFrame indexed by vessel_type with v_min, v_max columns
    """
    print("[Thresholds] Computing per-leg implied speeds from observed data...")

    records = []
    skipped_dist = 0
    skipped_short = 0
    skipped_speed = 0

    for _, row in obs.iterrows():
        ports    = safe_list(row['ports'])
        transits = safe_list(row['transits_hours'])
        vtype    = row['nbic_type_group']
        n        = int(row['n_ports'])

        for i in range(1, n):
            if i >= len(transits):
                continue
            t_h = transits[i]
            if t_h is None or t_h <= 0:
                continue

            p1 = ports[i-1].upper().strip()
            p2 = ports[i].upper().strip()
            dist = sea_nm.get((p1, p2)) or sea_nm.get((p2, p1))

            if dist is None or dist <= 0:
                skipped_dist += 1
                continue
            if dist < MIN_DIST_NM:
                skipped_short += 1
                continue

            speed_kt = dist / t_h
            if not (OBS_SPEED_MIN_KT <= speed_kt <= OBS_SPEED_MAX_KT):
                skipped_speed += 1
                continue

            records.append({
                'vessel_type': vtype,
                'dist_nm':     dist,
                'transit_h':   t_h,
                'speed_kt':    speed_kt,
            })

    legs_df = pd.DataFrame(records)
    total = len(records) + skipped_dist + skipped_short + skipped_speed
    print(f"         {len(legs_df):,} valid legs retained out of {total:,} total legs")
    print(f"         skipped — no matrix entry: {skipped_dist} | "
          f"dist<{MIN_DIST_NM}nm: {skipped_short} | "
          f"speed out of [{OBS_SPEED_MIN_KT},{OBS_SPEED_MAX_KT}]kt: {skipped_speed}")

    # Per-vessel-type thresholds
    rows = []
    print(f"\n{'Vessel type':<18} {'n_legs':>7} {'p5':>7} {'p25':>7} "
          f"{'p50':>7} {'p75':>7} {'p95':>7} {'obs_max':>8} "
          f"{'v_min(p5)':>10} {'v_max(max×1.05)':>16} {'fallback':>9}")
    print("-" * 115)

    for vtype, grp in legs_df.groupby('vessel_type'):
        s = grp['speed_kt'].values
        n = len(s)
        if n < MIN_OBS_LEGS:
            v_min = FALLBACK_VMIN_KT
            v_max = FALLBACK_VMAX_KT
            fallback = True
        else:
            v_min = float(np.percentile(s, 5))
            v_max = float(np.percentile(s, 95) * VMAX_BUFFER)
            fallback = False

        print(f"{vtype:<18} {n:>7} "
              f"{np.percentile(s,5):>7.2f} "
              f"{np.percentile(s,25):>7.2f} "
              f"{np.percentile(s,50):>7.2f} "
              f"{np.percentile(s,75):>7.2f} "
              f"{np.percentile(s,95):>7.2f} "
              f"{s.max():>8.2f} "
              f"{v_min:>10.2f} "
              f"{v_max:>16.2f} "
              f"{'YES' if fallback else '':>9}")

        rows.append({
            'vessel_type': vtype,
            'n_legs':      n,
            'p5_kt':       round(float(np.percentile(s, 5)), 2),
            'p25_kt':      round(float(np.percentile(s, 25)), 2),
            'p50_kt':      round(float(np.percentile(s, 50)), 2),
            'p75_kt':      round(float(np.percentile(s, 75)), 2),
            'p95_kt':      round(float(np.percentile(s, 95)), 2),
            'obs_max_kt':  round(float(s.max()), 2),
            'v_min_kt':    round(v_min, 2),
            'v_max_kt':    round(v_max, 2),
            'fallback':    fallback,
        })

    thresholds = pd.DataFrame(rows).set_index('vessel_type')
    return legs_df, thresholds


# ── Diagnostic plot ───────────────────────────────────────────────────────────

def make_plot(legs_df: pd.DataFrame, thresholds: pd.DataFrame, path: Path):
    vtypes = sorted(thresholds.index.tolist())
    ncols = 3
    nrows = (len(vtypes) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = axes.flatten()

    for ax, vtype in zip(axes, vtypes):
        grp = legs_df[legs_df['vessel_type'] == vtype]['speed_kt']
        ax.hist(grp, bins=40, color='#2E86AB', alpha=0.75, density=True)

        v_min = thresholds.loc[vtype, 'v_min_kt']
        v_max = thresholds.loc[vtype, 'v_max_kt']
        obs_max = thresholds.loc[vtype, 'obs_max_kt']

        ax.axvline(v_min, color='#E84855', lw=1.8, linestyle='--',
                   label=f'v_min=p5={v_min:.1f} kt')
        p95 = thresholds.loc[vtype, 'p95_kt']
        ax.axvline(p95, color='#555555', lw=1.2, linestyle=':',
                   label=f'p95={p95:.1f} kt')
        ax.axvline(v_max, color='#E07B39', lw=1.8, linestyle='--',
                   label=f'v_max=p95×1.05={v_max:.1f} kt')

        ax.set_title(f"{vtype}  (n={len(grp):,} legs)", fontsize=9, fontweight='bold')
        ax.set_xlabel("Implied speed (knots)", fontsize=8)
        ax.legend(fontsize=7, framealpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for ax in axes[len(vtypes):]:
        ax.set_visible(False)

    fig.suptitle(
        "Track 2 CVAE — Observed leg speed distributions by vessel type\n"
        "Red dashed = v_min (p5)  |  Orange dashed = v_max (p95 × 1.05)",
        fontsize=11, fontweight='bold', y=1.01,
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[Plot] Saved: {path.name}")


# ── Print suggested config update ─────────────────────────────────────────────

def print_config_suggestion(thresholds: pd.DataFrame):
    print("\n" + "=" * 60)
    print("Suggested VESSEL_SPEED update for config.py")
    print("(v_min=p5, v_max=p95×1.05, rounded to 1 decimal)")
    print("=" * 60)
    print("VESSEL_SPEED = {")
    for vtype, row in thresholds.iterrows():
        v_min = round(row['v_min_kt'], 1)
        v_max = round(row['v_max_kt'], 1)
        fb = "  # fallback — too few legs" if row['fallback'] else ""
        print(f"    '{vtype}':{' ' * max(1, 20-len(vtype))}({v_min}, {v_max}),{fb}")
    print("}")

    print("\nComparison with current hardcoded values:")
    CURRENT = {
        'Bulker':        (5.0, 16.0),
        'Container':     (8.0, 25.0),
        'General Cargo': (5.0, 15.0),
        'Other':         (4.0, 15.0),
        'Passenger':     (8.0, 24.0),
        'Reefer':        (6.0, 22.0),
        'RoRo':          (6.0, 22.0),
        'Tanker':        (5.0, 16.0),
    }
    print(f"\n{'Vessel type':<18} {'cur_min':>8} {'new_min':>8} "
          f"{'Δmin':>7} | {'cur_max':>8} {'new_max':>8} {'Δmax':>7}")
    print("-" * 72)
    for vtype, row in thresholds.iterrows():
        c_min, c_max = CURRENT.get(vtype, (None, None))
        n_min = round(row['v_min_kt'], 1)
        n_max = round(row['v_max_kt'], 1)
        if c_min is not None:
            print(f"{vtype:<18} {c_min:>8.1f} {n_min:>8.1f} "
                  f"{n_min-c_min:>+7.1f} | "
                  f"{c_max:>8.1f} {n_max:>8.1f} {n_max-c_max:>+7.1f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Track 2 CVAE — Data-driven Speed Threshold Derivation")
    print("=" * 65)

    print(f"\n[Load] Observed voyages: {OBS_PATH.name}")
    obs = pd.read_csv(OBS_PATH)
    for col in ['ports', 'transits_hours']:
        obs[col] = obs[col].apply(safe_list)
    print(f"       {len(obs):,} voyages loaded")

    print(f"\n[Load] Sea distance matrix: {SEA_DIST_PKL.name}")
    sea_nm = load_sea_matrix(SEA_DIST_PKL)

    print()
    legs_df, thresholds = derive_speed_thresholds(obs, sea_nm)

    thresholds.to_csv(OUT_DIR / 'speed_thresholds.csv')
    print(f"\n[CSV] Thresholds saved: {OUT_DIR / 'speed_thresholds.csv'}")

    make_plot(legs_df, thresholds, OUT_DIR / 'speed_distribution.png')

    print_config_suggestion(thresholds)

    print("\nDone.")


if __name__ == '__main__':
    sys.path.insert(0, str(BASE_DIR))
    main()
