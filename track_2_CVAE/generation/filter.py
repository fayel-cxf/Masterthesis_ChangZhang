"""
Track 2 CVAE — Layer 1 & Layer 2 Filters
"""
import os, sys, pickle
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (VESSEL_SPEED, LAYER2_MIN_DWELL, LAYER2_MAX_DAYS,
                    CIRCUITY_FACTOR, SEA_DIST_PKL, DATA_DIR, NZ_PORTS)

_NZ_PORT_SET = set(NZ_PORTS)


# ── Sea-distance matrix ───────────────────────────────────────────────────────

def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    lat1, lat2 = radians(lat1), radians(lat2)
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def get_sea_distance_km(lon1, lat1, lon2, lat2):
    try:
        import searoute as sr
        route = sr.searoute([lon1, lat1], [lon2, lat2], units='km')
        return route.properties['length']
    except Exception:
        return None


def get_distance_with_fallback(lon1, lat1, lon2, lat2):
    d = get_sea_distance_km(lon1, lat1, lon2, lat2)
    if d is not None:
        return d, 'searoute'
    d_gc = haversine_km(lon1, lat1, lon2, lat2)
    return d_gc * CIRCUITY_FACTOR, 'haversine'


def build_sea_distance_matrix(port_coords, save_path=SEA_DIST_PKL):
    """Build and cache the pairwise sea-distance matrix (km and nm)."""
    sea_dist_km, sea_dist_nm, source_log = {}, {}, {}
    ports = list(port_coords.keys())
    n     = len(ports)
    print(f"Computing sea distances for {n} ports ({n*(n-1)//2} pairs)…")

    for i, p1 in enumerate(ports):
        lat1, lon1 = port_coords[p1]
        for p2 in ports[i+1:]:
            lat2, lon2 = port_coords[p2]
            d_km, src  = get_distance_with_fallback(lon1, lat1, lon2, lat2)
            d_nm       = d_km * 0.539957

            sea_dist_km[(p1, p2)] = sea_dist_km[(p2, p1)] = d_km
            sea_dist_nm[(p1, p2)] = sea_dist_nm[(p2, p1)] = d_nm
            source_log[(p1, p2)]  = src

        if i % 50 == 0:
            print(f"  {i}/{n} ports done…")

    with open(save_path, 'wb') as f:
        pickle.dump({'km': sea_dist_km, 'nm': sea_dist_nm, 'source': source_log}, f)
    print(f"Distance matrix saved → {save_path}")
    return sea_dist_km, sea_dist_nm


def load_sea_distance_matrix(path=SEA_DIST_PKL):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            d = pickle.load(f)
        return d['nm']
    return None


# ── Layer 2: physical feasibility ────────────────────────────────────────────

def layer2_filter(voyage, sea_dist_nm, p99_dwell):
    """
    Returns (True, 'pass') or (False, reason).
    """
    ports    = voyage['ports']
    transits = voyage['transits_hours']
    dwells   = voyage['dwells_hours']
    vtype    = voyage.get('vessel_type', 'Other')

    min_spd, max_spd = VESSEL_SPEED.get(vtype, (5.0, 16.0))

    n = len(ports)
    if n < 5:
        return False, 'too_short'

    # duplicate intermediate ports (NZ final port excluded from check)
    if len(set(ports[:-1])) != n - 1:
        return False, 'duplicate_intermediate_port'

    # NZ ports must not appear as intermediate stops
    if any(p in _NZ_PORT_SET for p in ports[1:-1]):
        return False, 'nz_intermediate_port'

    # per-leg transit time check
    for i in range(1, n):
        key = (ports[i-1], ports[i])
        dist = sea_dist_nm.get(key) or sea_dist_nm.get((ports[i], ports[i-1]))
        if dist is None:
            # unknown port pair → skip physical check for this leg
            continue
        min_tr = dist / (max_spd * 1.05)
        max_tr = dist / min_spd
        tr     = transits[i]
        if tr < min_tr or tr > max_tr:
            return False, f'transit_out_of_range_leg_{i}'

    # dwell check (intermediate ports only, not first and not last)
    for dw in dwells[1:-1]:
        if dw < LAYER2_MIN_DWELL or dw > p99_dwell:
            return False, 'dwell_out_of_range'

    # total duration
    total_days = (sum(transits) + sum(dwells[:-1])) / 24.0
    if total_days > LAYER2_MAX_DAYS:
        return False, 'total_span_exceeded'

    return True, 'pass'


# ── Layer 1: statistical plausibility (post-generation summary) ──────────────

def layer1_metrics(observed_df, synthetic_voyages):
    """
    Compute JSD on port frequencies and KS on n_ports distribution.
    Returns a dict of metrics.
    """
    from scipy.stats import ks_2samp
    from scipy.spatial.distance import jensenshannon

    # port frequency JSD
    from collections import Counter
    obs_ports = Counter(p for row in observed_df['ports']
                        for p in (row if isinstance(row, list) else []))
    syn_ports = Counter(p for v in synthetic_voyages for p in v['ports'])

    all_ports = sorted(set(obs_ports) | set(syn_ports))
    obs_vec = np.array([obs_ports.get(p, 0) for p in all_ports], dtype=float)
    syn_vec = np.array([syn_ports.get(p, 0) for p in all_ports], dtype=float)
    obs_vec /= obs_vec.sum()
    syn_vec /= syn_vec.sum() if syn_vec.sum() > 0 else 1.0
    jsd = jensenshannon(obs_vec, syn_vec) ** 2  # squared JSD ∈ [0,1]

    # n_ports KS
    obs_n = observed_df['n_ports'].values
    syn_n = np.array([v['n_ports'] for v in synthetic_voyages])
    ks_stat, ks_p = ks_2samp(obs_n, syn_n)

    return {'jsd': jsd, 'ks_n_ports_p': ks_p, 'ks_n_ports_stat': ks_stat}


def nz_destination_jsd(observed_df, synthetic_voyages):
    """
    Compute JSD between observed and synthetic NZ destination distributions.
    Returns jsd_nz_dest and per-port counts (diagnostic metric only).
    """
    from scipy.spatial.distance import jensenshannon
    from collections import Counter

    obs_col = 'nz_destination' if 'nz_destination' in observed_df.columns else 'nz_dest_port'
    obs_nz = Counter(
        str(p).upper().strip()
        for p in observed_df[obs_col]
        if pd.notna(p)
    )
    syn_nz = Counter(v['nz_destination'] for v in synthetic_voyages)

    all_ports = sorted(set(obs_nz) | set(syn_nz))
    obs_vec = np.array([obs_nz.get(p, 0) for p in all_ports], dtype=float)
    syn_vec = np.array([syn_nz.get(p, 0) for p in all_ports], dtype=float)
    obs_vec /= obs_vec.sum()
    syn_vec /= syn_vec.sum() if syn_vec.sum() > 0 else 1.0

    jsd = float(jensenshannon(obs_vec, syn_vec) ** 2)
    return {
        'jsd_nz_dest':   jsd,
        'obs_nz_counts': dict(obs_nz),
        'syn_nz_counts': dict(syn_nz),
    }
