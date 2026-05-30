"""
markov_voyage_generator.py
==========================
Track 1: smoothed Markov-chain synthetic voyage generator.

Trains a model on cluster_m1_labels_k5.csv and generates synthetic voyages
ending at New Zealand ports, for use by the three-filter invasion risk model
(Seebens framework) downstream.

Usage:
    python markov_voyage_generator.py

Outputs:
    data/markov/output/mc.syn{k}.csv    Per-cluster output
    data/markov/output/mc.syn.all.csv   Combined output
"""

import ast
import os
import pickle
import random
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════
# 0. Path configuration
# ══════════════════════════════════════════════

# Script lives at <project_root>/track1_markovchain_code/mc_generator/<this file>,
# so the project root is three levels up.
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR   = SCRIPT_DIR.parent.parent

CSV_PATH   = BASE_DIR / 'data/clustering/cluster.m1.labels.k5.csv'
# Sea-distance matrix lives next to this script, in the same mc_generator folder.
PKL_PATH   = SCRIPT_DIR / 'sea_distance_matrix.pkl'
OUTPUT_DIR = BASE_DIR / 'data/markov/output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════
# 1. Hyperparameters
# ══════════════════════════════════════════════

ALPHA            = 0.05    # Geographic smoothing coefficient
N_TARGET         = 10000   # Target number of synthetic voyages 
MIN_PORTS        = 5       # Minimum ports per voyage (including NZ terminus)
MAX_PORTS        = 7       # Maximum ports per voyage (including NZ terminus)
MAX_SPAN_DAYS    = 60      # Maximum total voyage duration (days)
DWELL_MIN_H      = 6       # Dwell time lower bound (hours)
DWELL_MAX_H      = 136     # Dwell time upper bound (hours)
MIN_SAMPLE_FIT   = 10      # Minimum sample size for LogNormal fitting
PORT_FACTOR_MIN  = 20      # Minimum observations to compute a port adjustment factor
RESAMPLE_LIMIT   = 100     # Maximum number of truncated-resample attempts
TRANSIT_SPEED_BUFFER = 1.05  # Physical-lower-bound buffer factor (5%)
MIN_SPEED_KNOTS  = 6.0     # Minimum plausible vessel speed (knots)

# NZ port set (computed from the data; voyage terminus must lie in this set)
NZ_TERRITORIES = {'New Zealand'}

# ══════════════════════════════════════════════
# 2. Data loading
# ══════════════════════════════════════════════

def load_data(csv_path: Path, pkl_path: Path):
    """Load observed voyage data and the sea-distance matrix."""
    print(f"[Load] Reading voyage data: {csv_path}")
    df = pd.read_csv(csv_path)

    # Parse list-valued columns from their stringified form back into Python lists.
    list_cols = ['ports', 'port_territories', 'latitudes', 'longitudes',
                 'ecoprovinces', 'dwells_hours', 'transits_hours',
                 'env_similarity_risks']
    for col in list_cols:
        if col in df.columns:
            df[col] = df[col].apply(ast.literal_eval)

    print(f"       Voyages: {len(df)}, clusters: {df['cluster'].nunique()}")
    print(f"       Cluster distribution:\n{df['cluster'].value_counts().sort_index().to_string()}")

    print(f"\n[Load] Reading distance matrix: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        dist_data = pickle.load(f)
    sea_dist_nm = dist_data['nm']
    print(f"       Distance matrix entries: {len(sea_dist_nm):,}")

    return df, sea_dist_nm

# Ad-hoc patch for ports whose ecoprovince is missing in the source data.
ECO_PATCH = {
    'MANAUS': 'Tropical Southwestern Atlantic',
}
def extract_port_coords(df: pd.DataFrame) -> dict:
    """Extract coordinates and ecoprovince info for all unique ports."""
    port_coords      = {}   # {port_name: (lon, lat)}
    port_territories = {}   # {port_name: territory}
    port_ecoprovince = {}   # {port_name: ecoprovince}

    for _, row in df.iterrows():
        ports  = row['ports']
        lats   = row['latitudes']
        lons   = row['longitudes']
        terrs  = row['port_territories']
        ecos   = row['ecoprovinces']

        for i, p in enumerate(ports):
            if p not in port_coords:
                port_coords[p]      = (lons[i], lats[i])
                port_territories[p] = terrs[i]
                eco = ecos[i] if i < len(ecos) else ''
                port_ecoprovince[p] = ECO_PATCH.get(p) or ('' if pd.isna(eco) else str(eco))

    return port_coords, port_territories, port_ecoprovince


def identify_nz_ports(df: pd.DataFrame) -> set:
    """Identify all NZ terminus ports present in the data."""
    nz_ports = set()
    for _, row in df.iterrows():
        ports = row['ports']
        terrs = row['port_territories']
        for p, t in zip(ports, terrs):
            if t == 'New Zealand':
                nz_ports.add(p)
    print(f"\n[Info] Identified {len(nz_ports)} NZ ports: {sorted(nz_ports)}")
    return nz_ports


# ══════════════════════════════════════════════
# 3. Model training
# ══════════════════════════════════════════════

class MarkovModel:
    """
    Per-cluster Markov-chain model. Holds:
    - Smoothed transition matrices (1st- and 2nd-order)
    - Transit-time LogNormal distributions (three fallback layers)
    - Dwell-time LogNormal distributions (two fallback layers)
    - Port-level dwell adjustment factors
    - Start-port distribution
    - Vessel-speed lookup
    """

    def __init__(self, cluster_id: int):
        self.cluster_id       = cluster_id
        self.transition       = {}   # 1st-order fallback: {port_i: {port_j: prob}}
        self.transition_2     = {}   # 2nd-order primary matrix: {(port_{i-1}, port_i): {port_j: prob}}
        self.transit_models   = {}   # {key: {'mu': float, 'sigma': float}}
        self.dwell_models     = {}   # {key: {'mu': float, 'sigma': float}}
        self.port_dwell_factors = {}  # {port: float}
        self.start_port_dist  = {}   # {port: prob}
        self.vessel_speed        = {}   # {key: float}  Median speed (kept for reference)
        self.vessel_speed_p5     = {}   # {key: float}  p5 speed, used for transit upper bound (slowest)
        self.vessel_speed_p95    = {}   # {key: float}  p95 speed, used for transit lower bound (fastest)
        self.vessel_speed_params = {}   # {key: {'mu': float, 'sigma': float}}  Speed lognormal, for fallback sampling
        self.vessel_types     = []
        self.vessel_type_dist = {}   # {vtype: prob}
        self.dwt_pool         = {}   # {vtype: [dwt values]}
        self.nz_ports         = set()
        self.nz_dest_dist     = {}   # {nz_port: prob}  Empirical NZ destination frequency, used as fallback when force_nz degenerates


def fit_lognormal(observations: list) -> Optional[dict]:
    """
    Fit a LogNormal distribution to positive-valued observations.
    Returns {'mu': float, 'sigma': float}, or None if sample size is insufficient.
    """
    obs = np.array([x for x in observations if x > 0])
    if len(obs) < MIN_SAMPLE_FIT:
        return None
    log_obs = np.log(obs)
    sigma = log_obs.std()
    # When sigma is 0 (all values identical), use a small default to avoid degeneration.
    if sigma < 1e-6:
        sigma = 0.1
    return {'mu': log_obs.mean(), 'sigma': sigma}


def _smooth_and_normalise(raw_counts_row: dict,
                          current_port: str,
                          sea_dist_nm: dict,
                          alpha: float) -> dict:
    """
    Apply geographic smoothing to the observed-frequency row of a single
    departure state, then normalise.
        P_smooth = (1-α) × P_observed + α × P_geo
        P_geo ∝ 1 / sea_dist(current_port → p_j)
    `current_port` is used to compute the geographic prior (for 2nd-order
    states this is the current port p_i).
    """
    candidates = list(raw_counts_row.keys())
    if not candidates:
        return {}

    total = sum(raw_counts_row.values())
    p_obs = {pj: raw_counts_row[pj] / total for pj in candidates}

    geo_weights = {}
    for pj in candidates:
        d = sea_dist_nm.get((current_port, pj)) or sea_dist_nm.get((pj, current_port))
        geo_weights[pj] = (1.0 / d) if (d and d > 0) else 1e-6

    geo_total = sum(geo_weights.values())
    p_geo = {pj: geo_weights[pj] / geo_total for pj in candidates}

    p_smooth = {pj: (1 - alpha) * p_obs[pj] + alpha * p_geo[pj] for pj in candidates}
    s = sum(p_smooth.values())
    return {pj: v / s for pj, v in p_smooth.items()}


def build_transition_matrix(voyages: pd.DataFrame,
                             sea_dist_nm: dict,
                             all_ports: list,
                             alpha: float = ALPHA) -> tuple:
    """
    Build the 2nd-order and 1st-order geographically smoothed transition
    matrices simultaneously.

    Second-order matrix (primary): P̃(p_{i+1} | p_{i-1}, p_i)
      key = (p_{i-1}, p_i); conditions on the combination of current port and
      the previous port. Captures path-dependence (e.g. after crossing the
      Panama Canal, a vessel is more likely to continue trans-Pacific).

    First-order matrix (fallback): P̃(p_{i+1} | p_i)
      key = p_i; conditions on the current port only.
      Used when a 2nd-order state (p_{i-1}, p_i) is absent from the training data.

    Both use the same smoothing formula:
      P_smooth = (1-α) × P_observed + α × P_geo
      P_geo ∝ 1 / sea_dist(p_i → p_j)

    Returns: (transition_2, transition_1)
    """
    raw_2 = defaultdict(lambda: defaultdict(int))  # {(p_{i-1}, p_i): {p_j: count}}
    raw_1 = defaultdict(lambda: defaultdict(int))  # {p_i: {p_j: count}}

    for _, row in voyages.iterrows():
        ports = row['ports']
        n = len(ports)
        for i in range(n - 1):
            # 1st-order: p_i → p_{i+1}
            raw_1[ports[i]][ports[i + 1]] += 1
            # 2nd-order: (p_{i-1}, p_i) → p_{i+1} (starting from the second leg)
            if i >= 1:
                raw_2[(ports[i - 1], ports[i])][ports[i + 1]] += 1

    # Smooth the 2nd-order matrix (geographic prior is based on p_i, the
    # second port of the pair).
    transition_2 = {}
    for (p_prev, p_i), counts in raw_2.items():
        smoothed = _smooth_and_normalise(counts, p_i, sea_dist_nm, alpha)
        if smoothed:
            transition_2[(p_prev, p_i)] = smoothed

    # Smooth the 1st-order matrix (fallback).
    transition_1 = {}
    for p_i, counts in raw_1.items():
        smoothed = _smooth_and_normalise(counts, p_i, sea_dist_nm, alpha)
        if smoothed:
            transition_1[p_i] = smoothed

    return transition_2, transition_1


def fit_transit_models(voyages: pd.DataFrame) -> dict:
    """
    Fit LogNormal distributions for transit times.
    Three granularity levels: (cluster, vtype, pi, pj) → (cluster, vtype) → (cluster,)
    Also computes the vessel-speed lookup table.
    """
    # Collect observations.
    observations = defaultdict(list)  # key → [transit_hours]

    cluster_id = voyages['cluster'].iloc[0]

    for _, row in voyages.iterrows():
        vtype  = row['nbic_type_group']
        ports  = row['ports']
        trans  = row['transits_hours']

        for i in range(1, len(ports)):
            t = trans[i]
            if t <= 0:
                continue
            pi, pj = ports[i-1], ports[i]

            observations[(cluster_id, vtype, pi, pj)].append(t)
            observations[(cluster_id, vtype)].append(t)
            observations[(cluster_id,)].append(t)

    # Fit.
    models = {}
    for key, obs in observations.items():
        result = fit_lognormal(obs)
        if result:
            models[key] = result

    return models


def fit_dwell_models(voyages: pd.DataFrame) -> Tuple[dict, dict]:
    """
    Fit LogNormal distributions for dwell times.
    Excludes the first port (index 0) and the NZ destination port (index n-1).
    Two granularity levels: (cluster, vtype) → (cluster, 'all')
    Also computes per-port adjustment factors.
    """
    cluster_id = voyages['cluster'].iloc[0]
    all_intermediate_dwells = []

    observations      = defaultdict(list)
    port_observations = defaultdict(list)

    for _, row in voyages.iterrows():
        vtype  = row['nbic_type_group']
        ports  = row['ports']
        dwells = row['dwells_hours']
        n      = len(ports)

        # Intermediate ports only (exclude first port index 0 and NZ destination index n-1).
        for i in range(1, n - 1):
            d = dwells[i]
            if d <= 0:
                continue
            observations[(cluster_id, vtype)].append(d)
            observations[(cluster_id, 'all')].append(d)
            all_intermediate_dwells.append(d)
            port_observations[ports[i]].append(d)

    # Fit distributions.
    models = {}
    for key, obs in observations.items():
        result = fit_lognormal(obs)
        if result:
            models[key] = result

    # Per-port adjustment factors.
    port_factors = {}
    if all_intermediate_dwells:
        global_median = np.median(all_intermediate_dwells)
        if global_median > 0:
            for port, obs in port_observations.items():
                if len(obs) >= PORT_FACTOR_MIN:
                    port_factors[port] = np.median(obs) / global_median

    return models, port_factors


def compute_vessel_speeds(voyages: pd.DataFrame,
                          sea_dist_nm: dict) -> tuple:
    """
    Compute per-(cluster, vtype) vessel-speed distributions from observed data.

    Distances use the searoute matrix (consistent with sample_transit and Layer 2);
    when an entry is missing, fall back to great-circle distance × 1.3 as a
    circuity compensation.

    Returns four dicts:
      speed_median  {key: float}   Median speed (kept for reference)
      speed_p5      {key: float}   p5 speed, used for transit upper bound (slowest)
      speed_p95     {key: float}   p95 speed, used for transit lower bound (fastest)
      speed_params  {key: {'mu', 'sigma'}}  Speed LogNormal parameters, used for fallback sampling
    """
    from math import atan2, cos, radians, sin, sqrt

    speed_obs  = defaultdict(list)
    cluster_id = voyages['cluster'].iloc[0]

    for _, row in voyages.iterrows():
        vtype = row['nbic_type_group']
        ports = row['ports']
        trans = row['transits_hours']
        lats  = row['latitudes']
        lons  = row['longitudes']

        for i in range(1, len(ports)):
            t  = trans[i]
            pi = ports[i-1]
            pj = ports[i]
            if t <= 0:
                continue

            # Prefer the searoute matrix; fall back to great-circle × 1.3.
            d_nm = (sea_dist_nm.get((pi, pj))
                    or sea_dist_nm.get((pj, pi)))
            if d_nm is None:
                lat1r, lat2r = radians(lats[i-1]), radians(lats[i])
                dlon = radians(lons[i] - lons[i-1])
                dlat = lat2r - lat1r
                a = sin(dlat/2)**2 + cos(lat1r)*cos(lat2r)*sin(dlon/2)**2
                d_nm = 6371 * 2 * atan2(sqrt(a), sqrt(1-a)) * 0.5399 * 1.3

            if d_nm > 0:
                speed = d_nm / t
                if 3.0 <= speed <= 40.0:
                    speed_obs[(cluster_id, vtype)].append(speed)
                    speed_obs[(cluster_id,)].append(speed)

    speed_median = {}
    speed_p5     = {}
    speed_p95    = {}
    speed_params = {}
    for key, speeds in speed_obs.items():
        if len(speeds) >= 5:
            arr     = np.array(speeds)
            log_arr = np.log(arr)
            speed_median[key] = float(np.median(arr))
            speed_p5[key]     = float(np.percentile(arr, 5))
            speed_p95[key]    = float(np.percentile(arr, 95))
            speed_params[key] = {
                'mu':    float(log_arr.mean()),
                'sigma': float(max(log_arr.std(), 0.10)),
            }

    return speed_median, speed_p5, speed_p95, speed_params


def train_models(df: pd.DataFrame,
                 sea_dist_nm: dict,
                 nz_ports: set) -> dict[int, MarkovModel]:
    """
    Train an independent MarkovModel for each cluster.
    Returns {cluster_id: MarkovModel}.
    """
    print("\n[Train] Fitting models...")
    models = {}
    n_clusters = df['cluster'].nunique()

    for k in range(n_clusters):
        print(f"  Cluster {k}...", end=' ')
        voyages = df[df['cluster'] == k].copy()
        model   = MarkovModel(k)
        model.nz_ports = nz_ports

        # Transition matrices (2nd-order primary + 1st-order fallback).
        all_ports = list({p for ports in voyages['ports'] for p in ports})
        model.transition_2, model.transition = build_transition_matrix(
            voyages, sea_dist_nm, all_ports
        )

        # Transit-time distributions.
        model.transit_models = fit_transit_models(voyages)

        # Dwell-time distributions + per-port adjustment factors.
        model.dwell_models, model.port_dwell_factors = fit_dwell_models(voyages)

        # Vessel-speed lookup (median + p5/p95 + lognormal params).
        model.vessel_speed, model.vessel_speed_p5, \
        model.vessel_speed_p95, model.vessel_speed_params = \
            compute_vessel_speeds(voyages, sea_dist_nm)

        # Start-port distribution.
        start_counts = defaultdict(int)
        for _, row in voyages.iterrows():
            start_counts[row['ports'][0]] += 1
        total = sum(start_counts.values())
        model.start_port_dist = {p: c/total for p, c in start_counts.items()}

        # NZ-destination empirical distribution (used as fallback when force_nz
        # degenerates, replacing uniform-random selection).
        nz_dest_counts = defaultdict(int)
        for _, row in voyages.iterrows():
            last = row['ports'][-1]
            if last in nz_ports:
                nz_dest_counts[last] += 1
        nz_dest_total = sum(nz_dest_counts.values())
        if nz_dest_total > 0:
            model.nz_dest_dist = {p: c / nz_dest_total for p, c in nz_dest_counts.items()}
        else:
            # Edge case: no NZ destination records in this cluster — fall back
            # to a uniform distribution over all NZ ports.
            model.nz_dest_dist = {p: 1.0 / len(nz_ports) for p in nz_ports}

        # Vessel-type distribution.
        vtype_counts = voyages['nbic_type_group'].value_counts()
        total_v = vtype_counts.sum()
        model.vessel_types     = vtype_counts.index.tolist()
        model.vessel_type_dist = {v: c/total_v for v, c in vtype_counts.items()}

        # DWT pool (grouped by vessel type).
        for vtype in model.vessel_types:
            dwts = voyages[voyages['nbic_type_group'] == vtype]['DWT'].dropna().values
            if len(dwts) > 0:
                model.dwt_pool[vtype] = dwts

        n_trans = sum(1 for k2 in model.transit_models if len(k2) == 4)
        n_dwell = sum(1 for k2 in model.dwell_models if k2[1] != 'all')
        nz_dest_str = ', '.join(f"{p}:{v:.2f}" for p, v in
                                sorted(model.nz_dest_dist.items(), key=lambda x: -x[1]))
        print(f"{len(voyages)} voyages, "
              f"{len(model.transition_2)} 2nd-order states, {len(model.transition)} 1st-order fallback states, "
              f"{n_trans} transit port-pair models, "
              f"{n_dwell} dwell models, "
              f"{len(model.port_dwell_factors)} port adjustment factors\n"
              f"          NZ destination distribution ({len(model.nz_dest_dist)} ports): {nz_dest_str}")

        models[k] = model

    return models


# ══════════════════════════════════════════════
# 4. Sampling functions
# ══════════════════════════════════════════════

def sample_transit(model: MarkovModel,
                   vtype: str,
                   port_i: str,
                   port_j: str,
                   sea_dist_nm: dict) -> float:
    """
    Sample the transit time (hours) from port_i to port_j.

    Two paths:
      1. The port pair has sufficient data (≥10 records) → sample directly from
         the transit-time LogNormal.
      2. Insufficient data → sample from the speed LogNormal,
         transit = d_nm / speed (distance-independent, unbiased).

    Physical bounds are set by p95 speed (lower bound on transit) and p5 speed
    (upper bound on transit), consistent with Layer 2, and apply to both paths.
    """
    k = model.cluster_id

    # Look up the sea distance.
    d_nm = (sea_dist_nm.get((port_i, port_j))
            or sea_dist_nm.get((port_j, port_i))
            or 500)

    # Physical bounds (p95 speed sets the lower bound on transit time, p5 speed
    # sets the upper bound; consistent with Layer 2).
    speed_key    = (k, vtype) if (k, vtype) in model.vessel_speed_p95 else (k,)
    p95_speed    = model.vessel_speed_p95.get(speed_key,
                   model.vessel_speed_p95.get((k,), 18.0))
    p5_speed     = model.vessel_speed_p5.get(speed_key,
                   model.vessel_speed_p5.get((k,), 3.0))
    median_speed = model.vessel_speed.get(speed_key,
                   model.vessel_speed.get((k,), 14.0))   # Only used as a final fallback return value

    min_transit = d_nm / (p95_speed * TRANSIT_SPEED_BUFFER)  # Shortest: p95 speed + 5% buffer
    max_transit = d_nm / p5_speed                             # Longest: p5 speed

    # ── Path 1: port-pair-level data available → transit-time LogNormal ──
    key = (k, vtype, port_i, port_j)
    if key in model.transit_models:
        params = model.transit_models[key]
        for _ in range(RESAMPLE_LIMIT):
            sample = np.random.lognormal(mean=params['mu'], sigma=params['sigma'])
            if min_transit <= sample <= max_transit:
                return sample
        return float(np.clip(np.exp(params['mu']), min_transit, max_transit))

    # ── Path 2: fallback → sample from the speed LogNormal (distance-independent, unbiased) ──
    sp_key = (k, vtype) if (k, vtype) in model.vessel_speed_params else (k,)
    if sp_key in model.vessel_speed_params:
        sp = model.vessel_speed_params[sp_key]
        for _ in range(RESAMPLE_LIMIT):
            speed   = np.random.lognormal(mean=sp['mu'], sigma=sp['sigma'])
            transit = d_nm / speed
            if min_transit <= transit <= max_transit:
                return transit

    # No data at all: return the physical median.
    return d_nm / median_speed


def sample_dwell(model: MarkovModel,
                 vtype: str,
                 port_i: str) -> float:
    """
    Sample the dwell time (hours) at port_i.
    Two-level fallback + hard bounds + per-port adjustment factor.
    """
    k = model.cluster_id

    # Two-level fallback.
    key = (k, vtype)
    if key not in model.dwell_models:
        key = (k, 'all')
    if key not in model.dwell_models:
        return 24.0  # No data at all → return 24 hours.

    params = model.dwell_models[key]
    factor = model.port_dwell_factors.get(port_i, 1.0)

    # Truncated resampling.
    for _ in range(RESAMPLE_LIMIT):
        sample   = np.random.lognormal(mean=params['mu'], sigma=params['sigma'])
        adjusted = sample * factor
        if DWELL_MIN_H <= adjusted <= DWELL_MAX_H:
            return adjusted

    # Out of bounds: return the distribution median × factor (clipped to bounds).
    median = np.exp(params['mu']) * factor
    return float(np.clip(median, DWELL_MIN_H, DWELL_MAX_H))


def sample_next_port(model: MarkovModel,
                     current_port: str,
                     visited_ports: set,
                     nz_ports: set,
                     force_nz: bool = False,
                     prev_port: Optional[str] = None) -> Optional[str]:
    """
    Sample the next port from the transition matrix.

    Lookup priority:
      1. 2nd-order matrix transition_2[(prev_port, current_port)] (if prev_port
         is known and the state exists).
      2. 1st-order fallback matrix transition[current_port].

    - Excludes already-visited ports (no repeats).
    - When force_nz=True, the next port is forced to be an NZ port.
    """
    # Determine the candidate distribution: prefer 2nd-order, fall back to 1st-order.
    candidates = None
    if prev_port is not None:
        candidates = model.transition_2.get((prev_port, current_port))
    if candidates is None:
        candidates = model.transition.get(current_port)
    if candidates is None:
        return None

    if force_nz:
        # Forced termination: pick from NZ ports by their transition probability.
        nz_candidates = {p: w for p, w in candidates.items() if p in nz_ports}
        if not nz_candidates:
            # No NZ candidates in the transition matrix → sample by the empirical
            # NZ destination frequency (replaces uniform-random fallback).
            if model.nz_dest_dist:
                nz_list  = list(model.nz_dest_dist.keys())
                nz_probs = [model.nz_dest_dist[p] for p in nz_list]
                return str(np.random.choice(nz_list, p=nz_probs))
            return random.choice(list(nz_ports)) if nz_ports else None
        ports_list = list(nz_candidates.keys())
        weights    = [nz_candidates[p] for p in ports_list]
        s = sum(weights)
        probs = [w/s for w in weights]
        return str(np.random.choice(ports_list, p=probs))

    # Normal sampling: exclude already-visited ports.
    valid = {p: w for p, w in candidates.items() if p not in visited_ports}
    if not valid:
        return None

    ports_list = list(valid.keys())
    weights    = [valid[p] for p in ports_list]
    s = sum(weights)
    probs = [w/s for w in weights]
    return str(np.random.choice(ports_list, p=probs))


def compute_delta_t(transits_hours: list, dwells_hours: list) -> float:
    """
    Compute the cumulative Δt (hours) from the first port (index 0) to the NZ
    destination port (index n-1).

    Formula: Δt = Σ transit_hours[1:n] + Σ dwell_hours[1:n-1]
    (excludes the first-port dwell and the NZ-destination dwell).
    """
    n = len(transits_hours)
    if n < 2:
        return 0.0
    delta_t  = sum(transits_hours[1:])      # All transit legs.
    delta_t += sum(dwells_hours[1:n-1])     # Intermediate port dwells (excludes first port and NZ destination).
    return delta_t


# ══════════════════════════════════════════════
# 5. Single voyage generation
# ══════════════════════════════════════════════

def generate_one_voyage(model: MarkovModel,
                        sea_dist_nm: dict,
                        port_coords: dict,
                        port_territories: dict,
                        port_ecoprovince: dict,
                        target_n_ports: int) -> Optional[dict]:
    """
    Generate one synthetic voyage.
    Returns a dict of voyage fields, or None on failure.
    """
    nz_ports = model.nz_ports

    # Sample vessel type.
    vtypes  = list(model.vessel_type_dist.keys())
    vprobs  = [model.vessel_type_dist[v] for v in vtypes]
    vtype = str(np.random.choice(vtypes, p=vprobs))

    # Sample the start port (must not be an NZ port).
    start_candidates = {p: w for p, w in model.start_port_dist.items()
                        if p not in nz_ports}
    if not start_candidates:
        return None
    start_list  = list(start_candidates.keys())
    start_probs = [start_candidates[p] for p in start_list]
    s = sum(start_probs)
    start_probs = [p/s for p in start_probs]
    start_port = str(np.random.choice(start_list, p=start_probs))

    # Initialise the sequence.
    ports        = [start_port]
    transits     = [0.0]   # First-port transit is 0.
    dwells       = [sample_dwell(model, vtype, start_port)]
    last_port    = start_port   # Only the immediately previous port is forbidden (revisits allowed).
    total_hours  = dwells[0]

    # Extend step by step.
    for step in range(1, target_n_ports):
        current      = ports[-1]
        prev_port    = ports[-2] if len(ports) >= 2 else None   # Previous port required for the 2nd-order state.
        is_last_step = (step == target_n_ports - 1)

        force_nz = is_last_step

        next_port = sample_next_port(model, current, {last_port}, nz_ports,
                                     force_nz=force_nz, prev_port=prev_port)
        if next_port is None:
            # Cannot continue extending; force termination at an NZ port.
            next_port = sample_next_port(model, current, set(), nz_ports,
                                         force_nz=True, prev_port=prev_port)
            if next_port is None:
                return None

        # Sample transit time.
        t_hours = sample_transit(model, vtype, current, next_port, sea_dist_nm)

        # Check duration constraint.
        if (total_hours + t_hours) / 24 > MAX_SPAN_DAYS:
            # Duration exceeded: try to jump directly to an NZ port.
            next_port = sample_next_port(model, current, set(), nz_ports,
                                         force_nz=True, prev_port=prev_port)
            if next_port is None:
                return None
            t_hours = sample_transit(model, vtype, current, next_port, sea_dist_nm)

        ports.append(next_port)
        transits.append(t_hours)
        total_hours += t_hours
        last_port = next_port   # Update: only the immediately previous port is tracked, to prevent consecutive revisits.

        # Sample dwell (NZ destination port: no dwell, or dwell that is excluded from Δt).
        if next_port in nz_ports:
            dwells.append(0.0)  # NZ destination dwell is excluded from Δt.
            break
        else:
            d_hours = sample_dwell(model, vtype, next_port)
            dwells.append(d_hours)
            total_hours += d_hours

    # Ensure the terminus is an NZ port.
    if ports[-1] not in nz_ports:
        return None

    # Ensure the number of ports is within range.
    n_ports = len(ports)
    if n_ports < MIN_PORTS:
        return None

    # Extract coordinates, ecoprovince, and territory info.
    lats   = [port_coords[p][1] for p in ports]
    lons   = [port_coords[p][0] for p in ports]
    terrs  = [port_territories.get(p, '') for p in ports]
    ecos   = [port_ecoprovince.get(p, '') for p in ports]

    # Total voyage duration in days.
    total_span = (sum(transits) + sum(dwells)) / 24.0

    # Final check against the hard 60-day constraint.
    if total_span > MAX_SPAN_DAYS:
        return None

    # Cumulative Δt (used for P_intro).
    delta_t = compute_delta_t(transits, dwells)

    # DWT sampling.
    dwt_pool = model.dwt_pool.get(vtype)
    if dwt_pool is None or len(dwt_pool) == 0:
        # Fallback: use DWTs from all vessel types within the cluster.
        all_dwts = np.concatenate(list(model.dwt_pool.values())) if model.dwt_pool else np.array([50000])
        dwt = float(np.random.choice(all_dwts))
    else:
        dwt = float(np.random.choice(dwt_pool))

    return {
        'vessel_type':     vtype,
        'DWT':             dwt,
        'n_ports':         n_ports,
        'ports':           ports,
        'port_territories': terrs,
        'latitudes':       lats,
        'longitudes':      lons,
        'ecoprovinces':    ecos,
        'transits_hours':  transits,
        'dwells_hours':    dwells,
        'total_span_days': round(total_span, 4),
        'delta_t_hours':   round(delta_t, 4),
        'nz_dest_port':    ports[-1],
    }


# ══════════════════════════════════════════════
# 6. Batch generation
# ══════════════════════════════════════════════

def generate_voyages(models: Dict[int, MarkovModel],
                     sea_dist_nm: dict,
                     port_coords: dict,
                     port_territories: dict,
                     port_ecoprovince: dict,
                     df_observed: pd.DataFrame,
                     n_target: int = N_TARGET) -> pd.DataFrame:
    """
    Generate synthetic voyages in bulk, distributed across clusters.

    Stratified n_ports quota:
      Per cluster, compute exact quotas for each n_ports value from the
      observed distribution, and accept only voyages whose actual length
      matches the requested length exactly. This ensures the final
      distribution strictly matches the observed proportions and avoids the
      systematic under-representation of n=7 caused by 2nd-order matrix
      sparsity favouring shorter sequences.
    """
    print(f"\n[Generate] Target total: {n_target} voyages (stratified-quota mode)")

    # Allocate generation budget across clusters proportionally.
    cluster_sizes = df_observed['cluster'].value_counts().sort_index()
    n_per_cluster = {}
    for k in models:
        n_per_cluster[k] = max(10, int(n_target * cluster_sizes[k] / len(df_observed)))

    # Observed n_ports distribution — computed independently per cluster.
    nports_dist_by_cluster = {
        k: df_observed[df_observed['cluster'] == k]['n_ports']
                      .value_counts(normalize=True)
        for k in df_observed['cluster'].unique()
    }

    all_records = []
    voyage_id   = 1

    for k, model in models.items():
        n_gen = n_per_cluster[k]

        # ── Compute exact quotas ──────────────────────────────────────────
        _dist = nports_dist_by_cluster.get(
            k, nports_dist_by_cluster[next(iter(nports_dist_by_cluster))]
        )
        quota = {}
        for n_val in _dist.index:
            quota[int(n_val)] = max(1, round(n_gen * _dist[n_val]))

        # Correct rounding error so that the quotas sum exactly to n_gen.
        diff = n_gen - sum(quota.values())
        if diff != 0:
            largest = max(quota, key=quota.get)
            quota[largest] += diff

        filled   = {n_val: 0 for n_val in quota}
        success  = 0
        attempts = 0
        # Allocate enough attempts overall (n=7 is sparse in the 2nd-order
        # matrix and needs more tries).
        max_attempts = n_gen * 50
        records  = []

        print(f"  Cluster {k}: target {n_gen}  quota={dict(sorted(quota.items()))}...",
              end=' ', flush=True)

        while success < n_gen and attempts < max_attempts:
            # Among the lengths that still have unfilled quota, choose target_n
            # weighted by remaining demand.
            unfilled = {n_val: quota[n_val] - filled[n_val]
                        for n_val in quota
                        if filled[n_val] < quota[n_val]}
            if not unfilled:
                break

            n_vals    = list(unfilled.keys())
            remaining = list(unfilled.values())
            total_rem = sum(remaining)
            probs     = [r / total_rem for r in remaining]
            target_n  = int(np.random.choice(n_vals, p=probs))

            attempts += 1
            result = generate_one_voyage(
                model, sea_dist_nm,
                port_coords, port_territories, port_ecoprovince,
                target_n
            )

            # Accept only voyages whose actual length matches the requested
            # length exactly, and only when that bucket's quota is still open.
            if result is None:
                continue
            actual_n = result['n_ports']
            if actual_n != target_n:
                continue
            if filled[actual_n] >= quota[actual_n]:
                continue

            filled[actual_n] += 1
            result['voyage_id'] = f'syn_{voyage_id:05d}'
            result['cluster']   = k
            records.append(result)
            voyage_id += 1
            success   += 1

        rate = success / attempts * 100 if attempts > 0 else 0
        fill_str = '  '.join(f"n={n}:{filled[n]}/{quota[n]}" for n in sorted(quota))
        print(f"generated {success} (success rate {rate:.1f}%, {attempts} attempts)\n"
              f"          Fill status: {fill_str}")
        all_records.extend(records)

    df_syn = pd.DataFrame(all_records)

    # Standardise column order.
    col_order = [
        'voyage_id', 'cluster', 'vessel_type', 'DWT',
        'n_ports', 'ports', 'port_territories', 'latitudes', 'longitudes',
        'ecoprovinces', 'transits_hours', 'dwells_hours',
        'total_span_days', 'delta_t_hours', 'nz_dest_port'
    ]
    df_syn = df_syn[[c for c in col_order if c in df_syn.columns]]

    print(f"\n[Generate] Done. Total synthetic voyages: {len(df_syn)}")
    return df_syn


# ══════════════════════════════════════════════
# 7. Output
# ══════════════════════════════════════════════

def save_outputs(df_syn: pd.DataFrame, output_dir: Path) -> None:
    """Save per-cluster CSVs and a combined CSV."""
    print(f"\n[Output] Saving to {output_dir}")

    # Per-cluster files.
    for k in sorted(df_syn['cluster'].unique()):
        subset  = df_syn[df_syn['cluster'] == k]
        outpath = output_dir / f'mc.syn{k}.csv'
        subset.to_csv(outpath, index=False)
        print(f"  mc.syn{k}.csv: {len(subset)} voyages")

    # Combined file.
    all_path = output_dir / 'mc.syn.all.csv'
    df_syn.to_csv(all_path, index=False)
    print(f"  mc.syn.all.csv: {len(df_syn)} voyages (combined)")

    # Quick summary.
    print(f"\n[Stats]")
    print(f"  Cluster distribution:\n{df_syn['cluster'].value_counts().sort_index().to_string()}")
    print(f"  Vessel type distribution:\n{df_syn['vessel_type'].value_counts().to_string()}")
    print(f"  n_ports distribution:\n{df_syn['n_ports'].value_counts().sort_index().to_string()}")
    print(f"  Total voyage days: mean={df_syn['total_span_days'].mean():.1f}, "
          f"median={df_syn['total_span_days'].median():.1f}, "
          f"max={df_syn['total_span_days'].max():.1f}")
    print(f"  Δt (hours): mean={df_syn['delta_t_hours'].mean():.1f}, "
          f"median={df_syn['delta_t_hours'].median():.1f}")


# ══════════════════════════════════════════════
# 8. Main
# ══════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Track 1: smoothed Markov-chain synthetic voyage generator")
    print("=" * 60)

    np.random.seed(42)
    random.seed(42)

    # Load data.
    df, sea_dist_nm = load_data(CSV_PATH, PKL_PATH)

    # Extract port info.
    port_coords, port_territories, port_ecoprovince = extract_port_coords(df)
    nz_ports = identify_nz_ports(df)

    # Train models.
    models = train_models(df, sea_dist_nm, nz_ports)

    # Generate voyages.
    df_syn = generate_voyages(
        models, sea_dist_nm,
        port_coords, port_territories, port_ecoprovince,
        df_observed=df,
        n_target=N_TARGET
    )

    # Save.
    save_outputs(df_syn, OUTPUT_DIR)

    print("\nDone.")


if __name__ == '__main__':
    main()