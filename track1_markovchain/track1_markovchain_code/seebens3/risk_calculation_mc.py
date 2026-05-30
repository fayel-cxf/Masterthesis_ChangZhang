"""
risk_calculation_mc.py
=======================
Track 1 Markov Chain — Bioinvasion Risk Assessment
Improved Seebens three-filter model (Seebens 2013/2016; Tzeng 2024)

Input:
  Synthetic voyages : data/markov/validation.result/mc.validation/layer2_speed_filter/mc.syn.all.layer2.csv
  Observed voyages  : data/clustering/cluster.m1.labels.k5.csv
  ESR matrix        : data/seebens/ecoprov_envdist_scaled.csv
  W_r regression    : data/seebens/log_regression_results.csv
  Sea distance      : track1_markovchain_code/mc_generator/sea_distance_matrix.pkl

Three risk filters per source port i → NZ destination:
  Filter 1  P_alien(i)  — logistic distance curve (Seebens 2013)
  Filter 2  P_intro(i)  — ballast water introduction (W_r log-regression)
  Filter 3  P_estab(i)  — ecological establishment (Tzeng 2024 ESR matrix)

Full voyage risk (complementary product across all source ports):
  P_j(Inv) = 1 − ∏_i [ 1 − P_alien(i) × P_intro(i) × P_estab(i) ]

Output:
  data/seebens/track1_risk_results.csv    — per-voyage risk scores
  data/seebens/track1_risk_summary.txt    — summary statistics
"""

import ast
import pickle
import re
import warnings
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════
# 0. Configuration
# ══════════════════════════════════════════════════════════

SCRIPT_DIR  = Path(__file__).resolve().parent
BASE_DIR    = SCRIPT_DIR.parent.parent.parent   # → track1_markovchain/

SYN_PATH    = BASE_DIR / 'data/markov/validation.result/mc.validation/layer2_speed_filter/mc.syn.all.layer2.csv'
OBS_PATH    = BASE_DIR / 'data/clustering/cluster.m1.labels.k5.csv'
ESR_PATH    = BASE_DIR / 'data/seebens/ecoprov_envdist_scaled.csv'
REG_PATH    = BASE_DIR / 'data/seebens/log_regression_results.csv'
SEA_PATH    = SCRIPT_DIR.parent / 'mc_generator/sea_distance_matrix.pkl'
OUT_DIR     = BASE_DIR / 'data/seebens'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Risk parameters ──────────────────────────────────────
BETA        = 8.0       # P_alien logistic slope
GAMMA       = 1000.0    # P_alien distance midpoint (km)
MU          = 0.02      # organism die-off rate (day⁻¹)
LAMBDA_BW   = 0.002     # ballast-water concentration constant (m⁻³)
RHO         = 1.0       # treatment efficiency (1.0 = untreated)
Z           = 0.95      # the fraction of port calls at which a nonzero ballast discharge event occurs
VR_COEFF    = 0.3       # V_r = VR_COEFF × DWT  (note: Seebens 2013 uses 0.25)
ALPHA       = 1.5e-4    # baseline establishment probability
ESR_INTRA   = 0.99      # same-province ESR (diagonal = 0 in matrix, override here)
                        # Non-diagonal max = 0.987; 0.99 ensures intra > inter-province

# Fallback GC correction when port pair missing from sea distance matrix
GC_CORRECTION = 1.3

# Regression significance threshold for fallback decision
P_SIG = 0.05

# ── NZ port → province ID ────────────────────────────────
NZ_PORT_PROV = {
    'AUCKLAND':       53,
    'TAURANGA':       53,
    'GISBORNE':       53,
    'WHANGAREI':      53,
    'NAPIER':         54,
    'NELSON':         54,
    'NEW PLYMOUTH':   54,
    'PORT CHALMERS':  54,
    'LYTTELTON':      54,
    'TIMARU':         54,
    'WELLINGTON':     54,
    'BLUFF':          54,
}

# ══════════════════════════════════════════════════════════
# 1. Ecoprovince name → ID mapping
# ══════════════════════════════════════════════════════════

ECOPROV_NAME_TO_ID: dict = {
    'Arctic':                                   1,
    'Northern European Seas':                   2,
    'Lusitanian':                               3,
    'Mediterranean Sea':                        4,
    'Cold Temperate Northwest Atlantic':        5,
    'Warm Temperate Northwest Atlantic':        6,
    'Black Sea':                                7,
    'Cold Temperate Northwest Pacific':         8,
    'Warm Temperate Northwest Pacific':         9,
    'Cold Temperate Northeast Pacific':         10,
    'Warm Temperate Northeast Pacific':         11,
    'Tropical Northwestern Atlantic':           12,
    'Tropical Southwestern Atlantic':           14,
    'West African Transition':                  16,
    'Gulf of Guinea':                           17,
    'Red Sea and Gulf of Aden':                 18,
    'Somali/Arabian':                           19,
    'Western Indian Ocean':                     20,
    'West and South Indian Shelf':              21,
    'Bay of Bengal':                            23,
    'Andaman':                                  24,
    'South China Sea':                          25,
    'Sunda Shelf':                              26,
    'Java Transitional':                        27,
    'South Kuroshio':                           28,
    'Tropical Northwestern Pacific':            29,
    'Western Coral Triangle':                   30,
    'Eastern Coral Triangle':                   31,
    'Sahul Shelf':                              32,
    'Northeast Australian Shelf':               33,
    'Northwest Australian Shelf':               34,
    'Tropical Southwestern Pacific':            35,
    'Lord Howe and Norfolk Islands':            36,
    'Hawaii':                                   37,
    'Marshall, Gilbert and Ellis Islands':      38,
    'Central Polynesia':                        39,
    'Southeast Polynesia':                      40,
    'Tropical East Pacific':                    43,
    'Warm Temperate Southeastern Pacific':      45,
    'Warm Temperate Southwestern Atlantic':     47,
    'Magellanic':                               48,
    'Benguela':                                 50,
    'Agulhas':                                  51,
    'Northern New Zealand':                     53,
    'Southern New Zealand':                     54,
    'East Central Australian Shelf':            55,
    'Southeast Australian Shelf':               56,
    'Southwest Australian Shelf':               57,
    'West Central Australian Shelf':            58,
}

# ══════════════════════════════════════════════════════════
# 2. Utilities
# ══════════════════════════════════════════════════════════

def safe_eval(val):
    if pd.isna(val):
        return []
    s = re.sub(r'np\.\w+\(([^)]+)\)', r'\1', str(val))
    try:
        return ast.literal_eval(s)
    except Exception:
        return []


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1r, lat2r = radians(lat1), radians(lat2)
    dlat = lat2r - lat1r
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(lat1r) * cos(lat2r) * sin(dlon / 2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


# ══════════════════════════════════════════════════════════
# 3. Load data & build lookup tables
# ══════════════════════════════════════════════════════════

print('=' * 65)
print('Track 1 Markov Chain — Risk Assessment')
print('=' * 65)

# ── ESR matrix ───────────────────────────────────────────
print('\n[Load] ESR matrix...')
esr_df = pd.read_csv(ESR_PATH, index_col=0)
esr_df.index   = esr_df.index.astype(int)
esr_df.columns = esr_df.columns.astype(int)
print(f'       {esr_df.shape[0]}×{esr_df.shape[1]} province pairs')

# ── W_r regression model ─────────────────────────────────
print('\n[Load] W_r regression coefficients...')
reg = pd.read_csv(REG_PATH)
reg_model: dict = {}

sig_mask = reg['p_value'] < P_SIG
sig_types = reg[sig_mask]
fallback_intercept = float(sig_types['intercept'].mean())
fallback_slope     = float(sig_types['slope'].mean())

for _, r in reg.iterrows():
    if r['p_value'] < P_SIG:
        reg_model[r['group']] = (float(r['intercept']), float(r['slope']))
        status = 'significant'
    else:
        reg_model[r['group']] = (fallback_intercept, fallback_slope)
        status = f'NON-SIGNIFICANT (p={r["p_value"]:.3f}) → using fallback'
    print(f'       {r["group"]:15s}  intercept={r["intercept"]:7.4f}  '
          f'slope={r["slope"]:7.4f}  {status}')

reg_model['_default'] = (fallback_intercept, fallback_slope)
print(f'\n       Fallback (mean of significant):  '
      f'intercept={fallback_intercept:.4f}  slope={fallback_slope:.4f}')

# ── Sea distance matrix ───────────────────────────────────
print('\n[Load] Sea distance matrix...')
try:
    with open(SEA_PATH, 'rb') as f:
        sea_raw = pickle.load(f)
    sea_nm = sea_raw['nm']
    print(f'       {len(sea_nm):,} port pairs loaded (nm)')
except Exception as e:
    sea_nm = {}
    print(f'       WARNING: could not load ({e}) — using GC fallback only')

# ── Observed voyages ─────────────────────────────────────
print('\n[Load] Observed voyages...')
LIST_COLS = ['ports', 'latitudes', 'longitudes',
             'ecoprovinces', 'transits_hours', 'dwells_hours']
obs = pd.read_csv(OBS_PATH)
for col in LIST_COLS:
    if col in obs.columns:
        obs[col] = obs[col].apply(safe_eval)
# Observed uses 'nbic_type_group'; normalise to 'vessel_type'
if 'nbic_type_group' in obs.columns and 'vessel_type' not in obs.columns:
    obs = obs.rename(columns={'nbic_type_group': 'vessel_type'})
print(f'       {len(obs)} observed voyages')

# Build port → (lat, lon) lookup from observed data
port_coords: dict = {}
for row in obs.itertuples(index=False):
    for p, la, lo in zip(row.ports, row.latitudes, row.longitudes):
        if p not in port_coords:
            try:
                port_coords[p] = (float(la), float(lo))
            except (TypeError, ValueError):
                pass

# ── Synthetic voyages (already Layer 2 filtered) ─────────
print('\n[Load] Synthetic voyages (Layer 2 filtered)...')
syn = pd.read_csv(SYN_PATH)
for col in LIST_COLS:
    if col in syn.columns:
        syn[col] = syn[col].apply(safe_eval)

# Supplement port_coords from synthetic data
for row in syn.itertuples(index=False):
    for p, la, lo in zip(row.ports, row.latitudes, row.longitudes):
        if p not in port_coords:
            try:
                port_coords[p] = (float(la), float(lo))
            except (TypeError, ValueError):
                pass

print(f'       {len(syn)} synthetic voyages')
print(f'       {len(port_coords):,} unique ports with coordinates')

# ══════════════════════════════════════════════════════════
# 4. Distance helper
# ══════════════════════════════════════════════════════════

def get_dist_km(port_a: str, port_b: str):
    """
    Sea distance (km) between two ports.
    Priority: sea_distance_matrix (nm → km) → haversine × GC_CORRECTION.
    Returns None if coordinates are unavailable.
    """
    for key in [(port_a, port_b), (port_b, port_a)]:
        if key in sea_nm:
            return float(sea_nm[key]) / 0.5399   # nm → km
    # fallback
    ca = port_coords.get(port_a)
    cb = port_coords.get(port_b)
    if ca is None or cb is None:
        return None
    return haversine_km(ca[0], ca[1], cb[0], cb[1]) * GC_CORRECTION

# ══════════════════════════════════════════════════════════
# 5. Three-filter risk functions
# ══════════════════════════════════════════════════════════

def p_alien(src_port: str, nz_port: str) -> float:
    """
    Filter 1: logistic distance curve (Seebens 2013).
    P_alien = 1 / (1 + exp(-β × (d_km − γ) / γ))

    Uses pure great-circle (Haversine) distance — P_alien is a biogeographic
    measure of species pool dissimilarity, independent of actual shipping routes.
    Sea-route distance is only appropriate for transit-time calculations (P_intro).
    """
    ca = port_coords.get(src_port)
    cb = port_coords.get(nz_port)
    if ca is None or cb is None:
        return 0.5   # uninformative fallback when coords missing
    d = haversine_km(ca[0], ca[1], cb[0], cb[1])
    return float(1.0 / (1.0 + np.exp(-BETA * (d - GAMMA) / GAMMA)))


def predict_wr(vessel_type: str, dwt: float) -> float:
    """W_r (m³) from log-linear regression: W_r = exp(intercept) × DWT^slope."""
    intercept, slope = reg_model.get(vessel_type, reg_model['_default'])
    return float(np.exp(intercept + slope * np.log(max(dwt, 1.0))))


def p_intro(delta_t_hours: float, delta_r: int,
            dwt: float, vessel_type: str) -> float:
    """
    Filter 2: ballast water introduction probability (Seebens 2013).

    delta_t_hours : cumulative transit + dwell time from source port to NZ
    delta_r       : number of intermediate stops between source port and NZ

    Formula:
      W_r  = exp(intercept) × DWT^slope          (regression)
      V_r  = VR_COEFF × DWT                      (ballast tank volume)
      B_r  = Z × W_r × (1 − Z×W_r/V_r)^delta_r  (surviving organisms)
      P_intro = RHO × (1 − exp(−λ × B_r)) × exp(−μ × delta_t_days)
    """
    W_r = predict_wr(vessel_type, dwt)
    V_r = VR_COEFF * max(dwt, 1.0)
    if V_r <= 0 or W_r <= 0:
        return 0.0
    exchange_ratio = min(Z * W_r / V_r, 0.9999)
    B_r  = Z * W_r * ((1.0 - exchange_ratio) ** delta_r)
    prob = RHO * (1.0 - np.exp(-LAMBDA_BW * B_r)) * np.exp(-MU * delta_t_hours / 24.0)
    return float(np.clip(prob, 0.0, 1.0))


def p_estab(src_eco: str, nz_eco: str) -> float:
    """
    Filter 3: ecological establishment probability (Tzeng 2024 ESR).
    Higher ESR value = more similar environments = higher P_estab.

    P_estab = α × ESR(src_province, nz_province)
    """
    src_id = ECOPROV_NAME_TO_ID.get(src_eco)
    nz_id  = ECOPROV_NAME_TO_ID.get(nz_eco)
    if src_id is None or nz_id is None:
        return 0.0
    if src_id == nz_id:
        esr_val = ESR_INTRA
    elif src_id in esr_df.index and nz_id in esr_df.columns:
        esr_val = float(esr_df.loc[src_id, nz_id])
    else:
        return 0.0
    return float(ALPHA * esr_val)


# ══════════════════════════════════════════════════════════
# 6. Voyage-level risk computation
# ══════════════════════════════════════════════════════════

def compute_voyage_risk(ports: list, ecoprovinces: list,
                        transits: list, dwells: list,
                        n_ports: int, dwt: float,
                        vessel_type: str) -> dict:
    """
    Compute per-source-port risk components and full voyage invasion probability.

    Source ports: indices 0 … n_ports-2
    NZ destination: index n_ports-1

    Returns dict with per-port lists and aggregate P_j(Inv).
    """
    n       = n_ports - 1          # index of NZ destination
    nz_port = ports[n]
    nz_eco  = ecoprovinces[n] if n < len(ecoprovinces) else ''

    pa_list, pi_list, pe_list, pr_list = [], [], [], []

    for i in range(n):
        src_port = ports[i]
        src_eco  = ecoprovinces[i] if i < len(ecoprovinces) else ''

        # Filter 1: P_alien
        pa = p_alien(src_port, nz_port)

        # Filter 2: P_intro
        # delta_t = cumulative transit + dwell from port i to NZ (excluding NZ dwell)
        delta_t_h = sum(transits[i + 1: n + 1]) + sum(dwells[i + 1: n])
        delta_r   = n - i - 1   # intermediate stops between source and NZ
        pi = p_intro(delta_t_h, delta_r, dwt, vessel_type)

        # Filter 3: P_estab
        pe = p_estab(src_eco, nz_eco)

        pr = pa * pi * pe

        pa_list.append(round(pa, 6))
        pi_list.append(round(pi, 8))
        pe_list.append(round(pe, 8))
        pr_list.append(round(pr, 10))

    # Full voyage invasion probability (complementary product)
    p_inv = float(1.0 - np.prod([1.0 - p for p in pr_list]))

    return {
        'p_alien_per_port': pa_list,
        'p_intro_per_port': pi_list,
        'p_estab_per_port': pe_list,
        'pr_inv_per_port':  pr_list,
        'p_invasion':       round(p_inv, 10),
    }

# ══════════════════════════════════════════════════════════
# 7. Process observed voyages
# ══════════════════════════════════════════════════════════

print('\n[Risk] Processing observed voyages...')
obs_results = []
obs_skipped = 0

for row in obs.itertuples(index=False):
    n_ports = int(row.n_ports)
    if n_ports < 2 or len(row.ports) != n_ports:
        obs_skipped += 1
        continue

    risk = compute_voyage_risk(
        ports        = row.ports,
        ecoprovinces = row.ecoprovinces,
        transits     = row.transits_hours,
        dwells       = row.dwells_hours,
        n_ports      = n_ports,
        dwt          = float(row.DWT),
        vessel_type  = str(row.vessel_type),
    )
    obs_results.append({
        'source':       'observed',
        'voyage_id':    row.voyage_id,
        'vessel_type':  row.vessel_type,
        'cluster':      row.cluster,
        'dwt':          row.DWT,
        'n_ports':      n_ports,
        'nz_dest_port': row.nz_dest_port,
        **risk,
    })

obs_out = pd.DataFrame(obs_results)
print(f'       Processed: {len(obs_out):,}  skipped: {obs_skipped}')
print(f'       p_invasion  mean={obs_out["p_invasion"].mean():.4e}  '
      f'median={obs_out["p_invasion"].median():.4e}  '
      f'max={obs_out["p_invasion"].max():.4e}')

# ══════════════════════════════════════════════════════════
# 8. Process synthetic voyages
# ══════════════════════════════════════════════════════════

print('\n[Risk] Processing synthetic voyages...')
syn_results = []
syn_skipped = 0

for row in syn.itertuples(index=False):
    n_ports = int(row.n_ports)
    if n_ports < 2 or len(row.ports) != n_ports:
        syn_skipped += 1
        continue

    risk = compute_voyage_risk(
        ports        = row.ports,
        ecoprovinces = row.ecoprovinces,
        transits     = row.transits_hours,
        dwells       = row.dwells_hours,
        n_ports      = n_ports,
        dwt          = float(row.DWT),
        vessel_type  = str(row.vessel_type),
    )
    syn_results.append({
        'source':       'synthetic',
        'voyage_id':    row.voyage_id,
        'vessel_type':  row.vessel_type,
        'cluster':      row.cluster,
        'dwt':          row.DWT,
        'n_ports':      n_ports,
        'nz_dest_port': row.nz_dest_port,
        **risk,
    })

syn_out = pd.DataFrame(syn_results)
print(f'       Processed: {len(syn_out):,}  skipped: {syn_skipped}')
print(f'       p_invasion  mean={syn_out["p_invasion"].mean():.4e}  '
      f'median={syn_out["p_invasion"].median():.4e}  '
      f'max={syn_out["p_invasion"].max():.4e}')

# ══════════════════════════════════════════════════════════
# 9. Save results
# ══════════════════════════════════════════════════════════

print('\n[Save] Writing output files...')

combined = pd.concat([obs_out, syn_out], ignore_index=True)
results_path = OUT_DIR / 'track1_risk_results.csv'
combined.to_csv(results_path, index=False)
print(f'       {results_path.name}  ({len(combined):,} rows)')

# ── Summary report ────────────────────────────────────────
lines = []
lines.append('=' * 65)
lines.append('Track 1 Markov Chain — Risk Assessment Summary')
lines.append('=' * 65)
lines.append('')
lines.append(f'Observed voyages processed : {len(obs_out):,}')
lines.append(f'Synthetic voyages processed: {len(syn_out):,}')
lines.append('')

for label, df in [('Observed', obs_out), ('Synthetic', syn_out)]:
    p = df['p_invasion']
    lines.append(f'{label} P_j(Inv):')
    lines.append(f'  mean   = {p.mean():.4e}')
    lines.append(f'  median = {p.median():.4e}')
    lines.append(f'  p95    = {p.quantile(0.95):.4e}')
    lines.append(f'  max    = {p.max():.4e}')
    lines.append('')

lines.append('Synthetic breakdown by vessel type:')
lines.append(f'  {"Vessel type":20} {"N":>6} {"mean P_inv":>12} {"max P_inv":>12}')
lines.append('  ' + '-' * 54)
for vt, grp in syn_out.groupby('vessel_type'):
    lines.append(f'  {vt:20} {len(grp):>6} '
                 f'{grp["p_invasion"].mean():>12.4e} '
                 f'{grp["p_invasion"].max():>12.4e}')

lines.append('')
lines.append('Synthetic breakdown by cluster:')
lines.append(f'  {"Cluster":>8} {"N":>6} {"mean P_inv":>12} {"max P_inv":>12}')
lines.append('  ' + '-' * 42)
for cl, grp in syn_out.groupby('cluster'):
    lines.append(f'  {cl:>8} {len(grp):>6} '
                 f'{grp["p_invasion"].mean():>12.4e} '
                 f'{grp["p_invasion"].max():>12.4e}')

report_text = '\n'.join(lines)
report_path = OUT_DIR / 'track1_risk_summary.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_text)

print(f'       {report_path.name}')
print()
print(report_text)
print('\nDone.')